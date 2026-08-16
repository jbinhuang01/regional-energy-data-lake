#!/usr/bin/env python3
"""Convert Open-Meteo weather Bronze JSON into a typed Silver Parquet dataset.

Silver keeps one row per location and UTC hour. It does not impute missing
weather values; missingness and time gaps are written to quality_report.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_latest_batch(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*"))
    if not batches:
        raise FileNotFoundError(f"No weather Bronze batches found under {root}")
    return batches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bronze-root",
        type=Path,
        default=Path("data_lake/bronze/weather_open_meteo"),
    )
    parser.add_argument(
        "--bronze-batch",
        type=Path,
        help="Specific weather Bronze batch. Defaults to the latest batch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/silver/weather_open_meteo"),
    )
    return parser.parse_args()


def count_missing_hour_intervals(times: pd.Series) -> int:
    ordered = times.dropna().sort_values().drop_duplicates()
    if len(ordered) < 2:
        return 0
    gaps = ordered.diff().dropna()
    return int((gaps > pd.Timedelta(hours=1)).sum())


def main() -> int:
    args = parse_args()
    bronze_batch = args.bronze_batch or find_latest_batch(args.bronze_root)
    batch_id = bronze_batch.name.removeprefix("batch_id=")
    output_dir = args.output_root / f"ingestion_date={batch_id[:8]}" / f"batch_id={batch_id}"

    if output_dir.exists():
        raise SystemExit(f"Silver output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    location_reports: list[dict] = []

    for raw_path in sorted(bronze_batch.glob("location=*/weather.json")):
        location = raw_path.parent.name.removeprefix("location=")
        request_path = raw_path.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        hourly = payload["hourly"]

        frame = pd.DataFrame(hourly)
        frame = frame.rename(
            columns={
                "time": "event_time_utc",
                "temperature_2m": "temperature_2m_c",
                "relative_humidity_2m": "relative_humidity_2m_pct",
                "precipitation": "precipitation_mm",
                "wind_speed_10m": "wind_speed_10m_kmh",
                "shortwave_radiation": "shortwave_radiation_w_m2",
            }
        )
        frame["event_time_utc"] = pd.to_datetime(
            frame["event_time_utc"], errors="coerce", utc=True
        )
        frame["location"] = location
        frame["latitude"] = request["latitude"]
        frame["longitude"] = request["longitude"]
        frame["source"] = "open_meteo"
        frame["batch_id"] = batch_id
        frame["year"] = frame["event_time_utc"].dt.year.astype("int16")
        frame["month"] = frame["event_time_utc"].dt.month.astype("int8")

        weather_columns = [
            "temperature_2m_c",
            "relative_humidity_2m_pct",
            "precipitation_mm",
            "wind_speed_10m_kmh",
            "shortwave_radiation_w_m2",
            "weather_code",
        ]
        for column in weather_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        invalid_timestamps = int(frame["event_time_utc"].isna().sum())
        duplicate_keys = int(frame.duplicated(subset=["location", "event_time_utc"]).sum())
        missing_values = {
            column: int(frame[column].isna().sum()) for column in weather_columns
        }
        location_reports.append(
            {
                "location": location,
                "rows": len(frame),
                "start_time_utc": frame["event_time_utc"].min().isoformat(),
                "end_time_utc": frame["event_time_utc"].max().isoformat(),
                "invalid_timestamp_count": invalid_timestamps,
                "duplicate_key_count": duplicate_keys,
                "missing_hour_intervals": count_missing_hour_intervals(frame["event_time_utc"]),
                "missing_values": missing_values,
                "api_units": payload.get("hourly_units", {}),
            }
        )

        rows.append(
            frame[
                [
                    "event_time_utc",
                    "location",
                    "latitude",
                    "longitude",
                    *weather_columns,
                    "source",
                    "batch_id",
                    "year",
                    "month",
                ]
            ]
        )

    if not rows:
        raise SystemExit(f"No weather JSON files found under {bronze_batch}")

    weather = pd.concat(rows, ignore_index=True)
    table = pa.Table.from_pandas(weather, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["year", "month", "location"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    report = {
        "layer": "silver",
        "dataset": "weather_open_meteo",
        "bronze_batch": str(bronze_batch),
        "silver_output": str(output_dir),
        "created_at_utc": utc_now(),
        "rows": len(weather),
        "locations": sorted(weather["location"].unique().tolist()),
        "location_reports": location_reports,
        "partitioning": ["year", "month", "location"],
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Weather Silver output: {output_dir}")
    print(f"Rows: {len(weather):,}")
    print(f"Quality report: {output_dir / 'quality_report.json'}")
    print("Weather Silver transformation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
