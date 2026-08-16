#!/usr/bin/env python3
"""Convert forecast-vintage Bronze JSON into an event-level Silver table.

The output keeps forecast run time and valid time separately. No actual
weather values are joined here; this layer represents only what the model run
predicted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


FORECAST_COLUMNS = [
    "temperature_forecast_c",
    "wind_speed_forecast_kmh",
    "shortwave_radiation_forecast_w_m2",
    "precipitation_forecast_mm",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def latest_batch(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*/run=*"))
    if not batches:
        raise FileNotFoundError(
            f"No forecast-vintage Bronze runs found under {root}. "
            "Run ingest_weather_vintage_bronze.py first."
        )
    return batches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bronze-root",
        type=Path,
        default=Path("data_lake/bronze/weather_forecast_vintage"),
    )
    parser.add_argument("--bronze-batch", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/silver/weather_forecast_vintage"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bronze_run = args.bronze_batch or latest_batch(args.bronze_root)
    batch_id = next(
        part.removeprefix("batch_id=")
        for part in bronze_run.parts
        if part.startswith("batch_id=")
    )
    run_id = bronze_run.name.removeprefix("run=")
    output_dir = (
        args.output_root
        / f"ingestion_date={batch_id[:8]}"
        / f"batch_id={batch_id}"
        / f"run={run_id}"
    )
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    location_reports: list[dict] = []
    for forecast_path in sorted(bronze_run.glob("location=*/forecast.json")):
        location = forecast_path.parent.name.removeprefix("location=")
        request = json.loads(
            (forecast_path.parent / "request.json").read_text(encoding="utf-8")
        )
        payload = json.loads(forecast_path.read_text(encoding="utf-8"))
        hourly = payload["hourly"]
        frame = pd.DataFrame(hourly).rename(
            columns={
                "time": "valid_time_utc",
                "temperature_2m": "temperature_forecast_c",
                "wind_speed_10m": "wind_speed_forecast_kmh",
                "shortwave_radiation": "shortwave_radiation_forecast_w_m2",
                "precipitation": "precipitation_forecast_mm",
            }
        )
        frame["valid_time_utc"] = pd.to_datetime(
            frame["valid_time_utc"], errors="coerce", utc=True
        )
        frame["forecast_run_utc"] = pd.to_datetime(
            request["forecast_run_utc"], errors="coerce", utc=True
        )
        frame["forecast_horizon_hours"] = (
            frame["valid_time_utc"] - frame["forecast_run_utc"]
        ).dt.total_seconds() / 3600
        frame["location"] = location
        frame["latitude"] = request["latitude"]
        frame["longitude"] = request["longitude"]
        frame["model"] = request["model"]
        frame["source"] = "open_meteo_single_runs"
        frame["batch_id"] = batch_id
        for column in FORECAST_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        invalid_time = int(frame["valid_time_utc"].isna().sum())
        invalid_horizon = int((frame["forecast_horizon_hours"] < 0).sum())
        duplicates = int(
            frame.duplicated(subset=["location", "forecast_run_utc", "valid_time_utc"]).sum()
        )
        location_reports.append(
            {
                "location": location,
                "rows": len(frame),
                "forecast_run_utc": request["forecast_run_utc"],
                "model": request["model"],
                "valid_start_utc": frame["valid_time_utc"].min().isoformat(),
                "valid_end_utc": frame["valid_time_utc"].max().isoformat(),
                "horizon_min_hours": float(frame["forecast_horizon_hours"].min()),
                "horizon_max_hours": float(frame["forecast_horizon_hours"].max()),
                "invalid_time_count": invalid_time,
                "negative_horizon_count": invalid_horizon,
                "duplicate_key_count": duplicates,
                "missing_forecast_values": {
                    column: int(frame[column].isna().sum()) for column in FORECAST_COLUMNS
                },
            }
        )
        frames.append(
            frame[
                [
                    "forecast_run_utc",
                    "valid_time_utc",
                    "forecast_horizon_hours",
                    "location",
                    "latitude",
                    "longitude",
                    "model",
                    *FORECAST_COLUMNS,
                    "source",
                    "batch_id",
                ]
            ]
        )

    if not frames:
        raise SystemExit(f"No forecast JSON files found under {bronze_run}")
    forecast = pd.concat(frames, ignore_index=True)
    forecast["valid_year"] = forecast["valid_time_utc"].dt.year.astype("int16")
    forecast["valid_month"] = forecast["valid_time_utc"].dt.month.astype("int8")
    forecast["run_date"] = forecast["forecast_run_utc"].dt.strftime("%Y-%m-%d")

    table = pa.Table.from_pandas(forecast, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["run_date", "valid_year", "valid_month", "location"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    report = {
        "layer": "silver",
        "dataset": "weather_forecast_vintage",
        "bronze_run": str(bronze_run),
        "silver_output": str(output_dir),
        "created_at_utc": utc_now(),
        "rows": len(forecast),
        "locations": sorted(forecast["location"].unique().tolist()),
        "run_count": int(forecast["forecast_run_utc"].nunique()),
        "location_reports": location_reports,
        "partitioning": ["run_date", "valid_year", "valid_month", "location"],
        "time_semantics": {
            "forecast_run_utc": "model initialization time",
            "valid_time_utc": "time being forecast",
            "forecast_horizon_hours": "valid_time minus forecast_run",
        },
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Forecast-vintage Silver output: {output_dir}")
    print(f"Rows: {len(forecast):,}")
    print(f"Report: {output_dir / 'quality_report.json'}")
    print("Forecast-vintage Silver transformation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
