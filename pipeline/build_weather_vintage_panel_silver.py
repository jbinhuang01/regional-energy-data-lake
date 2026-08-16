#!/usr/bin/env python3
"""Normalize a multi-run forecast batch into event-level Silver rows."""

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
    batches = sorted(root.glob("ingestion_date=*/batch_id=*"))
    if not batches:
        raise FileNotFoundError(f"No forecast panel Bronze batches under {root}")
    return batches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-batch", type=Path)
    parser.add_argument(
        "--bronze-root",
        type=Path,
        default=Path("data_lake/bronze/weather_forecast_vintage"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/silver/weather_forecast_vintage_panel"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bronze_batch = args.bronze_batch or latest_batch(args.bronze_root)
    batch_id = next(
        part.removeprefix("batch_id=")
        for part in bronze_batch.parts
        if part.startswith("batch_id=")
    )
    output_dir = args.output_root / f"ingestion_date={batch_id[:8]}" / f"batch_id={batch_id}"
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")

    frames: list[pd.DataFrame] = []
    run_reports: list[dict] = []
    forecast_paths = sorted(bronze_batch.glob("run=*/location=*/forecast.json"))
    if not forecast_paths:
        raise SystemExit(f"No run/location forecast files under {bronze_batch}")

    for forecast_path in forecast_paths:
        location = forecast_path.parent.name.removeprefix("location=")
        run_id = forecast_path.parent.parent.name.removeprefix("run=")
        request = json.loads(
            (forecast_path.parent / "request.json").read_text(encoding="utf-8")
        )
        payload = json.loads(forecast_path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["hourly"]).rename(
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
        duplicates = int(
            frame.duplicated(
                subset=["location", "forecast_run_utc", "valid_time_utc"]
            ).sum()
        )
        run_reports.append(
            {
                "run": run_id,
                "location": location,
                "rows": len(frame),
                "valid_start_utc": frame["valid_time_utc"].min().isoformat(),
                "valid_end_utc": frame["valid_time_utc"].max().isoformat(),
                "duplicate_key_count": duplicates,
                "missing_forecast_values": {
                    column: int(frame[column].isna().sum())
                    for column in FORECAST_COLUMNS
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

    forecast = pd.concat(frames, ignore_index=True)
    forecast["valid_year"] = forecast["valid_time_utc"].dt.year.astype("int16")
    forecast["valid_month"] = forecast["valid_time_utc"].dt.month.astype("int8")
    forecast["run_date"] = forecast["forecast_run_utc"].dt.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.write_dataset(
        pa.Table.from_pandas(forecast, preserve_index=False),
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["run_date", "valid_year", "valid_month", "location"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )
    report = {
        "layer": "silver",
        "dataset": "weather_forecast_vintage_panel",
        "bronze_batch": str(bronze_batch),
        "silver_output": str(output_dir),
        "created_at_utc": utc_now(),
        "rows": len(forecast),
        "run_count": int(forecast["forecast_run_utc"].nunique()),
        "location_count": int(forecast["location"].nunique()),
        "locations": sorted(forecast["location"].unique().tolist()),
        "run_reports": run_reports,
        "time_semantics": {
            "forecast_run_utc": "model initialization time",
            "valid_time_utc": "time being forecast",
            "forecast_horizon_hours": "valid_time minus forecast_run",
        },
        "panel_note": "The same valid_time can occur in multiple forecast runs by design.",
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Forecast-vintage panel Silver output: {output_dir}")
    print(f"Rows: {len(forecast):,}; runs: {report['run_count']}")
    print(f"Report: {output_dir / 'quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
