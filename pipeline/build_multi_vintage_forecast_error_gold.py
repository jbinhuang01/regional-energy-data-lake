#!/usr/bin/env python3
"""Join multiple forecast runs to actual weather observations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


FORECAST_COLUMNS = {
    "temperature_forecast_c": "temperature_2m_c",
    "wind_speed_forecast_kmh": "wind_speed_10m_kmh",
    "shortwave_radiation_forecast_w_m2": "shortwave_radiation_w_m2",
    "precipitation_forecast_mm": "precipitation_mm",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-silver-batch", type=Path, required=True)
    parser.add_argument("--actual-weather-silver-batch", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data_lake/gold/forecast_error_panel")
    )
    parser.add_argument("--horizon-bin-hours", type=int, default=24)
    return parser.parse_args()


def batch_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("batch_id="):
            return part.removeprefix("batch_id=")
    raise ValueError(f"No batch_id= component in {path}")


def read_dataset(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {path}")
    return ds.dataset(
        [str(file) for file in files], format="parquet", partitioning="hive"
    ).to_table().to_pandas()


def main() -> int:
    args = parse_args()
    if args.horizon_bin_hours <= 0:
        raise SystemExit("--horizon-bin-hours must be positive")
    forecast_id = batch_id_from_path(args.forecast_silver_batch)
    actual_id = batch_id_from_path(args.actual_weather_silver_batch)
    output_dir = (
        args.output_root
        / f"forecast_batch={forecast_id}__actual_batch={actual_id}"
        / "multi_vintage"
    )
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")

    forecast = read_dataset(args.forecast_silver_batch)
    actual = read_dataset(args.actual_weather_silver_batch)
    required_forecast = {
        "forecast_run_utc",
        "valid_time_utc",
        "forecast_horizon_hours",
        "location",
        *FORECAST_COLUMNS.keys(),
    }
    required_actual = {"event_time_utc", "location", *FORECAST_COLUMNS.values()}
    missing_forecast = sorted(required_forecast - set(forecast.columns))
    missing_actual = sorted(required_actual - set(actual.columns))
    if missing_forecast or missing_actual:
        raise ValueError(
            f"Missing forecast columns={missing_forecast}; actual columns={missing_actual}"
        )

    for frame, column in [
        (forecast, "forecast_run_utc"),
        (forecast, "valid_time_utc"),
        (actual, "event_time_utc"),
    ]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    forecast["forecast_horizon_hours"] = pd.to_numeric(
        forecast["forecast_horizon_hours"], errors="coerce"
    )
    forecast_key = ["location", "forecast_run_utc", "valid_time_utc"]
    actual_key = ["location", "event_time_utc"]
    forecast_duplicates = int(forecast.duplicated(subset=forecast_key).sum())
    actual_duplicates = int(actual.duplicated(subset=actual_key).sum())
    if forecast_duplicates:
        raise ValueError(f"Forecast duplicate keys: {forecast_duplicates}")
    if actual_duplicates:
        raise ValueError(f"Actual duplicate keys: {actual_duplicates}")

    actual_for_join = actual.rename(columns={"event_time_utc": "valid_time_utc"})[
        ["location", "valid_time_utc", *FORECAST_COLUMNS.values(), "batch_id"]
    ].rename(
        columns={
            **{value: f"{value}_actual" for value in FORECAST_COLUMNS.values()},
            "batch_id": "actual_batch_id",
        }
    )
    joined = forecast.merge(
        actual_for_join,
        on=["location", "valid_time_utc"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    joined["actual_match_flag"] = joined["_merge"].eq("both")
    joined = joined.drop(columns=["_merge"])

    error_columns: list[str] = []
    for forecast_column, actual_column in FORECAST_COLUMNS.items():
        actual_name = f"{actual_column}_actual"
        stem = forecast_column.split("_forecast_")[0]
        error_name = f"{stem}_error"
        absolute_name = f"{stem}_absolute_error"
        joined[error_name] = joined[forecast_column] - joined[actual_name]
        joined[absolute_name] = joined[error_name].abs()
        error_columns.extend([error_name, absolute_name])

    joined["valid_year"] = joined["valid_time_utc"].dt.year.astype("Int16")
    joined["valid_month"] = joined["valid_time_utc"].dt.month.astype("Int8")
    joined["run_date"] = joined["forecast_run_utc"].dt.strftime("%Y-%m-%d")
    joined["horizon_bin_start_hours"] = (
        (joined["forecast_horizon_hours"] // args.horizon_bin_hours)
        * args.horizon_bin_hours
    ).astype("Int64")
    joined["horizon_bin"] = joined["horizon_bin_start_hours"].map(
        lambda value: (
            f"{int(value)}-{int(value) + args.horizon_bin_hours - 1}h"
            if pd.notna(value)
            else "unknown"
        )
    )
    actual_columns = [f"{column}_actual" for column in FORECAST_COLUMNS.values()]
    keep_columns = [
        "forecast_run_utc",
        "valid_time_utc",
        "forecast_horizon_hours",
        "horizon_bin",
        "horizon_bin_start_hours",
        "location",
        "latitude",
        "longitude",
        "model",
        "actual_match_flag",
        "batch_id",
        "actual_batch_id",
        *FORECAST_COLUMNS.keys(),
        *actual_columns,
        *error_columns,
        "run_date",
        "valid_year",
        "valid_month",
    ]
    gold = joined[keep_columns]
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.write_dataset(
        pa.Table.from_pandas(gold, preserve_index=False),
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["run_date", "valid_year", "valid_month", "location"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    metrics = []
    for forecast_column, actual_column in FORECAST_COLUMNS.items():
        stem = forecast_column.split("_forecast_")[0]
        error = joined[forecast_column] - joined[f"{actual_column}_actual"]
        valid = error.dropna()
        metrics.append(
            {
                "variable": stem,
                "n": int(len(valid)),
                "bias_forecast_minus_actual": float(valid.mean()) if len(valid) else None,
                "mae": float(valid.abs().mean()) if len(valid) else None,
                "rmse": float((valid.pow(2).mean()) ** 0.5) if len(valid) else None,
            }
        )
    pd.DataFrame(metrics).to_parquet(output_dir / "overall_error_metrics.parquet", index=False)

    horizon_rows = []
    for horizon, group in joined.groupby("horizon_bin", dropna=False, sort=True):
        for forecast_column, actual_column in FORECAST_COLUMNS.items():
            stem = forecast_column.split("_forecast_")[0]
            error = (group[forecast_column] - group[f"{actual_column}_actual"]).dropna()
            horizon_rows.append(
                {
                    "horizon_bin": horizon,
                    "horizon_bin_start_hours": group["horizon_bin_start_hours"].dropna().iloc[0]
                    if group["horizon_bin_start_hours"].notna().any()
                    else None,
                    "variable": stem,
                    "n": int(len(error)),
                    "mae": float(error.abs().mean()) if len(error) else None,
                    "rmse": float((error.pow(2).mean()) ** 0.5) if len(error) else None,
                }
            )
    pd.DataFrame(horizon_rows).to_parquet(output_dir / "horizon_error_metrics.parquet", index=False)

    report = {
        "layer": "gold",
        "dataset": "multi_vintage_forecast_error",
        "created_at_utc": utc_now(),
        "forecast_silver_batch": str(args.forecast_silver_batch),
        "actual_weather_silver_batch": str(args.actual_weather_silver_batch),
        "gold_output": str(output_dir),
        "rows_forecast": int(len(forecast)),
        "rows_actual": int(len(actual)),
        "rows_gold": int(len(gold)),
        "matched_rows": int(gold["actual_match_flag"].sum()),
        "match_rate": float(gold["actual_match_flag"].mean()) if len(gold) else None,
        "forecast_runs": int(gold["forecast_run_utc"].nunique()),
        "locations": sorted(gold["location"].dropna().unique().tolist()),
        "duplicate_key_count": forecast_duplicates,
        "actual_duplicate_key_count": actual_duplicates,
        "panel_note": (
            "Repeated valid_time_utc values across forecast_run_utc are intentional "
            "and represent forecast vintages."
        ),
        "error_definition": "forecast minus actual",
    }
    (output_dir / "error_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Multi-vintage forecast-error Gold output: {output_dir}")
    print(f"Rows: {len(gold):,}; runs: {report['forecast_runs']}; match: {report['match_rate']:.2%}")
    print(f"Report: {output_dir / 'error_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
