#!/usr/bin/env python3
"""Build a forecast-error Gold mart from forecast and actual weather Silver.

This job compares one or more forecast runs with the actual weather observed
at the same location and valid UTC hour.  The forecast run time is kept
separate from the valid time so that forecast horizon remains available for
research and model evaluation.

Error convention:
    signed_error = forecast - actual
    absolute_error = abs(forecast - actual)

The job is intentionally conservative: unmatched rows remain in the Gold
table and are counted in error_report.json instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


FORECAST_ROOT = Path("data_lake/silver/weather_forecast_vintage")
ACTUAL_ROOT = Path("data_lake/silver/weather_open_meteo")
OUTPUT_ROOT = Path("data_lake/gold/forecast_error")

FORECAST_COLUMNS = {
    "temperature_forecast_c": "temperature_2m_c",
    "wind_speed_forecast_kmh": "wind_speed_10m_kmh",
    "shortwave_radiation_forecast_w_m2": "shortwave_radiation_w_m2",
    "precipitation_forecast_mm": "precipitation_mm",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def latest_forecast_batch(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*/run=*"))
    if not batches:
        raise FileNotFoundError(
            f"No forecast-vintage Silver batches found under {root}. "
            "Run build_weather_vintage_silver.py first."
        )
    return batches[-1]


def latest_actual_batch(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*"))
    if not batches:
        raise FileNotFoundError(
            f"No actual-weather Silver batches found under {root}. "
            "Run ingest_weather_bronze.py and build_weather_silver.py first."
        )
    return batches[-1]


def batch_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("batch_id="):
            return part.removeprefix("batch_id=")
    raise ValueError(f"Could not find batch_id= in path: {path}")


def read_parquet_dataset(path: Path) -> pd.DataFrame:
    """Read a Hive-partitioned Parquet directory, excluding JSON reports."""
    if not path.exists():
        raise FileNotFoundError(f"Silver dataset does not exist: {path}")
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under: {path}")
    # The Silver directory also contains quality_report.json.  Pass the
    # explicit Parquet file list so Arrow never tries to inspect the report as
    # if it were a Parquet file; keep Hive partition parsing for location and
    # date fields that were removed from the physical Parquet files.
    table = ds.dataset(
        [str(file_path) for file_path in parquet_files],
        format="parquet",
        partitioning="hive",
    ).to_table()
    return table.to_pandas()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-silver-root",
        type=Path,
        default=FORECAST_ROOT,
        help=f"Forecast-vintage Silver root (default: {FORECAST_ROOT})",
    )
    parser.add_argument(
        "--forecast-silver-batch",
        type=Path,
        help="Specific forecast-vintage Silver run; defaults to the latest run.",
    )
    parser.add_argument(
        "--actual-weather-silver-root",
        type=Path,
        default=ACTUAL_ROOT,
        help=f"Actual-weather Silver root (default: {ACTUAL_ROOT})",
    )
    parser.add_argument(
        "--actual-weather-silver-batch",
        type=Path,
        help="Specific actual-weather Silver batch; defaults to the latest batch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Gold output root (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--horizon-bin-hours",
        type=int,
        default=24,
        help="Width of forecast-horizon metric bins in hours (default: 24).",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def metric_rows(
    joined: pd.DataFrame,
    forecast_column: str,
    actual_column: str,
    error_column: str,
    horizon_bin_hours: int,
) -> list[dict]:
    valid = joined[[forecast_column, actual_column]].dropna()
    if valid.empty:
        return [
            {
                "variable": error_column,
                "n": 0,
                "bias_forecast_minus_actual": None,
                "mae": None,
                "rmse": None,
                "max_absolute_error": None,
            }
        ]

    error = valid[forecast_column] - valid[actual_column]
    return [
        {
            "variable": error_column,
            "n": int(len(error)),
            "bias_forecast_minus_actual": float(error.mean()),
            "mae": float(error.abs().mean()),
            "rmse": float((error.pow(2).mean()) ** 0.5),
            "max_absolute_error": float(error.abs().max()),
        }
    ]


def build_horizon_metrics(
    joined: pd.DataFrame, horizon_bin_hours: int
) -> pd.DataFrame:
    """Return one row per horizon bin and weather variable."""
    if horizon_bin_hours <= 0:
        raise ValueError("--horizon-bin-hours must be positive")

    work = joined.copy()
    work["horizon_bin_start_hours"] = (
        (work["forecast_horizon_hours"] // horizon_bin_hours) * horizon_bin_hours
    ).astype("Int64")
    work["horizon_bin"] = work["horizon_bin_start_hours"].map(
        lambda value: (
            f"{int(value)}-{int(value) + horizon_bin_hours - 1}h"
            if pd.notna(value)
            else "unknown"
        )
    )

    rows: list[dict] = []
    for horizon_bin, group in work.groupby("horizon_bin", dropna=False, sort=True):
        for forecast_column, actual_column in FORECAST_COLUMNS.items():
            actual_column = f"{actual_column}_actual"
            valid = group[[forecast_column, actual_column]].dropna()
            error = valid[forecast_column] - valid[actual_column]
            rows.append(
                {
                    "horizon_bin": horizon_bin,
                    "horizon_bin_start_hours": (
                        None
                        if group["horizon_bin_start_hours"].isna().all()
                        else int(group["horizon_bin_start_hours"].dropna().iloc[0])
                    ),
                    "variable": forecast_column.removesuffix("_forecast_c")
                    if forecast_column == "temperature_forecast_c"
                    else forecast_column.removesuffix("_forecast_kmh")
                    if forecast_column == "wind_speed_forecast_kmh"
                    else forecast_column.removesuffix("_forecast_w_m2")
                    if forecast_column == "shortwave_radiation_forecast_w_m2"
                    else forecast_column.removesuffix("_forecast_mm"),
                    "n": int(len(error)),
                    "bias_forecast_minus_actual": (
                        float(error.mean()) if len(error) else None
                    ),
                    "mae": float(error.abs().mean()) if len(error) else None,
                    "rmse": float((error.pow(2).mean()) ** 0.5) if len(error) else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    forecast_batch = args.forecast_silver_batch or latest_forecast_batch(
        args.forecast_silver_root
    )
    actual_batch = args.actual_weather_silver_batch or latest_actual_batch(
        args.actual_weather_silver_root
    )

    forecast_batch_id = batch_id_from_path(forecast_batch)
    actual_batch_id = batch_id_from_path(actual_batch)
    run_part = next(
        part.removeprefix("run=")
        for part in forecast_batch.parts
        if part.startswith("run=")
    )
    output_dir = (
        args.output_root
        / f"forecast_batch={forecast_batch_id}__actual_batch={actual_batch_id}"
        / f"run={run_part}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Gold output already exists: {output_dir}")

    forecast = read_parquet_dataset(forecast_batch)
    actual = read_parquet_dataset(actual_batch)

    require_columns(
        forecast,
        [
            "forecast_run_utc",
            "valid_time_utc",
            "forecast_horizon_hours",
            "location",
            *FORECAST_COLUMNS.keys(),
        ],
        "Forecast Silver",
    )
    require_columns(
        actual,
        ["event_time_utc", "location", *FORECAST_COLUMNS.values()],
        "Actual-weather Silver",
    )

    for frame, time_column in [
        (forecast, "forecast_run_utc"),
        (forecast, "valid_time_utc"),
        (actual, "event_time_utc"),
    ]:
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce", utc=True)

    forecast_key = ["location", "forecast_run_utc", "valid_time_utc"]
    actual_key = ["location", "event_time_utc"]
    forecast_duplicate_count = int(forecast.duplicated(subset=forecast_key).sum())
    actual_duplicate_count = int(actual.duplicated(subset=actual_key).sum())
    if actual_duplicate_count:
        raise ValueError(
            "Actual-weather Silver contains duplicate location/time keys; "
            "refusing to compute ambiguous errors."
        )

    actual_for_join = actual.rename(columns={"event_time_utc": "valid_time_utc"})[
        ["location", "valid_time_utc", *FORECAST_COLUMNS.values(), "batch_id"]
    ].rename(columns={
        **{
            value: f"{value}_actual" for value in FORECAST_COLUMNS.values()
        },
        "batch_id": "actual_batch_id",
    })

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
        actual_column = f"{actual_column}_actual"
        stem = forecast_column.removesuffix("_forecast_c")
        if stem == forecast_column:
            stem = forecast_column.removesuffix("_forecast_kmh")
        if stem == forecast_column:
            stem = forecast_column.removesuffix("_forecast_w_m2")
        if stem == forecast_column:
            stem = forecast_column.removesuffix("_forecast_mm")
        error_column = f"{stem}_error"
        absolute_column = f"{stem}_absolute_error"
        joined[error_column] = joined[forecast_column] - joined[actual_column]
        joined[absolute_column] = joined[error_column].abs()
        error_columns.extend([error_column, absolute_column])

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
        *[f"{column}_actual" for column in FORECAST_COLUMNS.values()],
        *error_columns,
        "run_date",
        "valid_year",
        "valid_month",
    ]
    gold = joined[keep_columns]

    metric_rows_all: list[dict] = []
    for forecast_column, actual_column in FORECAST_COLUMNS.items():
        stem = forecast_column.removesuffix("_forecast_c")
        if stem == forecast_column:
            stem = forecast_column.removesuffix("_forecast_kmh")
        if stem == forecast_column:
            stem = forecast_column.removesuffix("_forecast_w_m2")
        if stem == forecast_column:
            stem = forecast_column.removesuffix("_forecast_mm")
        metric_rows_all.extend(
            metric_rows(
                joined,
                forecast_column,
                f"{actual_column}_actual",
                f"{stem}_error",
                args.horizon_bin_hours,
            )
        )
    metrics = pd.DataFrame(metric_rows_all)
    horizon_metrics = build_horizon_metrics(joined, args.horizon_bin_hours)

    # Create and populate the output only after all input validation, joining,
    # and metric calculations have succeeded. This keeps failed runs from
    # looking like complete Gold batches.
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(gold, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["run_date", "valid_year", "valid_month", "location"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )
    metrics.to_parquet(output_dir / "overall_error_metrics.parquet", index=False)
    horizon_metrics.to_parquet(output_dir / "horizon_error_metrics.parquet", index=False)

    matched = joined[joined["actual_match_flag"]]
    report = {
        "layer": "gold",
        "dataset": "forecast_error",
        "created_at_utc": utc_now(),
        "forecast_silver_batch": str(forecast_batch),
        "actual_weather_silver_batch": str(actual_batch),
        "gold_output": str(output_dir),
        "rows_forecast": int(len(forecast)),
        "rows_actual": int(len(actual)),
        "rows_gold": int(len(gold)),
        "matched_rows": int(len(matched)),
        "unmatched_rows": int((~joined["actual_match_flag"]).sum()),
        "match_rate": float(joined["actual_match_flag"].mean())
        if len(joined)
        else None,
        "locations_forecast": sorted(forecast["location"].dropna().unique().tolist()),
        "locations_actual": sorted(actual["location"].dropna().unique().tolist()),
        "forecast_runs": int(forecast["forecast_run_utc"].nunique()),
        "forecast_duplicate_key_count": forecast_duplicate_count,
        "actual_duplicate_key_count": actual_duplicate_count,
        "missing_actual_values": {
            column: int(joined[f"{column}_actual"].isna().sum())
            for column in FORECAST_COLUMNS.values()
        },
        "overall_metrics_file": str(output_dir / "overall_error_metrics.parquet"),
        "horizon_metrics_file": str(output_dir / "horizon_error_metrics.parquet"),
        "error_definition": "forecast minus actual",
        "partitioning": ["run_date", "valid_year", "valid_month", "location"],
    }
    (output_dir / "error_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Forecast-error Gold output: {output_dir}")
    print(f"Forecast rows: {len(forecast):,}")
    print(f"Actual rows:   {len(actual):,}")
    print(f"Matched rows:  {len(matched):,}")
    print(f"Match rate:    {report['match_rate']:.2%}" if len(joined) else "Match rate:    n/a")
    print(f"Report:        {output_dir / 'error_report.json'}")
    print("Forecast-error Gold transformation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
