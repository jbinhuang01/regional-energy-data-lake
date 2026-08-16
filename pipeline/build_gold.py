#!/usr/bin/env python3
"""Build a Germany/LU hourly research mart from the Silver dataset.

Research question for the first Gold mart:
    How do renewable penetration, load forecast error and negative/peak
    day-ahead prices move together in the DE-LU bidding zone?

This is an analytical mart, not a model-training result. It keeps the
Silver batch ID and source ingestion timestamp for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


REQUIRED_METRICS = {
    "load_actual_entsoe_transparency",
    "load_forecast_entsoe_transparency",
    "price_day_ahead",
    "solar_generation_actual",
    "wind_generation_actual",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_latest_silver(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*"))
    if not batches:
        raise FileNotFoundError(f"No Silver batches found under {root}")
    return batches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silver-root",
        type=Path,
        default=Path("data_lake/silver/opsd_time_series"),
    )
    parser.add_argument(
        "--silver-batch",
        type=Path,
        help="Specific Silver batch. Defaults to the latest batch.",
    )
    parser.add_argument(
        "--region",
        default="DE_LU",
        help="OPSD region to build; default: DE_LU.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/gold/energy_research"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    silver_batch = args.silver_batch or find_latest_silver(args.silver_root)
    parquet_files = [str(path) for path in silver_batch.rglob("*.parquet")]
    if not parquet_files:
        raise SystemExit(f"No Silver Parquet files found under {silver_batch}")

    batch_id = silver_batch.name.removeprefix("batch_id=")
    output_dir = args.output_root / f"region={args.region}" / f"silver_batch={batch_id}"
    if output_dir.exists():
        raise SystemExit(f"Gold output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(parquet_files, format="parquet", partitioning="hive")
    table = dataset.to_table(
        columns=[
            "region",
            "metric",
            "event_time_utc",
            "value",
            "ingested_at_utc",
            "batch_id",
        ],
        filter=ds.field("region") == args.region,
    )
    long = table.to_pandas()
    if long.empty:
        raise SystemExit(f"No Silver rows found for region={args.region}")

    available_metrics = set(long["metric"].unique())
    missing_metrics = sorted(REQUIRED_METRICS - available_metrics)
    if missing_metrics:
        raise SystemExit(
            f"Missing required metrics for {args.region}: {', '.join(missing_metrics)}"
        )

    # pivot() intentionally fails if the Silver table contains duplicate keys;
    # silently averaging duplicate observations would hide a data-quality bug.
    wide = long.pivot(index="event_time_utc", columns="metric", values="value").reset_index()
    wide.columns.name = None
    wide = wide.sort_values("event_time_utc").reset_index(drop=True)

    rename = {
        "load_actual_entsoe_transparency": "load_actual_mw",
        "load_forecast_entsoe_transparency": "load_forecast_mw",
        "price_day_ahead": "price_day_ahead_eur_mwh",
        "solar_generation_actual": "solar_generation_mw",
        "wind_generation_actual": "wind_generation_mw",
    }
    wide = wide.rename(columns=rename)
    wide["renewable_generation_mw"] = (
        wide["solar_generation_mw"] + wide["wind_generation_mw"]
    )
    wide["renewable_share"] = (
        wide["renewable_generation_mw"] / wide["load_actual_mw"].where(wide["load_actual_mw"] > 0)
    )
    wide["load_forecast_error_mw"] = (
        wide["load_actual_mw"] - wide["load_forecast_mw"]
    )
    wide["load_forecast_error_pct"] = (
        wide["load_forecast_error_mw"]
        / wide["load_forecast_mw"].where(wide["load_forecast_mw"] > 0)
    )
    wide["negative_price_flag"] = wide["price_day_ahead_eur_mwh"] < 0
    wide["price_spike_flag"] = wide["price_day_ahead_eur_mwh"] >= 100
    wide["hour_utc"] = wide["event_time_utc"].dt.hour.astype("int8")
    wide["day_of_week"] = wide["event_time_utc"].dt.dayofweek.astype("int8")
    wide["is_weekend"] = wide["day_of_week"] >= 5
    wide["year"] = wide["event_time_utc"].dt.year.astype("int16")
    wide["month"] = wide["event_time_utc"].dt.month.astype("int8")
    wide["region"] = args.region
    wide["source"] = "opsd"
    wide["silver_batch_id"] = batch_id
    wide["created_at_utc"] = utc_now()

    output_columns = [
        "event_time_utc",
        "region",
        "load_actual_mw",
        "load_forecast_mw",
        "load_forecast_error_mw",
        "load_forecast_error_pct",
        "solar_generation_mw",
        "wind_generation_mw",
        "renewable_generation_mw",
        "renewable_share",
        "price_day_ahead_eur_mwh",
        "negative_price_flag",
        "price_spike_flag",
        "hour_utc",
        "day_of_week",
        "is_weekend",
        "source",
        "silver_batch_id",
        "created_at_utc",
        "year",
        "month",
    ]
    wide = wide[output_columns]
    pa_table = pa.Table.from_pandas(wide, preserve_index=False)
    ds.write_dataset(
        pa_table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["year", "month"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    report = {
        "layer": "gold",
        "mart": "energy_research",
        "region": args.region,
        "silver_batch": str(silver_batch),
        "created_at_utc": utc_now(),
        "rows": len(wide),
        "start_time_utc": wide["event_time_utc"].min().isoformat(),
        "end_time_utc": wide["event_time_utc"].max().isoformat(),
        "missing_by_column": wide.isna().sum().to_dict(),
        "negative_price_rows": int(wide["negative_price_flag"].sum()),
        "price_spike_rows": int(wide["price_spike_flag"].sum()),
        "renewable_share_over_100pct_rows": int((wide["renewable_share"] > 1).sum()),
        "research_question": (
            "How do renewable penetration and load forecast error relate to "
            "negative and peak day-ahead prices in the DE-LU bidding zone?"
        ),
    }
    (output_dir / "gold_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Gold output: {output_dir}")
    print(f"Rows: {len(wide):,}")
    print(f"Report: {output_dir / 'gold_report.json'}")
    print("Gold research mart completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
