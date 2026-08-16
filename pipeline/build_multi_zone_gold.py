  #!/usr/bin/env python3
"""Build an unbalanced multi-zone electricity-market Gold panel.

The panel standardizes several bidding zones while preserving source-field
differences, especially the wind-generation metric used by each zone. It is
the input for fixed-effects and cross-zone robustness analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


DEFAULT_ZONES = ["AT", "DE_LU", "DK_1", "DK_2", "GB_GBN", "IT_NORD"]
COMMON_METRICS = {
    "load_actual_entsoe_transparency",
    "load_forecast_entsoe_transparency",
    "price_day_ahead",
    "solar_generation_actual",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_latest_batch(root: Path) -> Path:
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
    parser.add_argument("--silver-batch", type=Path)
    parser.add_argument(
        "--zones",
        nargs="+",
        default=DEFAULT_ZONES,
        help="Zones to include; default is six zones with common market fields.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/gold/multi_zone_panel"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    silver_batch = args.silver_batch or find_latest_batch(args.silver_root)
    batch_id = silver_batch.name.removeprefix("batch_id=")
    output_dir = args.output_root / f"silver_batch={batch_id}"
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [str(path) for path in silver_batch.rglob("*.parquet")]
    if not files:
        raise SystemExit(f"No Parquet files found under {silver_batch}")
    dataset = ds.dataset(files, format="parquet", partitioning="hive")
    table = dataset.to_table(
        columns=["region", "metric", "event_time_utc", "value", "batch_id"],
        filter=ds.field("region").isin(args.zones),
    )
    long = table.to_pandas()
    if long.empty:
        raise SystemExit(f"No rows found for zones: {args.zones}")
    long["event_time_utc"] = pd.to_datetime(long["event_time_utc"], utc=True)

    available_by_zone = {
        zone: set(long.loc[long["region"] == zone, "metric"].unique())
        for zone in args.zones
    }
    zone_reports = []
    for zone in args.zones:
        available = available_by_zone[zone]
        wind_metric = (
            "wind_generation_actual"
            if "wind_generation_actual" in available
            else "wind_onshore_generation_actual"
            if "wind_onshore_generation_actual" in available
            else None
        )
        missing = sorted(COMMON_METRICS - available)
        if wind_metric is None:
            missing.append("wind_generation_actual or wind_onshore_generation_actual")
        zone_reports.append(
            {
                "zone": zone,
                "wind_source_metric": wind_metric,
                "missing_required_metrics": missing,
                "available_metric_count": len(available),
            }
        )

    pivot = long.pivot(
        index=["region", "event_time_utc"],
        columns="metric",
        values="value",
    ).reset_index()
    pivot.columns.name = None

    pivot = pivot.rename(
        columns={
            "load_actual_entsoe_transparency": "load_actual_mw",
            "load_forecast_entsoe_transparency": "load_forecast_mw",
            "price_day_ahead": "price_day_ahead_eur_mwh",
            "solar_generation_actual": "solar_generation_mw",
        }
    )
    pivot["wind_generation_mw"] = pd.NA
    pivot["wind_source_metric"] = pd.NA
    for zone in args.zones:
        wind_metric = next(
            report["wind_source_metric"]
            for report in zone_reports
            if report["zone"] == zone
        )
        if wind_metric and wind_metric in pivot.columns:
            mask = pivot["region"] == zone
            pivot.loc[mask, "wind_generation_mw"] = pivot.loc[mask, wind_metric]
            pivot.loc[mask, "wind_source_metric"] = wind_metric

    pivot["renewable_generation_mw"] = (
        pivot["solar_generation_mw"] + pivot["wind_generation_mw"]
    )
    pivot["renewable_share"] = (
        pivot["renewable_generation_mw"]
        / pivot["load_actual_mw"].where(pivot["load_actual_mw"] > 0)
    )
    pivot["load_forecast_error_mw"] = (
        pivot["load_actual_mw"] - pivot["load_forecast_mw"]
    )
    pivot["load_forecast_error_pct"] = (
        pivot["load_forecast_error_mw"]
        / pivot["load_forecast_mw"].where(pivot["load_forecast_mw"] > 0)
    )
    pivot["negative_price_flag"] = pivot["price_day_ahead_eur_mwh"] < 0
    pivot["price_spike_flag"] = pivot["price_day_ahead_eur_mwh"] >= 100
    pivot["hour_utc"] = pivot["event_time_utc"].dt.hour.astype("int8")
    pivot["day_of_week"] = pivot["event_time_utc"].dt.dayofweek.astype("int8")
    pivot["is_weekend"] = pivot["day_of_week"] >= 5
    pivot["year"] = pivot["event_time_utc"].dt.year.astype("int16")
    pivot["month"] = pivot["event_time_utc"].dt.month.astype("int8")
    pivot["source"] = "opsd"
    pivot["silver_batch_id"] = batch_id

    output_columns = [
        "region",
        "event_time_utc",
        "load_actual_mw",
        "load_forecast_mw",
        "load_forecast_error_mw",
        "load_forecast_error_pct",
        "solar_generation_mw",
        "wind_generation_mw",
        "wind_source_metric",
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
        "year",
        "month",
    ]
    pivot = pivot[output_columns].sort_values(["region", "event_time_utc"])
    pa_table = pa.Table.from_pandas(pivot, preserve_index=False)
    ds.write_dataset(
        pa_table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["year", "month", "region"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    for report in zone_reports:
        zone_rows = pivot[pivot["region"] == report["zone"]]
        report.update(
            {
                "rows": len(zone_rows),
                "start_time_utc": (
                    zone_rows["event_time_utc"].min().isoformat()
                    if len(zone_rows)
                    else None
                ),
                "end_time_utc": (
                    zone_rows["event_time_utc"].max().isoformat()
                    if len(zone_rows)
                    else None
                ),
                "missing_price_rows": int(zone_rows["price_day_ahead_eur_mwh"].isna().sum()),
                "missing_renewable_share_rows": int(zone_rows["renewable_share"].isna().sum()),
                "negative_price_rows": int(zone_rows["negative_price_flag"].sum()),
            }
        )

    report = {
        "layer": "gold",
        "dataset": "multi_zone_electricity_panel",
        "silver_batch": str(silver_batch),
        "created_at_utc": utc_now(),
        "zones_requested": args.zones,
        "rows": len(pivot),
        "balanced_panel": False,
        "partitioning": ["year", "month", "region"],
        "price_spike_threshold_eur_mwh": 100,
        "zone_reports": zone_reports,
        "schema_note": (
            "wind_generation_mw is standardized, but wind_source_metric is "
            "retained because zones do not expose identical wind fields."
        ),
    }
    (output_dir / "panel_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Multi-zone Gold output: {output_dir}")
    print(f"Rows: {len(pivot):,}")
    for zone_report in zone_reports:
        print(
            f"{zone_report['zone']}: rows={zone_report.get('rows', 0):,} "
            f"wind={zone_report['wind_source_metric']} "
            f"missing={zone_report['missing_required_metrics']}"
        )
    print(f"Report: {output_dir / 'panel_report.json'}")
    print("Multi-zone Gold panel completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
