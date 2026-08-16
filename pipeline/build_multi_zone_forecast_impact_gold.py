#!/usr/bin/env python3
"""Join each bidding zone to its own weather representative."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_energy_forecast_impact_gold import (  # noqa: E402
    MARKET_METRICS,
    WEATHER_ERROR_COLUMNS,
    batch_id_from_path,
    build_market_hourly,
    read_partition_dataset,
)


DEFAULT_ZONE_LOCATION = {
    "DE-LU": "frankfurt",
    "FR": "paris",
    "AT": "vienna",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-silver-batch",
        action="append",
        required=True,
        help="Repeat once per market Silver batch.",
    )
    parser.add_argument("--weather-error-batch", type=Path, required=True)
    parser.add_argument(
        "--zone-location",
        action="append",
        help="Optional mapping such as FR=paris; defaults to DE-LU/FR/AT.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data_lake/gold/multi_zone_forecast_impact")
    )
    return parser.parse_args()


def read_weather_error(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No weather-error Gold Parquet files under {path}")
    return ds.dataset(
        [str(file) for file in files], format="parquet", partitioning="hive"
    ).to_table().to_pandas()


def build_weather_for_location(weather: pd.DataFrame, location: str) -> pd.DataFrame:
    subset = weather[weather["location"] == location].copy()
    if subset.empty:
        raise ValueError(f"No weather rows found for location={location}")
    for column in ["forecast_run_utc", "valid_time_utc"]:
        subset[column] = pd.to_datetime(subset[column], errors="coerce", utc=True)
    subset["forecast_horizon_hours"] = pd.to_numeric(
        subset["forecast_horizon_hours"], errors="coerce"
    )
    hourly = (
        subset.groupby(
            ["forecast_run_utc", "valid_time_utc", "forecast_horizon_hours"],
            as_index=False,
        )
        .agg({**{column: "mean" for column in WEATHER_ERROR_COLUMNS}, "location": "nunique"})
        .rename(columns={"location": "weather_location_count"})
    )
    hourly["weather_location_complete"] = hourly["weather_location_count"] >= 1
    return hourly


def main() -> int:
    args = parse_args()
    mapping = dict(DEFAULT_ZONE_LOCATION)
    for item in args.zone_location or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --zone-location value: {item}")
        zone, location = item.split("=", 1)
        mapping[zone] = location

    weather = read_weather_error(args.weather_error_batch)
    weather_batch_id = args.weather_error_batch.parent.name
    output_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"panel_batch={output_id}"
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")

    frames: list[pd.DataFrame] = []
    zone_reports: list[dict] = []
    for market_text in args.market_silver_batch:
        market_batch = Path(market_text)
        bzn = next(
            part.removeprefix("bzn=")
            for part in market_batch.parts
            if part.startswith("bzn=")
        )
        if bzn not in mapping:
            raise ValueError(
                f"No weather mapping for {bzn}; pass --zone-location {bzn}=location"
            )
        location = mapping[bzn]
        market = read_partition_dataset(market_batch)
        market_hourly, market_report = build_market_hourly(market)
        weather_hourly = build_weather_for_location(weather, location)
        panel = weather_hourly.merge(
            market_hourly,
            on="valid_time_utc",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        panel["market_match_flag"] = panel["_merge"].eq("both")
        panel = panel.drop(columns=["_merge"])
        panel["forecast_availability_proxy"] = (
            panel["forecast_run_utc"] <= panel["valid_time_utc"]
        )
        panel["run_date"] = panel["forecast_run_utc"].dt.strftime("%Y-%m-%d")
        panel["valid_year"] = panel["valid_time_utc"].dt.year.astype("Int16")
        panel["valid_month"] = panel["valid_time_utc"].dt.month.astype("Int8")
        panel["bzn"] = bzn
        panel["weather_location"] = location
        panel["market_batch_id"] = batch_id_from_path(market_batch)
        panel["weather_batch_id"] = weather_batch_id
        frames.append(panel)
        zone_reports.append(
            {
                "bzn": bzn,
                "weather_location": location,
                "market_batch": str(market_batch),
                "rows": int(len(panel)),
                "market_match_rate": float(panel["market_match_flag"].mean()),
                "forecast_runs": int(panel["forecast_run_utc"].nunique()),
                "market_report": market_report,
            }
        )

    gold = pd.concat(frames, ignore_index=True)
    selected = [
        "forecast_run_utc",
        "valid_time_utc",
        "forecast_horizon_hours",
        "weather_location_count",
        "weather_location_complete",
        "market_match_flag",
        "forecast_availability_proxy",
        "bzn",
        "weather_location",
        "market_batch_id",
        "weather_batch_id",
        *WEATHER_ERROR_COLUMNS,
        *MARKET_METRICS,
        "price_change_eur_mwh",
        "price_volatility_24h",
        "negative_price_flag",
        "load_forecast_error_mw",
        "solar_forecast_error_mw",
        "wind_onshore_forecast_error_mw",
        "wind_offshore_forecast_error_mw",
        "run_date",
        "valid_year",
        "valid_month",
    ]
    selected = [column for column in selected if column in gold.columns]
    gold = gold[selected]
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.write_dataset(
        pa.Table.from_pandas(gold, preserve_index=False),
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["bzn", "run_date", "valid_year", "valid_month"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )
    report = {
        "layer": "gold",
        "dataset": "multi_zone_energy_forecast_impact",
        "created_at_utc": utc_now(),
        "gold_output": str(output_dir),
        "weather_error_batch": str(args.weather_error_batch),
        "rows": int(len(gold)),
        "zones": sorted(gold["bzn"].unique().tolist()),
        "zone_count": int(gold["bzn"].nunique()),
        "forecast_runs": int(gold["forecast_run_utc"].nunique()),
        "market_match_rate": float(gold["market_match_flag"].mean()),
        "availability_proxy_violations": int((~gold["forecast_availability_proxy"]).sum()),
        "zone_reports": zone_reports,
        "research_caveat": (
            "Weather is represented by one point per zone; results are conditional "
            "associations, not causal effects."
        ),
    }
    (output_dir / "gold_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Multi-zone forecast impact Gold output: {output_dir}")
    print(f"Rows: {len(gold):,}; zones: {report['zone_count']}; match: {report['market_match_rate']:.2%}")
    print(f"Report: {output_dir / 'gold_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
