#!/usr/bin/env python3
"""Join one market zone to the weather forecast-error Gold table."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


DEFAULT_MARKET_ROOT = Path("data_lake/silver/energy_charts")
DEFAULT_WEATHER_ROOT = Path("data_lake/gold/forecast_error")
DEFAULT_OUTPUT_ROOT = Path("data_lake/gold/energy_forecast_impact")

MARKET_METRICS = [
    "price_day_ahead",
    "actual_load",
    "forecast_load",
    "actual_solar",
    "forecast_solar",
    "actual_wind_onshore",
    "forecast_wind_onshore",
    "actual_wind_offshore",
    "forecast_wind_offshore",
    "actual_renewable_share_of_load",
    "actual_residual_load",
]

WEATHER_ERROR_COLUMNS = [
    "temperature_error",
    "temperature_absolute_error",
    "wind_speed_error",
    "wind_speed_absolute_error",
    "shortwave_radiation_error",
    "shortwave_radiation_absolute_error",
    "precipitation_error",
    "precipitation_absolute_error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-silver-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--market-silver-batch", type=Path)
    parser.add_argument("--weather-error-root", type=Path, default=DEFAULT_WEATHER_ROOT)
    parser.add_argument("--weather-error-batch", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--volatility-window-hours",
        type=int,
        default=24,
        help="Rolling price-volatility window in hourly observations.",
    )
    return parser.parse_args()


def latest_market_batch(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*/country=*/bzn=*"))
    if not batches:
        raise FileNotFoundError(f"No market Silver batches under {root}")
    return batches[-1]


def latest_weather_batch(root: Path) -> Path:
    batches = sorted(root.glob("forecast_batch=*/run=*"))
    if not batches:
        raise FileNotFoundError(f"No weather forecast-error Gold batches under {root}")
    return batches[-1]


def batch_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("batch_id="):
            return part.removeprefix("batch_id=")
    raise ValueError(f"No batch_id= component in {path}")


def read_partition_dataset(path: Path, pattern: str = "part-*.parquet") -> pd.DataFrame:
    files = sorted(path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files under {path}")
    table = ds.dataset(
        [str(file_path) for file_path in files],
        format="parquet",
        partitioning="hive",
    ).to_table()
    return table.to_pandas()


def build_market_hourly(market: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"event_time_utc", "metric", "value", "batch_id"}
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ValueError(f"Market Silver missing columns: {missing}")

    market = market.copy()
    market["event_time_utc"] = pd.to_datetime(
        market["event_time_utc"], errors="coerce", utc=True
    )
    market["value"] = pd.to_numeric(market["value"], errors="coerce")
    market = market[market["metric"].isin(MARKET_METRICS)].copy()
    market["valid_time_utc"] = market["event_time_utc"].dt.floor("h")

    hourly = (
        market.groupby(["valid_time_utc", "metric"], as_index=False)["value"]
        .mean()
        .pivot(index="valid_time_utc", columns="metric", values="value")
        .reset_index()
    )
    hourly.columns.name = None
    for column in MARKET_METRICS:
        if column not in hourly.columns:
            hourly[column] = pd.NA

    hourly = hourly.sort_values("valid_time_utc").reset_index(drop=True)
    if "price_day_ahead" in hourly:
        hourly["price_change_eur_mwh"] = hourly["price_day_ahead"].diff()
        hourly["price_volatility_24h"] = hourly["price_day_ahead"].rolling(
            24, min_periods=12
        ).std()
        hourly["negative_price_flag"] = (
            hourly["price_day_ahead"] < 0
        ).astype("Int8")

    hourly["load_forecast_error_mw"] = (
        hourly["forecast_load"] - hourly["actual_load"]
    )
    hourly["solar_forecast_error_mw"] = (
        hourly["forecast_solar"] - hourly["actual_solar"]
    )
    hourly["wind_onshore_forecast_error_mw"] = (
        hourly["forecast_wind_onshore"] - hourly["actual_wind_onshore"]
    )
    hourly["wind_offshore_forecast_error_mw"] = (
        hourly["forecast_wind_offshore"] - hourly["actual_wind_offshore"]
    )
    report = {
        "rows_native": int(len(market)),
        "rows_hourly": int(len(hourly)),
        "native_metrics": sorted(market["metric"].dropna().unique().tolist()),
        "hourly_start_utc": hourly["valid_time_utc"].min().isoformat()
        if len(hourly)
        else None,
        "hourly_end_utc": hourly["valid_time_utc"].max().isoformat()
        if len(hourly)
        else None,
        "aggregation_rule": "arithmetic mean within UTC hour for every metric",
    }
    return hourly, report


def build_weather_hourly(weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "forecast_run_utc",
        "valid_time_utc",
        "forecast_horizon_hours",
        "location",
        *WEATHER_ERROR_COLUMNS,
    }
    missing = sorted(required.difference(weather.columns))
    if missing:
        raise ValueError(f"Weather forecast-error Gold missing columns: {missing}")

    weather = weather.copy()
    for column in ["forecast_run_utc", "valid_time_utc"]:
        weather[column] = pd.to_datetime(weather[column], errors="coerce", utc=True)
    weather["forecast_horizon_hours"] = pd.to_numeric(
        weather["forecast_horizon_hours"], errors="coerce"
    )
    weather_hourly = (
        weather.groupby(
            ["forecast_run_utc", "valid_time_utc", "forecast_horizon_hours"],
            as_index=False,
        )
        .agg(
            {
                **{column: "mean" for column in WEATHER_ERROR_COLUMNS},
                "location": "nunique",
            }
        )
        .rename(columns={"location": "weather_location_count"})
    )
    weather_hourly["weather_location_complete"] = (
        weather_hourly["weather_location_count"] >= 2
    )
    report = {
        "rows_location_level": int(len(weather)),
        "rows_time_level": int(len(weather_hourly)),
        "locations": sorted(weather["location"].dropna().unique().tolist()),
        "forecast_runs": int(weather["forecast_run_utc"].nunique()),
        "complete_location_hours": int(
            weather_hourly["weather_location_complete"].sum()
        ),
        "aggregation_rule": "arithmetic mean across available weather locations",
    }
    return weather_hourly, report


def main() -> int:
    args = parse_args()
    if args.volatility_window_hours < 2:
        raise SystemExit("--volatility-window-hours must be at least 2")

    market_batch = args.market_silver_batch or latest_market_batch(args.market_silver_root)
    weather_batch = args.weather_error_batch or latest_weather_batch(args.weather_error_root)
    market_batch_id = batch_id_from_path(market_batch)
    weather_batch_id = weather_batch.name.removeprefix("run=")
    bzn = next(
        part.removeprefix("bzn=")
        for part in market_batch.parts
        if part.startswith("bzn=")
    )
    output_dir = (
        args.output_root
        / f"market_batch={market_batch_id}__weather_batch={weather_batch_id}"
        / f"bzn={bzn}"
    )
    if output_dir.exists():
        raise SystemExit(f"Energy forecast impact Gold already exists: {output_dir}")

    market = read_partition_dataset(market_batch)
    weather = read_partition_dataset(weather_batch)
    market_hourly, market_report = build_market_hourly(market)
    weather_hourly, weather_report = build_weather_hourly(weather)

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
    panel["market_batch_id"] = market_batch_id
    panel["weather_batch_id"] = weather_batch_id

    selected = [
        "forecast_run_utc",
        "valid_time_utc",
        "forecast_horizon_hours",
        "weather_location_count",
        "weather_location_complete",
        "market_match_flag",
        "forecast_availability_proxy",
        "bzn",
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
    selected = [column for column in selected if column in panel.columns]
    gold = panel[selected]

    output_dir.mkdir(parents=True, exist_ok=False)
    ds.write_dataset(
        pa.Table.from_pandas(gold, preserve_index=False),
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["run_date", "valid_year", "valid_month", "bzn"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    report = {
        "layer": "gold",
        "dataset": "energy_forecast_impact",
        "created_at_utc": utc_now(),
        "market_silver_batch": str(market_batch),
        "weather_error_gold_batch": str(weather_batch),
        "gold_output": str(output_dir),
        "rows_gold": int(len(gold)),
        "market_matched_rows": int(gold["market_match_flag"].sum()),
        "market_unmatched_rows": int((~gold["market_match_flag"]).sum()),
        "market_match_rate": float(gold["market_match_flag"].mean())
        if len(gold)
        else None,
        "availability_proxy_violations": int(
            (~gold["forecast_availability_proxy"]).sum()
        ),
        "weather_location_complete_rate": float(
            gold["weather_location_complete"].mean()
        )
        if len(gold)
        else None,
        "market_report": market_report,
        "weather_report": weather_report,
        "lineage": {
            "market_batch_id": market_batch_id,
            "weather_batch_id": weather_batch_id,
            "bidding_zone": bzn,
        },
        "aggregation_rules": {
            "market_to_hour": "mean within UTC hour",
            "weather_locations": "mean across available locations",
            "price_volatility_24h": "rolling standard deviation of hourly day-ahead price",
        },
        "research_caveat": (
            "This panel supports conditional association analysis. It is not a causal estimate. "
            "forecast_run_utc is an availability proxy, not the provider publication timestamp."
        ),
        "partitioning": ["run_date", "valid_year", "valid_month", "bzn"],
    }
    (output_dir / "gold_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Energy forecast impact Gold output: {output_dir}")
    print(f"Rows: {len(gold):,}")
    print(f"Market match rate: {report['market_match_rate']:.2%}")
    print(f"Availability proxy violations: {report['availability_proxy_violations']}")
    print(f"Report: {output_dir / 'gold_report.json'}")
    print("Energy forecast impact Gold transformation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
