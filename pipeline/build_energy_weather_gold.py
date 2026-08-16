#!/usr/bin/env python3
"""Join the DE-LU energy Gold mart with the weather Silver dataset.

The output is the main research panel: one row per energy hour, with weather
aggregated across Frankfurt and Luxembourg. A left join preserves every
energy observation and makes missing weather coverage explicit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


WEATHER_COLUMNS = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "shortwave_radiation_w_m2",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def latest_batch(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*"))
    if not batches:
        raise FileNotFoundError(f"No batches found under {root}")
    return batches[-1]


def parquet_dataset(batch_dir: Path) -> ds.Dataset:
    files = [str(path) for path in batch_dir.rglob("*.parquet")]
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {batch_dir}")
    return ds.dataset(files, format="parquet", partitioning="hive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energy-gold-root",
        type=Path,
        default=Path("data_lake/gold/energy_research"),
    )
    parser.add_argument(
        "--energy-gold-batch",
        type=Path,
        help="Specific energy Gold batch. Defaults to latest.",
    )
    parser.add_argument(
        "--weather-silver-root",
        type=Path,
        default=Path("data_lake/silver/weather_open_meteo"),
    )
    parser.add_argument(
        "--weather-silver-batch",
        type=Path,
        help="Specific weather Silver batch. Defaults to latest.",
    )
    parser.add_argument("--region", default="DE_LU")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/gold/energy_weather_panel"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    energy_batch = args.energy_gold_batch or latest_batch(args.energy_gold_root)
    weather_batch = args.weather_silver_batch or latest_batch(args.weather_silver_root)
    energy_batch_id = energy_batch.name.removeprefix("silver_batch=")
    weather_batch_id = weather_batch.name.removeprefix("batch_id=")

    output_dir = (
        args.output_root
        / f"region={args.region}"
        / f"energy_batch={energy_batch_id}__weather_batch={weather_batch_id}"
    )
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_dataset = parquet_dataset(energy_batch)
    energy_table = energy_dataset.to_table()
    energy = energy_table.to_pandas()
    energy["event_time_utc"] = pd.to_datetime(energy["event_time_utc"], utc=True)
    energy = energy.sort_values("event_time_utc").reset_index(drop=True)

    weather_dataset = parquet_dataset(weather_batch)
    weather_table = weather_dataset.to_table(
        columns=["event_time_utc", "location", *WEATHER_COLUMNS]
    )
    weather = weather_table.to_pandas()
    weather["event_time_utc"] = pd.to_datetime(weather["event_time_utc"], utc=True)

    weather_hourly = (
        weather.groupby("event_time_utc", as_index=False)
        .agg(
            weather_location_count=("location", "nunique"),
            temperature_2m_c_mean=("temperature_2m_c", "mean"),
            relative_humidity_2m_pct_mean=("relative_humidity_2m_pct", "mean"),
            precipitation_mm_mean=("precipitation_mm", "mean"),
            precipitation_mm_max=("precipitation_mm", "max"),
            wind_speed_10m_kmh_mean=("wind_speed_10m_kmh", "mean"),
            wind_speed_10m_kmh_max=("wind_speed_10m_kmh", "max"),
            shortwave_radiation_w_m2_mean=("shortwave_radiation_w_m2", "mean"),
        )
    )

    panel = energy.merge(
        weather_hourly,
        on="event_time_utc",
        how="left",
        validate="one_to_one",
        indicator="weather_join_status",
    )
    panel["weather_complete_flag"] = panel["weather_location_count"].eq(2)
    panel["rain_flag"] = panel["precipitation_mm_mean"].fillna(0).gt(0)
    panel["heavy_rain_flag"] = panel["precipitation_mm_max"].fillna(0).ge(5)
    panel["high_wind_flag"] = panel["wind_speed_10m_kmh_max"].fillna(0).ge(30)
    panel["cold_weather_flag"] = panel["temperature_2m_c_mean"].fillna(99).lt(0)
    panel["weather_energy_joined_at_utc"] = utc_now()
    panel["weather_silver_batch_id"] = weather_batch_id

    panel["year"] = panel["event_time_utc"].dt.year.astype("int16")
    panel["month"] = panel["event_time_utc"].dt.month.astype("int8")
    pa_table = pa.Table.from_pandas(panel, preserve_index=False)
    ds.write_dataset(
        pa_table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["year", "month"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    matched = int(panel["weather_join_status"].eq("both").sum())
    report = {
        "layer": "gold",
        "dataset": "energy_weather_panel",
        "region": args.region,
        "energy_gold_batch": str(energy_batch),
        "weather_silver_batch": str(weather_batch),
        "created_at_utc": utc_now(),
        "rows": len(panel),
        "matched_weather_rows": matched,
        "unmatched_weather_rows": len(panel) - matched,
        "complete_weather_rows": int(panel["weather_complete_flag"].sum()),
        "energy_start_time_utc": energy["event_time_utc"].min().isoformat(),
        "energy_end_time_utc": energy["event_time_utc"].max().isoformat(),
        "weather_start_time_utc": weather["event_time_utc"].min().isoformat(),
        "weather_end_time_utc": weather["event_time_utc"].max().isoformat(),
        "rain_rows": int(panel["rain_flag"].sum()),
        "heavy_rain_rows": int(panel["heavy_rain_flag"].sum()),
        "high_wind_rows": int(panel["high_wind_flag"].sum()),
        "cold_weather_rows": int(panel["cold_weather_flag"].sum()),
        "missing_by_column": panel.isna().sum().to_dict(),
        "join_key": "event_time_utc",
        "join_type": "left",
    }
    (output_dir / "panel_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Energy-weather Gold output: {output_dir}")
    print(f"Rows: {len(panel):,}")
    print(f"Matched weather rows: {matched:,}")
    print(f"Complete two-location rows: {int(panel['weather_complete_flag'].sum()):,}")
    print(f"Report: {output_dir / 'panel_report.json'}")
    print("Energy-weather Gold panel completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
