#!/usr/bin/env python3
"""Normalize Energy-Charts responses without changing source resolution."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


DEFAULT_BRONZE_ROOT = Path("data_lake/bronze/energy_charts")
DEFAULT_OUTPUT_ROOT = Path("data_lake/silver/energy_charts")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-root", type=Path, default=DEFAULT_BRONZE_ROOT)
    parser.add_argument("--bronze-batch", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def latest_batch(root: Path) -> Path:
    batches = sorted(root.glob("ingestion_date=*/batch_id=*/country=*/bzn=*"))
    if not batches:
        raise FileNotFoundError(f"No Energy-Charts Bronze batches under {root}")
    return batches[-1]


def batch_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("batch_id="):
            return part.removeprefix("batch_id=")
    raise ValueError(f"No batch_id= component in {path}")


def path_value(path: Path, prefix: str) -> str:
    for part in path.parts:
        if part.startswith(prefix):
            return part.removeprefix(prefix)
    raise ValueError(f"No {prefix} component in {path}")


def normalize_name(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def resolution_minutes(times: pd.Series) -> float | None:
    ordered = times.dropna().sort_values().drop_duplicates()
    if len(ordered) < 2:
        return None
    return float(ordered.diff().dropna().dt.total_seconds().median() / 60)


def frame_from_series(
    timestamps: list[int],
    values: list[float | None],
    *,
    metric: str,
    series_type: str,
    unit: str,
    production_type: str | None,
    forecast_type: str | None,
    source_endpoint: str,
    batch_id: str,
    country: str,
    bzn: str,
) -> tuple[pd.DataFrame, dict]:
    if len(timestamps) != len(values):
        raise ValueError(
            f"{metric}: timestamp/value length mismatch "
            f"({len(timestamps)} vs {len(values)})"
        )
    frame = pd.DataFrame(
        {
            "source_timestamp_s": timestamps,
            "event_time_utc": pd.to_datetime(timestamps, unit="s", utc=True),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    )
    frame["metric"] = metric
    frame["series_type"] = series_type
    frame["unit"] = unit
    frame["production_type"] = production_type
    frame["forecast_type"] = forecast_type
    frame["source_endpoint"] = source_endpoint
    frame["source"] = "energy_charts_api"
    frame["country"] = country
    frame["bzn"] = bzn
    frame["batch_id"] = batch_id
    frame["resolution_minutes"] = resolution_minutes(frame["event_time_utc"])
    frame["year"] = frame["event_time_utc"].dt.year.astype("int16")
    frame["month"] = frame["event_time_utc"].dt.month.astype("int8")
    report = {
        "metric": metric,
        "series_type": series_type,
        "rows": len(frame),
        "invalid_timestamp_count": int(frame["event_time_utc"].isna().sum()),
        "duplicate_key_count": int(
            frame.duplicated(subset=["event_time_utc", "metric", "series_type"]).sum()
        ),
        "missing_value_count": int(frame["value"].isna().sum()),
        "resolution_minutes": frame["resolution_minutes"].iloc[0]
        if len(frame)
        else None,
    }
    return frame, report


def main() -> int:
    args = parse_args()
    bronze_batch = args.bronze_batch or latest_batch(args.bronze_root)
    batch_id = batch_id_from_path(bronze_batch)
    country = path_value(bronze_batch, "country=")
    bzn = path_value(bronze_batch, "bzn=")
    output_dir = (
        args.output_root
        / f"ingestion_date={batch_id[:8]}"
        / f"batch_id={batch_id}"
        / f"country={country}"
        / f"bzn={bzn}"
    )
    if output_dir.exists():
        raise SystemExit(f"Market Silver output already exists: {output_dir}")

    frames: list[pd.DataFrame] = []
    reports: list[dict] = []

    price_path = bronze_batch / "endpoint=price/price.json"
    if price_path.exists():
        payload = json.loads(price_path.read_text(encoding="utf-8"))
        frame, report = frame_from_series(
            payload["unix_seconds"],
            payload["price"],
            metric="price_day_ahead",
            series_type="market_outcome",
            unit=payload.get("unit", "EUR / MWh"),
            production_type=None,
            forecast_type=None,
            source_endpoint="/price",
            batch_id=batch_id,
            country=country,
            bzn=bzn,
        )
        frames.append(frame)
        reports.append(report)

    power_path = bronze_batch / "endpoint=public_power/public_power.json"
    if power_path.exists():
        payload = json.loads(power_path.read_text(encoding="utf-8"))
        for series in payload["production_types"]:
            production_type = series["name"]
            normalized = normalize_name(production_type)
            unit = "pct" if "share" in production_type.lower() else "MW"
            frame, report = frame_from_series(
                payload["unix_seconds"],
                series["data"],
                metric=f"actual_{normalized}",
                series_type="actual",
                unit=unit,
                production_type=production_type,
                forecast_type=None,
                source_endpoint="/public_power",
                batch_id=batch_id,
                country=country,
                bzn=bzn,
            )
            frames.append(frame)
            reports.append(report)

    for raw_path in sorted(
        (bronze_batch / "endpoint=public_power_forecast").glob(
            "production_type=*/forecast_*.json"
        )
    ):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        production_type = payload["production_type"]
        forecast_type = payload["forecast_type"]
        frame, report = frame_from_series(
            payload["unix_seconds"],
            payload["forecast_values"],
            metric=f"forecast_{normalize_name(production_type)}",
            series_type="forecast",
            unit="MW",
            production_type=production_type,
            forecast_type=forecast_type,
            source_endpoint="/public_power_forecast",
            batch_id=batch_id,
            country=country,
            bzn=bzn,
        )
        frames.append(frame)
        reports.append(report)

    if not frames:
        raise SystemExit(f"No recognized Energy-Charts JSON files under {bronze_batch}")

    silver = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    table = pa.Table.from_pandas(silver, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(output_dir),
        format="parquet",
        partitioning=["year", "month", "series_type", "metric"],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
    )

    report = {
        "layer": "silver",
        "dataset": "energy_charts",
        "bronze_batch": str(bronze_batch),
        "silver_output": str(output_dir),
        "created_at_utc": utc_now(),
        "rows": len(silver),
        "metrics": sorted(silver["metric"].unique().tolist()),
        "series_types": sorted(silver["series_type"].unique().tolist()),
        "native_resolution_minutes": sorted(
            silver["resolution_minutes"].dropna().unique().tolist()
        ),
        "metric_reports": reports,
        "partitioning": ["year", "month", "series_type", "metric"],
        "time_semantics": {
            "event_time_utc": "period start converted from provider unix seconds",
            "price": "day-ahead market price for the bidding zone",
            "actual": "provider public power time series",
            "forecast": "provider day-ahead forecast time series",
        },
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Energy market Silver output: {output_dir}")
    print(f"Rows: {len(silver):,}")
    print(f"Native resolutions: {report['native_resolution_minutes']}")
    print(f"Quality report: {output_dir / 'quality_report.json'}")
    print("Energy market Silver transformation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
