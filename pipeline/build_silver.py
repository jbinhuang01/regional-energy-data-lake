#!/usr/bin/env python3
"""Convert an OPSD Bronze batch from wide CSV to a partitioned Silver dataset.

Silver is long-format and typed:

    source, region, metric, event_time_utc, event_time_local,
    value, unit, ingested_at_utc, batch_id, year, month

The raw Bronze CSV is never modified. Null measurements are omitted from the
main Silver table, while data-quality counters are written to quality_report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_latest_batch(bronze_root: Path) -> Path:
    batches = sorted(bronze_root.glob("ingestion_date=*/batch_id=*"))
    if not batches:
        raise FileNotFoundError(f"No Bronze batches found under {bronze_root}")
    return batches[-1]


def load_column_mapping(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Read OPSD's field metadata instead of guessing regions from underscores."""

    package = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, str]] = {}

    for resource in package.get("resources", []):
        for field in resource.get("schema", {}).get("fields", []):
            properties = field.get("opsdProperties") or {}
            region = properties.get("Region")
            variable = properties.get("Variable")
            if region and variable:
                mapping[field["name"]] = {
                    "region": str(region),
                    "metric": str(variable),
                }

    return mapping


def infer_unit(metric: pd.Series) -> pd.Series:
    units = pd.Series("MW", index=metric.index, dtype="string")
    units.loc[metric.str.contains("price", case=False, na=False)] = "EUR/MWh"
    units.loc[metric.str.contains("profile", case=False, na=False)] = "fraction"
    return units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bronze-root",
        type=Path,
        default=Path("data_lake/bronze/opsd_time_series"),
    )
    parser.add_argument(
        "--bronze-batch",
        type=Path,
        help="Specific Bronze batch. Defaults to the latest completed batch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/silver/opsd_time_series"),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=20_000,
        help="CSV rows processed per chunk; default: 20000.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.bronze_batch or find_latest_batch(args.bronze_root)
    csv_path = batch_dir / "time_series_60min_singleindex.csv"
    metadata_path = batch_dir / "datapackage.json"

    if not csv_path.exists():
        raise SystemExit(f"Missing Bronze CSV: {csv_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Missing Bronze metadata: {metadata_path}")

    batch_id = batch_dir.name.removeprefix("batch_id=")
    output_dir = args.output_root / f"ingestion_date={batch_id[:8]}" / f"batch_id={batch_id}"
    if output_dir.exists():
        raise SystemExit(
            f"Silver output already exists: {output_dir}. "
            "Choose a new output root or remove it deliberately before rerunning."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_column_mapping(metadata_path)
    ingested_at = utc_now()
    rows_read = 0
    rows_written = 0
    non_null_values = 0
    invalid_timestamp_count = 0
    negative_physical_count = 0
    profile_out_of_range_count = 0
    duplicate_key_count = 0
    unknown_columns: set[str] = set()
    output_schema: pa.Schema | None = None

    print(f"Bronze batch: {batch_dir}")
    print(f"Silver output: {output_dir}")

    reader = pd.read_csv(
        csv_path,
        chunksize=args.chunksize,
        parse_dates=["utc_timestamp"],
        low_memory=False,
    )

    for chunk_number, wide in enumerate(reader):
        rows_read += len(wide)
        metric_columns = [
            column for column in wide.columns if column not in {"utc_timestamp", "cet_cest_timestamp"}
        ]
        long = wide.melt(
            id_vars=["utc_timestamp", "cet_cest_timestamp"],
            value_vars=metric_columns,
            var_name="source_column",
            value_name="value",
        )
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        non_null_values += int(long["value"].notna().sum())
        long = long.dropna(subset=["value"]).copy()

        long["event_time_utc"] = pd.to_datetime(
            long.pop("utc_timestamp"), errors="coerce", utc=True
        )
        invalid_timestamp_count += int(long["event_time_utc"].isna().sum())
        long = long.dropna(subset=["event_time_utc"])

        long["event_time_local"] = long.pop("cet_cest_timestamp").astype("string")
        long["region"] = long["source_column"].map(
            lambda column: mapping.get(column, {}).get("region", "UNKNOWN")
        )
        long["metric"] = long["source_column"].map(
            lambda column: mapping.get(column, {}).get("metric", column)
        )
        unknown_columns.update(
            column for column in long.loc[long["region"] == "UNKNOWN", "source_column"].unique()
        )

        long["unit"] = infer_unit(long["metric"])
        negative_physical_count += int(
            ((long["unit"] == "MW") & (long["value"] < 0)).sum()
        )
        profile_out_of_range_count += int(
            ((long["unit"] == "fraction") & ~long["value"].between(0, 1)).sum()
        )
        duplicate_key_count += int(
            long.duplicated(subset=["region", "metric", "event_time_utc"]).sum()
        )

        long["source"] = "opsd"
        long["ingested_at_utc"] = ingested_at
        long["batch_id"] = batch_id
        long["year"] = long["event_time_utc"].dt.year.astype("int16")
        long["month"] = long["event_time_utc"].dt.month.astype("int8")
        long = long[
            [
                "source",
                "region",
                "metric",
                "source_column",
                "event_time_utc",
                "event_time_local",
                "value",
                "unit",
                "ingested_at_utc",
                "batch_id",
                "year",
                "month",
            ]
        ]

        table = pa.Table.from_pandas(long, preserve_index=False)
        if output_schema is None:
            output_schema = table.schema
        ds.write_dataset(
            table,
            base_dir=str(output_dir),
            format="parquet",
            partitioning=["year", "month"],
            partitioning_flavor="hive",
            basename_template=f"part-{chunk_number:05d}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        rows_written += len(long)
        print(
            f"chunk={chunk_number:03d} input_rows={len(wide):,} "
            f"silver_rows={len(long):,}"
        )

    report = {
        "layer": "silver",
        "source": "open_power_system_data",
        "bronze_batch": str(batch_dir),
        "silver_output": str(output_dir),
        "created_at_utc": utc_now(),
        "rows_read": rows_read,
        "rows_written": rows_written,
        "non_null_values": non_null_values,
        "invalid_timestamp_count": invalid_timestamp_count,
        "negative_physical_value_count": negative_physical_count,
        "profile_out_of_range_count": profile_out_of_range_count,
        "duplicate_key_count_within_chunks": duplicate_key_count,
        "unknown_column_count": len(unknown_columns),
        "unknown_columns": sorted(unknown_columns),
        "partitioning": ["year", "month"],
        "schema": output_schema.to_string() if output_schema else None,
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Quality report: {output_dir / 'quality_report.json'}")
    print("Silver transformation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
