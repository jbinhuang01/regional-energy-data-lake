#!/usr/bin/env python3
"""Audit the data lake and build a batch-level metadata registry.

This is the control-plane companion to the Bronze/Silver/Gold data plane. It
does not rewrite data. It inventories completed batches, records quality and
lineage metadata, and applies simple quality gates so incomplete outputs are
visible before downstream analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_LAKE_ROOT = Path("data_lake")
DEFAULT_OUTPUT = Path("metadata/data_lake_registry.parquet")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake-root", type=Path, default=DEFAULT_LAKE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-match-rate",
        type=float,
        default=0.99,
        help="Minimum acceptable Gold match rate (default: 0.99).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def first_existing(path: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def batch_id_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("batch_id="):
            return part.removeprefix("batch_id=")
    return None


def dataset_name_from_path(path: Path, layer: str) -> str:
    try:
        layer_index = path.parts.index(layer)
        return path.parts[layer_index + 1]
    except (ValueError, IndexError):
        return path.name


def count_files(path: Path, pattern: str) -> tuple[int, int]:
    files = list(path.rglob(pattern))
    return len(files), sum(file.stat().st_size for file in files)


def lineage_values(report: dict) -> dict:
    lineage_keys = [
        "bronze_batch",
        "bronze_run",
        "silver_batch",
        "silver_output",
        "forecast_silver_batch",
        "actual_weather_silver_batch",
        "energy_batch",
        "weather_batch",
    ]
    return {
        f"lineage_{key}": str(report[key])
        for key in lineage_keys
        if key in report
    }


def audit_artifact(
    layer: str,
    path: Path,
    report_path: Path | None,
    min_match_rate: float,
) -> dict:
    report = read_json(report_path) if report_path else {}
    parquet_count, parquet_bytes = count_files(path, "*.parquet")
    manifest_count, manifest_bytes = count_files(path, "manifest.jsonl")
    batch_id = batch_id_from_path(path)

    checks: dict[str, bool] = {
        "report_present": report_path is not None,
        "parquet_present": parquet_count > 0 if layer != "bronze" else True,
        "manifest_present": manifest_count > 0 if layer == "bronze" else True,
    }
    match_rate = report.get("match_rate")
    if layer == "gold" and match_rate is not None:
        checks["match_rate_gate"] = float(match_rate) >= min_match_rate
    if layer == "silver":
        for key in ["invalid_timestamp_count", "invalid_timestamps"]:
            if key in report:
                checks["invalid_timestamp_gate"] = int(report[key]) == 0
        quality = report.get("quality_checks", {})
        if isinstance(quality, dict) and "invalid_timestamp_count" in quality:
            checks["invalid_timestamp_gate"] = (
                int(quality["invalid_timestamp_count"]) == 0
            )

    status = "COMPLETE" if all(checks.values()) else "WARN"
    rows = report.get("rows")
    if rows is None:
        rows = report.get("rows_gold", report.get("rows_written"))

    result = {
        "audited_at_utc": utc_now(),
        "layer": layer,
        "dataset": dataset_name_from_path(path, layer),
        "batch_id": batch_id,
        "artifact_path": str(path),
        "report_path": str(report_path) if report_path else None,
        "report_layer": report.get("layer"),
        "rows": rows,
        "parquet_file_count": parquet_count,
        "parquet_bytes": parquet_bytes,
        "manifest_file_count": manifest_count,
        "manifest_bytes": manifest_bytes,
        "match_rate": match_rate,
        "status": status,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
    }
    result.update(lineage_values(report))
    return result


def discover_artifacts(lake_root: Path) -> list[tuple[str, Path, Path | None]]:
    artifacts: list[tuple[str, Path, Path | None]] = []

    bronze_root = lake_root / "bronze"
    for path in sorted(bronze_root.glob("*/ingestion_date=*/batch_id=*")):
        report = first_existing(path, ["quality_report.json", "manifest.jsonl"])
        if report is None:
            nested_manifests = sorted(path.rglob("manifest.jsonl"))
            report = nested_manifests[-1] if nested_manifests else None
        artifacts.append(("bronze", path, report))

    silver_root = lake_root / "silver"
    for report in sorted(silver_root.glob("*/ingestion_date=*/batch_id=*/**/quality_report.json")):
        path = report.parent
        artifacts.append(("silver", path, report))

    gold_root = lake_root / "gold"
    report_names = ["gold_report.json", "error_report.json"]
    for report_name in report_names:
        for report in sorted(gold_root.glob(f"*/**/{report_name}")):
            path = report.parent
            artifacts.append(("gold", path, report))

    return artifacts


def main() -> int:
    args = parse_args()
    if not 0 <= args.min_match_rate <= 1:
        raise SystemExit("--min-match-rate must be between 0 and 1")
    if not args.lake_root.exists():
        raise FileNotFoundError(f"Data lake root does not exist: {args.lake_root}")

    artifacts = discover_artifacts(args.lake_root)
    if not artifacts:
        raise SystemExit(f"No Bronze/Silver/Gold artifacts found under {args.lake_root}")

    rows = [
        audit_artifact(layer, path, report, args.min_match_rate)
        for layer, path, report in artifacts
    ]
    registry = pd.DataFrame(rows).sort_values(
        ["layer", "dataset", "batch_id", "artifact_path"],
        na_position="last",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(args.output, index=False)
    registry.to_csv(args.output.with_suffix(".csv"), index=False)

    audit_report = {
        "created_at_utc": utc_now(),
        "lake_root": str(args.lake_root),
        "registry_file": str(args.output),
        "registry_csv": str(args.output.with_suffix(".csv")),
        "artifact_count": int(len(registry)),
        "complete_count": int((registry["status"] == "COMPLETE").sum()),
        "warning_count": int((registry["status"] == "WARN").sum()),
        "failed_artifacts": registry.loc[
            registry["status"] != "COMPLETE", "artifact_path"
        ].tolist(),
        "quality_gate": {
            "minimum_gold_match_rate": args.min_match_rate,
            "purpose": "prevent incomplete or poorly matched data from being treated as production-ready",
        },
    }
    report_path = args.output.with_name("data_lake_audit_report.json")
    report_path.write_text(
        json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Registry: {args.output}")
    print(f"CSV:      {args.output.with_suffix('.csv')}")
    print(f"Report:   {report_path}")
    print(f"Artifacts: {len(registry)}")
    print(f"Complete:  {audit_report['complete_count']}")
    print(f"Warnings:  {audit_report['warning_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
