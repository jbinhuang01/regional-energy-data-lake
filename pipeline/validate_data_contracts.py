#!/usr/bin/env python3
"""Check registered Silver and Gold batches against their contracts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


DEFAULT_REGISTRY = Path("metadata/data_lake_registry.parquet")
DEFAULT_CONTRACTS = Path("metadata/data_contracts.json")
DEFAULT_OUTPUT = Path("metadata/contract_validation_results.csv")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return exit code 1 when any batch is WARN or FAIL.",
    )
    return parser.parse_args()


def read_json(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def nested_values(value: Any, parts: list[str]) -> list[Any]:
    if not parts:
        return [value]
    part = parts[0]
    if part == "*":
        if not isinstance(value, list):
            return []
        values: list[Any] = []
        for item in value:
            values.extend(nested_values(item, parts[1:]))
        return values
    if isinstance(value, dict) and part in value:
        return nested_values(value[part], parts[1:])
    return []


def check_rule(report: dict, rule: dict) -> tuple[bool, str]:
    path = rule["path"]
    observed = nested_values(report, path.split("."))
    if not observed:
        return False, f"missing report field: {path}"
    operator = rule["op"]
    expected = rule["value"]
    failures = []
    for item in observed:
        try:
            passed = {
                "eq": item == expected,
                "ge": item >= expected,
                "gt": item > expected,
                "le": item <= expected,
                "lt": item < expected,
            }[operator]
        except (TypeError, KeyError):
            passed = False
        if not passed:
            failures.append(repr(item))
    if failures:
        return False, f"{path} {operator} {expected}; observed {failures}"
    return True, f"{path} {operator} {expected}"


def dataset_schema(path: Path) -> set[str]:
    files = sorted(path.rglob("part-*.parquet"))
    if not files:
        return set()
    table = ds.dataset(
        [str(file_path) for file_path in files],
        format="parquet",
        partitioning="hive",
    )
    return set(table.schema.names)


def validate_row(row: pd.Series, contract: dict | None) -> dict:
    layer = row["layer"]
    dataset = row["dataset"]
    key = f"{layer}/{dataset}"
    base = {
        "validated_at_utc": utc_now(),
        "layer": layer,
        "dataset": dataset,
        "batch_id": row.get("batch_id"),
        "artifact_path": row["artifact_path"],
        "registry_status": row["status"],
        "contract": key,
    }
    if layer == "bronze":
        if row["status"] != "COMPLETE":
            return {
                **base,
                "status": "WARN",
                "passed": False,
                "failure_count": 1,
                "failures": ["Bronze registry status is not COMPLETE"],
            }
        return {
            **base,
            "status": "SKIP_BRONZE",
            "passed": True,
            "failure_count": 0,
            "failures": [],
        }
    if row["status"] != "COMPLETE":
        return {
            **base,
            "status": "WARN",
            "passed": False,
            "failure_count": 1,
            "failures": ["registry status is not COMPLETE"],
        }
    if contract is None:
        return {
            **base,
            "status": "WARN",
            "passed": False,
            "failure_count": 1,
            "failures": [f"no contract defined for {key}"],
        }

    artifact = Path(row["artifact_path"])
    schema = dataset_schema(artifact)
    missing_columns = sorted(set(contract.get("required_columns", [])) - schema)
    failures = []
    if missing_columns:
        failures.append(f"missing required columns: {missing_columns}")

    report = read_json(row.get("report_path"))
    for rule in contract.get("report_rules", []):
        passed, message = check_rule(report, rule)
        if not passed:
            failures.append(message)

    return {
        **base,
        "status": "PASS" if not failures else "FAIL",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "required_column_count": len(contract.get("required_columns", [])),
        "observed_column_count": len(schema),
    }


def main() -> int:
    args = parse_args()
    registry = pd.read_parquet(args.registry)
    contracts = json.loads(args.contracts.read_text(encoding="utf-8"))
    results = []
    for _, row in registry.iterrows():
        key = f"{row['layer']}/{row['dataset']}"
        results.append(validate_row(row, contracts.get(key)))
    output = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    output.to_parquet(args.output.with_suffix(".parquet"), index=False)

    report = {
        "created_at_utc": utc_now(),
        "registry": str(args.registry),
        "contracts": str(args.contracts),
        "results_file": str(args.output),
        "total_artifacts": int(len(output)),
        "pass_count": int((output["status"] == "PASS").sum()),
        "fail_count": int((output["status"] == "FAIL").sum()),
        "warn_count": int((output["status"] == "WARN").sum()),
        "skip_bronze_count": int((output["status"] == "SKIP_BRONZE").sum()),
        "failed_artifacts": output.loc[
            output["status"].isin(["FAIL", "WARN"]), "artifact_path"
        ].tolist(),
    }
    report_path = args.output.with_name("contract_validation_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Results: {args.output}")
    print(f"Report:  {report_path}")
    print(f"PASS:    {report['pass_count']}")
    print(f"FAIL:    {report['fail_count']}")
    print(f"WARN:    {report['warn_count']}")
    if args.fail_on_warn and (report["fail_count"] or report["warn_count"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
