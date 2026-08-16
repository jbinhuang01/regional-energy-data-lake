#!/usr/bin/env python3
"""Run fixed-effects panel regressions on the multi-zone Gold panel.

The models estimate conditional associations after controlling for zone,
hour-of-day, weekday and year fixed effects. Standard errors are clustered by
zone. Because the current panel has only six zones, inference is explicitly
flagged as small-cluster and should be followed by a wild-cluster bootstrap.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import statsmodels.formula.api as smf


REQUIRED_COLUMNS = [
    "region",
    "event_time_utc",
    "price_day_ahead_eur_mwh",
    "negative_price_flag",
    "renewable_share",
    "load_forecast_error_mw",
    "hour_utc",
    "day_of_week",
    "year",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_latest_panel(root: Path) -> Path:
    panels = sorted(root.glob("silver_batch=*"))
    if not panels:
        raise FileNotFoundError(f"No multi-zone panels found under {root}")
    return panels[-1]


def read_panel(panel_dir: Path) -> pd.DataFrame:
    files = [str(path) for path in panel_dir.rglob("*.parquet")]
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {panel_dir}")
    dataset = ds.dataset(files, format="parquet", partitioning="hive")
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    frame = dataset.to_table(columns=REQUIRED_COLUMNS).to_pandas()
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True)
    frame["negative_price_flag"] = frame["negative_price_flag"].astype("int8")
    frame["renewable_share_pct"] = frame["renewable_share"] * 100
    frame["load_forecast_error_gw"] = frame["load_forecast_error_mw"] / 1000
    return frame


def fit_clustered_model(
    data: pd.DataFrame,
    outcome: str,
    formula_terms: str,
) -> tuple[pd.DataFrame, dict]:
    formula = f"{outcome} ~ {formula_terms}"
    result = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["region"], "use_correction": True},
    )
    confidence = result.conf_int()
    table = pd.DataFrame(
        {
            "outcome": outcome,
            "term": result.params.index,
            "coefficient": result.params.values,
            "std_error_clustered_by_region": result.bse.values,
            "t_value": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_low": confidence[0].values,
            "ci_high": confidence[1].values,
        }
    )
    metadata = {
        "formula": formula,
        "observations": int(result.nobs),
        "r_squared": float(result.rsquared),
        "adjusted_r_squared": float(result.rsquared_adj),
    }
    return table, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=Path("data_lake/gold/multi_zone_panel"),
    )
    parser.add_argument("--panel", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/multi_zone_panel"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel_dir = args.panel or find_latest_panel(args.panel_root)
    panel_id = panel_dir.name
    output_dir = args.output_root / panel_id
    if output_dir.exists():
        raise SystemExit(f"Analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_panel(panel_dir)
    model_columns = [
        "price_day_ahead_eur_mwh",
        "negative_price_flag",
        "renewable_share_pct",
        "load_forecast_error_gw",
        "region",
        "hour_utc",
        "day_of_week",
        "year",
    ]
    model_data = frame.dropna(subset=model_columns).copy()
    model_data["region"] = model_data["region"].astype("category")

    formula_terms = (
        "renewable_share_pct + load_forecast_error_gw + "
        "C(region) + C(hour_utc) + C(day_of_week) + C(year)"
    )
    price_results, price_metadata = fit_clustered_model(
        model_data,
        "price_day_ahead_eur_mwh",
        formula_terms,
    )
    negative_results, negative_metadata = fit_clustered_model(
        model_data,
        "negative_price_flag",
        formula_terms,
    )
    results = pd.concat([price_results, negative_results], ignore_index=True)
    results.to_csv(output_dir / "fixed_effects_results.csv", index=False)

    coverage = (
        frame.groupby("region", observed=True)
        .agg(
            rows=("event_time_utc", "size"),
            start_time_utc=("event_time_utc", "min"),
            end_time_utc=("event_time_utc", "max"),
            price_missing=("price_day_ahead_eur_mwh", lambda values: int(values.isna().sum())),
            model_rows=("price_day_ahead_eur_mwh", lambda values: int(values.notna().sum())),
        )
        .reset_index()
    )
    coverage["start_time_utc"] = coverage["start_time_utc"].astype(str)
    coverage["end_time_utc"] = coverage["end_time_utc"].astype(str)
    coverage.to_csv(output_dir / "zone_coverage.csv", index=False)

    report = {
        "analysis": "multi_zone_fixed_effects",
        "panel": str(panel_dir),
        "created_at_utc": utc_now(),
        "panel_rows": len(frame),
        "model_rows": len(model_data),
        "zones": sorted(frame["region"].unique().tolist()),
        "zone_count": int(frame["region"].nunique()),
        "formula_terms": formula_terms,
        "standard_errors": "clustered by region",
        "small_cluster_warning": (
            "Only six region clusters are available. Cluster-robust p-values "
            "are fragile and should be checked with a wild-cluster bootstrap."
        ),
        "price_model": price_metadata,
        "negative_price_model": negative_metadata,
        "price_spike_threshold_eur_mwh": 100,
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Panel analysis output: {output_dir}")
    print(f"Panel rows: {len(frame):,}")
    print(f"Model rows: {len(model_data):,}")
    print(f"Zones: {frame['region'].nunique()}")
    print(f"Report: {output_dir / 'analysis_report.json'}")
    print("Fixed-effects panel analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
