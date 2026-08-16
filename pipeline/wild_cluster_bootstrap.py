#!/usr/bin/env python3
"""Wild cluster bootstrap for the six-zone electricity panel.

This is a coefficient/t-statistic bootstrap under the restricted model. It
resamples residuals at the region level, preserving within-region dependence.
The implementation uses precomputed cross-products so 999 repetitions remain
practical for the 250k-row panel.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pyarrow.dataset as ds


MODEL_COLUMNS = [
    "region",
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


def latest_panel(root: Path) -> Path:
    panels = sorted(root.glob("silver_batch=*"))
    if not panels:
        raise FileNotFoundError(f"No multi-zone panel found under {root}")
    return panels[-1]


def read_panel(panel_dir: Path) -> pd.DataFrame:
    files = [str(path) for path in panel_dir.rglob("*.parquet")]
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {panel_dir}")
    dataset = ds.dataset(files, format="parquet", partitioning="hive")
    frame = dataset.to_table(columns=MODEL_COLUMNS).to_pandas()
    frame["negative_price_flag"] = frame["negative_price_flag"].astype("int8")
    frame["renewable_share_pct"] = frame["renewable_share"] * 100
    frame["load_forecast_error_gw"] = frame["load_forecast_error_mw"] / 1000
    required = [
        "price_day_ahead_eur_mwh",
        "negative_price_flag",
        "renewable_share_pct",
        "load_forecast_error_gw",
        "region",
        "hour_utc",
        "day_of_week",
        "year",
    ]
    return frame.dropna(subset=required).copy()


def cluster_covariance(x: np.ndarray, residual: np.ndarray, groups: np.ndarray) -> np.ndarray:
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in np.unique(groups):
        xg = x[groups == group]
        eg = residual[groups == group]
        score = xg.T @ eg
        meat += np.outer(score, score)
    return xtx_inv @ meat @ xtx_inv


def bootstrap_target(
    data: pd.DataFrame,
    outcome: str,
    target: str,
    formula_terms: str,
    reps: int,
    seed: int,
) -> dict:
    full_formula = f"{outcome} ~ {formula_terms}"
    restricted_terms = formula_terms.replace(f"{target} + ", "")
    restricted_formula = f"{outcome} ~ {restricted_terms}"

    y_full, x_full_df = patsy.dmatrices(full_formula, data, return_type="dataframe")
    y_restricted, x_restricted_df = patsy.dmatrices(
        restricted_formula, data, return_type="dataframe"
    )
    y = np.asarray(y_full).ravel()
    x = np.asarray(x_full_df)
    x_restricted = np.asarray(x_restricted_df)
    groups = data["region"].astype(str).to_numpy()
    target_index = list(x_full_df.columns).index(target)

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta_full = xtx_inv @ (x.T @ y)
    residual_full = y - x @ beta_full
    observed_cov = cluster_covariance(x, residual_full, groups)
    observed_se = float(np.sqrt(max(observed_cov[target_index, target_index], 0)))
    observed_t = float(beta_full[target_index] / observed_se)

    beta_restricted = np.linalg.pinv(x_restricted.T @ x_restricted) @ (
        x_restricted.T @ y
    )
    fitted_restricted = x_restricted @ beta_restricted
    residual_restricted = y - fitted_restricted

    unique_groups = np.unique(groups)
    group_xt_yhat = []
    group_xt_residual = []
    group_xtx = []
    for group in unique_groups:
        mask = groups == group
        xg = x[mask]
        group_xt_yhat.append(xg.T @ fitted_restricted[mask])
        group_xt_residual.append(xg.T @ residual_restricted[mask])
        group_xtx.append(xg.T @ xg)

    rng = np.random.default_rng(seed)
    bootstrap_t = np.empty(reps)
    for repetition in range(reps):
        weights = rng.choice(np.array([-1.0, 1.0]), size=len(unique_groups))
        xt_y_boot = sum(
            yhat + weight * residual
            for yhat, residual, weight in zip(
                group_xt_yhat, group_xt_residual, weights
            )
        )
        beta_boot = xtx_inv @ xt_y_boot

        meat = np.zeros((x.shape[1], x.shape[1]))
        for xt_yhat, xt_residual, xtx, weight in zip(
            group_xt_yhat, group_xt_residual, group_xtx, weights
        ):
            score = xt_yhat + weight * xt_residual - xtx @ beta_boot
            meat += np.outer(score, score)
        cov_boot = xtx_inv @ meat @ xtx_inv
        se_boot = float(np.sqrt(max(cov_boot[target_index, target_index], 0)))
        bootstrap_t[repetition] = beta_boot[target_index] / se_boot if se_boot else 0

    p_value = float(
        (1 + np.sum(np.abs(bootstrap_t) >= abs(observed_t))) / (reps + 1)
    )
    return {
        "outcome": outcome,
        "target": target,
        "coefficient": float(beta_full[target_index]),
        "cluster_se_uncorrected": observed_se,
        "observed_t": observed_t,
        "wild_bootstrap_p_value": p_value,
        "bootstrap_t_2_5pct": float(np.quantile(bootstrap_t, 0.025)),
        "bootstrap_t_50pct": float(np.quantile(bootstrap_t, 0.50)),
        "bootstrap_t_97_5pct": float(np.quantile(bootstrap_t, 0.975)),
        "observations": len(data),
        "clusters": len(unique_groups),
        "repetitions": reps,
        "seed": seed,
        "restricted_formula": restricted_formula,
        "full_formula": full_formula,
    }


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
        default=Path("analysis/wild_cluster_bootstrap"),
    )
    parser.add_argument("--reps", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel_dir = args.panel or latest_panel(args.panel_root)
    panel_id = panel_dir.name
    output_dir = args.output_root / panel_id
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_panel(panel_dir)
    formula_terms = (
        "renewable_share_pct + load_forecast_error_gw + "
        "C(region) + C(hour_utc) + C(day_of_week) + C(year)"
    )
    results = []
    for outcome in ["price_day_ahead_eur_mwh", "negative_price_flag"]:
        for target in ["renewable_share_pct", "load_forecast_error_gw"]:
            results.append(
                bootstrap_target(
                    data,
                    outcome,
                    target,
                    formula_terms,
                    reps=args.reps,
                    seed=args.seed,
                )
            )
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_dir / "wild_bootstrap_results.csv", index=False)

    report = {
        "analysis": "wild_cluster_bootstrap",
        "panel": str(panel_dir),
        "created_at_utc": utc_now(),
        "rows": len(data),
        "clusters": sorted(data["region"].unique().tolist()),
        "cluster_count": int(data["region"].nunique()),
        "repetitions": args.reps,
        "seed": args.seed,
        "weight_distribution": "Rademacher +/- 1 at region level",
        "formula_terms": formula_terms,
        "interpretation_note": (
            "Bootstrap p-values are more conservative for the six-cluster panel, "
            "but six clusters remain a hard limit on inferential reliability."
        ),
    }
    (output_dir / "bootstrap_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Bootstrap output: {output_dir}")
    print(f"Rows: {len(data):,}; clusters: {data['region'].nunique()}; reps: {args.reps}")
    print(f"Report: {output_dir / 'bootstrap_report.json'}")
    print("Wild cluster bootstrap completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
