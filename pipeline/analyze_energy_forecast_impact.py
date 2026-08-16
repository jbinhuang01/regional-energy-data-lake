#!/usr/bin/env python3
"""Analyze the Energy Forecast Impact Gold panel.

The models are deliberately descriptive. They use HAC standard errors with a
24-hour lag window, fixed effects for hour and weekday, and a high-renewables
interaction. A single forecast initialization and one bidding zone are not
enough for causal inference or external validity.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/regional_energy_matplotlib")
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.dataset as ds


DEFAULT_GOLD_ROOT = Path("data_lake/gold/energy_forecast_impact")
DEFAULT_OUTPUT_ROOT = Path("analysis/energy_forecast_impact")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-batch", type=Path)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def latest_batch(root: Path) -> Path:
    batches = sorted(root.glob("market_batch=*/bzn=*"))
    if not batches:
        raise FileNotFoundError(f"No energy forecast impact Gold batches under {root}")
    return batches[-1]


def read_gold(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Gold partition files under {path}")
    return ds.dataset(
        [str(file_path) for file_path in files],
        format="parquet",
        partitioning="hive",
    ).to_table().to_pandas()


def zscore(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame[column], errors="coerce")
    mean = values.mean()
    std = values.std()
    output = f"z_{column}"
    frame[output] = (values - mean) / std if std and np.isfinite(std) else 0.0
    return output


def fit_model(
    frame: pd.DataFrame,
    formula: str,
    outcome: str,
    model_name: str,
) -> tuple[pd.DataFrame, dict]:
    required_outcome = frame.dropna(subset=[outcome]).copy()
    if len(required_outcome) < 60:
        return pd.DataFrame(), {
            "model": model_name,
            "outcome": outcome,
            "status": "SKIPPED",
            "n": int(len(required_outcome)),
            "reason": "fewer than 60 complete observations",
        }
    try:
        result = smf.ols(formula, data=required_outcome).fit(
            cov_type="HAC", cov_kwds={"maxlags": 24}
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return pd.DataFrame(), {
            "model": model_name,
            "outcome": outcome,
            "status": "FAILED",
            "n": int(len(required_outcome)),
            "reason": str(exc),
        }

    rows = []
    for term in result.params.index:
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "term": term,
                "coefficient": float(result.params[term]),
                "std_error_hac24": float(result.bse[term]),
                "t_value": float(result.tvalues[term]),
                "p_value_hac24": float(result.pvalues[term]),
                "ci_low": float(result.conf_int().loc[term, 0]),
                "ci_high": float(result.conf_int().loc[term, 1]),
                "n": int(result.nobs),
                "r_squared": float(result.rsquared),
            }
        )
    summary = {
        "model": model_name,
        "outcome": outcome,
        "status": "PASS",
        "n": int(result.nobs),
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "hac_maxlags": 24,
    }
    return pd.DataFrame(rows), summary


def plot_price_and_regimes(frame: pd.DataFrame, output_dir: Path) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(frame["valid_time_utc"], frame["price_day_ahead"], color="#2E74B5")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Day-ahead price (EUR/MWh)")
    axes[0].set_title("DE-LU day-ahead price during the forecast-vintage window")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        frame["valid_time_utc"],
        frame["actual_renewable_share_of_load"],
        color="#2E8B57",
    )
    axes[1].axhline(
        frame["actual_renewable_share_of_load"].median(),
        color="#C45A00",
        linestyle="--",
        label="Median threshold",
    )
    axes[1].set_ylabel("Renewable share of load (%)")
    axes[1].set_xlabel("Valid time (UTC)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / "price_and_renewable_regimes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_error_price_scatter(frame: pd.DataFrame, output_dir: Path) -> Path:
    pairs = [
        ("temperature_absolute_error", "Temperature error"),
        ("wind_speed_absolute_error", "Wind error"),
        ("shortwave_radiation_absolute_error", "Radiation error"),
        ("load_forecast_error_mw", "Load forecast error"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for axis, (column, label) in zip(axes.ravel(), pairs):
        subset = frame[[column, "price_day_ahead"]].dropna()
        axis.scatter(subset[column], subset["price_day_ahead"], s=14, alpha=0.55)
        axis.set_xlabel(label)
        axis.set_ylabel("Price (EUR/MWh)")
        axis.grid(alpha=0.25)
    figure.suptitle("Market price and forecast-error associations", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "forecast_error_price_scatter.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    gold_batch = args.gold_batch or latest_batch(args.gold_root)
    frame = read_gold(gold_batch)
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    frame = frame[frame["market_match_flag"]].copy()
    if frame.empty:
        raise SystemExit("No market-matched rows available for analysis")

    frame = frame.sort_values("valid_time_utc").reset_index(drop=True)
    frame["hour_utc"] = frame["valid_time_utc"].dt.hour
    frame["day_of_week"] = frame["valid_time_utc"].dt.dayofweek
    frame["high_renewable_regime"] = (
        frame["actual_renewable_share_of_load"]
        >= frame["actual_renewable_share_of_load"].median()
    ).astype(int)
    predictors = [
        "temperature_absolute_error",
        "wind_speed_absolute_error",
        "shortwave_radiation_absolute_error",
        "precipitation_absolute_error",
        "load_forecast_error_mw",
        "solar_forecast_error_mw",
        "wind_onshore_forecast_error_mw",
        "actual_residual_load",
    ]
    z_predictors = [zscore(frame, column) for column in predictors]
    frame["price_volatility_24h"] = pd.to_numeric(
        frame["price_volatility_24h"], errors="coerce"
    )

    formula_terms = " + ".join(z_predictors + ["C(hour_utc)", "C(day_of_week)"])
    interaction_terms = " + ".join(
        [f"{column}:high_renewable_regime" for column in z_predictors[:4]]
    )
    formula_with_interactions = f"{formula_terms} + {interaction_terms}"
    models = [
        ("price_level_baseline", "price_day_ahead", f"price_day_ahead ~ {formula_terms}"),
        (
            "price_level_regime_interactions",
            "price_day_ahead",
            f"price_day_ahead ~ {formula_with_interactions}",
        ),
        (
            "price_volatility_baseline",
            "price_volatility_24h",
            f"price_volatility_24h ~ {formula_terms}",
        ),
        (
            "negative_price_lpm",
            "negative_price_flag",
            f"negative_price_flag ~ {formula_terms}",
        ),
    ]

    coefficient_frames = []
    model_summaries = []
    for model_name, outcome, formula in models:
        coefficients, summary = fit_model(frame, formula, outcome, model_name)
        if not coefficients.empty:
            coefficient_frames.append(coefficients)
        model_summaries.append(summary)

    output_dir = args.output_root / gold_batch.name
    if output_dir.exists() and (output_dir / "analysis_report.json").exists():
        raise SystemExit(f"Analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )
    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)
    pd.DataFrame(model_summaries).to_csv(
        output_dir / "model_summaries.csv", index=False
    )

    frame["horizon_bin_start_hours"] = (
        (frame["forecast_horizon_hours"] // 24) * 24
    ).astype("Int64")
    horizon_table = (
        frame.groupby("horizon_bin_start_hours", as_index=False)
        .agg(
            rows=("valid_time_utc", "size"),
            mean_price=("price_day_ahead", "mean"),
            price_std=("price_day_ahead", "std"),
            negative_price_rate=("negative_price_flag", "mean"),
            mean_temperature_error=("temperature_absolute_error", "mean"),
            mean_wind_error=("wind_speed_absolute_error", "mean"),
            mean_radiation_error=("shortwave_radiation_absolute_error", "mean"),
            mean_load_forecast_error=("load_forecast_error_mw", "mean"),
        )
        .sort_values("horizon_bin_start_hours")
    )
    horizon_table.to_csv(output_dir / "market_outcomes_by_horizon.csv", index=False)

    regime_table = (
        frame.groupby("high_renewable_regime", as_index=False)
        .agg(
            rows=("valid_time_utc", "size"),
            mean_price=("price_day_ahead", "mean"),
            price_volatility=("price_volatility_24h", "mean"),
            negative_price_rate=("negative_price_flag", "mean"),
            mean_temperature_error=("temperature_absolute_error", "mean"),
            mean_wind_error=("wind_speed_absolute_error", "mean"),
            mean_radiation_error=("shortwave_radiation_absolute_error", "mean"),
        )
    )
    regime_table.to_csv(output_dir / "market_outcomes_by_regime.csv", index=False)

    price_plot = plot_price_and_regimes(frame, output_dir)
    scatter_plot = plot_error_price_scatter(frame, output_dir)
    report = {
        "analysis": "energy_forecast_impact",
        "created_at_utc": utc_now(),
        "gold_batch": str(gold_batch),
        "analysis_output": str(output_dir),
        "rows_analyzed": int(len(frame)),
        "forecast_runs": int(frame["forecast_run_utc"].nunique()),
        "bidding_zones": sorted(frame["bzn"].dropna().unique().tolist()),
        "model_summaries_file": str(output_dir / "model_summaries.csv"),
        "coefficient_file": str(output_dir / "model_coefficients.csv"),
        "horizon_table_file": str(output_dir / "market_outcomes_by_horizon.csv"),
        "regime_table_file": str(output_dir / "market_outcomes_by_regime.csv"),
        "price_plot": str(price_plot),
        "scatter_plot": str(scatter_plot),
        "inference": {
            "standard_errors": "HAC with maxlags=24",
            "fixed_effects": ["hour_utc", "day_of_week"],
            "standardization": "continuous predictors standardized within this Gold batch",
            "interpretation": "conditional association, not causal effect",
        },
        "limitations": [
            "one forecast initialization",
            "one bidding zone",
            "forecast initialization is an availability proxy, not exact publication time",
            "Energy-Charts public power data and weather actuals have different source semantics",
        ],
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Energy forecast impact analysis output: {output_dir}")
    print(f"Rows analyzed: {len(frame):,}")
    print(f"Forecast runs: {report['forecast_runs']}")
    print(f"Models written: {len(model_summaries)}")
    print(f"Report: {output_dir / 'analysis_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
