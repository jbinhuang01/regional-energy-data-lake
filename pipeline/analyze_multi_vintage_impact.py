#!/usr/bin/env python3
"""Estimate forecast-error effects after adding multiple forecast vintages.

The model uses valid-date and hour fixed effects.  Standard errors are
two-way clustered by valid hour and forecast run, because observations from
the same event and observations from the same model run are not independent.
This is still an association design, but it is a more defensible robustness
check than fitting one time series with HAC errors.
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
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
import pyarrow.dataset as ds


PREDICTORS = [
    "temperature_absolute_error",
    "wind_speed_absolute_error",
    "shortwave_radiation_absolute_error",
    "precipitation_absolute_error",
    "load_forecast_error_mw",
    "solar_forecast_error_mw",
    "wind_onshore_forecast_error_mw",
    "actual_residual_load",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-batch", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("analysis/multi_vintage_impact")
    )
    return parser.parse_args()


def read_gold(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Gold Parquet files under {path}")
    return ds.dataset(
        [str(file) for file in files], format="parquet", partitioning="hive"
    ).to_table().to_pandas()


def zscore(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame[column], errors="coerce")
    output = f"z_{column}"
    std = values.std()
    frame[output] = (values - values.mean()) / std if std and np.isfinite(std) else 0.0
    return output


def fit_two_way_cluster(
    frame: pd.DataFrame, outcome: str, formula: str, model_name: str
) -> tuple[pd.DataFrame, dict]:
    data = frame.dropna(subset=[outcome]).copy()
    if len(data) < 100:
        return pd.DataFrame(), {
            "model": model_name,
            "outcome": outcome,
            "status": "SKIPPED",
            "n": int(len(data)),
            "reason": "fewer than 100 complete observations",
        }
    try:
        result = smf.ols(formula, data=data).fit()
        # Patsy/statsmodels may drop rows with missing predictors while
        # building the design matrix.  Cluster labels must be taken from the
        # exact rows used by the fitted model, not from the pre-fit frame.
        used_rows = data.loc[result.model.data.row_labels]
        valid_time_groups = pd.factorize(used_rows["valid_time_utc"].astype(str))[0]
        forecast_run_groups = pd.factorize(used_rows["forecast_run_utc"].astype(str))[0]
        covariance = cov_cluster_2groups(
            result,
            valid_time_groups,
            forecast_run_groups,
        )[0]
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
        cluster_df = max(
            1,
            min(
                used_rows["valid_time_utc"].nunique(),
                used_rows["forecast_run_utc"].nunique(),
            )
            - 1,
        )
        t_critical = stats.t.ppf(0.975, cluster_df)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return pd.DataFrame(), {
            "model": model_name,
            "outcome": outcome,
            "status": "FAILED",
            "n": int(len(data)),
            "reason": str(exc),
        }

    params = result.params.to_numpy()
    t_values = np.divide(
        params,
        standard_errors,
        out=np.full_like(params, np.nan, dtype=float),
        where=standard_errors > 0,
    )
    p_values = 2 * stats.t.sf(np.abs(t_values), df=cluster_df)
    rows = []
    for index, term in enumerate(result.params.index):
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "term": term,
                "coefficient": float(params[index]),
                "std_error_two_way_clustered": float(standard_errors[index]),
                "t_value": float(t_values[index]),
                "p_value": float(p_values[index]),
                "ci_low": float(params[index] - t_critical * standard_errors[index]),
                "ci_high": float(params[index] + t_critical * standard_errors[index]),
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
        "adjusted_r_squared": float(result.rsquared_adj),
        "cluster_dimensions": ["valid_time_utc", "forecast_run_utc"],
        "smallest_cluster_count": int(
                min(
                    used_rows["valid_time_utc"].nunique(),
                    used_rows["forecast_run_utc"].nunique(),
                )
        ),
        "inference_df": int(cluster_df),
    }
    return pd.DataFrame(rows), summary


def make_plots(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    plot_paths: list[str] = []
    horizon = (
        frame.groupby("horizon_bin_start_hours", as_index=False)
        .agg(
            temperature_mae=("temperature_absolute_error", "mean"),
            wind_mae=("wind_speed_absolute_error", "mean"),
            radiation_mae=("shortwave_radiation_absolute_error", "mean"),
            precipitation_mae=("precipitation_absolute_error", "mean"),
            rows=("valid_time_utc", "size"),
        )
        .sort_values("horizon_bin_start_hours")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for column, label in [
        ("temperature_mae", "Temperature"),
        ("wind_mae", "Wind"),
        ("radiation_mae", "Shortwave radiation"),
        ("precipitation_mae", "Precipitation"),
    ]:
        ax.plot(horizon["horizon_bin_start_hours"], horizon[column], marker="o", label=label)
    ax.set_xlabel("Forecast horizon bin start (hours)")
    ax.set_ylabel("Mean absolute forecast error")
    ax.set_title("Forecast error across vintages and horizons")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output_dir / "forecast_error_by_horizon.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path))

    price = frame.groupby("forecast_run_utc", as_index=False).agg(
        mean_price=("price_day_ahead", "mean"),
        negative_price_rate=("negative_price_flag", "mean"),
        rows=("valid_time_utc", "size"),
    )
    price["forecast_run_utc"] = pd.to_datetime(price["forecast_run_utc"])
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(price["forecast_run_utc"], price["mean_price"], marker="o", color="#2E74B5")
    ax1.set_ylabel("Mean day-ahead price (EUR/MWh)", color="#2E74B5")
    ax1.tick_params(axis="y", labelcolor="#2E74B5")
    ax2 = ax1.twinx()
    ax2.plot(price["forecast_run_utc"], price["negative_price_rate"], marker="s", color="#C45A00")
    ax2.set_ylabel("Negative-price rate", color="#C45A00")
    ax2.tick_params(axis="y", labelcolor="#C45A00")
    ax1.set_title("Same market window repeated across forecast vintages")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "market_outcomes_by_vintage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path))
    return plot_paths


def main() -> int:
    args = parse_args()
    frame = read_gold(args.gold_batch)
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    frame["forecast_run_utc"] = pd.to_datetime(frame["forecast_run_utc"], utc=True)
    frame = frame[frame["market_match_flag"]].copy()
    if frame.empty:
        raise SystemExit("No market-matched rows available")
    frame["hour_utc"] = frame["valid_time_utc"].dt.hour
    frame["valid_date"] = frame["valid_time_utc"].dt.strftime("%Y-%m-%d")
    frame["horizon_bin_start_hours"] = (
        (pd.to_numeric(frame["forecast_horizon_hours"], errors="coerce") // 24) * 24
    ).astype("Int64")
    frame["high_renewable_regime"] = (
        frame["actual_renewable_share_of_load"]
        >= frame["actual_renewable_share_of_load"].median()
    ).astype(int)
    z_predictors = [zscore(frame, column) for column in PREDICTORS]
    base_terms = " + ".join(z_predictors + ["C(hour_utc)", "C(valid_date)"])
    interaction_predictors = z_predictors[:4]
    interactions = " + ".join(
        f"{column}:high_renewable_regime" for column in interaction_predictors
    )
    interaction_terms = f"{base_terms} + high_renewable_regime + {interactions}"
    models = [
        ("price_level_vintage_panel", "price_day_ahead", f"price_day_ahead ~ {base_terms}"),
        (
            "price_level_regime_interactions",
            "price_day_ahead",
            f"price_day_ahead ~ {interaction_terms}",
        ),
        (
            "price_volatility_vintage_panel",
            "price_volatility_24h",
            f"price_volatility_24h ~ {base_terms}",
        ),
        (
            "negative_price_regime_interactions",
            "negative_price_flag",
            f"negative_price_flag ~ {interaction_terms}",
        ),
    ]
    coefficient_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for model_name, outcome, formula in models:
        coefficients, summary = fit_two_way_cluster(frame, outcome, formula, model_name)
        if not coefficients.empty:
            coefficient_frames.append(coefficients)
        summaries.append(summary)

    output_dir = args.output_root / args.gold_batch.name
    if output_dir.exists() and (output_dir / "analysis_report.json").exists():
        raise SystemExit(f"Analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(coefficient_frames, ignore_index=True).to_csv(
        output_dir / "model_coefficients.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(output_dir / "model_summaries.csv", index=False)

    vintage_coverage = (
        frame.groupby("forecast_run_utc", as_index=False)
        .agg(
            rows=("valid_time_utc", "size"),
            valid_start_utc=("valid_time_utc", "min"),
            valid_end_utc=("valid_time_utc", "max"),
            mean_price=("price_day_ahead", "mean"),
            negative_price_rate=("negative_price_flag", "mean"),
            mean_temperature_error=("temperature_absolute_error", "mean"),
            mean_radiation_error=("shortwave_radiation_absolute_error", "mean"),
        )
        .sort_values("forecast_run_utc")
    )
    vintage_coverage.to_csv(output_dir / "vintage_coverage.csv", index=False)

    horizon_stability = (
        frame.groupby(["horizon_bin_start_hours", "high_renewable_regime"], as_index=False)
        .agg(
            rows=("valid_time_utc", "size"),
            mean_price=("price_day_ahead", "mean"),
            negative_price_rate=("negative_price_flag", "mean"),
            mean_temperature_error=("temperature_absolute_error", "mean"),
            mean_radiation_error=("shortwave_radiation_absolute_error", "mean"),
        )
        .sort_values(["horizon_bin_start_hours", "high_renewable_regime"])
    )
    horizon_stability.to_csv(output_dir / "horizon_regime_stability.csv", index=False)
    plot_paths = make_plots(frame, output_dir)
    report = {
        "analysis": "multi_vintage_energy_forecast_impact",
        "created_at_utc": utc_now(),
        "gold_batch": str(args.gold_batch),
        "rows_analyzed": int(len(frame)),
        "forecast_runs": int(frame["forecast_run_utc"].nunique()),
        "valid_hours": int(frame["valid_time_utc"].nunique()),
        "models": [summary["model"] for summary in summaries],
        "model_summaries_file": str(output_dir / "model_summaries.csv"),
        "coefficient_file": str(output_dir / "model_coefficients.csv"),
        "vintage_coverage_file": str(output_dir / "vintage_coverage.csv"),
        "horizon_regime_file": str(output_dir / "horizon_regime_stability.csv"),
        "plots": plot_paths,
        "inference": {
            "fixed_effects": ["hour_utc", "valid_date"],
            "standard_errors": "two-way clustered by valid_time_utc and forecast_run_utc",
            "continuous_predictors": "standardized within the multi-vintage Gold panel",
            "interpretation": "conditional association, not causal effect",
        },
        "limitations": [
            "forecast run count is still small for cluster-based inference",
            "one bidding zone and one ten-day market window",
            "forecast_run_utc is model initialization time, not exact provider publication time",
            "repeated market outcomes across vintages require event/run clustered inference",
        ],
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Multi-vintage analysis output: {output_dir}")
    print(f"Rows: {len(frame):,}; runs: {report['forecast_runs']}; valid hours: {report['valid_hours']}")
    print(f"Report: {output_dir / 'analysis_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
