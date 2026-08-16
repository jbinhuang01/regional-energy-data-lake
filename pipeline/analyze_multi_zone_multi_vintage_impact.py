#!/usr/bin/env python3
"""Analyze the multi-zone, multi-vintage market-impact Gold panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_multi_vintage_impact import fit_two_way_cluster  # noqa: E402


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
        "--output-root", type=Path, default=Path("analysis/multi_zone_multi_vintage")
    )
    return parser.parse_args()


def read_gold(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Gold Parquet files under {path}")
    return ds.dataset(
        [str(file) for file in files], format="parquet", partitioning="hive"
    ).to_table().to_pandas()


def groupwise_zscore(frame: pd.DataFrame, column: str) -> str:
    output = f"z_{column}"
    values = pd.to_numeric(frame[column], errors="coerce")
    means = values.groupby(frame["bzn"]).transform("mean")
    stds = values.groupby(frame["bzn"]).transform("std")
    frame[output] = (values - means) / stds.replace(0, np.nan)
    return output


def make_plots(frame: pd.DataFrame, coefficients: pd.DataFrame, output_dir: Path) -> list[str]:
    paths: list[str] = []
    regime = (
        frame.groupby(["bzn", "high_renewable_regime"], as_index=False)
        .agg(
            mean_price=("price_day_ahead", "mean"),
            negative_price_rate=("negative_price_flag", "mean"),
            rows=("valid_time_utc", "size"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for zone, group in regime.groupby("bzn"):
        axes[0].plot(group["high_renewable_regime"], group["mean_price"], marker="o", label=zone)
        axes[1].plot(group["high_renewable_regime"], group["negative_price_rate"], marker="o", label=zone)
    axes[0].set_title("Mean price by renewable regime")
    axes[0].set_ylabel("EUR/MWh")
    axes[1].set_title("Negative-price rate by renewable regime")
    axes[1].set_ylabel("Probability")
    for axis in axes:
        axis.set_xlabel("High renewable regime (0/1)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "market_regimes_by_zone.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    interaction = coefficients[
        coefficients.term.str.contains(":high_renewable_regime", regex=False)
    ].copy()
    if not interaction.empty:
        interaction["label"] = interaction["term"].str.replace(
            "z_", "", regex=False
        ).str.replace(":high_renewable_regime", "", regex=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in interaction.groupby("model"):
            ax.errorbar(
                group["label"],
                group["coefficient"],
                yerr=[
                    group["coefficient"] - group["ci_low"],
                    group["ci_high"] - group["coefficient"],
                ],
                fmt="o",
                capsize=3,
                label=model,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Interaction coefficient")
        ax.set_title("Forecast-error heterogeneity by renewable regime")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output_dir / "regime_interactions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> int:
    args = parse_args()
    frame = read_gold(args.gold_batch)
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    frame["forecast_run_utc"] = pd.to_datetime(frame["forecast_run_utc"], utc=True)
    frame = frame[frame["market_match_flag"]].copy()
    frame["hour_utc"] = frame["valid_time_utc"].dt.hour
    frame["valid_date"] = frame["valid_time_utc"].dt.strftime("%Y-%m-%d")
    frame["horizon_bin_start_hours"] = (
        (pd.to_numeric(frame["forecast_horizon_hours"], errors="coerce") // 24) * 24
    ).astype("Int64")
    frame["high_renewable_regime"] = (
        frame["actual_renewable_share_of_load"]
        >= frame.groupby("bzn")["actual_renewable_share_of_load"].transform("median")
    ).astype(int)
    z_predictors = [groupwise_zscore(frame, column) for column in PREDICTORS]
    base_terms = " + ".join(z_predictors + ["C(bzn)", "C(hour_utc)", "C(valid_date)"])
    interactions = " + ".join(
        f"{column}:high_renewable_regime" for column in z_predictors[:4]
    )
    interaction_terms = f"{base_terms} + high_renewable_regime + {interactions}"
    models = [
        ("price_level_multi_zone", "price_day_ahead", f"price_day_ahead ~ {base_terms}"),
        (
            "price_level_regime_interactions_multi_zone",
            "price_day_ahead",
            f"price_day_ahead ~ {interaction_terms}",
        ),
        (
            "price_volatility_multi_zone",
            "price_volatility_24h",
            f"price_volatility_24h ~ {base_terms}",
        ),
        (
            "negative_price_regime_interactions_multi_zone",
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
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "model_summaries.csv", index=False)
    coverage = (
        frame.groupby("bzn", as_index=False)
        .agg(
            rows=("valid_time_utc", "size"),
            valid_hours=("valid_time_utc", "nunique"),
            forecast_runs=("forecast_run_utc", "nunique"),
            mean_price=("price_day_ahead", "mean"),
            negative_price_rate=("negative_price_flag", "mean"),
        )
    )
    coverage.to_csv(output_dir / "zone_coverage.csv", index=False)
    regime = (
        frame.groupby(["bzn", "high_renewable_regime"], as_index=False)
        .agg(
            rows=("valid_time_utc", "size"),
            mean_price=("price_day_ahead", "mean"),
            negative_price_rate=("negative_price_flag", "mean"),
            mean_temperature_error=("temperature_absolute_error", "mean"),
            mean_radiation_error=("shortwave_radiation_absolute_error", "mean"),
        )
    )
    regime.to_csv(output_dir / "zone_regime_summary.csv", index=False)
    plots = make_plots(frame, coefficients, output_dir)
    report = {
        "analysis": "multi_zone_multi_vintage_energy_forecast_impact",
        "created_at_utc": utc_now(),
        "gold_batch": str(args.gold_batch),
        "rows_analyzed": int(len(frame)),
        "zones": sorted(frame["bzn"].unique().tolist()),
        "zone_count": int(frame["bzn"].nunique()),
        "forecast_runs": int(frame["forecast_run_utc"].nunique()),
        "valid_hours_by_zone": coverage.set_index("bzn")["valid_hours"].to_dict(),
        "model_summaries_file": str(output_dir / "model_summaries.csv"),
        "coefficient_file": str(output_dir / "model_coefficients.csv"),
        "zone_coverage_file": str(output_dir / "zone_coverage.csv"),
        "zone_regime_file": str(output_dir / "zone_regime_summary.csv"),
        "plots": plots,
        "inference": {
            "fixed_effects": ["bzn", "hour_utc", "valid_date"],
            "standard_errors": "two-way clustered by valid_time_utc and forecast_run_utc",
            "standardization": "within-zone standardization for continuous predictors",
            "interpretation": "conditional association, not causal effect",
        },
        "limitations": [
            "one representative weather point per bidding zone",
            "forecast run count is common across zones and still limited by one month",
            "market outcomes are repeated across forecast vintages",
        ],
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Multi-zone analysis output: {output_dir}")
    print(f"Rows: {len(frame):,}; zones: {report['zone_count']}; runs: {report['forecast_runs']}")
    print(f"Report: {output_dir / 'analysis_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
