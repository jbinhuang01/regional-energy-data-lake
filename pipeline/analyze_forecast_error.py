#!/usr/bin/env python3
"""Summarize forecast error by horizon and location."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/regional_energy_matplotlib")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds


DEFAULT_GOLD_ROOT = Path("data_lake/gold/forecast_error")
DEFAULT_ANALYSIS_ROOT = Path("analysis/forecast_error")

ERROR_VARIABLES = {
    "temperature": {
        "error": "temperature_error",
        "absolute": "temperature_absolute_error",
    },
    "wind_speed": {
        "error": "wind_speed_error",
        "absolute": "wind_speed_absolute_error",
    },
    "shortwave_radiation": {
        "error": "shortwave_radiation_error",
        "absolute": "shortwave_radiation_absolute_error",
    },
    "precipitation": {
        "error": "precipitation_error",
        "absolute": "precipitation_absolute_error",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def latest_gold_batch(root: Path) -> Path:
    batches = sorted(root.glob("forecast_batch=*/run=*"))
    if not batches:
        raise FileNotFoundError(
            f"No forecast-error Gold batches found under {root}. "
            "Run build_forecast_error_gold.py first."
        )
    return batches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-batch",
        type=Path,
        help="Specific forecast-error Gold batch; defaults to the latest batch.",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=DEFAULT_GOLD_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in an existing analysis output directory.",
    )
    return parser.parse_args()


def variable_label(variable: str) -> str:
    return {
        "temperature_error": "Temperature",
        "wind_speed_error": "Wind speed",
        "shortwave_radiation_error": "Shortwave radiation",
        "precipitation_error": "Precipitation",
        "temperature": "Temperature",
        "wind_speed": "Wind speed",
        "shortwave_radiation": "Shortwave radiation",
        "precipitation": "Precipitation",
    }.get(variable, variable)


def read_gold_rows(gold_batch: Path) -> pd.DataFrame:
    """Read only the partition data files, not the metric Parquet files."""
    data_files = sorted(gold_batch.rglob("part-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No Gold partition files found under {gold_batch}")
    table = ds.dataset(
        [str(path) for path in data_files],
        format="parquet",
        partitioning="hive",
    ).to_table()
    return table.to_pandas()


def horizon_bin_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["horizon_bin_start_hours"] = (
        (result["forecast_horizon_hours"] // 24) * 24
    ).astype("Int64")
    result["horizon_bin"] = result["horizon_bin_start_hours"].map(
        lambda value: (
            f"{int(value)}-{int(value) + 23}h"
            if pd.notna(value)
            else "unknown"
        )
    )
    return result


def build_group_metrics(gold: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in gold.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys))
        for variable, columns in ERROR_VARIABLES.items():
            error = group[columns["error"]].dropna()
            rows.append(
                {
                    **key_values,
                    "variable": variable,
                    "variable_label": variable_label(variable),
                    "n": int(len(error)),
                    "bias_forecast_minus_actual": float(error.mean())
                    if len(error)
                    else None,
                    "mae": float(error.abs().mean()) if len(error) else None,
                    "rmse": float(np.sqrt((error**2).mean())) if len(error) else None,
                }
            )
    return pd.DataFrame(rows)


def save_horizon_bias_plot(horizon: pd.DataFrame, output_dir: Path) -> Path:
    variables = sorted(horizon["variable"].dropna().unique().tolist())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for axis, variable in zip(axes.ravel(), variables):
        group = horizon[horizon["variable"] == variable].sort_values(
            "horizon_bin_start_hours"
        )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axis.plot(
            group["horizon_bin_start_hours"],
            group["bias_forecast_minus_actual"],
            marker="o",
            linewidth=2,
        )
        axis.set_title(variable_label(variable))
        axis.set_xlabel("Forecast horizon (hours)")
        axis.set_ylabel("Bias: forecast − actual")
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(variables) :]:
        axis.set_visible(False)
    figure.suptitle("Forecast bias by forecast horizon", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "bias_by_forecast_horizon.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_location_plot(location_metrics: pd.DataFrame, output_dir: Path) -> Path:
    variables = sorted(location_metrics["variable"].dropna().unique().tolist())
    locations = sorted(location_metrics["location"].dropna().unique().tolist())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for axis, variable in zip(axes.ravel(), variables):
        group = location_metrics[location_metrics["variable"] == variable]
        pivot = group.pivot(index="location", columns="variable", values="mae")
        values = [pivot.loc[location, variable] for location in locations]
        axis.bar(locations, values, color="#2E74B5")
        axis.set_title(variable_label(variable))
        axis.set_ylabel("Overall MAE (native units)")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.ravel()[len(variables) :]:
        axis.set_visible(False)
    figure.suptitle("Forecast error by location", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "mae_by_location.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_distribution_plot(gold: pd.DataFrame, output_dir: Path) -> Path:
    variables = list(ERROR_VARIABLES.keys())
    horizon_order = sorted(gold["horizon_bin_start_hours"].dropna().unique())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for axis, variable in zip(axes.ravel(), variables):
        absolute_column = ERROR_VARIABLES[variable]["absolute"]
        values = []
        labels = []
        for start in horizon_order:
            group = gold.loc[
                gold["horizon_bin_start_hours"] == start, absolute_column
            ].dropna()
            values.append(group.to_numpy())
            labels.append(f"{int(start)}")
        axis.boxplot(values, labels=labels, showfliers=False)
        axis.set_title(variable_label(variable))
        axis.set_xlabel("Horizon bin start (hours)")
        axis.set_ylabel("Absolute error (native units)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Distribution of absolute forecast error", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "error_distribution_by_horizon.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    gold_batch = args.gold_batch or latest_gold_batch(args.gold_root)
    overall_path = gold_batch / "overall_error_metrics.parquet"
    horizon_path = gold_batch / "horizon_error_metrics.parquet"
    if not overall_path.exists() or not horizon_path.exists():
        raise FileNotFoundError(
            f"Gold batch is missing analysis inputs: {gold_batch}"
        )

    overall = pd.read_parquet(overall_path)
    horizon = pd.read_parquet(horizon_path)
    gold = horizon_bin_columns(read_gold_rows(gold_batch))
    required = {
        "horizon_bin",
        "horizon_bin_start_hours",
        "variable",
        "mae",
        "rmse",
        "bias_forecast_minus_actual",
        "n",
    }
    missing = sorted(required.difference(horizon.columns))
    if missing:
        raise ValueError(f"Horizon metrics missing columns: {missing}")

    horizon = horizon.copy()
    horizon["variable_label"] = horizon["variable"].map(variable_label)
    horizon = horizon.sort_values(
        ["variable", "horizon_bin_start_hours"]
    ).reset_index(drop=True)

    # A simple descriptive trend: regress MAE on horizon for each variable.
    # This is descriptive, not a claim that horizon alone causes the error.
    trend_rows: list[dict] = []
    for variable, group in horizon.groupby("variable", sort=True):
        valid = group.dropna(subset=["horizon_bin_start_hours", "mae"])
        if len(valid) >= 2:
            slope, intercept = np.polyfit(
                valid["horizon_bin_start_hours"], valid["mae"], 1
            )
            correlation = valid["horizon_bin_start_hours"].corr(valid["mae"])
            trend_rows.append(
                {
                    "variable": variable,
                    "variable_label": variable_label(variable),
                    "horizon_bins": int(len(valid)),
                    "mae_slope_per_hour": float(slope),
                    "mae_slope_per_24_hours": float(slope * 24),
                    "horizon_mae_correlation": float(correlation)
                    if pd.notna(correlation)
                    else None,
                    "interpretation": (
                        "descriptive upward MAE trend"
                        if slope > 0
                        else "no descriptive upward MAE trend"
                    ),
                }
            )
    trends = pd.DataFrame(trend_rows)

    output_dir = args.output_root / gold_batch.name
    if output_dir.exists() and (output_dir / "analysis_report.json").exists() and not args.overwrite:
        raise SystemExit(f"Analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    location_metrics = build_group_metrics(gold, ["location"])
    location_horizon_metrics = build_group_metrics(
        gold, ["location", "horizon_bin", "horizon_bin_start_hours"]
    )

    overall.to_csv(output_dir / "overall_error_metrics.csv", index=False)
    horizon.to_csv(output_dir / "horizon_error_metrics.csv", index=False)
    trends.to_csv(output_dir / "horizon_trends.csv", index=False)
    location_metrics.to_csv(output_dir / "location_error_metrics.csv", index=False)
    location_horizon_metrics.to_csv(
        output_dir / "location_horizon_error_metrics.csv", index=False
    )

    variables = sorted(horizon["variable"].dropna().unique().tolist())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    axes_flat = axes.ravel()
    for axis, variable in zip(axes_flat, variables):
        group = horizon[horizon["variable"] == variable].copy()
        group = group.sort_values("horizon_bin_start_hours")
        axis.plot(
            group["horizon_bin_start_hours"],
            group["mae"],
            marker="o",
            linewidth=2,
        )
        axis.set_title(variable_label(variable))
        axis.set_xlabel("Forecast horizon (hours)")
        axis.set_ylabel("MAE (native units)")
        axis.grid(True, alpha=0.25)
    for axis in axes_flat[len(variables) :]:
        axis.set_visible(False)
    figure.suptitle("Weather forecast error by forecast horizon", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = output_dir / "mae_by_forecast_horizon.png"
    figure.savefig(plot_path, dpi=180)
    plt.close()

    bias_plot_path = save_horizon_bias_plot(horizon, output_dir)
    location_plot_path = save_location_plot(location_metrics, output_dir)
    distribution_plot_path = save_distribution_plot(gold, output_dir)

    positive_trends = trends[trends["mae_slope_per_hour"] > 0]
    report = {
        "analysis": "forecast_error_by_horizon",
        "created_at_utc": utc_now(),
        "gold_batch": str(gold_batch),
        "analysis_output": str(output_dir),
        "overall_metrics_file": str(output_dir / "overall_error_metrics.csv"),
        "horizon_metrics_file": str(output_dir / "horizon_error_metrics.csv"),
        "trend_file": str(output_dir / "horizon_trends.csv"),
        "plot_file": str(plot_path),
        "bias_plot_file": str(bias_plot_path),
        "location_plot_file": str(location_plot_path),
        "distribution_plot_file": str(distribution_plot_path),
        "location_metrics_file": str(output_dir / "location_error_metrics.csv"),
        "location_horizon_metrics_file": str(
            output_dir / "location_horizon_error_metrics.csv"
        ),
        "variables": sorted(horizon["variable"].dropna().unique().tolist()),
        "horizon_bins": int(horizon["horizon_bin"].nunique()),
        "locations": sorted(gold["location"].dropna().unique().tolist()),
        "variables_with_descriptive_upward_mae_trend": positive_trends[
            "variable"
        ].tolist(),
        "interpretation_note": (
            "The horizon trend is descriptive. It does not establish causality, "
            "and this single forecast run is not sufficient for uncertainty "
            "estimation across forecast initializations."
        ),
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Forecast-error analysis output: {output_dir}")
    print(f"Horizon bins: {report['horizon_bins']}")
    print(
        "Upward descriptive MAE trends: "
        f"{report['variables_with_descriptive_upward_mae_trend']}"
    )
    print(f"Plot: {plot_path}")
    print(f"Report: {output_dir / 'analysis_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
