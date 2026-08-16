#!/usr/bin/env python3
"""Summarize the original energy/weather Gold panel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.formula.api as smf


CORE_COLUMNS = [
    "event_time_utc",
    "price_day_ahead_eur_mwh",
    "negative_price_flag",
    "price_spike_flag",
    "renewable_share",
    "load_actual_mw",
    "load_forecast_error_mw",
    "temperature_2m_c_mean",
    "wind_speed_10m_kmh_mean",
    "precipitation_mm_mean",
    "rain_flag",
    "high_wind_flag",
    "cold_weather_flag",
    "hour_utc",
    "day_of_week",
    "year",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_latest_panel(root: Path) -> Path:
    panels = sorted(root.glob("region=*/energy_batch=*__weather_batch=*"))
    if not panels:
        raise FileNotFoundError(f"No energy-weather panels found under {root}")
    return panels[-1]


def read_panel(panel_dir: Path) -> pd.DataFrame:
    parquet_files = [str(path) for path in panel_dir.rglob("*.parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {panel_dir}")
    dataset = ds.dataset(parquet_files, format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    missing = sorted(set(CORE_COLUMNS) - available)
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    frame = dataset.to_table(columns=CORE_COLUMNS).to_pandas()
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True)
    return frame.sort_values("event_time_utc").reset_index(drop=True)


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "all": pd.Series(True, index=frame.index),
        "rain": frame["rain_flag"],
        "no_rain": ~frame["rain_flag"],
        "high_wind": frame["high_wind_flag"],
        "not_high_wind": ~frame["high_wind_flag"],
        "cold": frame["cold_weather_flag"],
        "not_cold": ~frame["cold_weather_flag"],
    }
    rows = []
    for name, mask in conditions.items():
        subset = frame.loc[mask]
        rows.append(
            {
                "condition": name,
                "rows": len(subset),
                "mean_price_eur_mwh": subset["price_day_ahead_eur_mwh"].mean(),
                "median_price_eur_mwh": subset["price_day_ahead_eur_mwh"].median(),
                "negative_price_rate": subset["negative_price_flag"].mean(),
                "price_spike_rate": subset["price_spike_flag"].mean(),
                "mean_renewable_share": subset["renewable_share"].mean(),
            }
        )
    return pd.DataFrame(rows)


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    model = frame.copy()
    # Statsmodels formula models need an explicit numeric 0/1 outcome. A
    # boolean column can otherwise be expanded into two endogenous columns.
    model["negative_price_flag"] = model["negative_price_flag"].astype("int8")
    model["renewable_share_pct"] = model["renewable_share"] * 100
    model["load_forecast_error_gw"] = model["load_forecast_error_mw"] / 1000
    variables = [
        "price_day_ahead_eur_mwh",
        "negative_price_flag",
        "renewable_share_pct",
        "temperature_2m_c_mean",
        "wind_speed_10m_kmh_mean",
        "load_forecast_error_gw",
        "hour_utc",
        "day_of_week",
        "year",
    ]
    return model.dropna(subset=variables).copy()


def fit_model(data: pd.DataFrame, formula: str, outcome: str) -> pd.DataFrame:
    result = smf.ols(formula, data=data).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 24},
    )
    table = pd.DataFrame(
        {
            "outcome": outcome,
            "term": result.params.index,
            "coefficient": result.params.values,
            "std_error_hac": result.bse.values,
            "t_value": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_low": result.conf_int()[0].values,
            "ci_high": result.conf_int()[1].values,
        }
    )
    table.attrs["r_squared"] = float(result.rsquared)
    table.attrs["observations"] = int(result.nobs)
    return table


def create_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plot_data = frame.dropna(
        subset=["renewable_share", "price_day_ahead_eur_mwh"]
    ).copy()
    plot_data["renewable_share_pct"] = plot_data["renewable_share"] * 100
    plot_data["renewable_bin"] = pd.qcut(
        plot_data["renewable_share_pct"], q=20, duplicates="drop"
    )
    grouped = (
        plot_data.groupby("renewable_bin", observed=True)
        .agg(
            renewable_share_pct=("renewable_share_pct", "mean"),
            mean_price=("price_day_ahead_eur_mwh", "mean"),
            negative_rate=("negative_price_flag", "mean"),
        )
        .reset_index(drop=True)
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        grouped["renewable_share_pct"],
        grouped["mean_price"],
        marker="o",
        color="#1769aa",
    )
    axes[0].set_title("Mean day-ahead price by renewable share")
    axes[0].set_xlabel("Renewable share (%)")
    axes[0].set_ylabel("Mean price (EUR/MWh)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        grouped["renewable_share_pct"],
        grouped["negative_rate"] * 100,
        marker="o",
        color="#c44e52",
    )
    axes[1].set_title("Negative-price rate by renewable share")
    axes[1].set_xlabel("Renewable share (%)")
    axes[1].set_ylabel("Negative-price rate (%)")
    axes[1].grid(alpha=0.25)

    figure.suptitle("DE-LU energy-weather research panel")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=Path("data_lake/gold/energy_weather_panel"),
    )
    parser.add_argument("--panel", type=Path, help="Specific panel directory.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/energy_weather"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel_dir = args.panel or find_latest_panel(args.panel_root)
    run_id = panel_dir.name
    output_dir = args.output_root / run_id
    if output_dir.exists():
        raise SystemExit(f"Analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_panel(panel_dir)
    summary = condition_summary(frame)
    summary.to_csv(output_dir / "condition_summary.csv", index=False)

    correlation_columns = [
        "price_day_ahead_eur_mwh",
        "renewable_share",
        "load_actual_mw",
        "load_forecast_error_mw",
        "temperature_2m_c_mean",
        "wind_speed_10m_kmh_mean",
        "precipitation_mm_mean",
    ]
    frame[correlation_columns].corr(method="spearman").to_csv(
        output_dir / "spearman_correlation.csv"
    )

    model_data = model_frame(frame)
    formula_terms = (
        "renewable_share_pct + temperature_2m_c_mean + "
        "wind_speed_10m_kmh_mean + load_forecast_error_gw + "
        "C(hour_utc) + C(day_of_week) + C(year)"
    )
    price_model = fit_model(
        model_data,
        f"price_day_ahead_eur_mwh ~ {formula_terms}",
        "price_day_ahead_eur_mwh",
    )
    negative_model = fit_model(
        model_data,
        f"negative_price_flag ~ {formula_terms}",
        "negative_price_flag_lpm",
    )
    model_results = pd.concat([price_model, negative_model], ignore_index=True)
    model_results.to_csv(output_dir / "hac_regression_results.csv", index=False)

    create_plot(frame, output_dir / "renewable_share_price_relationship.png")

    report = {
        "panel": str(panel_dir),
        "created_at_utc": utc_now(),
        "rows_in_panel": len(frame),
        "rows_in_models": len(model_data),
        "start_time_utc": frame["event_time_utc"].min().isoformat(),
        "end_time_utc": frame["event_time_utc"].max().isoformat(),
        "negative_price_rows": int(frame["negative_price_flag"].sum()),
        "price_spike_rows": int(frame["price_spike_flag"].sum()),
        "price_spike_threshold_eur_mwh": 100,
        "model_formula_terms": formula_terms,
        "covariance": "HAC/Newey-West with 24 hourly lags",
        "interpretation_note": (
            "Results describe conditional associations. They are not causal "
            "estimates because weather, generation and market conditions are "
            "not randomized."
        ),
        "model_r_squared": {
            "price": float(price_model.attrs["r_squared"]),
            "negative_price_lpm": float(negative_model.attrs["r_squared"]),
        },
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Analysis output: {output_dir}")
    print(f"Panel rows: {len(frame):,}")
    print(f"Model rows: {len(model_data):,}")
    print(f"Report: {output_dir / 'analysis_report.json'}")
    print("Energy-weather analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
