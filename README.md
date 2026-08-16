# Regional Energy Data Lake

This project studies one practical question:

> When renewable generation is high, do weather forecast errors line up with
> unusual electricity prices?

I built the data pipeline first and added the market analysis after the time
semantics were stable. The current panel covers June 2024, 30 forecast runs,
and three bidding zones: `DE-LU`, `FR`, and `AT`.

## What is in the repository

```text
pipeline/    ingestion, validation and Bronze/Silver/Gold jobs
analysis/    exported tables, model output and figures
metadata/    registry, contracts and validation reports
docs/        design notes and supporting scripts
```

The raw downloads are not committed. They are stored locally under
`data_lake/`, which is ignored by Git. Every download writes its request
parameters, retrieval time and checksum so the raw layer can be rebuilt.

## Current panel

| Item | Value |
|---|---:|
| Forecast runs | 30, every 12 hours |
| Weather points | Frankfurt, Paris, Vienna |
| Bidding zones | DE-LU, FR, AT |
| Common valid hours | 588 per zone |
| Market-impact rows | 21,600 |
| Market match rate | 100% |

The final cross-zone analysis is under:

```text
analysis/multi_zone_multi_vintage_v2/panel_batch=20260816T110610Z/
```

The main files are:

- `model_coefficients.csv`
- `model_summaries.csv`
- `zone_coverage.csv`
- `zone_regime_summary.csv`
- `market_regimes_by_zone.png`
- `regime_interactions.png`

The model includes bidding-zone, date and hour fixed effects. Standard errors
are clustered by valid hour and forecast run because the same market outcome
appears in several forecast vintages.

## Data path

```text
Bronze  raw API responses, request metadata and manifests
   ↓
Silver  typed event-level Parquet, preserving source resolution
   ↓
Gold    hourly market/weather panel with forecast horizon and lineage
```

Forecast run time and forecast valid time are kept as separate fields. This
matters because the same delivery hour can be predicted by many different
model runs.

## Reproducing the main branch

The following commands show the shape of the current workflow. The actual
download commands use the same scripts with explicit batch paths when a batch
has already been ingested.

```bash
python3 pipeline/ingest_weather_bronze.py \
  --start-date 2024-06-01 \
  --end-date 2024-06-30 \
  --locations frankfurt paris vienna

python3 pipeline/ingest_weather_vintage_panel.py \
  --start-run 2024-06-01T00:00 \
  --run-count 30 \
  --interval-hours 12 \
  --forecast-days 10 \
  --locations frankfurt paris vienna

python3 pipeline/build_weather_silver.py \
  --bronze-batch <weather-bronze-batch>

python3 pipeline/build_weather_vintage_panel_silver.py \
  --bronze-batch <forecast-bronze-batch>

python3 pipeline/build_multi_vintage_forecast_error_gold.py \
  --forecast-silver-batch <forecast-silver-batch> \
  --actual-weather-silver-batch <actual-weather-silver-batch>

python3 pipeline/analyze_multi_zone_multi_vintage_impact.py \
  --gold-batch <multi-zone-gold-batch>
```

Market ingestion is run once per zone with the Energy-Charts `/price`,
`/public_power` and optional `/public_power_forecast` endpoints. Source
availability is recorded when a zone does not publish a series such as
offshore wind forecasts.

## What the current results say

High-renewable hours are cheaper in all three zones. In the June sample, the
mean price was €105.63/MWh in the low-renewable DE-LU group and €35.75/MWh in
the high-renewable group. The corresponding values were €90.03 and €40.87 for
AT, and €51.95 and €12.26 for FR.

The cross-zone model finds a strong residual-load relationship with price. It
also finds that the relationship between forecast errors and market outcomes
changes by renewable regime. These are sample-specific associations, not a
claim that a weather error by itself caused a price event. The study currently
uses one representative weather point per zone and one month of data.

## Design notes

The decisions that shaped the current layout are recorded in
[`docs/decision_log.md`](docs/decision_log.md). In particular, it explains the
UTC boundary issue in the Energy-Charts API, the unavailable 06 UTC forecast
run, and the missing Austrian offshore forecast series.

## Sources

- [Open Power System Data time series](https://data.open-power-system-data.org/time_series/2020-10-06)
- [Energy-Charts API](https://api.energy-charts.info/)
- [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api)
