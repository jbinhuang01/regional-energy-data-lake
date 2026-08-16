# Regional Energy Data Lake

The first milestone is a reproducible Bronze ingestion pipeline for the
Open Power System Data time-series package. The package provides hourly load,
wind, solar and price data for European power systems.

## First milestone: Bronze

Bronze keeps the downloaded source files unchanged and writes a JSON Lines
manifest containing the source URL, ingestion timestamp, file size and SHA-256
checksum. Each run creates a new batch directory and does not overwrite prior
runs.

```bash
cd /Users/j.bhuang/Documents/working/regional-energy-data-lake
python3 pipeline/ingest_bronze.py
```

The downloader retries transient network failures and keeps a `.part` file
when the connection breaks. To continue a failed batch, reuse its batch ID:

```bash
python3 pipeline/ingest_bronze.py \
  --batch-id 20260804T114958Z
```

The source server must support HTTP Range requests for byte-level resume. If
it does not, the script safely restarts that file rather than appending a
duplicate response.

The default output is:

```text
data_lake/bronze/opsd_time_series/
└── ingestion_date=YYYYMMDD/
    └── batch_id=YYYYMMDDTHHMMSSZ/
        ├── time_series_60min_singleindex.csv
        ├── README.md
        ├── datapackage.json
        └── manifest.jsonl
```

## Current result: multi-zone forecast-impact panel

The current research branch uses 30 archived forecast vintages at 00/12 UTC,
30 days of hourly weather, and three bidding zones: `DE-LU`, `FR` and `AT`.
Each zone is joined to one representative weather point: Frankfurt, Paris or
Vienna. The Gold panel retains `forecast_run_utc`, `valid_time_utc`,
`forecast_horizon_hours`, `bzn` and source batch IDs.

The pipeline can be rerun with:

```bash
python3 pipeline/ingest_weather_bronze.py \
  --start-date 2024-06-01 --end-date 2024-06-30 \
  --locations frankfurt paris vienna

python3 pipeline/ingest_weather_vintage_panel.py \
  --start-run 2024-06-01T00:00 --run-count 30 \
  --interval-hours 12 --forecast-days 10 \
  --locations frankfurt paris vienna

python3 pipeline/analyze_multi_zone_multi_vintage_impact.py \
  --gold-batch data_lake/gold/multi_zone_forecast_impact_v2/panel_batch=20260816T110610Z
```

The latest analysis is under
`analysis/multi_zone_multi_vintage_v2/panel_batch=20260816T110610Z/` and
contains model coefficients, zone coverage, regime summaries and PNG figures.
Raw downloaded data remains outside Git under `data_lake/`; it is reproducible
from the recorded request metadata and manifests.

The 60-minute CSV is intentionally large. To preview the pipeline without
downloading anything, use:

```bash
python3 pipeline/ingest_bronze.py --help
```

Do not commit `data_lake/` to Git. The raw files are reproducible from the
source URL and can be restored using the ingestion script.

## Data source

Open Power System Data, Time series package, version 2020-10-06:

https://data.open-power-system-data.org/time_series/2020-10-06

The next milestone will convert the raw CSV into a Silver Parquet table with
UTC timestamps, normalized column names, typed numeric values and data-quality
checks.

## Second milestone: Silver

The Silver job reads the latest completed Bronze batch, converts the wide CSV
to a long-format Parquet dataset, and partitions it by year and month:

```bash
python3 pipeline/build_silver.py
```

To process a specific Bronze batch:

```bash
python3 pipeline/build_silver.py \
  --bronze-batch data_lake/bronze/opsd_time_series/ingestion_date=20260804/batch_id=20260804T115229Z
```

Silver rows contain `source`, `region`, `metric`, `event_time_utc`,
`event_time_local`, `value`, `unit`, `ingested_at_utc` and `batch_id`.
The output also includes `quality_report.json`. The script requires `pandas`
and `pyarrow`.

## Third milestone: Gold research mart

The first Gold mart uses the DE-LU bidding zone and creates hourly features
for renewable penetration, load forecast error and price events:

```bash
python3 pipeline/build_gold.py
```

The result is partitioned Parquet under `data_lake/gold/energy_research/` and
includes `gold_report.json`. The price spike threshold is fixed at
`100 EUR/MWh` and is recorded in the source code so the research result is
reproducible.

## Fifth milestone: Weather Silver

Convert the weather Bronze JSON into one row per location and UTC hour:

```bash
python3 pipeline/build_weather_silver.py
```

The output is partitioned by `year`, `month` and `location`. Missing values are
not imputed in Silver; the job records missing values, duplicate keys and
missing hourly intervals in `quality_report.json`.

## Sixth milestone: Energy-weather Gold panel

Join the energy Gold mart with the weather Silver dataset using UTC hour:

```bash
python3 pipeline/build_energy_weather_gold.py
```

The join is a left join from energy to weather, so missing weather coverage is
visible instead of silently deleting energy observations. Weather features are
aggregated across the two locations, while `weather_location_count` preserves
coverage information.

## Seventh milestone: research analysis

Run the descriptive statistics, Spearman correlations, HAC-robust association
models and plots:

```bash
python3 pipeline/analyze_energy_weather.py
```

The analysis writes results under `analysis/energy_weather/`. It controls for
hour of day, day of week and year, and uses 24 hourly HAC lags. The results are
conditional associations, not causal claims.

## Hard Mode, step 1: multi-zone panel

Build an unbalanced panel across six zones with common load, forecast, price,
solar and wind fields:

```bash
python3 pipeline/build_multi_zone_gold.py
```

The panel keeps `wind_source_metric` because some zones expose total wind
generation while others expose onshore wind only. This is intentional schema
lineage, not a hidden normalization assumption. The next Hard Mode step is to
add true forecast-vintage fields such as `issued_at_utc` and
`forecast_horizon_hours`.

## Hard Mode, step 2: fixed-effects panel analysis

Run the multi-zone regression with region-clustered standard errors:

```bash
python3 pipeline/analyze_multi_zone_panel.py
```

The model includes region, hour, weekday and year fixed effects. Because the
first panel has only six regions, the output explicitly warns that clustered
inference is small-sample fragile. A wild-cluster bootstrap is the next
statistical upgrade.

## Hard Mode, step 3: wild cluster bootstrap

Run the small-cluster inference check:

```bash
python3 pipeline/wild_cluster_bootstrap.py --reps 999
```

The bootstrap resamples residuals at the region level and reports a bootstrap
p-value for renewable share and forecast error in both the price and negative-
price models.

## Hard Mode, step 4: true forecast vintage Bronze

The current OPSD data does not contain the original publication history of its
load forecasts. Start a separate 2024+ weather forecast-vintage branch using
the Open-Meteo Single Runs API:

```bash
python3 pipeline/ingest_weather_vintage_bronze.py \
  --run 2024-06-01T00:00 \
  --model ecmwf_ifs \
  --forecast-days 10 \
  --retries 10
```

The output stores the model initialization time as `forecast_run_utc` and
keeps the exact request URL. Do not call it `issued_at_utc`: the API documents
that initialization time and public release time are different concepts.

## Hard Mode, step 5: forecast-vintage Silver

After the forecast Bronze download completes, convert it into one row per
forecast event:

```bash
python3 pipeline/build_weather_vintage_silver.py
```

The key columns are `forecast_run_utc`, `valid_time_utc` and
`forecast_horizon_hours`. Actual weather is intentionally not joined in this
step; that comparison belongs in the next Gold layer.

## Fourth milestone: second Bronze source — weather

The next source is Open-Meteo Historical Weather. It retrieves hourly ERA5
weather variables for Frankfurt and Luxembourg over the same date range as the
DE-LU energy mart. The API supports historical hourly weather by coordinate;
the exact request is saved beside each raw response.

```bash
python3 pipeline/ingest_weather_bronze.py --retries 10
```

Output:

```text
data_lake/bronze/weather_open_meteo/
└── ingestion_date=YYYYMMDD/
    └── batch_id=YYYYMMDDTHHMMSSZ/
        ├── location=frankfurt/
        │   ├── weather.json
        │   └── request.json
        ├── location=luxembourg/
        │   ├── weather.json
        │   └── request.json
        └── manifest.jsonl
```
