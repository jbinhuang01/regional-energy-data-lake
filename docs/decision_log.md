# Design notes

This is a short record of decisions that changed the implementation.

## Separate run time from valid time

The forecast branch keeps both `forecast_run_utc` and `valid_time_utc`. A
forecast made at 12:00 can describe a delivery hour several days later, and
that same delivery hour can appear in later runs. Collapsing the two timestamps
would remove the forecast revision information.

## Use exact UTC timestamps for market requests

The first Energy-Charts request used date-only parameters. The API interpreted
those dates in the local timezone of the bidding zone, which left a two-hour
boundary mismatch in the market join. The market ingestion job now accepts
explicit `Z` timestamps and stores both query values in the request metadata.

## Use 00 and 12 UTC forecast runs

The Single Runs endpoint returned HTTP 400 for the 06 UTC run used in an early
test. The working panel therefore uses 00 and 12 UTC runs. The failed batch is
left in the local data lake as a trace of the failed request, but it is not
used in the final analysis.

## Keep source resolution in Silver

France returned hourly market data while Germany and Austria returned a mix of
15-minute and hourly series. Silver keeps those resolutions as received. The
Gold job performs the explicit UTC-hour aggregation used by the regression.

## Treat unavailable series as metadata

Austria does not publish the offshore wind forecast series through the selected
Energy-Charts endpoint. The Bronze job records a 404 in `availability.json` and
continues with the other forecast types. This is different from a failed
download of a required price or power series.

## Keep raw data out of Git

The local Bronze data is several hundred megabytes and contains files that can
be regenerated from the stored URLs. The repository contains code, manifests,
quality reports, model output and figures instead. This keeps the Git history
usable while preserving the information needed to reproduce the pipeline.

## Interpretation boundary

The panel is observational. Date, hour and zone effects are controlled in the
current model, and errors are clustered by valid hour and forecast run. That
helps with dependence created by the vintage layout, but it does not turn the
result into a causal estimate.
