# Method notes

This page describes the choices behind the current result. It is deliberately
short; the raw requests and batch-level checks live in `metadata/`.

## Unit of observation

The final table is hourly. Market series that arrive every 15 minutes are
averaged within a UTC hour in the Gold job. Weather errors are averaged over
the representative point assigned to each zone; the current panel uses one
point for each of Frankfurt, Paris and Vienna.

The forecast table keeps both:

- `forecast_run_utc`: when the model run was initialized
- `valid_time_utc`: the hour being predicted

The same valid hour therefore appears more than once. That is expected.

## Outcomes and predictors

The main outcomes are day-ahead price, a 24-hour rolling price standard
deviation, and a negative-price indicator. Forecast-error predictors are the
absolute errors for temperature, wind speed, shortwave radiation and
precipitation. Market controls include load forecast error and actual residual
load.

Continuous predictors are standardized within zone before fitting the pooled
model. Coefficients therefore describe a one-standard-deviation change within
the relevant zone.

## Model

The pooled specifications include bidding-zone, valid-date and hour fixed
effects. The regime models add interactions between forecast error and a
zone-specific high-renewable indicator. Standard errors are two-way clustered
by `valid_time_utc` and `forecast_run_utc`.

There are 30 forecast-run clusters in the current sample. This is better than
the earlier eight-run check, but it is still not a long historical panel.

## What the result supports

The June sample shows a clear price difference between high- and
low-renewable hours in all three zones. Residual load is also strongly related
to price after the fixed effects are included. Some forecast-error effects
change with the renewable regime.

## What it does not support

The estimates do not identify a causal effect of forecast error. The weather
point is a proxy for a bidding zone, forecast initialization is not the same as
the provider's publication time, and the market outcomes are repeated across
vintages. A longer time range, more zones and an actual publication timestamp
would be needed for a stronger design.
