#!/usr/bin/env python3
"""Ingest one archived weather forecast run with its run timestamp.

Unlike the continuous Historical Forecast API, Single Runs preserves the
original model-run structure. The run timestamp is stored as
``forecast_run_utc``; it is the model initialization time, not an assumed
public-release timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
DEFAULT_RUN = "2024-06-01T00:00"
# Single Runs uses the archived ECMWF IFS HRES run identifier. The 0.25-degree
# ecmwf_ifs025 model ID is a different product and is not valid for this
# Single Runs archive request.
DEFAULT_MODEL = "ecmwf_ifs"
DEFAULT_FORECAST_DAYS = 10
HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "precipitation",
]
LOCATIONS = {
    "frankfurt": {"latitude": 50.1109, "longitude": 8.6821},
    "luxembourg": {"latitude": 49.6116, "longitude": 6.1319},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(run: str) -> None:
    parsed = datetime.strptime(run, "%Y-%m-%dT%H:%M")
    if parsed.year < 2024:
        raise SystemExit("Single Runs ECMWF archive begins in 2024; use a 2024+ run.")
    if parsed.hour not in {0, 6, 12, 18}:
        raise SystemExit("ECMWF global runs must use 00, 06, 12 or 18 UTC.")


def build_url(
    latitude: float,
    longitude: float,
    run: str,
    model: str,
    forecast_days: int,
) -> str:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "run": run,
        "models": model,
        "forecast_days": forecast_days,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    return f"{API_URL}?{urlencode(params)}"


def download_json(url: str, destination: Path, retries: int) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temp_path: Path | None = None
        try:
            request = Request(url, headers={"User-Agent": "regional-energy-data-lake/0.1"})
            with urlopen(request, timeout=120) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, prefix=".forecast-", delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temp_file.write(chunk)
            os.replace(temp_path, destination)
            return {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
        except Exception as exc:  # noqa: BLE001 - retry network failures.
            last_error = exc
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            print(f"  attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"forecast run download failed after {retries} attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="UTC model run, e.g. 2024-06-01T00:00")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--forecast-days", type=int, default=DEFAULT_FORECAST_DAYS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/bronze/weather_forecast_vintage"),
    )
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_run(args.run)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = args.run.replace(":", "-") + "Z"
    batch_dir = args.output_root / f"ingestion_date={batch_id[:8]}" / f"batch_id={batch_id}" / f"run={run_path}"
    entries = []

    print(f"Forecast-vintage Bronze batch: {batch_id}")
    print(f"Forecast run initialization: {args.run}Z")
    for location, coordinates in LOCATIONS.items():
        location_dir = batch_dir / f"location={location}"
        raw_path = location_dir / "forecast.json"
        url = build_url(
            coordinates["latitude"],
            coordinates["longitude"],
            args.run,
            args.model,
            args.forecast_days,
        )
        print(f"Downloading run for {location} ...")
        file_info = download_json(url, raw_path, retries=args.retries)
        request_path = location_dir / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "source": "open_meteo_single_runs_api",
                    "url": url,
                    "location": location,
                    **coordinates,
                    "forecast_run_utc": f"{args.run}Z",
                    "model": args.model,
                    "forecast_days": args.forecast_days,
                    "hourly_variables": HOURLY_VARIABLES,
                    "retrieved_at_utc": utc_now(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "dataset": "open_meteo_weather_forecast_vintage",
                "layer": "bronze",
                "batch_id": batch_id,
                "forecast_run_utc": f"{args.run}Z",
                "model": args.model,
                "location": location,
                "path": str(raw_path),
                "source_url": url,
                "retrieved_at_utc": utc_now(),
                **coordinates,
                **file_info,
            }
        )
        print(f"  {file_info['bytes']:,} bytes  sha256={file_info['sha256'][:16]}...")

    manifest_path = batch_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Manifest: {manifest_path}")
    print("Forecast-vintage Bronze ingestion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
