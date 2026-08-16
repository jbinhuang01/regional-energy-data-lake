#!/usr/bin/env python3
"""Ingest hourly historical weather for the DE-LU energy study.

This creates a second independent Bronze source. It stores the raw Open-Meteo
JSON response, the exact request parameters, and a SHA-256 manifest entry.
Weather is fetched at two representative points: Frankfurt and Luxembourg.
The date range matches the current DE-LU Gold research mart.
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


API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2018-09-30"
END_DATE = "2020-09-30"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
    "weather_code",
]
LOCATIONS = {
    "frankfurt": {"latitude": 50.1109, "longitude": 8.6821},
    "luxembourg": {"latitude": 49.6116, "longitude": 6.1319},
    "paris": {"latitude": 48.8566, "longitude": 2.3522},
    "vienna": {"latitude": 48.2082, "longitude": 16.3738},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_url(latitude: float, longitude: float, start_date: str, end_date: str) -> str:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "models": "era5",
    }
    return f"{API_URL}?{urlencode(params)}"


def download_json(url: str, destination: Path, retries: int) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f".{destination.name}.part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "regional-energy-data-lake/0.1"})
            with urlopen(request, timeout=120) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, prefix=".weather-", delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temp_file.write(chunk)
            os.replace(temp_path, destination)
            return {
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        except Exception as exc:  # noqa: BLE001 - retry network failures.
            last_error = exc
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink()
            print(f"  attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"weather download failed after {retries} attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/bronze/weather_open_meteo"),
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument(
        "--locations",
        nargs="+",
        choices=sorted(LOCATIONS),
        default=["frankfurt", "luxembourg"],
        help="Representative weather points to ingest.",
    )
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = args.output_root / f"ingestion_date={batch_id[:8]}" / f"batch_id={batch_id}"
    manifest_path = batch_dir / "manifest.jsonl"
    entries = []

    print(f"Weather Bronze batch: {batch_id}")
    selected_locations = {name: LOCATIONS[name] for name in args.locations}
    for location, coordinates in selected_locations.items():
        location_dir = batch_dir / f"location={location}"
        url = build_url(
            coordinates["latitude"],
            coordinates["longitude"],
            args.start_date,
            args.end_date,
        )
        raw_path = location_dir / "weather.json"
        request_path = location_dir / "request.json"
        print(f"Downloading weather for {location} ...")
        file_info = download_json(url, raw_path, retries=args.retries)
        request_path.write_text(
            json.dumps(
                {
                    "source": "open_meteo_historical_weather_api",
                    "url": url,
                    "location": location,
                    **coordinates,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "hourly_variables": HOURLY_VARIABLES,
                    "retrieved_at_utc": utc_now(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "dataset": "open_meteo_historical_weather",
                "layer": "bronze",
                "batch_id": batch_id,
                "location": location,
                "path": str(raw_path),
                "source_url": url,
                "retrieved_at_utc": utc_now(),
                **coordinates,
                **file_info,
            }
        )
        print(f"  {file_info['bytes']:,} bytes  sha256={file_info['sha256'][:16]}...")

    batch_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Manifest: {manifest_path}")
    print("Weather Bronze ingestion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
