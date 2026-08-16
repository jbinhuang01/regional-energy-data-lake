#!/usr/bin/env python3
"""Ingest several forecast vintages into one Bronze parent batch.

The important distinction is between ``forecast_run_utc`` and
``valid_time_utc``.  Multiple runs can predict the same valid hour.  Keeping
both timestamps makes it possible to study forecast revision and horizon
effects instead of treating one forecast as if it were the only forecast.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import ingest_weather_vintage_bronze as vintage  # noqa: E402


LOCATION_CATALOG = {
    **vintage.LOCATIONS,
    "paris": {"latitude": 48.8566, "longitude": 2.3522},
    "vienna": {"latitude": 48.2082, "longitude": 16.3738},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-run", default="2024-06-01T00:00")
    parser.add_argument(
        "--run-count",
        type=int,
        default=8,
        help="Number of runs.",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=12,
        choices=[6, 12, 24],
        help="Spacing between model runs; 12 hours matches the available archive runs.",
    )
    parser.add_argument("--forecast-days", type=int, default=10)
    parser.add_argument("--model", default=vintage.DEFAULT_MODEL)
    parser.add_argument(
        "--locations",
        nargs="+",
        choices=sorted(LOCATION_CATALOG),
        default=["frankfurt", "luxembourg"],
        help="Representative weather points to forecast.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/bronze/weather_forecast_vintage"),
    )
    parser.add_argument("--retries", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_count <= 0:
        raise SystemExit("--run-count must be positive")
    start = datetime.strptime(args.start_run, "%Y-%m-%dT%H:%M").replace(
        tzinfo=timezone.utc
    )
    if start.hour not in {0, 6, 12, 18}:
        raise SystemExit("--start-run must use 00, 06, 12 or 18 UTC")
    for offset in range(args.run_count):
        vintage.validate_run(
            (start + timedelta(hours=args.interval_hours * offset)).strftime(
                "%Y-%m-%dT%H:%M"
            )
        )

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = args.output_root / f"ingestion_date={batch_id[:8]}" / f"batch_id={batch_id}"
    if batch_dir.exists():
        raise SystemExit(f"Output already exists: {batch_dir}")

    entries: list[dict] = []
    print(f"Forecast-vintage panel Bronze batch: {batch_id}")
    print(f"Runs: {args.run_count}; interval: {args.interval_hours} hours")
    for offset in range(args.run_count):
        run_dt = start + timedelta(hours=args.interval_hours * offset)
        run_text = run_dt.strftime("%Y-%m-%dT%H:%M")
        run_id = run_text.replace(":", "-") + "Z"
        print(f"Run {offset + 1}/{args.run_count}: {run_text}Z")
        selected_locations = {name: LOCATION_CATALOG[name] for name in args.locations}
        for location, coordinates in selected_locations.items():
            location_dir = batch_dir / f"run={run_id}" / f"location={location}"
            raw_path = location_dir / "forecast.json"
            url = vintage.build_url(
                coordinates["latitude"],
                coordinates["longitude"],
                run_text,
                args.model,
                args.forecast_days,
            )
            file_info = vintage.download_json(url, raw_path, retries=args.retries)
            request = {
                "source": "open_meteo_single_runs_api",
                "url": url,
                "location": location,
                **coordinates,
                "forecast_run_utc": f"{run_text}Z",
                "model": args.model,
                "forecast_days": args.forecast_days,
                "hourly_variables": vintage.HOURLY_VARIABLES,
                "retrieved_at_utc": utc_now(),
                "panel_batch_id": batch_id,
            }
            (location_dir / "request.json").write_text(
                json.dumps(request, indent=2), encoding="utf-8"
            )
            entries.append(
                {
                    "dataset": "open_meteo_weather_forecast_vintage_panel",
                    "layer": "bronze",
                    "batch_id": batch_id,
                    "forecast_run_utc": f"{run_text}Z",
                    "model": args.model,
                    "location": location,
                    "path": str(raw_path),
                    "source_url": url,
                    "retrieved_at_utc": utc_now(),
                    **coordinates,
                    **file_info,
                }
            )
            print(f"  {location}: {file_info['bytes']:,} bytes")

    manifest_path = batch_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Manifest: {manifest_path}")
    print("Forecast-vintage panel Bronze ingestion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
