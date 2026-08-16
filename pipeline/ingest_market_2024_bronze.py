#!/usr/bin/env python3
"""Ingest 2024 Energy-Charts market data into an immutable Bronze batch.

The script keeps the provider JSON unchanged and stores request metadata next
to each response. The default scope is the DE-LU bidding zone and Germany's
public power data for the same UTC date range as the forecast-vintage branch.

Endpoints:
    /price                  day-ahead price for a bidding zone
    /public_power           actual public net generation by production type
    /public_power_forecast  optional day-ahead forecasts for load/renewables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://api.energy-charts.info"
DEFAULT_OUTPUT_ROOT = Path("data_lake/bronze/energy_charts")
USER_AGENT = "regional-energy-data-lake/1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2024-06-01")
    parser.add_argument("--end-date", default="2024-06-10")
    parser.add_argument(
        "--start-timestamp",
        help="Exact API start timestamp, e.g. 2024-06-01T00:00:00Z.",
    )
    parser.add_argument(
        "--end-timestamp",
        help="Exact API end timestamp, e.g. 2024-06-10T23:00:00Z.",
    )
    parser.add_argument("--country", default="de")
    parser.add_argument("--bzn", default="DE-LU")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-id")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--include-forecasts",
        action="store_true",
        help="Also request historical day-ahead forecasts for load, solar and wind.",
    )
    return parser.parse_args()


def request_url(endpoint: str, params: dict[str, str]) -> str:
    return f"{API_BASE}{endpoint}?{urlencode(params)}"


def download_json(
    url: str,
    retries: int,
    timeout_seconds: int,
) -> tuple[bytes, dict]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
                if not body:
                    raise RuntimeError("Empty response body")
                payload = json.loads(body.decode("utf-8"))
                return body, payload
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if isinstance(exc, HTTPError) and exc.code == 404:
                raise
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 20))
    raise RuntimeError(f"Download failed after {retries} attempts: {last_error}")


def validate_response(endpoint: str, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{endpoint} returned a non-object JSON response")
    if payload.get("error"):
        raise ValueError(f"{endpoint} returned an API error: {payload['error']}")
    if endpoint == "/price" and "price" not in payload:
        raise ValueError("/price response has no price series")
    if endpoint == "/public_power" and "production_types" not in payload:
        raise ValueError("/public_power response has no production_types")
    if endpoint == "/public_power_forecast" and "forecast_values" not in payload:
        raise ValueError("/public_power_forecast response has no forecast_values")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def manifest_record(
    path: Path,
    root: Path,
    url: str,
    body: bytes,
    endpoint: str,
    request_metadata: dict,
) -> dict:
    return {
        "relative_path": str(path.relative_to(root)),
        "source_url": url,
        "endpoint": endpoint,
        "retrieved_at_utc": request_metadata["retrieved_at_utc"],
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "request": request_metadata,
    }


def main() -> int:
    args = parse_args()
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")
    if args.start_date > args.end_date:
        raise SystemExit("--start-date must be before or equal to --end-date")

    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    query_start = args.start_timestamp or args.start_date
    query_end = args.end_timestamp or args.end_date
    if query_start > query_end:
        raise SystemExit("The API start timestamp must be before the end timestamp")
    output_dir = (
        args.output_root
        / f"ingestion_date={batch_id[:8]}"
        / f"batch_id={batch_id}"
        / f"country={args.country}"
        / f"bzn={args.bzn}"
    )
    if output_dir.exists():
        raise SystemExit(f"Bronze output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    base_params = {
        "start": query_start,
        "end": query_end,
    }
    requests: list[tuple[str, str, dict[str, str], Path]] = [
        (
            "/price",
            "price.json",
            {"bzn": args.bzn, **base_params},
            output_dir / "endpoint=price" / "price.json",
        ),
        (
            "/public_power",
            "public_power.json",
            {"country": args.country, **base_params},
            output_dir / "endpoint=public_power" / "public_power.json",
        ),
    ]
    if args.include_forecasts:
        for production_type in ["load", "solar", "wind_onshore", "wind_offshore"]:
            requests.append(
                (
                    "/public_power_forecast",
                    f"forecast_{production_type}.json",
                    {
                        "country": args.country,
                        "production_type": production_type,
                        "forecast_type": "day-ahead",
                        **base_params,
                    },
                    output_dir
                    / "endpoint=public_power_forecast"
                    / f"production_type={production_type}"
                    / f"forecast_{production_type}.json",
                )
            )

    manifest: list[dict] = []
    for endpoint, filename, params, path in requests:
        url = request_url(endpoint, params)
        print(f"Downloading {endpoint} -> {filename} ...")
        try:
            body, payload = download_json(url, args.retries, args.timeout_seconds)
        except HTTPError as exc:
            # Some zones do not expose every production type.  A missing
            # optional forecast series is a valid source-level condition, not
            # a reason to invalidate the complete market batch.
            if endpoint == "/public_power_forecast" and exc.code == 404:
                availability_path = path.parent / "availability.json"
                availability = {
                    "source": "energy_charts_api",
                    "endpoint": endpoint,
                    "url": url,
                    "country": args.country,
                    "bzn": args.bzn,
                    "start_query": query_start,
                    "end_query": query_end,
                    "status": "unavailable",
                    "http_status": 404,
                    "reason": "Production type is not published for this country/zone.",
                    "retrieved_at_utc": utc_now(),
                }
                availability_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(availability_path, availability)
                manifest.append(
                    {
                        "relative_path": str(availability_path.relative_to(args.output_root)),
                        "source_url": url,
                        "endpoint": endpoint,
                        "status": "unavailable",
                        "http_status": 404,
                        "retrieved_at_utc": availability["retrieved_at_utc"],
                        "request": availability,
                    }
                )
                print("  optional production type unavailable; continuing")
                continue
            raise
        validate_response(endpoint, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        retrieved_at = utc_now()
        request_metadata = {
            "source": "energy_charts_api",
            "endpoint": endpoint,
            "url": url,
            "country": args.country,
            "bzn": args.bzn,
            "start_query": query_start,
            "end_query": query_end,
            "retrieved_at_utc": retrieved_at,
            "api_response_deprecated": bool(payload.get("deprecated", False)),
        }
        write_json(path.parent / "request.json", request_metadata)
        manifest.append(
            manifest_record(path, args.output_root, url, body, endpoint, request_metadata)
        )
        print(f"  {len(body):,} bytes sha256={hashlib.sha256(body).hexdigest()[:16]}...")

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in manifest) + "\n",
        encoding="utf-8",
    )
    batch_metadata = {
        "layer": "bronze",
        "dataset": "energy_charts",
        "source": "Energy-Charts API",
        "api_base": API_BASE,
        "api_version": "v1 endpoint response",
        "country": args.country,
        "bzn": args.bzn,
        "start_query": query_start,
        "end_query": query_end,
        "time_semantics": (
            "exact ISO UTC timestamps when supplied; plain dates are interpreted "
            "in the bidding zone's local timezone by Energy-Charts"
        ),
        "include_forecasts": args.include_forecasts,
        "batch_id": batch_id,
        "created_at_utc": utc_now(),
        "endpoint_count": len(requests),
        "manifest": str(manifest_path),
    }
    write_json(output_dir / "batch_metadata.json", batch_metadata)

    print(f"Energy market Bronze batch: {batch_id}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("Energy market Bronze ingestion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
