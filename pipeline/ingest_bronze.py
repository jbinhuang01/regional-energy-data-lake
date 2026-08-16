
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_VERSION = "2020-10-06"
BASE_URL = (
    "https://data.open-power-system-data.org/"
    f"time_series/{DATASET_VERSION}"
)

FILES = {
    "time_series_60min_singleindex.csv": f"{BASE_URL}/time_series_60min_singleindex.csv",
    "README.md": f"{BASE_URL}/README.md",
    "datapackage.json": f"{BASE_URL}/datapackage.json",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    force: bool = False,
    retries: int = 6,
) -> dict:
    """Download one file with retry/resume support and return its metadata.

    A ``.part`` file is deliberately retained after a failed attempt. If the
    server supports HTTP Range requests, the next attempt continues from the
    existing byte offset instead of restarting the large CSV download.
    """

    if destination.exists() and not force:
        print(f"  already present; verifying {destination.name}")
        return {
            "file_name": destination.name,
            "path": str(destination),
            "source_url": url,
            "downloaded_at_utc": None,
            "observed_at_utc": utc_now(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused_existing": True,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "regional-energy-data-lake/0.1"})

    partial_path = destination.with_name(f".{destination.name}.part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            offset = partial_path.stat().st_size if partial_path.exists() else 0
            headers = {"User-Agent": "regional-energy-data-lake/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"

            request = Request(url, headers=headers)
            with urlopen(request, timeout=120) as response:
                response_is_partial = response.status == 206
                if offset and not response_is_partial:
                    # The server ignored Range. Restart safely rather than
                    # appending a duplicate copy of the full response.
                    offset = 0

                mode = "ab" if offset else "wb"
                with partial_path.open(mode) as partial_file:
                    shutil.copyfileobj(response, partial_file, length=1024 * 1024)

            os.replace(partial_path, destination)
            break
        except Exception as exc:  # noqa: BLE001 - retry network failures.
            last_error = exc
            print(
                f"  attempt {attempt}/{retries} failed: {exc}; "
                "partial download retained",
                file=sys.stderr,
            )
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    else:
        raise RuntimeError(
            f"download failed after {retries} attempts; partial file is at "
            f"{partial_path}"
        ) from last_error

    return {
        "file_name": destination.name,
        "path": str(destination),
        "source_url": url,
        "downloaded_at_utc": utc_now(),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def append_manifest(manifest_path: Path, entries: list[dict]) -> None:
    """Append one JSON Lines record per downloaded file."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_lake/bronze/opsd_time_series"),
        help="Bronze directory for this ingestion batch.",
    )
    parser.add_argument(
        "--version",
        default=DATASET_VERSION,
        help=f"OPSD dataset version; default: {DATASET_VERSION}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files already present in the selected output directory.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Number of attempts per file; default: 6.",
    )
    parser.add_argument(
        "--batch-id",
        help="Reuse a prior batch ID to continue a .part download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.version != DATASET_VERSION:
        raise SystemExit(
            "This first ingestion script is pinned to the tested dataset version "
            f"{DATASET_VERSION}. Update BASE_URL and FILES before using another version."
        )

    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = args.output_root / f"ingestion_date={batch_id[:8]}" / f"batch_id={batch_id}"
    manifest_path = batch_dir / "manifest.jsonl"
    entries: list[dict] = []

    print(f"Bronze batch: {batch_id}")
    print(f"Output:       {batch_dir}")

    try:
        for file_name, url in FILES.items():
            destination = batch_dir / file_name
            print(f"Downloading {file_name} ...")
            entry = download_file(
                url,
                destination,
                force=args.force,
                retries=args.retries,
            )
            entry.update(
                {
                    "dataset": "open_power_system_data_time_series",
                    "dataset_version": DATASET_VERSION,
                    "layer": "bronze",
                    "batch_id": batch_id,
                }
            )
            entries.append(entry)
            print(f"  {entry['bytes']:,} bytes  sha256={entry['sha256'][:16]}...")
    except Exception as exc:  # noqa: BLE001 - CLI should report a useful failure.
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    append_manifest(manifest_path, entries)
    print(f"Manifest:     {manifest_path}")
    print("Bronze ingestion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
