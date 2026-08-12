"""
Nexa Phase 8.2B — Project Gutenberg Downloader
================================================
Downloads 200 English plain-text books from Project Gutenberg using
the official /robot/harvest endpoint (authorized bulk access).

Policy:
  - Uses ONLY the official harvest endpoint (not scraping the main site)
  - Polite rate limiting (2s between downloads)
  - Stores raw .zip files in data/raw/project_gutenberg/
  - Does NOT extract, clean, or transform downloaded files
  - Writes SHA-256 checksums and _metadata.yaml
  - Idempotent: skips already-downloaded files on re-run

Reference:
  https://www.gutenberg.org/policy/robot_access.html

Usage:
  python scripts/data/download_gutenberg.py [--count 200] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HARVEST_URL = "https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"
OUT_DIR = Path("data/raw/project_gutenberg")
METADATA_FILE = OUT_DIR / "_metadata.yaml"
LOG_FILE = OUT_DIR / "_download.log"
RATE_LIMIT_S = 2.0
DEFAULT_COUNT = 200
USER_AGENT = (
    "NexaDatasetCollector/1.0 "
    "(academic research; https://github.com/vimalluuu/Nexa; "
    "contact: 7770vijayan@gmail.com)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pg_downloader")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_harvest_urls(session: requests.Session) -> list[str]:
    """
    Fetch the Gutenberg harvest page and extract all .zip download URLs.
    The harvest endpoint returns HTML with <a href="..."> links to the mirror.
    We paginate using offset parameter until we have enough.
    """
    all_urls: list[str] = []
    offset = 0
    step = 100  # PG returns 100 results per page

    while True:
        url = f"{HARVEST_URL}&offset={offset}"
        log.info("Fetching harvest page offset=%d ...", offset)
        r = session.get(url, timeout=60)
        r.raise_for_status()

        links = re.findall(r'href="(https://[^"]+\.zip)"', r.text)
        if not links:
            log.info("No more links at offset=%d — end of harvest.", offset)
            break

        all_urls.extend(links)
        log.info("  offset=%d: found %d links (total: %d)", offset, len(links), len(all_urls))
        offset += step
        time.sleep(RATE_LIMIT_S)

    log.info("Total harvest URLs found: %d", len(all_urls))
    return all_urls


def extract_pg_id(url: str) -> str | None:
    """Extract Gutenberg book ID from a download URL."""
    m = re.search(r'/(\d+)/\d+', url)
    if m:
        return m.group(1)
    m = re.search(r'/(\d+)\.zip', url)
    if m:
        return m.group(1)
    return None


def select_best_urls(all_urls: list[str], count: int) -> list[tuple[str, str]]:
    """
    Select `count` download URLs, preferring plain UTF-8 text over legacy 8-bit,
    one file per book ID.
    Returns list of (url, book_id) tuples.
    """
    # Group by book ID, prefer non-8 suffix (UTF-8) over -8 (Latin-1)
    by_id: dict[str, str] = {}
    for url in all_urls:
        bid = extract_pg_id(url)
        if not bid:
            continue
        existing = by_id.get(bid)
        if existing is None:
            by_id[bid] = url
        elif "-8.zip" in existing and "-8.zip" not in url:
            # Prefer the non-8 (UTF-8) version
            by_id[bid] = url

    selected = list(by_id.values())[:count]
    return [(url, extract_pg_id(url) or "unknown") for url in selected]


def download_zip(url: str, out_path: Path, session: requests.Session) -> int:
    """Download a zip file. Returns bytes downloaded."""
    r = session.get(url, stream=True, timeout=120)
    r.raise_for_status()
    bytes_written = 0
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
    return bytes_written


def load_metadata(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "source": "Project Gutenberg",
        "official_url": "https://www.gutenberg.org",
        "license_note": (
            "Books downloaded from Project Gutenberg are in the U.S. public domain. "
            "Public-domain status may differ outside the U.S. "
            "See https://www.gutenberg.org/policy/license.html"
        ),
        "download_method": "official /robot/harvest endpoint, polite rate-limited",
        "harvest_url": HARVEST_URL,
        "phase": "8.2B",
        "collection_started": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }


def save_metadata(meta: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Project Gutenberg English text books")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of books to download (default: {DEFAULT_COUNT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show URLs that would be downloaded, without downloading")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    meta = load_metadata(METADATA_FILE)
    already_done = {e["source_url"] for e in meta.get("files", [])}

    log.info("=== Project Gutenberg Downloader ===")
    log.info("Target: %d books | Dry-run: %s", args.count, args.dry_run)
    log.info("Output: %s", OUT_DIR.resolve())

    # --- Step 1: Get harvest URLs ---
    all_urls = get_harvest_urls(session)
    candidates = select_best_urls(all_urls, args.count)
    log.info("Selected %d unique books for download", len(candidates))

    if args.dry_run:
        for url, bid in candidates:
            print(f"{bid:>8s}  {url}")
        return

    # --- Step 2: Download ---
    downloaded = 0
    skipped = 0
    failed = 0

    for i, (url, book_id) in enumerate(candidates, 1):
        log.info("[%d/%d] Book ID %s: %s", i, len(candidates), book_id, url)

        if url in already_done:
            log.info("  Already downloaded — skipping")
            skipped += 1
            continue

        filename = f"pg{book_id}.zip"
        out_path = OUT_DIR / filename

        if out_path.exists():
            log.info("  File exists — computing checksum and recording in metadata")
            checksum = sha256_file(out_path)
            size = out_path.stat().st_size
            skipped += 1
        else:
            try:
                time.sleep(RATE_LIMIT_S)
                size = download_zip(url, out_path, session)
                checksum = sha256_file(out_path)
                log.info("  OK %s  (%d KB)  sha256: %s...", filename, size // 1024, checksum[:16])
                downloaded += 1
            except Exception as exc:
                log.error("  ✗ Failed %s: %s", url, exc)
                if out_path.exists():
                    out_path.unlink()
                failed += 1
                continue

        meta.setdefault("files", []).append({
            "filename": filename,
            "source_url": url,
            "gutenberg_id": book_id,
            "sha256": checksum,
            "size_bytes": size,
            "download_date": datetime.now(timezone.utc).date().isoformat(),
            "license_note": "U.S. public domain. See https://www.gutenberg.org/policy/license.html",
        })
        already_done.add(url)
        save_metadata(meta, METADATA_FILE)

    meta["collection_completed"] = datetime.now(timezone.utc).isoformat()
    meta["summary"] = {
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "total_files": len(meta.get("files", [])),
        "total_size_bytes": sum(e["size_bytes"] for e in meta.get("files", [])),
    }
    save_metadata(meta, METADATA_FILE)

    log.info("=== DONE ===")
    log.info("Downloaded: %d | Skipped: %d | Failed: %d", downloaded, skipped, failed)
    log.info("Total files: %d | Total size: %.1f MB",
             meta["summary"]["total_files"],
             meta["summary"]["total_size_bytes"] / 1024 / 1024)


if __name__ == "__main__":
    main()
