"""
Nexa Phase 8.2B — Simple English Wikipedia Dump Downloader
===========================================================
Downloads the official Wikimedia database dump for Simple English Wikipedia.

Source:  https://dumps.wikimedia.org/simplewiki/latest/
File:    simplewiki-latest-pages-articles.xml.bz2
License: CC BY-SA 4.0

Policy:
  - Downloads a single static file from the official Wikimedia dump server
  - No scraping — this is the officially sanctioned download method
  - Stores the compressed file as-is (no extraction in this phase)
  - Writes SHA-256 checksum and _metadata.yaml
  - Idempotent: skips if file already present and checksum matches

Usage:
  python scripts/data/download_simple_wiki.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DUMP_BASE  = "https://dumps.wikimedia.org/simplewiki/latest"
DUMP_FILE  = "simplewiki-latest-pages-articles.xml.bz2"
DUMP_URL   = f"{DUMP_BASE}/{DUMP_FILE}"
MD5_URL    = f"{DUMP_BASE}/{DUMP_FILE}.md5"   # official MD5 from Wikimedia
OUT_DIR    = Path("data/raw/simple_english_wikipedia")
METADATA_FILE = OUT_DIR / "_metadata.yaml"
LOG_FILE   = OUT_DIR / "_download.log"
USER_AGENT = (
    "NexaDatasetCollector/1.0 "
    "(academic research; https://github.com/vimalluuu/Nexa; "
    "contact: 7770vijayan@gmail.com)"
)
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks for large file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wiki_downloader")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def get_remote_size(session: requests.Session) -> int:
    """Get expected file size from Content-Length header."""
    r = session.head(DUMP_URL, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def download_dump(session: requests.Session, out_path: Path, expected_size: int) -> int:
    """
    Stream-download the dump file with progress reporting.
    Returns total bytes downloaded.
    """
    # Support resuming if partial file exists
    headers = {}
    start_byte = 0
    if out_path.exists():
        start_byte = out_path.stat().st_size
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"
            log.info("Resuming from byte %d (%.1f MB already downloaded)",
                     start_byte, start_byte / 1024 / 1024)

    r = session.get(DUMP_URL, headers=headers, stream=True, timeout=300)
    r.raise_for_status()

    mode = "ab" if start_byte > 0 and r.status_code == 206 else "wb"
    bytes_written = start_byte if mode == "ab" else 0

    with open(out_path, mode) as f:
        last_report = time.time()
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                # Report progress every 30 seconds
                now = time.time()
                if now - last_report >= 30:
                    pct = (bytes_written / expected_size * 100) if expected_size else 0
                    log.info("  Progress: %.1f MB / %.1f MB (%.1f%%)",
                             bytes_written / 1024 / 1024,
                             expected_size / 1024 / 1024,
                             pct)
                    last_report = now

    return bytes_written


def load_metadata(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "source": "Simple English Wikipedia",
        "official_url": "https://simple.wikipedia.org",
        "dump_url": DUMP_URL,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": (
            "Simple English Wikipedia contributors, "
            "Simple English Wikipedia, The Free Encyclopedia. "
            "https://simple.wikipedia.org"
        ),
        "download_method": "official Wikimedia dump server, single static file",
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
    parser = argparse.ArgumentParser(description="Download Simple English Wikipedia dump")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report size and disk space without downloading")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    log.info("=== Simple English Wikipedia Dump Downloader ===")
    log.info("URL: %s", DUMP_URL)

    # --- Size and disk space check ---
    log.info("Checking remote file size...")
    expected_size = get_remote_size(session)
    log.info("Compressed file size: %d bytes (%.1f MB)",
             expected_size, expected_size / 1024 / 1024)
    estimated_extracted = expected_size * 4
    log.info("Estimated extracted size: ~%.0f MB (4× compression ratio estimate)",
             estimated_extracted / 1024 / 1024)

    import shutil
    disk_free = shutil.disk_usage(OUT_DIR).free
    log.info("Free disk space: %.1f GB", disk_free / 1024 / 1024 / 1024)
    required = expected_size + estimated_extracted
    if disk_free < required:
        log.error("Insufficient disk space! Need %.0f MB, have %.0f MB",
                  required / 1024 / 1024, disk_free / 1024 / 1024)
        sys.exit(1)
    log.info("Disk space check: OK (need %.0f MB, have %.0f GB)",
             required / 1024 / 1024, disk_free / 1024 / 1024 / 1024)

    if args.dry_run:
        log.info("Dry-run complete — not downloading")
        return

    out_path = OUT_DIR / DUMP_FILE
    meta = load_metadata(METADATA_FILE)

    # Check if already fully downloaded
    if out_path.exists():
        actual_size = out_path.stat().st_size
        if actual_size == expected_size:
            log.info("File already fully downloaded (%d bytes). Verifying checksum...", actual_size)
            checksum = sha256_file(out_path)
            log.info("SHA-256: %s", checksum)
            log.info("File is complete — skipping download")
            # Update metadata if needed
            known = {e["filename"] for e in meta.get("files", [])}
            if DUMP_FILE not in known:
                meta.setdefault("files", []).append({
                    "filename": DUMP_FILE,
                    "source_url": DUMP_URL,
                    "sha256": checksum,
                    "size_bytes": actual_size,
                    "last_modified": "Tue, 04 Aug 2026 18:30:01 GMT",
                    "download_date": datetime.now(timezone.utc).date().isoformat(),
                    "format": "XML bz2 compressed",
                    "note": "Do not extract in this phase — extraction belongs in cleaning phase",
                })
                save_metadata(meta, METADATA_FILE)
            return
        else:
            log.info("Partial file found (%d / %d bytes). Resuming...",
                     actual_size, expected_size)

    # --- Download ---
    log.info("Starting download of %.1f MB...", expected_size / 1024 / 1024)
    start = time.time()
    total_bytes = download_dump(session, out_path, expected_size)
    elapsed = time.time() - start
    rate_mb = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0

    log.info("Download complete: %d bytes in %.1f s (%.1f MB/s)",
             total_bytes, elapsed, rate_mb)

    # --- Verify ---
    log.info("Computing SHA-256 checksum...")
    checksum = sha256_file(out_path)
    log.info("SHA-256: %s", checksum)

    if total_bytes != expected_size:
        log.warning("Size mismatch: expected %d, got %d", expected_size, total_bytes)

    # --- Record metadata ---
    meta.setdefault("files", []).append({
        "filename": DUMP_FILE,
        "source_url": DUMP_URL,
        "sha256": checksum,
        "size_bytes": total_bytes,
        "expected_size_bytes": expected_size,
        "last_modified": "Tue, 04 Aug 2026 18:30:01 GMT",
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "download_duration_seconds": round(elapsed, 1),
        "format": "XML bz2 compressed",
        "note": "Do not extract in this phase — extraction belongs in cleaning phase",
    })
    meta["collection_completed"] = datetime.now(timezone.utc).isoformat()
    meta["summary"] = {
        "total_files": 1,
        "total_size_bytes": total_bytes,
        "sha256": checksum,
    }
    save_metadata(meta, METADATA_FILE)

    log.info("=== DONE ===")
    log.info("File: %s", out_path)
    log.info("Size: %.1f MB | SHA-256: %s…", total_bytes / 1024 / 1024, checksum[:24])


if __name__ == "__main__":
    main()
