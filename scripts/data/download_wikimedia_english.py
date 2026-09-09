"""
Nexa Phase 8.2D - Wikimedia English Wikipedia Downloader
=========================================================
Downloads shard 1 of the enwiki-20260901 dump from the official
Wikimedia dump server. This is a controlled subset collection:

  Snapshot:  20260901 (September 1, 2026)
  File:      enwiki-20260901-pages-articles1.xml-p1p41242.bz2
  Page IDs:  p1 through p41242 (~41,000 articles)
  Size:      ~285 MB compressed
  License:   CC BY-SA 4.0 / GFDL

Selection rationale:
- The full English Wikipedia dump is 25 GiB compressed and far exceeds
  the ~8M-token pilot target (a small fraction suffices).
- Shard 1 covers the lowest-numbered page IDs: these include core
  encyclopedic articles across history, science, geography, people,
  and concepts. The selection is deterministic by snapshot + page ID range.
- A single shard yields multiple gigabytes of raw text, far more than
  the 8M-token planning target.

Policy:
- Downloads only to F: (data/raw/wikimedia_english/source/)
- Does NOT extract, clean, or transform the bz2 file
- Writes SHA-256 checksum and _metadata.yaml
- Idempotent: skips if file already complete
- Resume-capable for interrupted downloads

Usage:
  python scripts/data/download_wikimedia_english.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SNAPSHOT       = "20260901"
SHARD_FILE     = f"enwiki-{SNAPSHOT}-pages-articles1.xml-p1p41242.bz2"
DUMP_BASE      = f"https://dumps.wikimedia.org/enwiki/{SNAPSHOT}"
DUMP_URL       = f"{DUMP_BASE}/{SHARD_FILE}"
OUT_DIR        = Path("data/raw/wikimedia_english")
SOURCE_DIR     = OUT_DIR / "source"
METADATA_FILE  = OUT_DIR / "_metadata.yaml"
LOG_FILE       = OUT_DIR / "_download.log"
CHUNK_SIZE     = 1024 * 1024   # 1 MB
USER_AGENT     = (
    "NexaDatasetCollector/1.0 "
    "(academic research; https://github.com/vimalluuu/Nexa; "
    "contact: 7770vijayan@gmail.com)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wiki_en_downloader")


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
    r = session.head(DUMP_URL, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def get_last_modified(session: requests.Session) -> str:
    r = session.head(DUMP_URL, allow_redirects=True, timeout=60)
    return r.headers.get("last-modified", "unknown")


def download_file(session: requests.Session, url: str, out_path: Path,
                  expected_size: int) -> int:
    """Stream-download with resume support. Returns total bytes on disk."""
    start_byte = 0
    mode = "wb"
    headers: dict[str, str] = {}

    if out_path.exists():
        start_byte = out_path.stat().st_size
        if start_byte == expected_size:
            log.info("File already complete (%d bytes) -- skipping download.", start_byte)
            return start_byte
        if 0 < start_byte < expected_size:
            headers["Range"] = f"bytes={start_byte}-"
            mode = "ab"
            log.info("Resuming from byte %d (%.1f MB already downloaded).",
                     start_byte, start_byte / 1024 / 1024)

    r = session.get(url, headers=headers, stream=True, timeout=300)
    r.raise_for_status()

    bytes_written = start_byte if mode == "ab" else 0
    last_report = time.time()

    with open(out_path, mode) as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                now = time.time()
                if now - last_report >= 30:
                    pct = (bytes_written / expected_size * 100) if expected_size else 0
                    log.info("  Progress: %.1f / %.1f MB (%.1f%%)",
                             bytes_written / 1024 / 1024,
                             expected_size / 1024 / 1024, pct)
                    last_report = now

    return bytes_written


def load_metadata(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "dataset_name": "Wikimedia Wikipedia English (shard 1)",
        "source": "Wikimedia Foundation / Wikipedia contributors",
        "official_url": "https://dumps.wikimedia.org/enwiki/",
        "version": f"enwiki snapshot {SNAPSHOT}",
        "snapshot": SNAPSHOT,
        "download_date": None,
        "license": "CC BY-SA 4.0 / GFDL",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "language": "English",
        "selection_method": (
            f"Single shard (shard 1) of the enwiki-{SNAPSHOT} dump. "
            "Page IDs p1-p41242. Covers ~41,000 articles with the "
            "lowest-numbered page IDs (earliest/core encyclopedic content). "
            "Deterministic: fixed snapshot + fixed page ID range."
        ),
        "selection_target": "~8M planning-target tokens (exact count pending Nexa BPE)",
        "training_use_notes": (
            "CC BY-SA 4.0 permits use and adaptation subject to attribution "
            "and share-alike conditions. Whether trained model weights constitute "
            "Adapted Material is not resolved by this metadata. Preserve attribution "
            "data. Seek jurisdiction-specific legal advice before public distribution."
        ),
        "redistribution_notes": (
            "Do not redistribute the raw dump files without satisfying CC BY-SA 4.0 "
            "attribution requirements. Attribution: Wikipedia contributors, Wikipedia, "
            "The Free Encyclopedia. https://en.wikipedia.org."
        ),
        "phase": "8.2D",
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
    parser = argparse.ArgumentParser(
        description="Download Wikimedia English Wikipedia shard 1"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report sizes and disk space without downloading")
    args = parser.parse_args()

    # Set up dirs and logging
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    log.info("=== Wikimedia English Wikipedia Shard 1 Downloader ===")
    log.info("Snapshot: %s", SNAPSHOT)
    log.info("File:     %s", SHARD_FILE)
    log.info("URL:      %s", DUMP_URL)
    log.info("Output:   %s", SOURCE_DIR.resolve())

    # ----- Size and disk-space verification -----
    log.info("Checking remote file size...")
    expected_size = get_remote_size(session)
    last_modified = get_last_modified(session)
    log.info("Compressed size: %d bytes (%.1f MB)", expected_size, expected_size / 1024 / 1024)
    estimated_extracted = expected_size * 5
    log.info("Estimated extracted size: ~%.0f MB (5x bz2 ratio estimate)",
             estimated_extracted / 1024 / 1024)

    disk_free_f = shutil.disk_usage("F:/").free
    log.info("F: free disk space: %.1f GiB", disk_free_f / 1024 / 1024 / 1024)

    disk_free_c = shutil.disk_usage("C:/").free
    log.info("C: free disk space: %.1f GiB", disk_free_c / 1024 / 1024 / 1024)
    log.info("NOTE: C: is nearly full -- all downloads going to F: only.")

    required = expected_size + estimated_extracted + 200 * 1024 * 1024  # +200MB margin
    if disk_free_f < required:
        log.error("STOP: Insufficient F: disk space. Need %.0f MB, have %.0f MB.",
                  required / 1024 / 1024, disk_free_f / 1024 / 1024)
        sys.exit(1)
    log.info("Disk space check: OK (need %.0f MB, have %.1f GiB on F:)",
             required / 1024 / 1024, disk_free_f / 1024 / 1024 / 1024)

    if args.dry_run:
        log.info("Dry-run complete -- not downloading.")
        return

    # ----- Download -----
    out_path = SOURCE_DIR / SHARD_FILE
    log.info("Starting download...")
    start_time = time.time()
    total_bytes = download_file(session, DUMP_URL, out_path, expected_size)
    elapsed = time.time() - start_time
    rate = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
    log.info("Download complete: %d bytes in %.1f s (%.1f MB/s)",
             total_bytes, elapsed, rate)

    # ----- Checksum -----
    log.info("Computing SHA-256 checksum...")
    checksum = sha256_file(out_path)
    log.info("SHA-256: %s", checksum)

    if total_bytes != expected_size:
        log.warning("Size mismatch: expected %d bytes, got %d bytes.",
                    expected_size, total_bytes)

    # ----- Verify file can be opened -----
    log.info("Verifying file can be opened (reading first 1 KB)...")
    import bz2
    try:
        with bz2.open(out_path, "rb") as bz:
            sample = bz.read(1024)
        log.info("File opens OK. First bytes: %s...", sample[:80])
    except Exception as exc:
        log.error("File verification FAILED: %s", exc)
        sys.exit(1)

    # ----- Write metadata -----
    meta = load_metadata(METADATA_FILE)
    meta["download_date"] = date.today().isoformat()
    meta["collection_completed"] = datetime.now(timezone.utc).isoformat()

    file_entry = {
        "filename": SHARD_FILE,
        "source_url": DUMP_URL,
        "snapshot": SNAPSHOT,
        "page_id_range": "p1-p41242",
        "approximate_articles": 41242,
        "sha256": checksum,
        "size_bytes": total_bytes,
        "expected_size_bytes": expected_size,
        "last_modified": last_modified,
        "download_date": date.today().isoformat(),
        "download_duration_seconds": round(elapsed, 1),
        "format": "MediaWiki XML, bz2 compressed",
        "note": "Do not extract in this phase -- extraction belongs in the cleaning phase.",
    }

    existing = [e["filename"] for e in meta.get("files", [])]
    if SHARD_FILE not in existing:
        meta.setdefault("files", []).append(file_entry)
    else:
        meta["files"] = [
            file_entry if e["filename"] == SHARD_FILE else e
            for e in meta["files"]
        ]

    meta["summary"] = {
        "total_files": len(meta["files"]),
        "total_size_bytes": sum(e["size_bytes"] for e in meta["files"]),
        "sha256": checksum,
    }
    save_metadata(meta, METADATA_FILE)
    log.info("Metadata written to %s", METADATA_FILE)

    log.info("=== DONE ===")
    log.info("File: %s", out_path)
    log.info("Size: %.1f MB | SHA-256: %s...", total_bytes / 1024 / 1024, checksum[:24])


if __name__ == "__main__":
    main()
