"""
Nexa Phase 8.2D - PG-19 Book Downloader
=========================================
Downloads a selected subset of PG-19 books from the official
Google Cloud Storage bucket (publicly accessible).

Selection method:
- Fetch metadata.csv from the PG-19 GitHub repository
- Sort books by book_id (ascending)
- Select the first N books (default 25) from the train split
- Download the .txt file for each selected book
- For each book: record ID, title, publication date, and Project Gutenberg
  license notes (pre-1919 PG books are U.S. public domain per PG policy)

PG-19 source:
  Repository:  https://github.com/google-deepmind/pg19
  GCS bucket:  gs://deepmind-gutenberg
  HTTP access: https://storage.googleapis.com/deepmind-gutenberg/train/<id>.txt
  License:     Apache-2.0 (repository); underlying texts are Project Gutenberg
               works published before 1919 (U.S. public domain per PG policy)

Policy:
- Downloads only to F: (data/raw/pg19/source/)
- Does NOT clean, normalize, or transform downloaded text files
- Writes SHA-256 checksums and _metadata.yaml with per-book records
- Idempotent: skips already-downloaded files
- Records book ID, title, publication date for every downloaded book

Usage:
  python scripts/data/download_pg19.py [--count 25] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METADATA_CSV_URL = (
    "https://raw.githubusercontent.com/google-deepmind/pg19/master/metadata.csv"
)
GCS_BASE        = "https://storage.googleapis.com/deepmind-gutenberg"
TRAIN_SPLIT     = "train"
OUT_DIR         = Path("data/raw/pg19")
SOURCE_DIR      = OUT_DIR / "source"
METADATA_FILE   = OUT_DIR / "_metadata.yaml"
LOG_FILE        = OUT_DIR / "_download.log"
RATE_LIMIT_S    = 1.0
DEFAULT_COUNT   = 25
USER_AGENT      = (
    "NexaDatasetCollector/1.0 "
    "(academic research; https://github.com/vimalluuu/Nexa; "
    "contact: 7770vijayan@gmail.com)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pg19_downloader")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_metadata_csv(session: requests.Session) -> list[dict]:
    """
    Download and parse PG-19 metadata.csv.
    The CSV has NO header row. Columns are:
      0: book_id (Project Gutenberg ID)
      1: title
      2: publication_year
      3: gutenberg_url
    Returns list of {book_id, title, publication_date, gutenberg_url} dicts,
    sorted by book_id ascending.
    """
    log.info("Fetching PG-19 metadata.csv...")
    r = session.get(METADATA_CSV_URL, timeout=30)
    r.raise_for_status()
    # No header: parse with positional columns
    reader = csv.reader(io.StringIO(r.text))
    books = []
    for row in reader:
        if len(row) < 3:
            continue
        books.append({
            "book_id":          row[0].strip(),
            "title":            row[1].strip(),
            "publication_date": row[2].strip(),
            "gutenberg_url":    row[3].strip() if len(row) > 3 else "",
        })
    log.info("Loaded %d books from metadata.csv", len(books))
    # Sort by book_id ascending (integer sort)
    books.sort(key=lambda b: int(b["book_id"]) if b["book_id"].isdigit() else 0)
    return books


def download_book(session: requests.Session, book_id: str, out_path: Path) -> int:
    """Download a single book .txt file. Returns bytes written."""
    url = f"{GCS_BASE}/{TRAIN_SPLIT}/{book_id}.txt"
    r = session.get(url, stream=True, timeout=120)
    r.raise_for_status()
    bytes_written = 0
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
    return bytes_written


def verify_pg_public_domain(pub_date: str) -> tuple[bool, str]:
    """
    Check publication date against U.S. public domain threshold.
    PG-19 only contains books published before 1919.
    Project Gutenberg's own policy states these are generally public domain
    in the U.S. under U.S. copyright law.
    Returns (is_clear, note).
    """
    try:
        year = int(str(pub_date).strip()[:4])
    except (ValueError, TypeError):
        return False, f"Could not parse publication year: {pub_date!r} -- SKIPPING"
    if year < 1919:
        return True, (
            f"Published {year}, before 1919. Per Project Gutenberg policy, "
            "generally U.S. public domain. See https://www.gutenberg.org/policy/license.html"
        )
    # PG-19 should not have post-1919 books, but guard anyway
    return False, (
        f"Published {year}: year >= 1919. Unexpected in PG-19 -- SKIPPING for safety."
    )


def load_metadata(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "dataset_name": "PG-19 (selected subset)",
        "source": "Google DeepMind PG-19 / Project Gutenberg",
        "official_url": "https://github.com/google-deepmind/pg19",
        "gcs_bucket": "gs://deepmind-gutenberg",
        "version": "PG-19 train split, sorted by book_id ascending",
        "snapshot": "PG-19 as of metadata.csv commit (accessed 2026-09-09)",
        "download_date": None,
        "license": (
            "Repository: Apache-2.0 (metadata, scripts). "
            "Texts: Project Gutenberg works published before 1919; "
            "generally U.S. public domain per PG license policy."
        ),
        "license_url": "https://www.gutenberg.org/policy/license.html",
        "language": "English",
        "selection_method": (
            "First N books by book_id ascending from the PG-19 train split. "
            "Only books with verified pre-1919 publication date are downloaded. "
            "Books where publication year cannot be parsed are skipped."
        ),
        "selection_target": (
            "~2M planning-target tokens (exact count pending Nexa BPE). "
            "Capped at 20% of initial 10M pilot."
        ),
        "training_use_notes": (
            "PG-19 books are pre-1919 Project Gutenberg works. "
            "Per Project Gutenberg policy, U.S. copyright law treats these as "
            "public domain, but country-specific review is still advisable. "
            "Individual books may have additional restrictions; inspect as needed. "
            "PG-19 pre-processed to remove boilerplate and map certain offensive terms."
        ),
        "redistribution_notes": (
            "Do not redistribute files without reviewing the Project Gutenberg "
            "License and trademark requirements. See "
            "https://www.gutenberg.org/policy/license.html"
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
    parser = argparse.ArgumentParser(description="Download selected PG-19 books")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of books to download (default: {DEFAULT_COUNT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="List books that would be downloaded without downloading")
    args = parser.parse_args()

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    log.info("=== PG-19 Book Downloader ===")
    log.info("Target count: %d | Dry-run: %s", args.count, args.dry_run)
    log.info("Output: %s", SOURCE_DIR.resolve())

    meta = load_metadata(METADATA_FILE)
    already_downloaded = {e["book_id"] for e in meta.get("files", [])}

    # ----- Fetch and select books -----
    all_books = fetch_metadata_csv(session)

    selected: list[dict] = []
    skipped_pd = 0
    for book in all_books:
        if len(selected) >= args.count:
            break
        book_id  = book["book_id"]
        title    = book["title"]
        pub_date = book["publication_date"]
        pg_url   = book.get("gutenberg_url", "")

        if not book_id.isdigit():
            continue  # skip malformed rows

        ok, note = verify_pg_public_domain(pub_date)
        if not ok:
            log.warning("Skipping book %s (%r): %s", book_id, title[:40], note)
            skipped_pd += 1
            continue

        selected.append({
            "book_id":          book_id,
            "title":            title,
            "publication_date": pub_date,
            "gutenberg_url":    pg_url,
            "pd_note":          note,
        })

    log.info("Selected %d books (skipped %d with PD uncertainty)", len(selected), skipped_pd)

    if args.dry_run:
        for i, b in enumerate(selected, 1):
            print(f"{i:3d}. ID={b['book_id']:>6s}  year={b['publication_date']:>4s}  {b['title'][:60]}")
        return

    # ----- Download -----
    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0

    for i, book in enumerate(selected, 1):
        bid = book["book_id"]
        log.info("[%d/%d] Book %s: %s (%s)",
                 i, len(selected), bid, book["title"][:50], book["publication_date"])

        if bid in already_downloaded:
            log.info("  Already downloaded -- skipping")
            skipped += 1
            continue

        out_path = SOURCE_DIR / f"{bid}.txt"
        if out_path.exists():
            log.info("  File exists -- computing checksum and recording")
            checksum = sha256_file(out_path)
            size = out_path.stat().st_size
            skipped += 1
        else:
            try:
                time.sleep(RATE_LIMIT_S)
                size = download_book(session, bid, out_path)
                checksum = sha256_file(out_path)
                log.info("  OK %s.txt  (%d KB)  sha256: %s...",
                         bid, size // 1024, checksum[:16])
                downloaded += 1
            except Exception as exc:
                log.error("  FAILED book %s: %s", bid, exc)
                if out_path.exists():
                    out_path.unlink()
                failed += 1
                continue

        total_bytes += size
        file_entry = {
            "filename":             f"{bid}.txt",
            "book_id":              bid,
            "title":                book["title"],
            "publication_date":     book["publication_date"],
            "source_url":           f"{GCS_BASE}/{TRAIN_SPLIT}/{bid}.txt",
            "project_gutenberg_id": bid,
            "gutenberg_url":        book.get("gutenberg_url", ""),
            "license_note":         book["pd_note"],
            "sha256":               checksum,
            "size_bytes":           size,
            "download_date":        date.today().isoformat(),
        }
        meta.setdefault("files", []).append(file_entry)
        already_downloaded.add(bid)
        save_metadata(meta, METADATA_FILE)

    meta["download_date"] = date.today().isoformat()
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
    log.info("Downloaded: %d | Skipped: %d | Failed: %d",
             downloaded, skipped, failed)
    log.info("Total files: %d | Total size: %.1f MB",
             meta["summary"]["total_files"],
             meta["summary"]["total_size_bytes"] / 1024 / 1024)


if __name__ == "__main__":
    main()
