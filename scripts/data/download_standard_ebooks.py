"""
Nexa Phase 8.2B — Standard Ebooks Fiction Downloader
=====================================================
Downloads approximately 200 fiction EPUBs from standardebooks.org.

Policy:
  - Uses polite rate limiting (2s between requests)
  - Downloads individual EPUB files only (no bulk ZIP access)
  - Stores files as-is in data/raw/standard_ebooks/
  - Writes SHA-256 checksums and _metadata.yaml
  - Does NOT extract, clean, or transform downloaded files
  - Idempotent: skips already-downloaded files on re-run

Usage:
  python scripts/data/download_standard_ebooks.py [--limit 200] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
BASE_URL = "https://standardebooks.org"
FICTION_BASE = f"{BASE_URL}/subjects/fiction"  # correct fiction subject URL
OUT_DIR = Path("data/raw/standard_ebooks")
METADATA_FILE = OUT_DIR / "_metadata.yaml"
LOG_FILE = OUT_DIR / "_download.log"
RATE_LIMIT_S = 2.0          # seconds between requests
DEFAULT_LIMIT = 200         # max books to collect
USER_AGENT = (
    "NexaDatasetCollector/1.0 "
    "(academic research; https://github.com/vimalluuu/Nexa; "
    "contact: 7770vijayan@gmail.com)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("se_downloader")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(url: str, session: requests.Session, stream: bool = False) -> requests.Response:
    """Polite GET with user-agent and error handling."""
    r = session.get(url, stream=stream, timeout=60)
    r.raise_for_status()
    return r


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fiction_page(html: str) -> list[dict]:
    """
    Extract book slugs from a Standard Ebooks subjects/fiction page.
    Each book has a <li> with about="/ebooks/<author>/<title>".
    Returns list of {slug, author_slug, title_slug} dicts.
    """
    books = []
    # Pattern: about="/ebooks/author/title"
    for m in re.finditer(r'about="(/ebooks/([^/"]+)/([^/"]+))"', html):
        slug_path   = m.group(1)
        author_slug = m.group(2)
        title_slug  = m.group(3)
        books.append({
            "slug":        slug_path,
            "author_slug": author_slug,
            "title_slug":  title_slug,
        })
    # Deduplicate preserving order
    seen = set()
    unique = []
    for b in books:
        if b["slug"] not in seen:
            seen.add(b["slug"])
            unique.append(b)
    return unique


def get_epub_url(book: dict) -> str:
    """
    Construct the EPUB download URL directly from the book slug.
    Pattern: /ebooks/<author>/<title>/downloads/<author>_<title>.epub
    This avoids an extra HTTP request per book.
    """
    slug = book["slug"]          # e.g. /ebooks/jane-austen/pride-and-prejudice
    slug_path = slug.strip("/")  # e.g. ebooks/jane-austen/pride-and-prejudice
    parts = slug_path.split("/")
    author = parts[1] if len(parts) > 1 else "unknown"
    title  = parts[2] if len(parts) > 2 else "unknown"
    filename = f"{author}_{title}.epub"
    return f"{BASE_URL}/{slug_path}/downloads/{filename}"


def collect_fiction_slugs(session: requests.Session, limit: int) -> list[dict]:
    """
    Paginate through the subjects/fiction catalog and collect up to `limit` book slugs.
    Standard Ebooks uses ?page=N&per-page=48 for pagination.
    """
    all_books: list[dict] = []
    page = 1
    per_page = 48
    while len(all_books) < limit:
        url = f"{FICTION_BASE}?per-page={per_page}" + (f"&page={page}" if page > 1 else "")
        log.info("Fetching catalog page %d (%s)...", page, url)
        try:
            html = get(url, session).text
        except Exception as exc:
            log.warning("Failed to fetch catalog page %d: %s", page, exc)
            break

        books = parse_fiction_page(html)
        if not books:
            log.info("No more books found on page %d — done paginating.", page)
            break

        known = {b["slug"] for b in all_books}
        new_books = [b for b in books if b["slug"] not in known]
        all_books.extend(new_books)
        log.info("Page %d: found %d new books (total so far: %d)",
                 page, len(new_books), len(all_books))

        if len(new_books) == 0:
            break

        page += 1
        time.sleep(RATE_LIMIT_S)

    return all_books[:limit]


def download_epub(epub_url: str, out_path: Path, session: requests.Session) -> int:
    """Download an EPUB file to out_path. Returns bytes downloaded."""
    r = get(epub_url, session, stream=True)
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
        "source": "Standard Ebooks",
        "official_url": "https://standardebooks.org",
        "license_note": (
            "Standard Ebooks productions are CC0 / public domain. "
            "Individual works have public-domain status varying by jurisdiction. "
            "See https://standardebooks.org/about/standard-ebooks-and-the-public-domain"
        ),
        "download_method": "individual EPUB, polite rate-limited scraper",
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
    parser = argparse.ArgumentParser(description="Download Standard Ebooks fiction EPUBs")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Max books to download (default: {DEFAULT_LIMIT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="List books that would be downloaded without downloading")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    meta = load_metadata(METADATA_FILE)
    already_done = {e["epub_url"] for e in meta.get("files", [])}

    log.info("=== Standard Ebooks Fiction Downloader ===")
    log.info("Target: %d books | Dry-run: %s", args.limit, args.dry_run)
    log.info("Output: %s", OUT_DIR.resolve())

    # --- Step 1: Collect fiction slugs ---
    slugs = collect_fiction_slugs(session, args.limit)
    log.info("Collected %d fiction book slugs", len(slugs))

    if args.dry_run:
        for b in slugs:
            print(b["slug"])
        return

    # --- Step 2: For each slug, get EPUB URL and download ---
    downloaded = 0
    skipped = 0
    failed = 0

    for i, book in enumerate(slugs, 1):
        log.info("[%d/%d] Processing: %s", i, len(slugs), book["slug"])

        time.sleep(RATE_LIMIT_S)
        epub_url = get_epub_url(book)
        if not epub_url:
            log.warning("  No EPUB URL found for %s — skipping", book["slug"])
            failed += 1
            continue

        if epub_url in already_done:
            log.info("  Already downloaded — skipping")
            skipped += 1
            continue

        # Derive filename
        slug_parts = book["slug"].strip("/").split("/")
        author = slug_parts[1] if len(slug_parts) > 1 else "unknown"
        title  = slug_parts[2] if len(slug_parts) > 2 else "unknown"
        filename = f"{author}_{title}.epub"
        out_path = OUT_DIR / filename

        if out_path.exists():
            log.info("  File exists — skipping (not in metadata, adding)")
            checksum = sha256_file(out_path)
            size = out_path.stat().st_size
        else:
            log.info("  Downloading: %s -> %s", epub_url, filename)
            try:
                time.sleep(RATE_LIMIT_S)
                size = download_epub(epub_url, out_path, session)
                checksum = sha256_file(out_path)
                log.info("  OK %s  (%d KB)  sha256: %s...", filename, size // 1024, checksum[:16])
                downloaded += 1
            except Exception as exc:
                log.error("  ✗ Failed to download %s: %s", epub_url, exc)
                if out_path.exists():
                    out_path.unlink()
                failed += 1
                continue

        meta.setdefault("files", []).append({
            "filename": filename,
            "epub_url": epub_url,
            "book_slug": book["slug"],
            "author_slug": book.get("author_slug"),
            "title_slug": book.get("title_slug"),
            "sha256": checksum,
            "size_bytes": size,
            "download_date": datetime.now(timezone.utc).date().isoformat(),
            "license_note": "See https://standardebooks.org/about/standard-ebooks-and-the-public-domain",
        })
        already_done.add(epub_url)
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
    log.info("Total files: %d", len(meta.get("files", [])))
    total_mb = meta["summary"]["total_size_bytes"] / 1024 / 1024
    log.info("Total size: %.1f MB", total_mb)


if __name__ == "__main__":
    main()
