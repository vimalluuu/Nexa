"""
Nexa Phase 8.3 - Cleaning Pipeline Configuration
=================================================
Configurable thresholds and paths for the extraction/cleaning pipeline.
All paths relative to the project root (run scripts from repo root).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_WIKI_DIR     = Path("data/raw/wikimedia_english/source")
RAW_PG19_DIR     = Path("data/raw/pg19/source")

CLEANED_WIKI_DIR = Path("data/cleaned/wikimedia_english")
CLEANED_PG19_DIR = Path("data/cleaned/pg19")

REPORTS_DIR      = Path("data/reports")

WIKI_OUTPUT_FILE = CLEANED_WIKI_DIR / "docs.jsonl"
PG19_OUTPUT_FILE = CLEANED_PG19_DIR / "docs.jsonl"

RAW_REPORT_FILE  = REPORTS_DIR / "raw_extraction_report.json"
CLEAN_REPORT_FILE = REPORTS_DIR / "cleaning_report.json"

# ---------------------------------------------------------------------------
# Wikimedia settings
# ---------------------------------------------------------------------------
WIKI_SNAPSHOT        = "20260901"
WIKI_NAMESPACE_MAIN  = "0"          # Only main article namespace
WIKI_BASE_URL        = "https://en.wikipedia.org/wiki/"

# ---------------------------------------------------------------------------
# Document filtering thresholds (configurable)
# ---------------------------------------------------------------------------
# Minimum text length (characters) to keep a document after cleaning
MIN_CHARS = 200
# Minimum word count to keep a document
MIN_WORDS = 30
# Maximum text length (chars) — very large docs are usually OK; set high
MAX_CHARS = 10_000_000

# ---------------------------------------------------------------------------
# PG-19 settings
# ---------------------------------------------------------------------------
PG19_GCS_BASE = "https://storage.googleapis.com/deepmind-gutenberg/train"
