"""
Tests for scripts.dedup.exact_dedup
"""
import json
import bz2
from pathlib import Path

import pytest

from scripts.dedup.exact_dedup import (
    content_hash,
    iter_jsonl,
    dedup_source,
    run_deduplication,
    DeduplicationStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _wiki_record(source_id: str, text: str, title: str = "") -> dict:
    return {
        "dataset": "wikimedia_english",
        "source_id": source_id,
        "title": title or f"Article {source_id}",
        "source_url": f"https://en.wikipedia.org/wiki/Article_{source_id}",
        "snapshot": "20260901",
        "text": text,
    }


def _pg19_record(source_id: str, text: str, title: str = "") -> dict:
    return {
        "dataset": "pg19",
        "source_id": source_id,
        "title": title or f"Book {source_id}",
        "publication_year": "1900",
        "gutenberg_url": f"https://www.gutenberg.org/ebooks/{source_id}",
        "source_url": f"https://storage.googleapis.com/deepmind-gutenberg/train/{source_id}.txt",
        "text": text,
    }


# ---------------------------------------------------------------------------
# Tests: content_hash
# ---------------------------------------------------------------------------
class TestContentHash:
    def test_returns_sha256_hex(self):
        h = content_hash("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert content_hash("same text") == content_hash("same text")

    def test_different_texts_different_hashes(self):
        assert content_hash("text one") != content_hash("text two")

    def test_whitespace_stripped(self):
        # Leading/trailing whitespace should not change the hash
        assert content_hash("hello") == content_hash("  hello  ")
        assert content_hash("hello") == content_hash("hello\n")

    def test_empty_string(self):
        h = content_hash("")
        assert len(h) == 64  # still valid SHA-256

    def test_unicode_text(self):
        h = content_hash("café naïve résumé")
        assert len(h) == 64

    def test_nfc_normalization(self):
        # Decomposed and precomposed forms should hash identically
        composed   = "caf\u00e9"
        decomposed = "cafe\u0301"
        assert content_hash(composed) == content_hash(decomposed)


# ---------------------------------------------------------------------------
# Tests: iter_jsonl
# ---------------------------------------------------------------------------
class TestIterJsonl:
    def test_yields_all_records(self, tmp_path):
        records = [{"id": i, "text": f"text {i}"} for i in range(5)]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = list(iter_jsonl(p))
        assert len(result) == 5
        assert result[0]["id"] == 0
        assert result[4]["id"] == 4

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
        result = list(iter_jsonl(p))
        assert len(result) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert list(iter_jsonl(p)) == []


# ---------------------------------------------------------------------------
# Tests: dedup_source
# ---------------------------------------------------------------------------
class TestDedupSource:
    def test_no_duplicates(self, tmp_path):
        records = [_wiki_record(str(i), f"unique text number {i} " * 10) for i in range(5)]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        seen = set()
        stats = dedup_source(inp, out, seen, "wikimedia_english")
        assert stats.kept_docs == 5
        assert stats.duplicate_docs == 0
        result = _read_jsonl(out)
        assert len(result) == 5

    def test_detects_exact_duplicates(self, tmp_path):
        repeated_text = "this is the exact same text " * 20
        records = [
            _wiki_record("1", repeated_text, "Article A"),
            _wiki_record("2", repeated_text, "Article B"),  # duplicate
            _wiki_record("3", "different text " * 20, "Article C"),
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        seen = set()
        stats = dedup_source(inp, out, seen, "wikimedia_english")
        assert stats.kept_docs == 2
        assert stats.duplicate_docs == 1
        result = _read_jsonl(out)
        assert len(result) == 2

    def test_first_occurrence_kept(self, tmp_path):
        text = "same text " * 20
        records = [
            _wiki_record("1", text, "First"),
            _wiki_record("2", text, "Second"),
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        seen = set()
        dedup_source(inp, out, seen, "wikimedia_english")
        result = _read_jsonl(out)
        assert result[0]["source_id"] == "1"

    def test_cross_source_dedup(self, tmp_path):
        """Documents first seen in wiki are deduped from pg19."""
        shared_text = "shared text that appears in both sources " * 10
        wiki_records = [_wiki_record("100", shared_text, "Wiki article")]
        pg19_records = [_pg19_record("11", shared_text, "PG book")]

        wiki_in  = tmp_path / "wiki_in.jsonl"
        wiki_out = tmp_path / "wiki_out.jsonl"
        pg19_in  = tmp_path / "pg19_in.jsonl"
        pg19_out = tmp_path / "pg19_out.jsonl"

        _write_jsonl(wiki_in, wiki_records)
        _write_jsonl(pg19_in, pg19_records)

        seen = set()
        wiki_stats = dedup_source(wiki_in, wiki_out, seen, "wikimedia_english")
        pg19_stats = dedup_source(pg19_in, pg19_out, seen, "pg19")

        assert wiki_stats.kept_docs == 1
        assert pg19_stats.kept_docs == 0
        assert pg19_stats.duplicate_docs == 1

    def test_provenance_preserved(self, tmp_path):
        records = [_wiki_record("42", "unique prose text " * 20, "My Article")]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        dedup_source(inp, out, set(), "wikimedia_english")
        result = _read_jsonl(out)
        assert result[0]["source_id"] == "42"
        assert result[0]["title"] == "My Article"
        assert result[0]["dataset"] == "wikimedia_english"
        assert "source_url" in result[0]

    def test_does_not_modify_input(self, tmp_path):
        records = [_wiki_record("1", "some text " * 20)]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        original_content = inp.read_bytes()
        dedup_source(inp, out, set(), "wikimedia_english")
        assert inp.read_bytes() == original_content

    def test_stats_total_input_counted(self, tmp_path):
        records = [_wiki_record(str(i), f"text {i} " * 20) for i in range(10)]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        stats = dedup_source(inp, out, set(), "wikimedia_english")
        assert stats.input_docs == 10

    def test_whitespace_variants_treated_as_duplicate(self, tmp_path):
        """Text differing only in leading/trailing whitespace → same hash → duplicate."""
        text = "the content of this document is important " * 10
        records = [
            _wiki_record("1", text),
            _wiki_record("2", "  " + text + "  "),  # same after strip → duplicate
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        stats = dedup_source(inp, out, set(), "wikimedia_english")
        assert stats.duplicate_docs == 1


# ---------------------------------------------------------------------------
# Tests: run_deduplication
# ---------------------------------------------------------------------------
class TestRunDeduplication:
    def test_combined_stats_correct(self, tmp_path):
        wiki_records = [_wiki_record(str(i), f"unique wiki text {i} " * 20) for i in range(5)]
        pg19_records = [_pg19_record(str(i), f"unique pg19 text {i} " * 20) for i in range(3)]

        wiki_in  = tmp_path / "wiki_in.jsonl"
        wiki_out = tmp_path / "wiki_out.jsonl"
        pg19_in  = tmp_path / "pg19_in.jsonl"
        pg19_out = tmp_path / "pg19_out.jsonl"

        _write_jsonl(wiki_in, wiki_records)
        _write_jsonl(pg19_in, pg19_records)

        sources = [
            (wiki_in, wiki_out, "wikimedia_english"),
            (pg19_in, pg19_out, "pg19"),
        ]
        _, combined = run_deduplication(sources)

        assert combined["total_input_docs"] == 8
        assert combined["total_kept_docs"] == 8
        assert combined["total_duplicate_docs"] == 0

    def test_missing_input_skipped(self, tmp_path):
        missing = tmp_path / "missing.jsonl"
        out = tmp_path / "out.jsonl"
        sources = [(missing, out, "test")]
        _, combined = run_deduplication(sources)
        assert combined["total_input_docs"] == 0

    def test_duplicate_pct_calculated(self, tmp_path):
        shared_text = "shared content text " * 20
        records = [
            _wiki_record("1", shared_text),
            _wiki_record("2", shared_text),  # duplicate
            _wiki_record("3", "different text " * 20),
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, records)
        _, combined = run_deduplication([(inp, out, "test")])
        assert combined["total_duplicate_docs"] == 1
        assert combined["total_duplicate_pct"] == pytest.approx(100 / 3, abs=0.01)
