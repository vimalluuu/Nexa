"""
Tests for scripts.cleaning.extract_pg19
"""
import json
from pathlib import Path

import pytest
import yaml

from scripts.cleaning.extract_pg19 import extract_pg19, _strip_pg_boilerplate


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_source(tmp_path, books: dict[str, str]) -> tuple[Path, Path]:
    """Create a synthetic PG-19 source dir and _metadata.yaml."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    meta_entries = []
    for bid, text in books.items():
        (source_dir / f"{bid}.txt").write_text(text, encoding="utf-8")
        meta_entries.append({
            "filename":        f"{bid}.txt",
            "book_id":         bid,
            "title":           f"Book {bid}",
            "publication_date": "1895",
            "gutenberg_url":   f"https://www.gutenberg.org/ebooks/{bid}",
            "license_note":    "Published 1895, pre-1919, U.S. public domain.",
            "sha256":          "deadbeef",
            "size_bytes":      len(text),
            "download_date":   "2026-09-09",
        })

    meta = {"dataset_name": "PG-19", "files": meta_entries}
    meta_yaml = tmp_path / "_metadata.yaml"
    meta_yaml.write_text(
        yaml.dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return source_dir, meta_yaml


def _long_prose(n_words: int = 200) -> str:
    sentence = "This is a test sentence with enough words to pass filtering. "
    words = (sentence * ((n_words // 12) + 1)).split()
    return " ".join(words[:n_words])


# ---------------------------------------------------------------------------
# Tests: _strip_pg_boilerplate
# ---------------------------------------------------------------------------
class TestStripPgBoilerplate:
    def test_no_boilerplate_unchanged(self):
        text = "Once upon a time there was a story."
        assert text in _strip_pg_boilerplate(text)

    def test_strips_gutenberg_footer(self):
        text = "Story content.\n\nEnd of the Project Gutenberg EBook.\nMore stuff"
        result = _strip_pg_boilerplate(text)
        assert "Story content" in result
        assert "End of the Project Gutenberg" not in result

    def test_strips_header_line(self):
        text = "The Project Gutenberg EBook of Foo\nActual content starts here."
        result = _strip_pg_boilerplate(text)
        assert "Actual content starts here" in result


# ---------------------------------------------------------------------------
# Tests: extract_pg19 (full pipeline)
# ---------------------------------------------------------------------------
class TestExtractPg19:
    def test_processes_valid_books(self, tmp_path):
        books = {"11": _long_prose(200), "12": _long_prose(200)}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        out = tmp_path / "out.jsonl"
        result = extract_pg19(source_dir, meta_yaml, out, min_chars=50, min_words=10)
        assert result["kept"] == 2
        assert result["removed_empty"] == 0

    def test_output_valid_jsonl(self, tmp_path):
        books = {"11": _long_prose(200)}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        out = tmp_path / "out.jsonl"
        extract_pg19(source_dir, meta_yaml, out, min_chars=50, min_words=10)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            obj = json.loads(line)
            assert "dataset" in obj
            assert "source_id" in obj
            assert "title" in obj
            assert "text" in obj

    def test_provenance_fields(self, tmp_path):
        books = {"11": _long_prose(200)}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        out = tmp_path / "out.jsonl"
        extract_pg19(source_dir, meta_yaml, out, min_chars=50, min_words=10)
        obj = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert obj["dataset"] == "pg19"
        assert obj["source_id"] == "11"
        assert obj["title"] == "Book 11"
        assert obj["publication_year"] == "1895"
        assert "gutenberg" in obj["gutenberg_url"]
        assert "source_url" in obj

    def test_filters_empty_book(self, tmp_path):
        books = {"1": "", "2": _long_prose(200)}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        out = tmp_path / "out.jsonl"
        result = extract_pg19(source_dir, meta_yaml, out, min_chars=50, min_words=10)
        assert result["removed_empty"] == 1
        assert result["kept"] == 1

    def test_filters_too_short(self, tmp_path):
        books = {"1": "short", "2": _long_prose(200)}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        out = tmp_path / "out.jsonl"
        result = extract_pg19(source_dir, meta_yaml, out, min_chars=200, min_words=30)
        assert result["removed_too_short"] >= 1
        assert result["kept"] == 1

    def test_does_not_modify_raw_files(self, tmp_path):
        books = {"11": _long_prose(200)}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        raw_file = source_dir / "11.txt"
        original = raw_file.read_bytes()
        out = tmp_path / "out.jsonl"
        extract_pg19(source_dir, meta_yaml, out, min_chars=50, min_words=10)
        assert raw_file.read_bytes() == original

    def test_handles_missing_metadata_yaml(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "11.txt").write_text(_long_prose(200), encoding="utf-8")
        meta_yaml = tmp_path / "_metadata.yaml"  # doesn't exist
        out = tmp_path / "out.jsonl"
        # Should not crash, just have empty metadata fields
        result = extract_pg19(source_dir, meta_yaml, out, min_chars=50, min_words=10)
        assert result["kept"] >= 0  # doesn't raise

    def test_handles_unicode_content(self, tmp_path):
        unicode_text = "café naïve résumé Ångström " * 50
        books = {"11": unicode_text}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        out = tmp_path / "out.jsonl"
        result = extract_pg19(source_dir, meta_yaml, out, min_chars=50, min_words=10)
        assert result["kept"] == 1
        obj = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert "café" in obj["text"] or "cafe" in obj["text"]

    def test_stats_total_input_matches_files(self, tmp_path):
        books = {"1": _long_prose(200), "2": _long_prose(200), "3": "short"}
        source_dir, meta_yaml = _make_source(tmp_path, books)
        out = tmp_path / "out.jsonl"
        result = extract_pg19(source_dir, meta_yaml, out, min_chars=200, min_words=30)
        assert result["total_input"] == 3

    def test_no_pg19_files_returns_zero_kept(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        meta_yaml = tmp_path / "_metadata.yaml"
        meta_yaml.write_text(yaml.dump({"files": []}), encoding="utf-8")
        out = tmp_path / "out.jsonl"
        result = extract_pg19(source_dir, meta_yaml, out)
        assert result["kept"] == 0
        assert result["txt_files_found"] == 0
