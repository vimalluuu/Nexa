"""
Tests for scripts.cleaning.extract_wikimedia

Uses synthetic in-memory XML to test streaming logic without touching raw files.
"""
import bz2
import io
import json
import textwrap
from pathlib import Path

import pytest

from scripts.cleaning.extract_wikimedia import iter_wiki_pages, extract_wikimedia

# ---------------------------------------------------------------------------
# Helpers: build synthetic MediaWiki XML
# ---------------------------------------------------------------------------
MW_HEADER = '''<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"
           version="0.11" xml:lang="en">
  <siteinfo>
    <sitename>Wikipedia</sitename>
    <dbname>enwiki</dbname>
    <namespaces>
      <namespace key="0" case="first-letter" />
      <namespace key="1" case="first-letter">Talk</namespace>
      <namespace key="10" case="first-letter">Template</namespace>
      <namespace key="14" case="first-letter">Category</namespace>
    </namespaces>
  </siteinfo>
'''

MW_FOOTER = '</mediawiki>\n'


def _make_page(title, ns, page_id, wikitext, redirect_title=None):
    redirect = f'  <redirect title="{redirect_title}" />\n' if redirect_title else ""
    return f'''
  <page>
    <title>{title}</title>
    <ns>{ns}</ns>
    <id>{page_id}</id>
    {redirect}
    <revision>
      <id>999{page_id}</id>
      <timestamp>2026-01-01T00:00:00Z</timestamp>
      <text bytes="100" xml:space="preserve">{wikitext}</text>
    </revision>
  </page>
'''


def _build_xml(*pages) -> bytes:
    return (MW_HEADER + "".join(pages) + MW_FOOTER).encode("utf-8")


def _bz2_bytes(data: bytes) -> bytes:
    return bz2.compress(data)


# ---------------------------------------------------------------------------
# Tests: iter_wiki_pages
# ---------------------------------------------------------------------------
class TestIterWikiPages:
    def _make_bz2_file(self, tmp_path, *pages):
        xml_bytes = _build_xml(*pages)
        bz2_path = tmp_path / "test.xml.bz2"
        bz2_path.write_bytes(_bz2_bytes(xml_bytes))
        return bz2_path

    def test_yields_main_namespace_article(self, tmp_path):
        page = _make_page("TestArticle", 0, 100, "Some article wikitext.")
        bz2_path = self._make_bz2_file(tmp_path, page)
        pages = list(iter_wiki_pages(bz2_path))
        assert len(pages) == 1
        assert pages[0]["title"] == "TestArticle"
        assert pages[0]["namespace"] == "0"
        assert pages[0]["is_redirect"] is False
        assert "Some article wikitext" in pages[0]["wikitext"]

    def test_detects_redirect(self, tmp_path):
        page = _make_page("RedirectPage", 0, 101,
                          "#REDIRECT [[Target]]", redirect_title="Target")
        bz2_path = self._make_bz2_file(tmp_path, page)
        pages = list(iter_wiki_pages(bz2_path))
        assert pages[0]["is_redirect"] is True

    def test_preserves_namespace(self, tmp_path):
        page = _make_page("Template:Foo", 10, 200, "{{Template content}}")
        bz2_path = self._make_bz2_file(tmp_path, page)
        pages = list(iter_wiki_pages(bz2_path))
        assert pages[0]["namespace"] == "10"

    def test_multiple_pages(self, tmp_path):
        pages_xml = [
            _make_page("Article1", 0, 1, "Text one"),
            _make_page("Article2", 0, 2, "Text two"),
            _make_page("Article3", 0, 3, "Text three"),
        ]
        bz2_path = self._make_bz2_file(tmp_path, *pages_xml)
        pages = list(iter_wiki_pages(bz2_path))
        assert len(pages) == 3
        titles = [p["title"] for p in pages]
        assert "Article1" in titles
        assert "Article3" in titles

    def test_preserves_page_id(self, tmp_path):
        page = _make_page("MyPage", 0, 12345, "Text")
        bz2_path = self._make_bz2_file(tmp_path, page)
        pages = list(iter_wiki_pages(bz2_path))
        assert pages[0]["page_id"] == "12345"

    def test_handles_empty_wikitext(self, tmp_path):
        page = _make_page("EmptyPage", 0, 999, "")
        bz2_path = self._make_bz2_file(tmp_path, page)
        pages = list(iter_wiki_pages(bz2_path))
        assert pages[0]["wikitext"] == ""

    def test_preserves_timestamp(self, tmp_path):
        page = _make_page("TimedPage", 0, 1, "Content")
        bz2_path = self._make_bz2_file(tmp_path, page)
        pages = list(iter_wiki_pages(bz2_path))
        assert pages[0]["timestamp"] == "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Tests: extract_wikimedia (full pipeline)
# ---------------------------------------------------------------------------
class TestExtractWikimedia:
    def _make_bz2_file(self, tmp_path, *pages):
        xml_bytes = _build_xml(*pages)
        bz2_path = tmp_path / "dump.xml.bz2"
        bz2_path.write_bytes(_bz2_bytes(xml_bytes))
        return bz2_path

    def _long_wikitext(self, words=100):
        return " ".join(["word"] * words)

    def test_filters_ns_nonzero(self, tmp_path):
        pages_xml = [
            _make_page("Talk:Foo", 1, 1, self._long_wikitext(100)),
            _make_page("Template:Bar", 10, 2, self._long_wikitext(100)),
            _make_page("Article", 0, 3, self._long_wikitext(100)),
        ]
        bz2_path = self._make_bz2_file(tmp_path, *pages_xml)
        out = tmp_path / "out.jsonl"
        result = extract_wikimedia(bz2_path, out, min_chars=10, min_words=5)
        assert result["skipped_wrong_ns"] == 2
        assert result["kept"] == 1

    def test_filters_redirects(self, tmp_path):
        pages_xml = [
            _make_page("Redir", 0, 1, "#REDIRECT [[Target]]", "Target"),
            _make_page("Real", 0, 2, self._long_wikitext(100)),
        ]
        bz2_path = self._make_bz2_file(tmp_path, *pages_xml)
        out = tmp_path / "out.jsonl"
        result = extract_wikimedia(bz2_path, out, min_chars=10, min_words=5)
        assert result["skipped_redirects"] == 1
        assert result["kept"] == 1

    def test_output_is_valid_jsonl(self, tmp_path):
        text = "This is a test article with plenty of words to keep."
        # Repeat enough times to exceed min_chars
        long_text = (text + " ") * 10
        page = _make_page("MyArticle", 0, 1, long_text)
        bz2_path = self._make_bz2_file(tmp_path, page)
        out = tmp_path / "out.jsonl"
        extract_wikimedia(bz2_path, out, min_chars=50, min_words=10)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            obj = json.loads(line)
            assert "dataset" in obj
            assert "source_id" in obj
            assert "title" in obj
            assert "text" in obj

    def test_provenance_fields_preserved(self, tmp_path):
        long_text = "Article prose text. " * 20
        page = _make_page("ProvenanceTest", 0, 42, long_text)
        bz2_path = self._make_bz2_file(tmp_path, page)
        out = tmp_path / "out.jsonl"
        extract_wikimedia(bz2_path, out, snapshot="20260901",
                          min_chars=10, min_words=5)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        obj = json.loads(lines[0])
        assert obj["dataset"] == "wikimedia_english"
        assert obj["source_id"] == "42"
        assert obj["title"] == "ProvenanceTest"
        assert obj["snapshot"] == "20260901"
        assert "en.wikipedia.org" in obj["source_url"]

    def test_filters_empty_after_cleaning(self, tmp_path):
        # Wikitext that cleans to nothing
        page = _make_page("EmptyAfterClean", 0, 1,
                          "{{Template}} [[Category:Foo]] <!-- comment -->")
        bz2_path = self._make_bz2_file(tmp_path, page)
        out = tmp_path / "out.jsonl"
        result = extract_wikimedia(bz2_path, out, min_chars=200, min_words=30)
        assert result["kept"] == 0
        assert result["removed_empty"] + result["removed_too_short"] >= 1

    def test_does_not_modify_raw_file(self, tmp_path):
        page = _make_page("Test", 0, 1, "word " * 100)
        bz2_path = self._make_bz2_file(tmp_path, page)
        original_bytes = bz2_path.read_bytes()
        out = tmp_path / "out.jsonl"
        extract_wikimedia(bz2_path, out, min_chars=10, min_words=5)
        assert bz2_path.read_bytes() == original_bytes

    def test_stats_total_input_counts_ns0_nonredirect(self, tmp_path):
        """total_input in filter stats counts only after ns/redirect filter."""
        pages_xml = [
            _make_page("TalkPage", 1, 1, "word " * 50),
            _make_page("Redir", 0, 2, "#REDIRECT [[X]]", "X"),
            _make_page("Article", 0, 3, "word " * 100),
        ]
        bz2_path = self._make_bz2_file(tmp_path, *pages_xml)
        out = tmp_path / "out.jsonl"
        result = extract_wikimedia(bz2_path, out, min_chars=10, min_words=5)
        # Only Article (ns=0, non-redirect) reaches filter; talk and redir don't
        assert result["total_input"] == 1
