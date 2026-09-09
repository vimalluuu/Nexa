"""
Tests for scripts.dedup.corpus_stats and scripts.dedup.balance_analysis
"""
import json
from pathlib import Path

import pytest

from scripts.dedup.corpus_stats import (
    CorpusStats,
    DocRecord,
    compute_stats,
    compute_combined_stats,
    _word_count,
    _type_token_ratio,
    _percentile,
)
from scripts.dedup.balance_analysis import (
    analyze_balance,
    select_pilot_candidates,
    PilotSelection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_record(source_id: str, text: str, dataset: str = "wikimedia_english",
                 pub_year: str = "") -> dict:
    r = {
        "dataset": dataset,
        "source_id": source_id,
        "title": f"Doc {source_id}",
        "source_url": f"https://example.com/{source_id}",
        "text": text,
    }
    if pub_year:
        r["publication_year"] = pub_year
        r["gutenberg_url"] = f"https://gutenberg.org/ebooks/{source_id}"
    return r


def _long_text(words: int = 100) -> str:
    return " ".join([f"word{i}" for i in range(words)])


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_word_count(self):
        assert _word_count("one two three") == 3
        assert _word_count("") == 0
        assert _word_count("  spaces  around  ") == 2

    def test_type_token_ratio_diverse(self):
        text = " ".join([f"word{i}" for i in range(100)])
        ttr = _type_token_ratio(text)
        assert ttr == pytest.approx(1.0)

    def test_type_token_ratio_repetitive(self):
        text = " ".join(["same"] * 100)
        ttr = _type_token_ratio(text)
        assert ttr == pytest.approx(1 / 100)

    def test_type_token_ratio_empty(self):
        assert _type_token_ratio("") == 0.0

    def test_percentile_median(self):
        vals = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _percentile(vals, 50) == pytest.approx(3.0)

    def test_percentile_min(self):
        vals = sorted([10.0, 20.0, 30.0])
        assert _percentile(vals, 0) == pytest.approx(10.0)

    def test_percentile_max(self):
        vals = sorted([10.0, 20.0, 30.0])
        assert _percentile(vals, 100) == pytest.approx(30.0)

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0


# ---------------------------------------------------------------------------
# Tests: CorpusStats
# ---------------------------------------------------------------------------
class TestCorpusStats:
    def test_initial_state(self):
        s = CorpusStats(dataset="test")
        assert s.doc_count == 0
        assert s.char_total == 0

    def test_update_increments(self):
        s = CorpusStats()
        s.update(DocRecord(chars=100, words=20, ttr=0.5))
        assert s.doc_count == 1
        assert s.char_total == 100
        assert s.word_total == 20

    def test_min_max_tracked(self):
        s = CorpusStats()
        s.update(DocRecord(chars=50,  words=10, ttr=0.5))
        s.update(DocRecord(chars=200, words=40, ttr=0.5))
        assert s.char_min == 50
        assert s.char_max == 200
        assert s.word_min == 10
        assert s.word_max == 40

    def test_avg_chars(self):
        s = CorpusStats()
        s.update(DocRecord(chars=100, words=10, ttr=0.5))
        s.update(DocRecord(chars=200, words=20, ttr=0.5))
        assert s.avg_chars == pytest.approx(150.0)

    def test_low_ttr_counted(self):
        s = CorpusStats()
        # Repetitive doc: TTR < 0.10, enough words
        s.update(DocRecord(chars=500, words=100, ttr=0.05))
        assert s.low_ttr_count == 1

    def test_low_ttr_not_counted_for_short_docs(self):
        s = CorpusStats()
        # Too short (< 50 words) — don't flag
        s.update(DocRecord(chars=100, words=10, ttr=0.02))
        assert s.low_ttr_count == 0

    def test_to_dict_has_required_keys(self):
        s = CorpusStats(dataset="wiki")
        s.update(DocRecord(chars=500, words=100, ttr=0.5))
        d = s.to_dict()
        for key in ["dataset", "doc_count", "char_total", "word_total",
                    "avg_chars", "avg_words", "char_min", "char_max",
                    "median_chars", "char_percentiles", "word_percentiles",
                    "low_ttr_docs", "low_ttr_pct"]:
            assert key in d

    def test_char_percentiles_ordered(self):
        s = CorpusStats()
        for i in range(1, 101):
            s.update(DocRecord(chars=i * 100, words=i * 10, ttr=0.5))
        pct = s.char_percentiles()
        assert pct["p10"] <= pct["p25"] <= pct["p50"] <= pct["p75"] <= pct["p90"]


# ---------------------------------------------------------------------------
# Tests: compute_stats
# ---------------------------------------------------------------------------
class TestComputeStats:
    def test_basic_count(self, tmp_path):
        records = [_make_record(str(i), _long_text(50)) for i in range(10)]
        p = tmp_path / "docs.jsonl"
        _write_jsonl(p, records)
        stats = compute_stats(p, "test")
        assert stats.doc_count == 10

    def test_char_total_correct(self, tmp_path):
        texts = ["hello world", "foo bar baz"]
        records = [_make_record(str(i), t) for i, t in enumerate(texts)]
        p = tmp_path / "docs.jsonl"
        _write_jsonl(p, records)
        stats = compute_stats(p, "test")
        assert stats.char_total == sum(len(t) for t in texts)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "docs.jsonl"
        p.write_text("", encoding="utf-8")
        stats = compute_stats(p, "test")
        assert stats.doc_count == 0

    def test_deterministic(self, tmp_path):
        records = [_make_record(str(i), _long_text(50)) for i in range(20)]
        p = tmp_path / "docs.jsonl"
        _write_jsonl(p, records)
        s1 = compute_stats(p, "test", seed=42)
        s2 = compute_stats(p, "test", seed=42)
        assert s1.doc_count == s2.doc_count
        assert s1.char_total == s2.char_total


# ---------------------------------------------------------------------------
# Tests: compute_combined_stats
# ---------------------------------------------------------------------------
class TestComputeCombinedStats:
    def test_totals(self, tmp_path):
        s1 = CorpusStats(dataset="a")
        s2 = CorpusStats(dataset="b")
        for _ in range(5):
            s1.update(DocRecord(chars=100, words=20, ttr=0.5))
            s2.update(DocRecord(chars=200, words=40, ttr=0.5))
        result = compute_combined_stats([s1, s2])
        assert result["total_docs"] == 10
        assert result["total_chars"] == 5 * 100 + 5 * 200
        assert result["total_words"] == 5 * 20 + 5 * 40

    def test_source_proportions(self, tmp_path):
        s1 = CorpusStats(dataset="a")
        s2 = CorpusStats(dataset="b")
        for _ in range(3):
            s1.update(DocRecord(chars=100, words=10, ttr=0.5))
        for _ in range(1):
            s2.update(DocRecord(chars=100, words=10, ttr=0.5))
        result = compute_combined_stats([s1, s2])
        props = result["source_proportions"]
        assert props["a"]["doc_pct"] == pytest.approx(75.0)
        assert props["b"]["doc_pct"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Tests: analyze_balance
# ---------------------------------------------------------------------------
class TestAnalyzeBalance:
    def _source_stats(self):
        return {
            "wikimedia_english": {"doc_count": 19536, "word_total": 49_000_000},
            "pg19":              {"doc_count": 25,    "word_total": 3_000_000},
        }

    def test_returns_three_scenarios(self):
        result = analyze_balance(self._source_stats())
        assert len(result) == 3

    def test_ratios_labeled_correctly(self):
        result = analyze_balance(self._source_stats())
        labels = [s["ratio"] for s in result]
        assert any("90/10" in l for l in labels)
        assert any("80/20" in l for l in labels)
        assert any("70/30" in l for l in labels)

    def test_fractions_sum_to_one(self):
        result = analyze_balance(self._source_stats())
        for sc in result:
            total = sc["wiki_fraction"] + sc["pg19_fraction"]
            assert total == pytest.approx(1.0)

    def test_word_counts_within_available(self):
        source = self._source_stats()
        result = analyze_balance(source)
        for sc in result:
            assert sc["wikimedia_english"]["available_words"] <= source["wikimedia_english"]["word_total"]
            assert sc["pg19"]["available_words"] <= source["pg19"]["word_total"]

    def test_custom_ratios(self):
        result = analyze_balance(self._source_stats(), ratios=[(0.50, 0.50)])
        assert len(result) == 1
        assert "50/50" in result[0]["ratio"]

    def test_empty_source_stats(self):
        result = analyze_balance({})
        for sc in result:
            assert sc["wikimedia_english"]["available_words"] == 0
            assert sc["pg19"]["available_words"] == 0


# ---------------------------------------------------------------------------
# Tests: select_pilot_candidates
# ---------------------------------------------------------------------------
class TestSelectPilotCandidates:
    def _make_wiki_jsonl(self, tmp_path: Path, n: int = 10, words_per: int = 200) -> Path:
        records = [
            _make_record(str(100 + i), " ".join([f"word{j}" for j in range(words_per)]))
            for i in range(n)
        ]
        p = tmp_path / "wiki.jsonl"
        _write_jsonl(p, records)
        return p

    def _make_pg19_jsonl(self, tmp_path: Path, n: int = 5, words_per: int = 200) -> Path:
        records = [
            _make_record(str(i), " ".join([f"book{j}" for j in range(words_per)]),
                         dataset="pg19", pub_year="1900")
            for i in range(n)
        ]
        p = tmp_path / "pg19.jsonl"
        _write_jsonl(p, records)
        return p

    def test_includes_all_pg19(self, tmp_path):
        wiki = self._make_wiki_jsonl(tmp_path, n=100)
        pg19 = self._make_pg19_jsonl(tmp_path, n=5)
        idx  = tmp_path / "pilot_index.jsonl"
        sel = select_pilot_candidates(wiki, pg19, idx, target_words=50_000, seed=42)
        assert sel.pg19_selected_docs == 5

    def test_deterministic_with_same_seed(self, tmp_path):
        wiki = self._make_wiki_jsonl(tmp_path, n=50, words_per=100)
        pg19 = self._make_pg19_jsonl(tmp_path, n=3, words_per=100)
        idx1 = tmp_path / "idx1.jsonl"
        idx2 = tmp_path / "idx2.jsonl"
        s1 = select_pilot_candidates(wiki, pg19, idx1, target_words=1000, seed=42)
        s2 = select_pilot_candidates(wiki, pg19, idx2, target_words=1000, seed=42)
        assert s1.wiki_selected_docs == s2.wiki_selected_docs
        assert s1.total_actual_words == s2.total_actual_words

    def test_output_index_is_valid_jsonl(self, tmp_path):
        wiki = self._make_wiki_jsonl(tmp_path, n=10, words_per=200)
        pg19 = self._make_pg19_jsonl(tmp_path, n=3, words_per=200)
        idx  = tmp_path / "pilot.jsonl"
        select_pilot_candidates(wiki, pg19, idx, target_words=5000, seed=42)
        with open(idx, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) > 0
        for rec in lines:
            assert "dataset"   in rec
            assert "source_id" in rec
            assert "words"     in rec

    def test_index_has_no_text_field(self, tmp_path):
        """Index should be lightweight — no text field."""
        wiki = self._make_wiki_jsonl(tmp_path, n=5, words_per=200)
        pg19 = self._make_pg19_jsonl(tmp_path, n=2, words_per=200)
        idx  = tmp_path / "pilot.jsonl"
        select_pilot_candidates(wiki, pg19, idx, target_words=5000, seed=42)
        with open(idx, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    assert "text" not in rec

    def test_to_dict_includes_disclaimer(self, tmp_path):
        wiki = self._make_wiki_jsonl(tmp_path, n=5)
        pg19 = self._make_pg19_jsonl(tmp_path, n=2)
        idx  = tmp_path / "pilot.jsonl"
        sel = select_pilot_candidates(wiki, pg19, idx, target_words=5000, seed=42)
        d = sel.to_dict()
        assert "UNKNOWN" in d["exact_nexa_token_count"]
        assert "disclaimer" in d

    def test_missing_wiki_doesnt_crash(self, tmp_path):
        missing = tmp_path / "missing_wiki.jsonl"
        pg19    = self._make_pg19_jsonl(tmp_path, n=3)
        idx     = tmp_path / "pilot.jsonl"
        sel = select_pilot_candidates(missing, pg19, idx, target_words=5000, seed=42)
        assert sel.wiki_selected_docs == 0
        assert sel.pg19_selected_docs == 3

    def test_word_target_respected(self, tmp_path):
        """Should not exceed the wiki word target significantly."""
        wiki = self._make_wiki_jsonl(tmp_path, n=1000, words_per=100)
        pg19 = self._make_pg19_jsonl(tmp_path, n=2, words_per=100)
        idx  = tmp_path / "pilot.jsonl"
        target = 5000
        sel = select_pilot_candidates(wiki, pg19, idx, target_words=target,
                                      wiki_fraction=1.0, pg19_fraction=0.0,
                                      seed=42)
        # Should not wildly overshoot
        assert sel.wiki_actual_words <= target + 200  # at most one extra doc
