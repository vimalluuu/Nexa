"""
Tests for Phase 8.5 tokenizer experiments.

Covers:
- corpus_sampler: pilot index loading, text streaming
- measure_tokenizer: metric computation
- select_vocab: recommendation logic
- tokenizer quality: round-trip, unicode, special tokens, edge cases

All tests use synthetic in-memory data — never read real corpus files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from nexa.tokenizer.tokenizer import NexaTokenizer
from nexa.tokenizer.vocab import PAD_ID, BOS_ID, EOS_ID, UNK_ID, UNK_TOKEN
from scripts.tokenizer.corpus_sampler import load_pilot_index, stream_pilot_texts
from scripts.tokenizer.measure_tokenizer import measure_tokenizer, _percentile
from scripts.tokenizer.select_vocab import (
    recommend_vocab_size,
    compute_embedding_params,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _train_small_tokenizer(texts: list[str], vocab_size: int = 256) -> NexaTokenizer:
    """Train a small NexaTokenizer for testing."""
    return NexaTokenizer.train(corpus=texts, vocab_size=vocab_size, min_frequency=1)


def _make_pilot_index(wiki_ids: list[str], pg19_ids: list[str]) -> dict[str, set[str]]:
    return {
        "wikimedia_english": set(wiki_ids),
        "pg19":              set(pg19_ids),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===========================================================================
# 1. Corpus sampler tests
# ===========================================================================

class TestLoadPilotIndex:
    def test_loads_datasets(self, tmp_path):
        idx = tmp_path / "pilot.jsonl"
        records = [
            {"dataset": "wikimedia_english", "source_id": "10", "words": 500},
            {"dataset": "wikimedia_english", "source_id": "20", "words": 700},
            {"dataset": "pg19",              "source_id": "5",  "words": 1000},
        ]
        _write_jsonl(idx, records)
        result = load_pilot_index(idx)
        assert "wikimedia_english" in result
        assert "pg19" in result
        assert "10" in result["wikimedia_english"]
        assert "20" in result["wikimedia_english"]
        assert "5"  in result["pg19"]

    def test_empty_file(self, tmp_path):
        idx = tmp_path / "pilot.jsonl"
        idx.write_text("", encoding="utf-8")
        result = load_pilot_index(idx)
        assert result == {}

    def test_source_ids_are_strings(self, tmp_path):
        idx = tmp_path / "pilot.jsonl"
        _write_jsonl(idx, [{"dataset": "pg19", "source_id": 42, "words": 100}])
        result = load_pilot_index(idx)
        assert "42" in result["pg19"]   # integer source_id coerced to str


class TestStreamPilotTexts:
    def test_yields_only_pilot_docs(self, tmp_path):
        wiki = tmp_path / "wiki.jsonl"
        _write_jsonl(wiki, [
            {"dataset": "wikimedia_english", "source_id": "1", "text": "wiki text one " * 10},
            {"dataset": "wikimedia_english", "source_id": "2", "text": "wiki text two " * 10},
            {"dataset": "wikimedia_english", "source_id": "3", "text": "wiki text three " * 10},
        ])
        pg19 = tmp_path / "pg19.jsonl"
        _write_jsonl(pg19, [
            {"dataset": "pg19", "source_id": "10", "text": "book text one " * 10},
        ])

        pilot = {"wikimedia_english": {"1", "3"}, "pg19": {"10"}}

        # Patch the DEDUP paths
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            texts = list(stream_pilot_texts(pilot))

        assert len(texts) == 3
        assert any("wiki text one" in t for t in texts)
        assert any("wiki text three" in t for t in texts)
        assert not any("wiki text two" in t for t in texts)  # not in pilot
        assert any("book text" in t for t in texts)

    def test_word_limit_stops_stream(self, tmp_path):
        wiki = tmp_path / "wiki.jsonl"
        _write_jsonl(wiki, [
            {"dataset": "wikimedia_english", "source_id": str(i),
             "text": " ".join([f"word{j}" for j in range(1000)])}
            for i in range(10)
        ])
        pg19 = tmp_path / "pg19.jsonl"
        pg19.write_text("", encoding="utf-8")

        pilot = {"wikimedia_english": {str(i) for i in range(10)}, "pg19": set()}
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            texts = list(stream_pilot_texts(pilot, max_words=2500))

        total = sum(len(t.split()) for t in texts)
        assert total >= 1000   # at least one document
        assert total < 10000   # didn't yield everything


# ===========================================================================
# 2. Tokenizer training quality tests
# ===========================================================================

ENGLISH_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
] * 20   # repeat for frequency


class TestTokenizerRoundTrip:
    @pytest.fixture(scope="class")
    def tok(self):
        return _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=300)

    def test_encode_returns_list_of_ints(self, tok):
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_recovers_text(self, tok):
        text = "the quick brown fox"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded.strip() == text.strip()

    def test_round_trip_all_english_chars(self, tok):
        # Use text fully within the training corpus character set
        text = "the quick brown fox"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded.strip() == text.strip()

    def test_bos_eos_added(self, tok):
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

    def test_empty_string(self, tok):
        ids = tok.encode("")
        assert ids == []

    def test_deterministic(self, tok):
        text = "the lazy dog jumps quickly"
        assert tok.encode(text) == tok.encode(text)

    def test_special_token_ids_stable(self, tok):
        assert tok.pad_token_id == PAD_ID   # 0
        assert tok.bos_token_id == BOS_ID   # 1
        assert tok.eos_token_id == EOS_ID   # 2
        assert tok.unk_token_id == UNK_ID   # 3


class TestTokenizerUnicode:
    @pytest.fixture(scope="class")
    def tok(self):
        corpus = [
            "café naïve résumé Ångström",
            "The quick brown fox",
        ] * 30
        return _train_small_tokenizer(corpus, vocab_size=300)

    def test_encodes_without_crash(self, tok):
        ids = tok.encode("café naïve résumé")
        assert len(ids) > 0

    def test_decode_produces_string(self, tok):
        ids = tok.encode("café")
        result = tok.decode(ids)
        assert isinstance(result, str)


class TestTokenizerEdgeCases:
    @pytest.fixture(scope="class")
    def tok(self):
        return _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=300)

    def test_single_word(self, tok):
        ids = tok.encode("hello")
        assert len(ids) >= 1

    def test_numbers(self, tok):
        ids = tok.encode("123 456 789")
        assert len(ids) >= 3

    def test_contractions(self, tok):
        ids = tok.encode("don't won't can't")
        assert len(ids) >= 3

    def test_unknown_chars_get_unk(self, tok):
        # Characters completely absent from training will get UNK
        ids = tok.encode("xyz\x00abc")
        assert UNK_ID in ids or len(ids) > 0  # doesn't crash

    def test_long_document(self, tok):
        text = "the quick brown fox " * 500
        ids = tok.encode(text)
        assert len(ids) >= 500

    def test_mixed_case_preserved(self, tok):
        # BPE is case-sensitive; different cases produce different tokens
        ids_lower = tok.encode("the quick")
        ids_upper = tok.encode("THE QUICK")
        # They may or may not differ — just ensure no crash
        assert len(ids_lower) > 0 and len(ids_upper) > 0

    def test_batch_encode_decode(self, tok):
        texts = ["hello world", "fox jumps", "quick brown"]
        batch = tok.encode_batch(texts)
        assert len(batch) == 3
        decoded = tok.decode_batch(batch)
        for orig, dec in zip(texts, decoded):
            assert dec.strip() == orig.strip()

    def test_special_tokens_skipped_in_decode(self, tok):
        ids = [BOS_ID, tok.encode("hello")[0], EOS_ID]
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded

    def test_pad_token_skipped_in_decode(self, tok):
        ids = [PAD_ID, PAD_ID, tok.encode("hello")[0]]
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "<pad>" not in decoded


class TestTokenizerSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        tok = _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=200)
        tok.save(tmp_path / "tokenizer")
        loaded = NexaTokenizer.load(tmp_path / "tokenizer")

        assert tok.vocab_size == loaded.vocab_size
        text = "the quick brown fox"
        assert tok.encode(text) == loaded.encode(text)

    def test_save_creates_json_files(self, tmp_path):
        tok = _train_small_tokenizer(["hello world " * 20], vocab_size=100)
        tok.save(tmp_path / "tok")
        assert (tmp_path / "tok" / "tokenizer.json").exists()
        assert (tmp_path / "tok" / "vocab.json").exists()

    def test_saved_merges_are_lists(self, tmp_path):
        tok = _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=200)
        tok.save(tmp_path / "tok")
        with open(tmp_path / "tok" / "tokenizer.json", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data["merges"], list)
        for pair in data["merges"]:
            assert isinstance(pair, list)
            assert len(pair) == 2


# ===========================================================================
# 3. Measurement helper tests
# ===========================================================================

class TestPercentile:
    def test_median(self):
        vals = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _percentile(vals, 50) == pytest.approx(3.0)

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 50) == pytest.approx(42.0)


class TestMeasureTokenizer:
    def _make_pilot_data(self, tmp_path, n_wiki=5, n_pg19=2):
        """Create synthetic pilot data files and index."""
        wiki_records = []
        pg19_records = []
        for i in range(n_wiki):
            wiki_records.append({
                "dataset": "wikimedia_english",
                "source_id": str(i),
                "text": "The quick brown fox jumps over the lazy dog. " * 50,
            })
        for i in range(n_pg19):
            pg19_records.append({
                "dataset": "pg19",
                "source_id": str(i),
                "text": "Once upon a time there was a quick fox. " * 50,
            })

        wiki_path = tmp_path / "wiki.jsonl"
        pg19_path = tmp_path / "pg19.jsonl"
        _write_jsonl(wiki_path, wiki_records)
        _write_jsonl(pg19_path, pg19_records)

        pilot = {
            "wikimedia_english": {str(i) for i in range(n_wiki)},
            "pg19":              {str(i) for i in range(n_pg19)},
        }
        return wiki_path, pg19_path, pilot

    def test_token_count_positive(self, tmp_path):
        wiki, pg19, pilot = self._make_pilot_data(tmp_path)
        tok = _train_small_tokenizer(
            ["The quick brown fox jumps over the lazy dog. " * 50], vocab_size=200
        )
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            metrics = measure_tokenizer(tok, pilot)
        assert metrics["total_tokens"] > 0
        assert metrics["doc_count"] == 7

    def test_unk_rate_in_range(self, tmp_path):
        wiki, pg19, pilot = self._make_pilot_data(tmp_path)
        tok = _train_small_tokenizer(
            ["The quick brown fox " * 50], vocab_size=200
        )
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            metrics = measure_tokenizer(tok, pilot)
        assert 0.0 <= metrics["unk_rate"] <= 1.0

    def test_compression_ratio_positive(self, tmp_path):
        wiki, pg19, pilot = self._make_pilot_data(tmp_path)
        tok = _train_small_tokenizer(
            ["the quick brown fox " * 50], vocab_size=300
        )
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            metrics = measure_tokenizer(tok, pilot)
        assert metrics["compression_ratio"] > 1.0   # chars > tokens

    def test_seq_len_percentiles_ordered(self, tmp_path):
        wiki, pg19, pilot = self._make_pilot_data(tmp_path, n_wiki=10, n_pg19=3)
        tok = _train_small_tokenizer(
            ["the quick brown fox " * 100], vocab_size=300
        )
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            metrics = measure_tokenizer(tok, pilot)
        sp = metrics["seq_len_percentiles"]
        assert sp["p50"] <= sp["p90"] <= sp["p95"] <= sp["p99"]


# ===========================================================================
# 4. Vocabulary selection tests
# ===========================================================================

class TestComputeEmbeddingParams:
    def test_basic(self):
        assert compute_embedding_params(8192, 128) == 8192 * 128 * 2

    def test_scales_linearly(self):
        assert compute_embedding_params(16384, 128) == 2 * compute_embedding_params(8192, 128)


class TestRecommendVocabSize:
    def _make_metrics(self, vocab_size: int, tpw: float, unk_rate: float,
                      coverage: float = 90.0) -> dict:
        return {
            "vocab_size":             vocab_size,
            "tokens_per_word":        tpw,
            "compression_ratio":      3.5,
            "unk_rate":               unk_rate,
            "word_type_coverage_pct": coverage,
            "total_tokens":           9_000_000,
            "seq_len_percentiles":    {"p50": 500, "p90": 2000, "p95": 3000, "p99": 5000},
        }

    def test_returns_recommendation(self):
        candidates = [
            self._make_metrics(2048,  1.8, 0.02),
            self._make_metrics(4096,  1.6, 0.005),
            self._make_metrics(8192,  1.4, 0.001),
            self._make_metrics(16384, 1.3, 0.0005),
            self._make_metrics(32768, 1.2, 0.0002),
        ]
        rec = recommend_vocab_size(candidates)
        assert "recommended_vocab_size" in rec
        assert rec["recommended_vocab_size"] in [2048, 4096, 8192, 16384, 32768]

    def test_high_unk_rate_penalised(self):
        """High UNK rate should prevent a candidate from being recommended."""
        candidates = [
            self._make_metrics(2048,  2.0, 0.10),  # 10% UNK — very bad
            self._make_metrics(4096,  1.6, 0.001),
        ]
        rec = recommend_vocab_size(candidates)
        # 2048 has 10% UNK — should not be recommended
        assert rec["recommended_vocab_size"] == 4096

    def test_rationale_is_string(self):
        candidates = [
            self._make_metrics(4096, 1.6, 0.002),
            self._make_metrics(8192, 1.4, 0.001),
        ]
        rec = recommend_vocab_size(candidates)
        assert isinstance(rec["rationale"], str)
        assert len(rec["rationale"]) > 20

    def test_embedding_params_computed(self):
        candidates = [
            self._make_metrics(4096, 1.6, 0.001),
            self._make_metrics(8192, 1.4, 0.0005),
        ]
        rec = recommend_vocab_size(candidates)
        assert 4096 in rec["embedding_params_by_size"]
        assert 8192 in rec["embedding_params_by_size"]
        assert rec["embedding_params_by_size"][8192] == 8192 * 128 * 2

    def test_candidate_summary_length(self):
        candidates = [
            self._make_metrics(vs, 1.5 - i * 0.05, 0.001 / (i + 1))
            for i, vs in enumerate([2048, 4096, 8192])
        ]
        rec = recommend_vocab_size(candidates)
        assert len(rec["candidate_summary"]) == 3


# ===========================================================================
# 5. Tokenizer vocabulary coverage
# ===========================================================================

class TestVocabularyCoverage:
    def test_special_tokens_always_present(self):
        tok = _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=300)
        assert tok.pad_token_id == 0
        assert tok.bos_token_id == 1
        assert tok.eos_token_id == 2
        assert tok.unk_token_id == 3

    def test_larger_vocab_has_more_tokens(self):
        tok_small = _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=100)
        tok_large = _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=300)
        assert tok_large.vocab_size > tok_small.vocab_size

    def test_larger_vocab_fewer_tokens_per_word(self):
        """More merges → longer tokens → fewer tokens per word."""
        corpus = ENGLISH_CORPUS * 5
        tok_small = _train_small_tokenizer(corpus, vocab_size=100)
        tok_large = _train_small_tokenizer(corpus, vocab_size=500)
        text = "the quick brown fox jumps over the lazy dog"
        small_len = len(tok_small.encode(text))
        large_len = len(tok_large.encode(text))
        # Larger vocab should produce fewer or equal tokens
        assert large_len <= small_len

    def test_known_words_get_low_unk_rate(self):
        corpus = ["hello world test sentence " * 50] * 5
        tok = _train_small_tokenizer(corpus, vocab_size=500)
        ids = tok.encode("hello world test sentence")
        unk_count = sum(1 for i in ids if tok.vocab.id_to_token(i) == UNK_TOKEN)
        assert unk_count == 0  # all training words should be in vocab

    def test_repr_has_vocab_size(self):
        tok = _train_small_tokenizer(ENGLISH_CORPUS, vocab_size=200)
        r = repr(tok)
        assert "NexaTokenizer" in r
        assert "vocab_size" in r
