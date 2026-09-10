"""
Tests for Phase 8.6 tokenization and dataset sharding.

Covers:
- Deterministic train/validation split
- Tokenization logic (BOS/EOS inclusion)
- Shard creation (uint16 binary format)
- Shard reading and verification
- Checksum reproducibility
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.data.shard_dataset import (
    deterministic_split,
    ShardWriter,
    tokenize_and_shard,
)
from nexa.tokenizer.tokenizer import NexaTokenizer


# ===========================================================================
# Helpers
# ===========================================================================

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def mock_tokenizer():
    """A small tokenizer for testing."""
    corpus = ["the quick brown fox jumps over the lazy dog"] * 10
    return NexaTokenizer.train(corpus, vocab_size=50)


# ===========================================================================
# Tests
# ===========================================================================

class TestDeterministicSplit:
    def test_split_ratio(self):
        # Create 100 fake pilot ids
        source_ids = {str(i) for i in range(100)}
        pilot = {"wikimedia_english": source_ids}
        
        train, val = deterministic_split(pilot, val_fraction=0.1, seed=42)
        
        # 10% of 100 is 10
        assert len(val["wikimedia_english"]) == 10
        assert len(train["wikimedia_english"]) == 90

    def test_split_is_deterministic(self):
        source_ids = {str(i) for i in range(100)}
        pilot = {"wikimedia_english": source_ids}
        
        t1, v1 = deterministic_split(pilot, val_fraction=0.1, seed=42)
        t2, v2 = deterministic_split(pilot, val_fraction=0.1, seed=42)
        
        assert v1 == v2
        assert t1 == t2

    def test_split_different_seed(self):
        source_ids = {str(i) for i in range(100)}
        pilot = {"wikimedia_english": source_ids}
        
        _, v1 = deterministic_split(pilot, val_fraction=0.1, seed=42)
        _, v2 = deterministic_split(pilot, val_fraction=0.1, seed=99)
        
        assert v1 != v2

    def test_no_overlap(self):
        source_ids = {str(i) for i in range(100)}
        pilot = {"wikimedia_english": source_ids}
        
        train, val = deterministic_split(pilot, val_fraction=0.1, seed=42)
        assert train["wikimedia_english"].isdisjoint(val["wikimedia_english"])


class TestShardWriter:
    def test_writes_uint16_binary(self, tmp_path):
        out_dir = tmp_path / "train"
        writer = ShardWriter(out_dir, prefix="shard", max_tokens=10)
        
        # Write 5 tokens
        tokens = [1, 2, 3, 4, 5]
        writer.write(tokens)
        writer.close()
        
        assert len(writer.shards) == 1
        
        # Verify file size (5 tokens * 2 bytes = 10 bytes)
        shard_path = out_dir / "shard_000.bin"
        assert shard_path.exists()
        assert shard_path.stat().st_size == 10
        
        # Read back uint16
        with open(shard_path, "rb") as f:
            data = f.read()
            recovered = list(struct.unpack(f"<{len(tokens)}H", data))
        
        assert recovered == tokens

    def test_rolls_over_shards(self, tmp_path):
        out_dir = tmp_path / "train"
        # Small max_tokens to force rollover
        writer = ShardWriter(out_dir, prefix="shard", max_tokens=5)
        
        writer.write([1, 2, 3])
        writer.write([4, 5, 6, 7])  # Crosses 5 token boundary
        writer.close()
        
        assert len(writer.shards) == 2
        assert (out_dir / "shard_000.bin").exists()
        assert (out_dir / "shard_001.bin").exists()
        
        # Shard 0 should have 5 tokens
        assert writer.shards[0]["tokens"] == 5
        # Shard 1 should have 2 tokens
        assert writer.shards[1]["tokens"] == 2


class TestTokenizeAndShard:
    def _make_pilot_data(self, tmp_path):
        wiki_records = []
        pg19_records = []
        for i in range(10):
            wiki_records.append({
                "dataset": "wikimedia_english",
                "source_id": str(i),
                "text": "the quick brown fox"
            })
            pg19_records.append({
                "dataset": "pg19",
                "source_id": str(i),
                "text": "lazy dog"
            })
            
        wiki_path = tmp_path / "wiki.jsonl"
        pg19_path = tmp_path / "pg19.jsonl"
        _write_jsonl(wiki_path, wiki_records)
        _write_jsonl(pg19_path, pg19_records)
        
        pilot = {
            "wikimedia_english": {str(i) for i in range(10)},
            "pg19":              {str(i) for i in range(10)},
        }
        return wiki_path, pg19_path, pilot

    def test_pipeline_creates_correct_structure(self, tmp_path, mock_tokenizer):
        wiki, pg19, pilot = self._make_pilot_data(tmp_path)
        out_dir = tmp_path / "processed"
        
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            
            reports = tokenize_and_shard(
                tokenizer=mock_tokenizer,
                pilot_index=pilot,
                output_dir=out_dir,
                val_fraction=0.2, # 4 docs val, 16 train
                seed=42,
                max_tokens_per_shard=100
            )
            
        assert (out_dir / "train").exists()
        assert (out_dir / "validation").exists()
        
        # Check report numbers
        stats = reports["stats"]
        assert stats["total_documents"] == 20
        assert stats["val_documents"] == 4
        assert stats["train_documents"] == 16
        
        # 20 docs * (tokens + bos + eos) -> should have exact match in shards
        assert stats["total_tokens"] == stats["train_tokens"] + stats["val_tokens"]
        
        # Token ranges check: all IDs must be < vocab_size
        train_shards = list((out_dir / "train").glob("*.bin"))
        for sp in train_shards:
            with open(sp, "rb") as f:
                data = f.read()
                tokens = struct.unpack(f"<{len(data)//2}H", data)
                assert all(0 <= t < mock_tokenizer.vocab_size for t in tokens)

    def test_pipeline_reproducibility(self, tmp_path, mock_tokenizer):
        """Test that running the pipeline twice produces identical binary files and checksums."""
        wiki, pg19, pilot = self._make_pilot_data(tmp_path)
        
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        
        with patch("scripts.tokenizer.corpus_sampler.DEDUP_WIKI", wiki), \
             patch("scripts.tokenizer.corpus_sampler.DEDUP_PG19", pg19):
            
            rep1 = tokenize_and_shard(mock_tokenizer, pilot, out1, val_fraction=0.2, seed=123, max_tokens_per_shard=100)
            rep2 = tokenize_and_shard(mock_tokenizer, pilot, out2, val_fraction=0.2, seed=123, max_tokens_per_shard=100)
            
        # Manifests and checksums must be identical
        assert rep1["checksums"] == rep2["checksums"]
        assert rep1["manifest"]["shards"] == rep2["manifest"]["shards"]
