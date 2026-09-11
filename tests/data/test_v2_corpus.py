"""
tests/data/test_v2_corpus.py
==============================
Tests the deterministic behavior, token conservation, and V2 sharding pipeline.
"""

import json
from pathlib import Path
from scripts.data.build_v2_corpus import normalize_text
from scripts.data.tokenize_v2_corpus import generate_shards
from nexa.tokenizer.byte_tokenizer import NexaByteTokenizer
import struct

def test_normalized_text():
    t1 = "Hello, World! "
    t2 = "helloworld"
    t3 = "H-e-l-l-o, world."
    assert normalize_text(t1) == normalize_text(t2)
    assert normalize_text(t1) == normalize_text(t3)

def test_v2_sharding_and_token_conservation(tmp_path):
    docs = [
        {"source": "test", "text": "This is doc 1."},
        {"source": "test", "text": "Another document here."},
        {"source": "test", "text": "Short."}
    ]
    corpus = [d["text"] for d in docs]
    tokenizer = NexaByteTokenizer.train(corpus, vocab_size=270)
    
    docs = [
        {"source": "test", "text": "This is doc 1."},
        {"source": "test", "text": "Another document here."},
        {"source": "test", "text": "Short."}
    ]
    
    # Overwrite OUT_DIR for testing
    import scripts.data.tokenize_v2_corpus as tok_module
    tok_module.OUT_DIR = tmp_path
    tok_module.SHARD_SIZE_TOKENS = 10 # very small shard size to test spillover
    
    stats = tok_module.generate_shards("test_split", docs, tokenizer)
    
    assert stats["docs"] == 3
    assert stats["total_tokens"] > 0
    assert stats["total_unks"] == 0 # Byte BPE has 0 unks for valid UTF-8
    
    # Verify shard files
    split_dir = tmp_path / "test_split"
    shards = list(split_dir.glob("*.bin"))
    assert len(shards) > 0
    
    # Read back tokens to verify conservation
    read_tokens = 0
    for shard in shards:
        with open(shard, "rb") as f:
            while True:
                chunk = f.read(2)
                if not chunk: break
                read_tokens += 1
                
    assert read_tokens == stats["total_tokens"]

def test_v2_corpus_no_invalid_tokens():
    # Byte BPE tokens should always be < vocab_size
    text = "Some random text with 🚀 and \n and \t"
    tokenizer = NexaByteTokenizer.train([text], vocab_size=300)
    encoded = tokenizer.encode(text)
    
    for t in encoded:
        assert 0 <= t < 300
        assert t != 3 # UNK_ID is 3, shouldn't appear
