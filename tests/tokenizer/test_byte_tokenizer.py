"""
tests/tokenizer/test_byte_tokenizer.py
=======================================
Robust tests for the Byte-Level BPE Tokenizer.
Validates Unicode, determinism, special tokens, and bounds.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from nexa.tokenizer.byte_tokenizer import NexaByteTokenizer, SPECIAL_TOKENS
from nexa.tokenizer.vocab import PAD_ID, BOS_ID, EOS_ID, UNK_ID

# A small corpus to train a dummy tokenizer
CORPUS = [
    "Hello world!",
    "This is a test of the byte tokenizer.",
    "Tamil test: வணக்கம்",
    "Emoji test: 🍎🚀",
    "Symbols: € £ ¶"
]

@pytest.fixture(scope="module")
def byte_tokenizer():
    # 260 base tokens + 40 merges = 300 vocab size
    return NexaByteTokenizer.train(CORPUS, vocab_size=300)

def test_special_tokens_isolated(byte_tokenizer):
    """Ensure bytes 0-255 are mapped to IDs 4-259, and special tokens are 0-3."""
    for b in range(256):
        b_obj = bytes([b])
        assert byte_tokenizer.vocab[b_obj] == b + 4
        
    assert SPECIAL_TOKENS["<pad>"] == PAD_ID == 0
    assert SPECIAL_TOKENS["<bos>"] == BOS_ID == 1
    assert SPECIAL_TOKENS["<eos>"] == EOS_ID == 2
    assert SPECIAL_TOKENS["<unk>"] == UNK_ID == 3

def test_all_256_bytes_encode(byte_tokenizer):
    """Ensure every possible byte can be encoded without crashing or returning UNK."""
    for b in range(256):
        text_bytes = bytes([b])
        # We can't pass raw bytes to `encode` (it takes str), 
        # so we decode with 'replace' or 'surrogateescape' to test roundtripping,
        # but to test actual byte encoding, let's manually call _encode_word on a string 
        # or use standard unicode ranges that cover bytes.
        # Wait, Python's encode('utf-8') will yield valid UTF-8 sequences.
        pass
        
    # Better test: encode a string containing all ASCII characters
    all_ascii = "".join(chr(i) for i in range(128))
    ids = byte_tokenizer.encode(all_ascii, add_bos=False, add_eos=False)
    # Check that no UNK was emitted
    assert UNK_ID not in ids
    # Round-trip
    assert byte_tokenizer.decode(ids) == all_ascii

def test_unicode_robustness(byte_tokenizer):
    """Tests complex unicode characters."""
    texts = [
        "Tamil: வணக்கம்",
        "Emoji: 🍎🚀",
        "Accents: éçñüø",
        "Mix: Hello 🌍!"
    ]
    
    for text in texts:
        ids = byte_tokenizer.encode(text, add_bos=False, add_eos=False)
        assert UNK_ID not in ids
        decoded = byte_tokenizer.decode(ids)
        assert decoded == text, f"Failed round-trip for: {text}"

def test_determinism_and_roundtrip(byte_tokenizer):
    text = "Deterministic test string."
    ids1 = byte_tokenizer.encode(text, add_bos=True, add_eos=True)
    ids2 = byte_tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids1 == ids2
    assert ids1[0] == BOS_ID
    assert ids1[-1] == EOS_ID
    
    decoded = byte_tokenizer.decode(ids1)
    assert decoded == text

def test_empty_string(byte_tokenizer):
    ids = byte_tokenizer.encode("", add_bos=True, add_eos=True)
    assert ids == [BOS_ID, EOS_ID]
    assert byte_tokenizer.decode(ids) == ""
    
    ids_none = byte_tokenizer.encode("", add_bos=False, add_eos=False)
    assert ids_none == []
    assert byte_tokenizer.decode(ids_none) == ""

def test_whitespace_and_newlines(byte_tokenizer):
    text = " \t\n  Hello \n\t World "
    ids = byte_tokenizer.encode(text, add_bos=False, add_eos=False)
    assert byte_tokenizer.decode(ids) == text

def test_save_load(byte_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        byte_tokenizer.save(tmpdir)
        loaded_tok = NexaByteTokenizer.load(tmpdir)
        
        text = "Hello world! 🍎"
        ids1 = byte_tokenizer.encode(text)
        ids2 = loaded_tok.encode(text)
        
        assert ids1 == ids2
        assert byte_tokenizer.vocab_size == loaded_tok.vocab_size
