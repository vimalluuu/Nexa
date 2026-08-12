"""
nexa/tokenizer/__init__.py
===========================
Public API for Nexa's BPE tokenizer sub-package.

The primary entry point is NexaTokenizer:

    from nexa.tokenizer import NexaTokenizer

    # Train from scratch on your own corpus:
    tokenizer = NexaTokenizer.train(corpus, vocab_size=1000)

    # Encode and decode:
    ids  = tokenizer.encode("hello world", add_bos=True, add_eos=True)
    text = tokenizer.decode(ids)

    # Save and reload:
    tokenizer.save("data/processed/tokenizer/")
    tokenizer = NexaTokenizer.load("data/processed/tokenizer/")

Independence guarantee
----------------------
NexaTokenizer has no from_pretrained() method.
No pretrained vocabulary or merge rules are ever downloaded or imported.
Every tokenizer instance is trained from scratch using train().
"""

from nexa.tokenizer.vocab import (
    Vocabulary,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
    DEFAULT_SPECIAL_TOKENS, SPECIAL_TOKEN_SET,
)
from nexa.tokenizer.bpe import (
    Pair,
    word_to_tokens,
    get_word_freqs,
    count_pairs,
    apply_merge,
    get_base_tokens,
    train_bpe,
)
from nexa.tokenizer.tokenizer import NexaTokenizer

__all__ = [
    # Main API
    "NexaTokenizer",
    # Vocabulary
    "Vocabulary",
    # Token string constants
    "PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "UNK_TOKEN",
    # Token ID constants
    "PAD_ID", "BOS_ID", "EOS_ID", "UNK_ID",
    "DEFAULT_SPECIAL_TOKENS", "SPECIAL_TOKEN_SET",
    # BPE primitives (exposed for testing and advanced use)
    "Pair",
    "word_to_tokens", "get_word_freqs", "count_pairs",
    "apply_merge", "get_base_tokens", "train_bpe",
]
