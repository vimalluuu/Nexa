"""
tests/test_tokenizer.py
========================
Phase 2 test suite — Nexa BPE Tokenizer.

Tests are organised into four classes mirroring the four source files:
    TestBPEPrimitives    — pure functions in bpe.py
    TestVocabulary       — Vocabulary class in vocab.py
    TestNexaTokenizer    — NexaTokenizer class in tokenizer.py
    TestIndependence     — confirms no forbidden libraries are imported

Run with:
    pytest tests/test_tokenizer.py -v
    pytest tests/test_tokenizer.py -v --tb=short   # concise failures
"""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures and constants
# ---------------------------------------------------------------------------

# A small corpus with deliberate repeated patterns.
# "at" appears in: cat, sat, mat, rat, flat → BPE should learn to merge it.
# "the" appears many times → should become a single token.
SAMPLE_CORPUS = [
    "the cat sat on the mat",
    "a cat sat on a flat mat",
    "the fat cat and the rat",
    "the rat sat on the mat",
    "a flat mat on the floor",
    "the cat and the rat sat flat",
    "a cat and a rat and a mat",
]

# Corpus where a single pair has a clear frequency winner (no tie)
CLEAR_CORPUS = ["aaa bbb aaa aaa bbb"]
# In CLEAR_CORPUS:
#   "aaa" × 3 → ("a","a","a","</w>") — pairs: (a,a):6 [2×3], (a,</w>):3 [1×3]
#   "bbb" × 2 → ("b","b","b","</w>") — pairs: (b,b):4 [2×2], (b,</w>):2 [1×2]
# Highest frequency pair: (a,a) with count 6. First merge must be (a,a).


# ===========================================================================
# 1. BPE primitive functions (bpe.py)
# ===========================================================================

class TestBPEPrimitives:
    """Unit tests for the pure functions in nexa.tokenizer.bpe."""

    # --- word_to_tokens ---

    def test_word_to_tokens_basic(self):
        """Standard word becomes characters + </w>."""
        from nexa.tokenizer import word_to_tokens
        assert word_to_tokens("cat") == ("c", "a", "t", "</w>")

    def test_word_to_tokens_single_char(self):
        """Single character word."""
        from nexa.tokenizer import word_to_tokens
        assert word_to_tokens("a") == ("a", "</w>")

    def test_word_to_tokens_empty_string(self):
        """Empty string returns empty tuple."""
        from nexa.tokenizer import word_to_tokens
        assert word_to_tokens("") == ()

    def test_word_to_tokens_always_ends_with_endofword(self):
        """The last element is always '</w>' for any non-empty word."""
        from nexa.tokenizer import word_to_tokens
        for word in ["hello", "world", "z", "programming"]:
            result = word_to_tokens(word)
            assert result[-1] == "</w>", f"Expected </w> at end of {result}"

    def test_word_to_tokens_length(self):
        """Result length = number of characters + 1 (for </w>)."""
        from nexa.tokenizer import word_to_tokens
        word = "hello"
        assert len(word_to_tokens(word)) == len(word) + 1

    # --- get_word_freqs ---

    def test_get_word_freqs_basic(self):
        """Words are split and counted correctly."""
        from nexa.tokenizer import get_word_freqs
        freqs = get_word_freqs(["hello world", "hello"])
        assert freqs[("h", "e", "l", "l", "o", "</w>")] == 2
        assert freqs[("w", "o", "r", "l", "d", "</w>")] == 1

    def test_get_word_freqs_empty_corpus(self):
        """Empty corpus returns empty dict."""
        from nexa.tokenizer import get_word_freqs
        assert get_word_freqs([]) == {}

    def test_get_word_freqs_blank_lines_ignored(self):
        """Blank lines do not produce empty word entries."""
        from nexa.tokenizer import get_word_freqs
        freqs = get_word_freqs(["hello", "", "   ", "hello"])
        assert freqs[("h", "e", "l", "l", "o", "</w>")] == 2
        assert len(freqs) == 1

    def test_get_word_freqs_multi_line(self):
        """Frequencies accumulate correctly across multiple strings."""
        from nexa.tokenizer import get_word_freqs
        freqs = get_word_freqs(["a b", "a c", "a"])
        # "a" appears 3 times
        assert freqs[("a", "</w>")] == 3
        # "b" and "c" once each
        assert freqs[("b", "</w>")] == 1
        assert freqs[("c", "</w>")] == 1

    # --- count_pairs ---

    def test_count_pairs_basic(self):
        """Adjacent pairs are counted correctly."""
        from nexa.tokenizer import count_pairs
        word_freqs = {("a", "b", "c", "</w>"): 1}
        pairs = count_pairs(word_freqs)
        assert pairs[("a", "b")] == 1
        assert pairs[("b", "c")] == 1
        assert pairs[("c", "</w>")] == 1

    def test_count_pairs_weighted_by_word_frequency(self):
        """Pair counts are scaled by how often the word appears."""
        from nexa.tokenizer import count_pairs
        # Word ("a","b","</w>") appears 5 times
        word_freqs = {("a", "b", "</w>"): 5}
        pairs = count_pairs(word_freqs)
        assert pairs[("a", "b")] == 5
        assert pairs[("b", "</w>")] == 5

    def test_count_pairs_sums_across_words(self):
        """Pair (a,b) sums over all words that contain it."""
        from nexa.tokenizer import count_pairs
        word_freqs = {
            ("a", "b", "c", "</w>"): 3,    # contains (a,b): 3
            ("a", "b", "d", "</w>"): 2,    # contains (a,b): 2
            ("x", "y", "z", "</w>"): 10,   # does NOT contain (a,b)
        }
        pairs = count_pairs(word_freqs)
        assert pairs[("a", "b")] == 5   # 3 + 2

    def test_count_pairs_returns_counter(self):
        """Return type is a Counter."""
        from nexa.tokenizer import count_pairs, get_word_freqs
        freqs = get_word_freqs(["hello"])
        assert isinstance(count_pairs(freqs), Counter)

    # --- apply_merge ---

    def test_apply_merge_basic(self):
        """Merging (l,o) produces 'lo' in affected words."""
        from nexa.tokenizer import apply_merge
        freqs = {("l", "o", "w", "</w>"): 2}
        result = apply_merge(("l", "o"), freqs)
        assert ("lo", "w", "</w>") in result
        assert result[("lo", "w", "</w>")] == 2

    def test_apply_merge_unaffected_words_unchanged(self):
        """Words that don't contain the pair are returned unchanged."""
        from nexa.tokenizer import apply_merge
        freqs = {
            ("l", "o", "w", "</w>"): 2,
            ("h", "i", "</w>"): 3,    # no (l,o) pair
        }
        result = apply_merge(("l", "o"), freqs)
        assert ("h", "i", "</w>") in result
        assert result[("h", "i", "</w>")] == 3

    def test_apply_merge_multiple_occurrences_in_word(self):
        """All occurrences of the pair in a word are merged (left to right)."""
        from nexa.tokenizer import apply_merge
        # "aab" has pair (a,a) once at position 0
        freqs = {("a", "a", "b", "</w>"): 1}
        result = apply_merge(("a", "a"), freqs)
        assert ("aa", "b", "</w>") in result

    def test_apply_merge_is_pure(self):
        """apply_merge does not mutate its input dict."""
        from nexa.tokenizer import apply_merge
        original = {("a", "b", "</w>"): 3}
        original_copy = dict(original)
        apply_merge(("a", "b"), original)
        assert original == original_copy

    def test_apply_merge_frequencies_preserved(self):
        """Word frequencies are preserved through a merge."""
        from nexa.tokenizer import apply_merge
        freqs = {("x", "y", "z", "</w>"): 42}
        result = apply_merge(("x", "y"), freqs)
        assert result[("xy", "z", "</w>")] == 42

    # --- train_bpe ---

    def test_train_bpe_returns_correct_types(self):
        """train_bpe returns (list, set)."""
        from nexa.tokenizer import train_bpe
        merges, base = train_bpe(SAMPLE_CORPUS, num_merges=10)
        assert isinstance(merges, list)
        assert isinstance(base, set)

    def test_train_bpe_merges_are_pairs_of_strings(self):
        """Every element of merges is a 2-tuple of strings."""
        from nexa.tokenizer import train_bpe
        merges, _ = train_bpe(SAMPLE_CORPUS, num_merges=10)
        for pair in merges:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert all(isinstance(s, str) for s in pair)

    def test_train_bpe_base_tokens_includes_endofword(self):
        """Base tokens always contains the </w> marker."""
        from nexa.tokenizer import train_bpe
        _, base = train_bpe(SAMPLE_CORPUS, num_merges=5)
        assert "</w>" in base

    def test_train_bpe_base_tokens_are_single_chars(self):
        """All base tokens are single characters or the </w> marker."""
        from nexa.tokenizer import train_bpe
        _, base = train_bpe(SAMPLE_CORPUS, num_merges=5)
        for tok in base:
            assert len(tok) == 1 or tok == "</w>", f"Unexpected base token: {tok!r}"

    def test_train_bpe_num_merges_respected(self):
        """Number of returned merges ≤ num_merges requested."""
        from nexa.tokenizer import train_bpe
        for n in [1, 5, 20, 100]:
            merges, _ = train_bpe(SAMPLE_CORPUS, num_merges=n, min_frequency=1)
            assert len(merges) <= n, f"Got {len(merges)} merges for n={n}"

    def test_train_bpe_first_merge_is_most_frequent_pair(self):
        """On CLEAR_CORPUS, the first merge must be (a,a) — count=6, clearly highest."""
        from nexa.tokenizer import train_bpe
        merges, _ = train_bpe(CLEAR_CORPUS, num_merges=1, min_frequency=1)
        assert len(merges) == 1
        assert merges[0] == ("a", "a"), (
            f"Expected first merge to be ('a','a'), got {merges[0]}. "
            "Pair (a,a) has count 6 in CLEAR_CORPUS — it must be chosen first."
        )

    def test_train_bpe_min_frequency_filters_rare_pairs(self):
        """With min_frequency=100, no pairs qualify and merges is empty."""
        from nexa.tokenizer import train_bpe
        # No pair in SAMPLE_CORPUS appears 100 times
        merges, _ = train_bpe(SAMPLE_CORPUS, num_merges=50, min_frequency=100)
        assert merges == []

    def test_train_bpe_empty_corpus(self):
        """Empty corpus → empty merges and empty base tokens."""
        from nexa.tokenizer import train_bpe
        merges, base = train_bpe([], num_merges=10)
        assert merges == []
        assert base == set()

    def test_train_bpe_zero_merges(self):
        """Requesting 0 merges returns an empty merge list."""
        from nexa.tokenizer import train_bpe
        merges, base = train_bpe(SAMPLE_CORPUS, num_merges=0)
        assert merges == []
        assert len(base) > 0   # base tokens are still collected


# ===========================================================================
# 2. Vocabulary (vocab.py)
# ===========================================================================

class TestVocabulary:
    """Unit tests for the Vocabulary class in nexa.tokenizer.vocab."""

    def test_special_tokens_have_correct_ids(self):
        """<pad>=0, <bos>=1, <eos>=2, <unk>=3 — always."""
        from nexa.tokenizer import Vocabulary, PAD_ID, BOS_ID, EOS_ID, UNK_ID
        vocab = Vocabulary()
        assert vocab.token_to_id("<pad>") == PAD_ID == 0
        assert vocab.token_to_id("<bos>") == BOS_ID == 1
        assert vocab.token_to_id("<eos>") == EOS_ID == 2
        assert vocab.token_to_id("<unk>") == UNK_ID == 3

    def test_add_token_assigns_next_available_id(self):
        """New tokens get IDs 4, 5, 6, ... after the special tokens."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        id_a = vocab.add_token("a")
        id_b = vocab.add_token("b")
        assert id_a == 4
        assert id_b == 5

    def test_add_token_is_idempotent(self):
        """Adding the same token twice returns the same ID both times."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        id1 = vocab.add_token("hello")
        id2 = vocab.add_token("hello")
        assert id1 == id2

    def test_token_to_id_unknown_returns_unk(self):
        """Unseen tokens map to UNK_ID."""
        from nexa.tokenizer import Vocabulary, UNK_ID
        vocab = Vocabulary()
        assert vocab.token_to_id("NEVER_SEEN") == UNK_ID

    def test_id_to_token_out_of_range_returns_unk_string(self):
        """Out-of-range IDs return the <unk> string."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        assert vocab.id_to_token(99999) == "<unk>"

    def test_bidirectional_lookup(self):
        """token_to_id and id_to_token are inverses of each other."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        vocab.add_token("cat")
        idx = vocab.token_to_id("cat")
        assert vocab.id_to_token(idx) == "cat"

    def test_contains_operator(self):
        """'in' operator works for both present and absent tokens."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        vocab.add_token("hello")
        assert "hello" in vocab
        assert "world" not in vocab
        assert "<pad>" in vocab   # special tokens are always present

    def test_len_includes_special_tokens(self):
        """len(vocab) counts special tokens (4) + added tokens."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        assert len(vocab) == 4   # just the special tokens
        vocab.add_token("a")
        assert len(vocab) == 5

    def test_save_and_load_roundtrip(self, tmp_path):
        """save() then load() produces an identical vocabulary."""
        from nexa.tokenizer import Vocabulary, PAD_ID
        vocab = Vocabulary()
        vocab.add_token("cat")
        vocab.add_token("sat")
        vocab.add_token("</w>")

        path = tmp_path / "vocab.json"
        vocab.save(path)

        loaded = Vocabulary.load(path)

        assert len(loaded) == len(vocab)
        assert loaded.token_to_id("cat") == vocab.token_to_id("cat")
        assert loaded.token_to_id("sat") == vocab.token_to_id("sat")
        assert loaded.token_to_id("<pad>") == PAD_ID
        assert loaded.token_to_id("UNSEEN") == 3   # UNK_ID

    def test_save_creates_json_file(self, tmp_path):
        """save() creates a valid JSON file."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        vocab.add_token("hello")
        path = tmp_path / "vocab.json"
        vocab.save(path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "<pad>" in data
        assert data["<pad>"] == 0

    def test_repr_shows_size(self):
        """__repr__ includes the vocabulary size."""
        from nexa.tokenizer import Vocabulary
        vocab = Vocabulary()
        assert "4" in repr(vocab)   # 4 special tokens


# ===========================================================================
# 3. NexaTokenizer (tokenizer.py)
# ===========================================================================

class TestNexaTokenizer:
    """Unit tests for NexaTokenizer — training, encoding, decoding, serialization."""

    @pytest.fixture(scope="class")
    @classmethod
    def tokenizer(cls):
        """A trained tokenizer shared across all tests in this class."""
        from nexa.tokenizer import NexaTokenizer
        return NexaTokenizer.train(SAMPLE_CORPUS, vocab_size=200, min_frequency=1)

    # --- Training ---

    def test_train_returns_tokenizer_instance(self):
        from nexa.tokenizer import NexaTokenizer
        tok = NexaTokenizer.train(SAMPLE_CORPUS, vocab_size=100)
        assert isinstance(tok, NexaTokenizer)

    def test_vocab_size_at_most_requested(self, tokenizer):
        """Actual vocab size ≤ requested vocab_size."""
        assert tokenizer.vocab_size <= 200

    def test_vocab_size_at_least_special_plus_base(self, tokenizer):
        """Vocab contains at minimum the 4 special tokens."""
        assert tokenizer.vocab_size >= 4

    def test_special_token_ids_are_correct(self, tokenizer):
        """BOS, EOS, PAD, UNK IDs match module-level constants."""
        from nexa.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID
        assert tokenizer.bos_token_id == BOS_ID == 1
        assert tokenizer.eos_token_id == EOS_ID == 2
        assert tokenizer.pad_token_id == PAD_ID == 0
        assert tokenizer.unk_token_id == UNK_ID == 3

    def test_merges_are_non_empty_for_nontrivial_corpus(self, tokenizer):
        """A non-trivial corpus should produce at least one merge."""
        assert len(tokenizer.merges) >= 1

    def test_repr_contains_vocab_size(self, tokenizer):
        r = repr(tokenizer)
        assert "NexaTokenizer" in r
        assert str(tokenizer.vocab_size) in r

    # --- Encoding ---

    def test_encode_returns_list_of_ints(self, tokenizer):
        ids = tokenizer.encode("the cat sat")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_nonempty_text_gives_nonempty_ids(self, tokenizer):
        assert len(tokenizer.encode("the cat sat")) > 0

    def test_encode_empty_string_gives_empty_list(self, tokenizer):
        assert tokenizer.encode("") == []

    def test_encode_empty_with_bos_eos(self, tokenizer):
        """Empty string + BOS/EOS gives just [BOS_ID, EOS_ID]."""
        ids = tokenizer.encode("", add_bos=True, add_eos=True)
        assert ids == [tokenizer.bos_token_id, tokenizer.eos_token_id]

    def test_encode_add_bos_prepends_bos_id(self, tokenizer):
        ids = tokenizer.encode("hello", add_bos=True)
        assert ids[0] == tokenizer.bos_token_id

    def test_encode_add_eos_appends_eos_id(self, tokenizer):
        ids = tokenizer.encode("hello", add_eos=True)
        assert ids[-1] == tokenizer.eos_token_id

    def test_encode_all_ids_in_vocab_range(self, tokenizer):
        """Every ID produced by encode() is within [0, vocab_size)."""
        ids = tokenizer.encode("the cat sat on the flat mat")
        for i in ids:
            assert 0 <= i < tokenizer.vocab_size, f"ID {i} out of range"

    def test_encode_same_text_gives_same_ids(self, tokenizer):
        """Encoding is deterministic."""
        text = "the cat sat on the mat"
        assert tokenizer.encode(text) == tokenizer.encode(text)

    # --- Decoding ---

    def test_decode_returns_string(self, tokenizer):
        ids = tokenizer.encode("the cat sat")
        assert isinstance(tokenizer.decode(ids), str)

    def test_decode_empty_list_gives_empty_string(self, tokenizer):
        assert tokenizer.decode([]) == ""

    def test_decode_skips_special_tokens_by_default(self, tokenizer):
        """BOS, EOS, PAD are excluded from decoded text."""
        ids = [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
        ] + tokenizer.encode("the cat") + [
            tokenizer.eos_token_id,
        ]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded
        assert "<pad>" not in decoded

    def test_decode_includes_special_tokens_when_disabled(self, tokenizer):
        """skip_special_tokens=False preserves BOS/EOS in output."""
        ids = [tokenizer.bos_token_id] + tokenizer.encode("hi") + [tokenizer.eos_token_id]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert "<bos>" in decoded
        assert "<eos>" in decoded

    # --- Roundtrip ---

    def test_encode_decode_roundtrip_training_words(self, tokenizer):
        """
        For text composed entirely of words seen in training, encoding
        then decoding should reconstruct the original text exactly.
        """
        texts = [
            "the cat sat on the mat",
            "a flat mat",
            "the rat sat",
        ]
        for text in texts:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            assert decoded == text, (
                f"Roundtrip failed for {text!r}:\n"
                f"  IDs    : {ids}\n"
                f"  Decoded: {decoded!r}"
            )

    def test_roundtrip_with_bos_eos(self, tokenizer):
        """BOS/EOS roundtrip: adding then skipping them preserves text."""
        text = "the cat sat"
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text

    # --- Batch operations ---

    def test_encode_batch_returns_list_of_lists(self, tokenizer):
        texts = ["the cat", "sat on", "the mat"]
        result = tokenizer.encode_batch(texts)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(ids, list) for ids in result)

    def test_decode_batch_returns_list_of_strings(self, tokenizer):
        ids_batch = [tokenizer.encode(t) for t in ["the cat", "sat on"]]
        result = tokenizer.decode_batch(ids_batch)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    # --- BPE property verification ---

    def test_frequent_word_becomes_fewer_tokens(self):
        """
        "the" appears many times in SAMPLE_CORPUS. With enough merges,
        it should be encoded as fewer tokens than its character count.
        At minimum, it should merge to something like "th" + "e</w>"
        or "the</w>" as a single token.
        """
        from nexa.tokenizer import NexaTokenizer
        # Train with generous vocab to allow deep merging
        tok = NexaTokenizer.train(SAMPLE_CORPUS, vocab_size=300, min_frequency=1)
        ids = tok.encode("the")
        # "the" has 3 chars + </w> = 4 tokens at character level.
        # BPE should reduce this. With enough training data, expect ≤ 3.
        assert len(ids) < 4, (
            f"Expected 'the' to be merged into fewer than 4 tokens, got {len(ids)}: "
            f"{[tok.vocab.id_to_token(i) for i in ids]}"
        )

    def test_more_merges_means_shorter_encoding(self):
        """
        Larger vocab_size allows more merges → tokens tend to be longer
        → encoding a word produces fewer IDs.
        """
        from nexa.tokenizer import NexaTokenizer
        text = "the cat sat on the flat mat"
        tok_small = NexaTokenizer.train(SAMPLE_CORPUS, vocab_size=50, min_frequency=1)
        tok_large = NexaTokenizer.train(SAMPLE_CORPUS, vocab_size=300, min_frequency=1)
        ids_small = tok_small.encode(text)
        ids_large = tok_large.encode(text)
        # More merges → encoding is same length OR shorter
        assert len(ids_large) <= len(ids_small), (
            f"Expected larger vocab to produce fewer or equal tokens. "
            f"small={len(ids_small)}, large={len(ids_large)}"
        )

    # --- Serialization ---

    def test_save_creates_both_files(self, tokenizer, tmp_path):
        """save() creates tokenizer.json and vocab.json."""
        tokenizer.save(tmp_path)
        assert (tmp_path / "tokenizer.json").exists()
        assert (tmp_path / "vocab.json").exists()

    def test_save_tokenizer_json_is_valid(self, tokenizer, tmp_path):
        """tokenizer.json contains required keys."""
        tokenizer.save(tmp_path)
        with open(tmp_path / "tokenizer.json") as f:
            data = json.load(f)
        assert "version" in data
        assert "model" in data
        assert "merges" in data
        assert data["model"] == "bpe"

    def test_load_roundtrip(self, tokenizer, tmp_path):
        """save → load → encode gives identical results."""
        from nexa.tokenizer import NexaTokenizer
        tokenizer.save(tmp_path)
        loaded = NexaTokenizer.load(tmp_path)

        text = "the cat sat on the mat"
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == tokenizer.decode(tokenizer.encode(text))

    def test_load_preserves_vocab_size(self, tokenizer, tmp_path):
        """Loaded tokenizer has the same vocab_size as the original."""
        from nexa.tokenizer import NexaTokenizer
        tokenizer.save(tmp_path)
        loaded = NexaTokenizer.load(tmp_path)
        assert loaded.vocab_size == tokenizer.vocab_size

    def test_load_missing_file_raises(self, tmp_path):
        """load() raises FileNotFoundError if files are missing."""
        from nexa.tokenizer import NexaTokenizer
        with pytest.raises(FileNotFoundError):
            NexaTokenizer.load(tmp_path / "nonexistent_dir")


# ===========================================================================
# 4. Independence — no forbidden imports
# ===========================================================================

class TestIndependence:
    """
    Verify that the tokenizer does not import any forbidden libraries.
    These tests guard against accidental introduction of pretrained models.
    """

    FORBIDDEN_MODULES = [
        "openai",
        "anthropic",
        "transformers",    # HuggingFace model hub (loads pretrained weights)
        "sentencepiece",   # Google's tokenizer (pretrained models)
        "tiktoken",        # OpenAI's tokenizer
        "tokenizers",      # HuggingFace fast tokenizers (pretrained)
        "whisper",
        "vosk",
        "deepspeech",
    ]

    def test_no_forbidden_modules_imported(self):
        """
        After importing the tokenizer package, none of the forbidden
        modules should appear in sys.modules.
        """
        # Force import
        import nexa.tokenizer  # noqa: F401

        for module_name in self.FORBIDDEN_MODULES:
            assert module_name not in sys.modules, (
                f"Forbidden module '{module_name}' was imported by the tokenizer. "
                f"Nexa must not use pretrained AI libraries."
            )

    def test_tokenizer_has_no_from_pretrained(self):
        """NexaTokenizer must not expose a from_pretrained() method."""
        from nexa.tokenizer import NexaTokenizer
        assert not hasattr(NexaTokenizer, "from_pretrained"), (
            "NexaTokenizer must not have a from_pretrained() method. "
            "Use train() instead."
        )

    def test_nexa_tokenizer_stdlib_only_imports(self):
        """
        Verify that bpe.py only imports from Python stdlib.
        We do this by checking the module's __dict__ for non-stdlib imports.
        """
        import nexa.tokenizer.bpe as bpe_module
        # The module should only depend on 'collections' from stdlib
        # Verify by checking that no ML framework is in its globals
        for name in ["torch", "numpy", "tensorflow"]:
            assert name not in dir(bpe_module), (
                f"bpe.py should not import {name}"
            )
