"""
nexa/tokenizer/tokenizer.py
============================
NexaTokenizer — the user-facing tokenizer API.

This class orchestrates the BPE algorithm (bpe.py) and the vocabulary
(vocab.py) into a clean interface with four responsibilities:
  1. Training   — NexaTokenizer.train(corpus, vocab_size)
  2. Encoding   — tokenizer.encode(text) → list[int]
  3. Decoding   — tokenizer.decode(ids) → str
  4. Persistence — tokenizer.save(dir) / NexaTokenizer.load(dir)

Independence
------------
This class has ZERO pretrained data. The only way to create one is to
call NexaTokenizer.train() with your own corpus, or to load a file
that you previously saved from NexaTokenizer.train().
No model weights are downloaded or imported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from nexa.tokenizer.vocab import (
    Vocabulary,
    DEFAULT_SPECIAL_TOKENS,
    SPECIAL_TOKEN_SET,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID, UNK_TOKEN,
)
from nexa.tokenizer.bpe import (
    Pair,
    train_bpe,
    word_to_tokens,
)
from nexa.utils import get_logger

log = get_logger(__name__, log_to_file=False)


class NexaTokenizer:
    """
    Nexa's BPE tokenizer — trained from scratch, no pretrained weights.

    Workflow
    --------
    **Training** (done once, then save):

        corpus = ["hello world", "hello there", ...]
        tokenizer = NexaTokenizer.train(corpus, vocab_size=500)
        tokenizer.save("data/processed/tokenizer/")

    **Inference** (load once, then reuse):

        tokenizer = NexaTokenizer.load("data/processed/tokenizer/")
        ids = tokenizer.encode("hello world", add_bos=True, add_eos=True)
        # → [1, 423, 287, 2]   (BOS + tokens + EOS)
        text = tokenizer.decode(ids)
        # → "hello world"

    Parameters
    ----------
    merges : list of Pair
        Ordered BPE merge rules (output of train_bpe).
        Index 0 = highest priority (applied first during encoding).
    vocab : Vocabulary
        The token ↔ ID mapping (output of Vocabulary construction in .train()).

    Notes
    -----
    - Words are split on whitespace. Punctuation attached to words is
      tokenized along with the word (e.g., "world!" → "w","o","r","l","d","!","</w>").
    - OOV characters (not seen during training) are mapped to <unk>.
    - The end-of-word marker "</w>" is not a special token; it becomes a
      space during decoding.
    """

    def __init__(self, merges: list[Pair], vocab: Vocabulary) -> None:
        self.merges = merges
        self.vocab = vocab

        # Pre-build a merge-rank lookup for O(1) priority checks during encode.
        # merge_rank[pair] = 0 means "merge this first" (it was learned first).
        self._merge_rank: dict[Pair, int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus: list[str],
        vocab_size: int = 1000,
        min_frequency: int = 1,
        special_tokens: Optional[list[str]] = None,
    ) -> "NexaTokenizer":
        """
        Train a BPE tokenizer from scratch on a text corpus.

        This is the ONLY legitimate way to create a NexaTokenizer —
        there is no "from_pretrained()" method by design.

        Vocabulary construction order (for deterministic ID assignment):
            1. Special tokens (0..len(specials)-1).
            2. Base characters + "</w>", sorted alphabetically.
            3. Merged tokens, in the order they were learned (merge rank).
               This means frequently-merged tokens get lower IDs.
        The loop stops as soon as len(vocab) reaches vocab_size.

        Parameters
        ----------
        corpus : list of str
            Raw training texts. Each element is one document or sentence.
            Use as much data as you can — more data → better tokenizer.
        vocab_size : int
            Target total vocabulary size (including the 4 special tokens).
            The actual size may be smaller if the corpus runs out of
            eligible pairs before vocab_size is reached.
        min_frequency : int
            Minimum pair frequency to be eligible for merging.
            Use 1 for small corpora (< 1 MB), 2–5 for larger ones.
        special_tokens : list of str, optional
            Override the default special tokens [<pad>,<bos>,<eos>,<unk>].

        Returns
        -------
        NexaTokenizer
            A trained, ready-to-use tokenizer.
        """
        _specials = special_tokens or DEFAULT_SPECIAL_TOKENS

        log.info(
            "BPE training started | corpus=%d texts | "
            "target_vocab=%d | min_freq=%d",
            len(corpus), vocab_size, min_frequency,
        )

        # We ask train_bpe for up to vocab_size merges.
        # The actual number learned may be fewer (corpus exhausted),
        # and we'll further cap at vocab construction time.
        merges_all, base_tokens = train_bpe(
            corpus=corpus,
            num_merges=vocab_size,        # generous upper bound
            min_frequency=min_frequency,
        )

        # ------------------------------------------------------------------
        # Build vocabulary with deterministic ID assignment
        # ------------------------------------------------------------------
        vocab = Vocabulary(special_tokens=_specials)

        # Add base tokens in alphabetical order (deterministic across runs)
        for token in sorted(base_tokens - set(_specials)):
            vocab.add_token(token)

        # Add merged tokens in merge order — stop when vocab_size is reached
        merges_used: list[Pair] = []
        for pair in merges_all:
            if len(vocab) >= vocab_size:
                break
            vocab.add_token(pair[0] + pair[1])
            merges_used.append(pair)

        log.info(
            "BPE training complete | vocab_size=%d | merges_learned=%d",
            len(vocab), len(merges_used),
        )

        return cls(merges=merges_used, vocab=vocab)

    # ------------------------------------------------------------------
    # Encoding — text → token IDs
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list[str]:
        """
        Apply BPE merge rules to a single word, returning its final tokens.

        Algorithm
        ---------
        1. Convert word → (char, char, ..., "</w>") using word_to_tokens().
        2. Repeatedly find the pair with the LOWEST merge rank among all
           adjacent pairs in the current token list.
           (Lower rank = was learned earlier = higher priority.)
        3. Merge that pair (replace both tokens with their concatenation).
        4. Repeat until no applicable merge exists.
        5. Map any remaining token not in vocab → "<unk>".

        Time complexity: O(len(word)² × len(merges)) in the worst case.
        This is fine for inference; KV-cache handles latency, not tokenizer.

        Parameters
        ----------
        word : str
            A single whitespace-stripped word (no spaces).

        Returns
        -------
        list of str
            Final BPE token strings for this word.
        """
        tokens = list(word_to_tokens(word))

        if not tokens:
            return []

        # --- Iteratively apply the highest-priority available merge ---
        while len(tokens) >= 2:
            # Find the adjacent pair with the lowest merge rank
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx < 0 or best_rank == float("inf"):
                break    # No more applicable merges for this word

            # Perform the merge at best_idx
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        # Any token not in vocab (e.g. unseen character) → <unk>
        return [t if t in self.vocab else UNK_TOKEN for t in tokens]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Encode a text string into a list of integer token IDs.

        Parameters
        ----------
        text : str
            Raw input text. Splits on whitespace — each whitespace-
            delimited chunk is tokenized as one word.
        add_bos : bool
            If True, prepend BOS_ID (1) to the output.
        add_eos : bool
            If True, append EOS_ID (2) to the output.

        Returns
        -------
        list of int
            Token IDs ready to feed into the Nexa transformer.
            Empty list if text is empty (plus BOS/EOS if requested).

        Examples
        --------
        >>> ids = tokenizer.encode("hello world", add_bos=True, add_eos=True)
        >>> ids[0] == tokenizer.bos_token_id
        True
        >>> ids[-1] == tokenizer.eos_token_id
        True
        """
        ids: list[int] = []

        if add_bos:
            ids.append(BOS_ID)

        words = text.strip().split()
        for word in words:
            for token in self._tokenize_word(word):
                ids.append(self.vocab.token_to_id(token))

        if add_eos:
            ids.append(EOS_ID)

        return ids

    def encode_batch(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[list[int]]:
        """
        Encode a list of strings into a list of token ID lists.

        Parameters
        ----------
        texts : list of str
            Input strings to encode.
        **kwargs
            Forwarded to encode() (e.g., add_bos=True, add_eos=True).

        Returns
        -------
        list of list of int
            One ID list per input string. Lists may have different lengths
            (use padding for batched model inputs — handled in training/).
        """
        return [self.encode(t, **kwargs) for t in texts]

    # ------------------------------------------------------------------
    # Decoding — token IDs → text
    # ------------------------------------------------------------------

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token IDs back into a human-readable string.

        How word boundaries are recovered
        ----------------------------------
        During encoding, every word's last character is merged with "</w>".
        So "cat" → ("c","a","t","</w>") → after BPE → perhaps ("cat","</w>").
        When we join all tokens and replace "</w>" → " ", we recover:
        "cat</w>" + "sat</w>" → "cat sat ".  strip() removes the trailing space.

        Parameters
        ----------
        ids : list of int
            Token IDs (output of encode()).
        skip_special_tokens : bool
            If True, <pad>, <bos>, <eos>, and <unk> are silently omitted
            from the output. Note: "</w>" is NOT a special token — it is
            always converted to a space.

        Returns
        -------
        str
            Reconstructed text string.
        """
        parts: list[str] = []

        for idx in ids:
            token = self.vocab.id_to_token(idx)
            if skip_special_tokens and token in SPECIAL_TOKEN_SET:
                continue
            parts.append(token)

        # Join all token strings, then convert </w> → " " and clean up
        return "".join(parts).replace("</w>", " ").rstrip()

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        **kwargs,
    ) -> list[str]:
        """Decode a batch of token ID sequences into strings."""
        return [self.decode(ids, **kwargs) for ids in batch_ids]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_ID

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        """
        Save the tokenizer to a directory.

        Creates two human-readable JSON files:
            tokenizer.json  — BPE model type, version, and merge rules
            vocab.json      — complete token ↔ ID mapping

        These files are the ONLY source of truth for a saved tokenizer.
        No binary blobs, no weights, no external dependencies to load them.

        Parameters
        ----------
        directory : Path or str
            Directory to save into (created if it doesn't exist).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        tokenizer_data = {
            "version": "1.0",
            "model": "bpe",
            "description": (
                "Nexa BPE tokenizer — built and trained from scratch. "
                "No pretrained weights. "
                "Reload with NexaTokenizer.load()."
            ),
            "merges": [list(pair) for pair in self.merges],
        }
        with open(directory / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

        self.vocab.save(directory / "vocab.json")
        log.debug("Tokenizer saved → %s", directory)

    @classmethod
    def load(cls, directory: Path | str) -> "NexaTokenizer":
        """
        Load a tokenizer from a directory saved by .save().

        Parameters
        ----------
        directory : Path or str
            Directory containing tokenizer.json and vocab.json.

        Returns
        -------
        NexaTokenizer
            Fully restored tokenizer, identical to the one that was saved.

        Raises
        ------
        FileNotFoundError
            If tokenizer.json or vocab.json are missing.
        """
        directory = Path(directory)

        tok_path = directory / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {directory}")

        with open(tok_path, encoding="utf-8") as f:
            data = json.load(f)

        merges: list[Pair] = [tuple(p) for p in data["merges"]]  # type: ignore[misc]
        vocab = Vocabulary.load(directory / "vocab.json")

        log.debug(
            "Tokenizer loaded ← %s  (vocab_size=%d, merges=%d)",
            directory, len(vocab), len(merges),
        )
        return cls(merges=merges, vocab=vocab)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NexaTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"num_merges={len(self.merges)})"
        )
