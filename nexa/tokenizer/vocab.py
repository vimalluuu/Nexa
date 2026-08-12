"""
nexa/tokenizer/vocab.py
========================
Vocabulary — bidirectional token ↔ integer ID mapping.

Concept
--------
A vocabulary is the bridge between the string world (text) and the integer
world (tensors). Every token has exactly one ID, and every ID maps back to
exactly one token.

We always reserve the four lowest IDs for special tokens:
    0  <pad>  Padding — used to make all sequences the same length in a batch.
              The model is trained to ignore <pad> positions.
    1  <bos>  Beginning of Sequence — prepended to every input at inference.
    2  <eos>  End of Sequence — the model learns to emit this when done.
    3  <unk>  Unknown — any token not in the vocabulary maps here.

Independence
------------
This module contains ZERO pretrained data. The vocabulary is populated
entirely by NexaTokenizer.train() from the corpus you provide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants — IDs are fixed across all Nexa models
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DEFAULT_SPECIAL_TOKENS: list[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# The set of tokens that are "special" in the sense that they should be
# hidden during human-readable decoding. </w> is NOT in this set —
# it is a regular BPE token that gets converted to a space.
SPECIAL_TOKEN_SET: frozenset[str] = frozenset(DEFAULT_SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Vocabulary class
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Bidirectional mapping: token string ↔ integer ID.

    Tokens are registered in insertion order. The caller controls the order,
    and therefore the ID assignment. By convention:
        1. Special tokens are inserted first (IDs 0–3).
        2. Base characters are inserted next (sorted for determinism).
        3. Merged BPE tokens are inserted last (in merge order).

    Parameters
    ----------
    special_tokens : list of str, optional
        The special tokens to seed the vocabulary with.
        Defaults to [<pad>, <bos>, <eos>, <unk>].

    Examples
    --------
    >>> vocab = Vocabulary()
    >>> vocab.add_token("hello")
    4
    >>> vocab.token_to_id("hello")
    4
    >>> vocab.id_to_token(4)
    'hello'
    >>> vocab.token_to_id("UNSEEN")
    3   # UNK_ID
    """

    def __init__(self, special_tokens: Optional[list[str]] = None) -> None:
        self._tok2id: dict[str, int] = {}
        self._id2tok: dict[int, str] = {}

        # Seed with special tokens — these always get the lowest IDs
        for tok in (special_tokens or DEFAULT_SPECIAL_TOKENS):
            self._register(tok)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(self, token: str) -> int:
        """Add a token unconditionally (used internally during init)."""
        if token not in self._tok2id:
            idx = len(self._tok2id)
            self._tok2id[token] = idx
            self._id2tok[idx] = token
        return self._tok2id[token]

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def add_token(self, token: str) -> int:
        """
        Add a token if not already present and return its ID.

        If the token already exists, this is a no-op and returns
        the existing ID. IDs are never reassigned.

        Parameters
        ----------
        token : str
            The token string to register.

        Returns
        -------
        int
            The assigned or existing integer ID.
        """
        return self._register(token)

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> int:
        """
        Return the integer ID for a token.

        Parameters
        ----------
        token : str
            A token string.

        Returns
        -------
        int
            The token's ID, or UNK_ID (3) if the token is not in the vocabulary.
        """
        return self._tok2id.get(token, UNK_ID)

    def id_to_token(self, idx: int) -> str:
        """
        Return the token string for an integer ID.

        Parameters
        ----------
        idx : int
            An integer ID.

        Returns
        -------
        str
            The corresponding token, or '<unk>' if the ID is out of range.
        """
        return self._id2tok.get(idx, UNK_TOKEN)

    def __len__(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._tok2id)

    def __contains__(self, token: str) -> bool:
        """Check whether a token string is in the vocabulary."""
        return token in self._tok2id

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save the vocabulary to a JSON file.

        The file maps every token string to its integer ID.
        JSON is chosen for human-readability and portability.

        Parameters
        ----------
        path : Path or str
            Destination file path (e.g., "data/processed/tokenizer/vocab.json").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._tok2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """
        Load a vocabulary from a JSON file previously saved by .save().

        Parameters
        ----------
        path : Path or str
            Path to the vocab.json file.

        Returns
        -------
        Vocabulary
            A fully restored vocabulary instance.
        """
        with open(Path(path), encoding="utf-8") as f:
            tok2id: dict[str, int] = json.load(f)

        vocab = cls.__new__(cls)
        vocab._tok2id = tok2id
        # JSON keys are always strings; values are ints → reverse cleanly
        vocab._id2tok = {v: k for k, v in tok2id.items()}
        return vocab
