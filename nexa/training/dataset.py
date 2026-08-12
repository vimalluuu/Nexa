"""
nexa/training/dataset.py
=========================
NexaDataset — a sliding-window dataset for language model training.

Concept: How Does the Model See Training Data?
-----------------------------------------------
A language model is trained on the "next-token prediction" task:
given a prefix of tokens, predict the next one.

We implement this as a **sliding window** over a flat list of token IDs:

    Full token sequence:  [3, 17, 42, 91, 55, 8, 22, ...]
    block_size = 4

    Window 0:  x = [3,  17, 42, 91]   y = [17, 42, 91, 55]
    Window 1:  x = [17, 42, 91, 55]   y = [42, 91, 55,  8]
    Window 2:  x = [42, 91, 55,  8]   y = [91, 55,  8, 22]
    ...

Every (x, y) pair teaches the model S separate predictions:
  - "given [3]          → predict 17"
  - "given [3, 17]      → predict 42"
  - "given [3, 17, 42]  → predict 91"
  - "given [3, 17, 42, 91] → predict 55"

The causal mask in the Transformer ensures that token i can only attend
to tokens 0..i — so all S predictions are learned from a single forward pass.

Independence
------------
No external data is downloaded. The dataset is built from text you provide
and a NexaTokenizer trained on the same text in Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from nexa.tokenizer import NexaTokenizer


class NexaDataset(Dataset):
    """
    Sliding-window language model dataset.

    Wraps a flat sequence of token IDs and yields (input, target) pairs
    of length `block_size` for causal language model training.

    Parameters
    ----------
    token_ids : list[int] or Tensor
        The full tokenized sequence. Any length.
    block_size : int
        Length of each (input, target) window.
        Must be ≤ len(token_ids) - 1 for the dataset to be non-empty.

    Examples
    --------
    >>> ids = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> ds = NexaDataset(ids, block_size=4)
    >>> len(ds)   # 8 - 4 = 4
    4
    >>> x, y = ds[0]
    >>> x.tolist()   # [1, 2, 3, 4]
    >>> y.tolist()   # [2, 3, 4, 5]   ← y is x shifted by 1
    """

    def __init__(
        self,
        token_ids: list[int] | Tensor,
        block_size: int,
    ) -> None:
        if isinstance(token_ids, list):
            self.data = torch.tensor(token_ids, dtype=torch.long)
        else:
            self.data = token_ids.to(dtype=torch.long)

        self.block_size = block_size

    def __len__(self) -> int:
        """
        Number of valid windows.

        Each window needs block_size tokens for x PLUS one more for the
        last target in y → total window = block_size + 1.
        Valid start positions: 0 .. len(data) - block_size - 1  inclusive.
        """
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Return (input, target) pair at position idx.

        Parameters
        ----------
        idx : int
            Start index of the window. Must be in [0, len(self)).

        Returns
        -------
        x : Tensor  shape [block_size]   — input token IDs
        y : Tensor  shape [block_size]   — target token IDs (x shifted by 1)
        """
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: "NexaTokenizer",
        block_size: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> "NexaDataset":
        """
        Tokenize text (line by line) and return a NexaDataset.

        Each non-empty line is encoded separately with optional BOS/EOS tokens.
        All token ID sequences are concatenated into one flat list.

        Parameters
        ----------
        text : str
            Raw text. Each line is treated as one document.
        tokenizer : NexaTokenizer
            A trained NexaTokenizer (from Phase 2).
        block_size : int
            Sliding window size.
        add_bos, add_eos : bool
            Whether to wrap each line with BOS and EOS tokens.

        Returns
        -------
        NexaDataset
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        token_ids: list[int] = []
        for line in lines:
            ids = tokenizer.encode(line, add_bos=add_bos, add_eos=add_eos)
            token_ids.extend(ids)
        return cls(token_ids, block_size)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: "NexaTokenizer",
        block_size: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> "NexaDataset":
        """
        Load a text file, tokenize, and return a NexaDataset.

        Parameters
        ----------
        path : str or Path
            Path to the text file (UTF-8).
        tokenizer : NexaTokenizer
            A trained NexaTokenizer.
        block_size : int
            Sliding window size.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(text, tokenizer, block_size, add_bos, add_eos)

    @classmethod
    def train_val_split(
        cls,
        token_ids: list[int],
        block_size: int,
        val_fraction: float = 0.1,
    ) -> tuple["NexaDataset", "NexaDataset"]:
        """
        Split a flat token ID sequence into train and validation datasets.

        The split is done at token level (not document level).
        A small overlap of `block_size` tokens is kept at the boundary so
        the validation set has proper context.

        Parameters
        ----------
        token_ids : list[int]
            The full tokenized corpus.
        block_size : int
            Sliding window size.
        val_fraction : float
            Fraction of tokens to reserve for validation. Default: 0.1 (10%).

        Returns
        -------
        (train_dataset, val_dataset)
        """
        if len(token_ids) <= block_size + 1:
            raise ValueError(
                f"Corpus too short ({len(token_ids)} tokens) for block_size={block_size}. "
                f"Need at least {block_size + 2} tokens to form any training window."
            )

        # Split index: at least block_size+1 tokens in training
        min_train = block_size + 1
        split = max(min_train, int(len(token_ids) * (1.0 - val_fraction)))

        train_ids = token_ids[:split]
        # Val set starts block_size before split so it has context from the boundary
        val_start = max(0, split - block_size)
        val_ids   = token_ids[val_start:]

        return cls(train_ids, block_size), cls(val_ids, block_size)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def token_count(self) -> int:
        """Total number of tokens in the underlying sequence."""
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"NexaDataset(windows={len(self)}, "
            f"tokens={self.token_count()}, "
            f"block_size={self.block_size})"
        )
