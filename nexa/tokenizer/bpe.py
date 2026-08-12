"""
nexa/tokenizer/bpe.py
======================
Byte-Pair Encoding — the training algorithm, implemented from scratch.

This module contains PURE FUNCTIONS only — no classes, no state.
Every function takes data in and returns new data out.
This makes the algorithm easy to test, reason about, and optimize later.

The BPE Algorithm (Sennrich et al., 2016 — implemented from first principles)
------------------------------------------------------------------------------
1. Split every word into characters + end-of-word marker ("</w>").
2. Count how often each word occurs in the corpus.
3. Repeat until `num_merges` merge operations have been learned:
   a. Count all adjacent token pairs across all words (weighted by word freq).
   b. Find the most frequent pair.
   c. Record it as a merge rule.
   d. Replace every occurrence of that pair with the new merged token.
4. Return: (ordered list of merge rules, set of base character tokens).

Independence
------------
This module uses ONLY Python's standard library (collections).
No external AI libraries, no pretrained data.
"""

from __future__ import annotations

from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Type aliases — these make the function signatures self-documenting
# ---------------------------------------------------------------------------

Pair = tuple[str, str]              # An adjacent token pair, e.g. ("l", "o")
WordTokens = tuple[str, ...]        # A word as a tuple of tokens
WordFreqs = dict[WordTokens, int]   # Word representation → corpus frequency


# ---------------------------------------------------------------------------
# Step 1: Convert raw text → per-word token representations
# ---------------------------------------------------------------------------

def word_to_tokens(word: str) -> WordTokens:
    """
    Convert a word string into an initial tuple of character tokens.

    The last element is always the special end-of-word token ``"</w>"``.
    This marker is what allows BPE to distinguish word-final subwords
    (e.g., "est" in "newest") from word-internal ones ("est" in "estimate").

    Parameters
    ----------
    word : str
        A single word with no leading/trailing whitespace.

    Returns
    -------
    tuple of str
        Character-level token representation, e.g. ("c","a","t","</w>").
        Returns an empty tuple for an empty string.

    Examples
    --------
    >>> word_to_tokens("cat")
    ('c', 'a', 't', '</w>')
    >>> word_to_tokens("a")
    ('a', '</w>')
    >>> word_to_tokens("")
    ()
    """
    if not word:
        return ()
    return tuple(word) + ("</w>",)


def get_word_freqs(corpus: list[str]) -> WordFreqs:
    """
    Build a word-frequency dictionary from a text corpus.

    Splits each text on whitespace to produce words, then converts
    each word to its character-token tuple and counts occurrences.

    Parameters
    ----------
    corpus : list of str
        Raw text strings. One string per document or sentence.

    Returns
    -------
    WordFreqs
        Maps each unique word-token-tuple to how many times it appears
        across the entire corpus.

    Examples
    --------
    >>> get_word_freqs(["hello world", "hello"])
    {('h','e','l','l','o','</w>'): 2, ('w','o','r','l','d','</w>'): 1}
    """
    freqs: WordFreqs = defaultdict(int)
    for text in corpus:
        for raw_word in text.strip().split():
            tokens = word_to_tokens(raw_word)
            if tokens:
                freqs[tokens] += 1
    return dict(freqs)


# ---------------------------------------------------------------------------
# Step 2: Count adjacent pairs (the core of BPE statistics)
# ---------------------------------------------------------------------------

def count_pairs(word_freqs: WordFreqs) -> Counter[Pair]:
    """
    Count the frequency of every adjacent token pair across all words.

    The count of each pair is weighted by word frequency:
    if the word "low" appears 5 times and contains the pair ("l","o"),
    that pair receives +5 to its count (not +1).

    This weighting is essential — without it, a word that appears once
    would have the same influence as a word that appears 1000 times.

    Parameters
    ----------
    word_freqs : WordFreqs
        Current word → token-tuple → frequency mapping.

    Returns
    -------
    Counter[Pair]
        Maps each adjacent pair to its total weighted frequency.

    Examples
    --------
    >>> count_pairs({("l","o","w","</w>"): 3, ("l","o","g","</w>"): 2})
    Counter({('l','o'): 5, ('o','w'): 3, ('w','</w>'): 3, ('o','g'): 2, ('g','</w>'): 2})
    """
    pairs: Counter[Pair] = Counter()
    for tokens, freq in word_freqs.items():
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += freq
    return pairs


# ---------------------------------------------------------------------------
# Step 3: Apply a merge — pure transformation of WordFreqs
# ---------------------------------------------------------------------------

def apply_merge(pair: Pair, word_freqs: WordFreqs) -> WordFreqs:
    """
    Apply one BPE merge to every word in the corpus representation.

    Scans each word left-to-right and replaces every consecutive occurrence
    of (pair[0], pair[1]) with the concatenated string pair[0]+pair[1].

    This is a PURE function — it returns a new WordFreqs dict and does
    not modify its input.

    Parameters
    ----------
    pair : Pair
        The adjacent token pair to merge, e.g. ("l", "o").
    word_freqs : WordFreqs
        Current word representations before this merge.

    Returns
    -------
    WordFreqs
        Updated word representations with the pair merged everywhere.

    Examples
    --------
    >>> apply_merge(("l","o"), {("l","o","w","</w>"): 2, ("n","o","t","</w>"): 1})
    {("lo","w","</w>"): 2, ("n","o","t","</w>"): 1}
    """
    merged_token = pair[0] + pair[1]   # e.g. ("l","o") → "lo"
    new_freqs: WordFreqs = {}

    for tokens, freq in word_freqs.items():
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            # Merge if current and next token match the pair exactly
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged_token)
                i += 2    # Consumed two tokens — skip both
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_freqs[tuple(new_tokens)] = freq

    return new_freqs


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_base_tokens(word_freqs: WordFreqs) -> set[str]:
    """
    Extract the complete set of base tokens from word_freqs.

    Base tokens are the individual characters (+ "</w>") present in the
    corpus BEFORE any merges have been applied. They form the "alphabet"
    of the tokenizer — the minimum granularity we can represent.

    Parameters
    ----------
    word_freqs : WordFreqs
        Word representations (should be called BEFORE the first merge).

    Returns
    -------
    set of str
        All character-level tokens including "</w>".
    """
    base: set[str] = set()
    for tokens in word_freqs:
        base.update(tokens)
    return base


# ---------------------------------------------------------------------------
# The full BPE training loop
# ---------------------------------------------------------------------------

def train_bpe(
    corpus: list[str],
    num_merges: int,
    min_frequency: int = 2,
) -> tuple[list[Pair], set[str]]:
    """
    Run BPE training on a text corpus.

    This is the central function of the tokenizer training pipeline.
    It learns an ordered list of merge rules from scratch — no pretrained
    data, no external tokenizer libraries.

    Algorithm
    ---------
    1. Build character-level word representations with word frequencies.
    2. Record all base character tokens (the starting "alphabet").
    3. For each merge step:
       a. Count all adjacent pair frequencies.
       b. Filter by min_frequency (ignore rare pairs).
       c. Select the pair with the highest frequency (alphabetically
          stable on ties for determinism).
       d. Record the merge.
       e. Apply the merge to all word representations.
    4. Stop when num_merges is reached or no eligible pairs remain.

    Parameters
    ----------
    corpus : list of str
        Raw training texts. Words are whitespace-separated.
    num_merges : int
        Maximum number of merge operations to learn. Each merge adds one
        new token to the vocabulary.
    min_frequency : int
        A pair must appear at least this many times to be eligible for
        merging. Use 1 for small corpora; 2–5 for large ones.

    Returns
    -------
    merges : list of Pair
        Ordered merge rules. Index 0 = highest-priority merge.
        Apply them in this order during encoding.
    base_tokens : set of str
        All single-character tokens (+ "</w>") seen in the corpus.
        These are the tokens present before any merges.

    Examples
    --------
    >>> merges, base = train_bpe(["low low lower newest"], num_merges=5)
    >>> merges[0]          # First (most frequent) merge
    ('o', 'w')             # or similar, depending on corpus
    >>> "</w>" in base
    True
    """
    # Build initial word frequency table (character-level)
    word_freqs = get_word_freqs(corpus)

    if not word_freqs:
        return [], set()

    # Capture base tokens BEFORE any merges
    base_tokens = get_base_tokens(word_freqs)

    merges: list[Pair] = []

    for _step in range(num_merges):
        # --- Count all adjacent pairs ---
        pairs = count_pairs(word_freqs)

        if not pairs:
            break    # Corpus is fully merged (single token per word)

        # --- Filter by minimum frequency ---
        eligible = {p: c for p, c in pairs.items() if c >= min_frequency}
        if not eligible:
            break

        # --- Select best pair ---
        # Primary key: frequency (higher is better).
        # Secondary key: the pair tuple itself (alphabetical, for determinism).
        best_pair: Pair = max(eligible, key=lambda p: (eligible[p], p))

        # --- Record and apply the merge ---
        merges.append(best_pair)
        word_freqs = apply_merge(best_pair, word_freqs)

    return merges, base_tokens
