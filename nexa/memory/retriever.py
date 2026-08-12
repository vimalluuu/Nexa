"""
nexa/memory/retriever.py
=========================
TF-IDF retriever — find the most relevant stored memories for a query.

Theory
------
Given a corpus of N documents (each memory is a document), we want to rank
them by relevance to a new query string.  We use TF-IDF + cosine similarity:

  1. Tokenise every document and the query into a bag-of-words.
  2. Build a vocabulary V of all unique tokens.
  3. Represent every document as a |V|-dimensional vector where component j is:

        tfidf(token_j, doc_i) = TF(token_j, doc_i) × IDF(token_j)

     where:
        TF(t, d)  = count(t in d) / |d|          (relative frequency)
        IDF(t)    = log((1 + N) / (1 + df(t))) + 1   (smooth IDF)
        df(t)     = number of documents containing t

     The "+1" inside the log prevents division by zero for unseen terms.
     The outer "+1" prevents IDF from reaching zero for terms that appear
     in every document (which would make them uninformative).

  4. Represent the query as a TF-IDF vector using the *fitted* IDF weights
     (terms not in the training vocabulary score 0).

  5. Rank documents by cosine similarity to the query vector:

        cosine(u, v) = dot(u, v) / (||u|| * ||v||)

     Cosine is length-invariant: a long memory isn't automatically ranked
     higher than a short one just because it has more words.

Why not FAISS?
--------------
FAISS needs real dense embeddings (e.g. from a pre-trained encoder).  Since
Nexa's transformer isn't trained as an encoder, we would need a pretrained
embedding model — which breaks the independence rule.  TF-IDF is purely
algorithmic and requires no learned weights.

Limitations
-----------
TF-IDF is *lexical*, not *semantic*.  "cat" and "feline" won't match unless
both words appear in the same document.  Semantic retrieval is reserved for
a future phase (once Nexa has an encoder trained from scratch).

Independence
------------
NumPy only.  No sklearn, no pretrained models, no external APIs.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np


# ===========================================================================
# Tokeniser
# ===========================================================================

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenise(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split into tokens.

    Keeps apostrophes so contractions ("don't", "it's") stay intact.
    Returns an empty list for empty/whitespace-only input.

    Examples
    --------
    >>> tokenise("The cat sat on the mat!")
    ['the', 'cat', 'sat', 'on', 'the', 'mat']
    >>> tokenise("   ")
    []
    """
    return _TOKEN_RE.findall(text.lower())


# ===========================================================================
# TFIDFRetriever
# ===========================================================================

class TFIDFRetriever:
    """
    Fit a TF-IDF index over a list of documents and search it by query.

    Workflow
    --------
    >>> retriever = TFIDFRetriever()
    >>> docs = ["the cat sat on the mat",
    ...         "the rat ran from the cat",
    ...         "morning noon and night"]
    >>> retriever.fit(docs)
    >>> results = retriever.search("where did the cat go?", top_k=2)
    >>> # results → [(idx, score), ...] sorted descending by score

    Parameters
    ----------
    min_df : int
        Minimum document frequency for a token to enter the vocabulary.
        Tokens appearing in fewer than `min_df` documents are ignored.
        Set to 1 to keep all tokens (useful when the corpus is tiny).
    """

    def __init__(self, min_df: int = 1) -> None:
        self.min_df:  int = min_df

        # Set after fit()
        self._vocab:      dict[str, int] = {}   # token → column index
        self._idf:        np.ndarray     = np.array([])
        self._matrix:     np.ndarray     = np.zeros((0, 0))  # (N_docs, |V|)
        self._doc_norms:  np.ndarray     = np.array([])
        self._n_docs:     int            = 0
        self._is_fitted:  bool           = False

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self, documents: list[str]) -> "TFIDFRetriever":
        """
        Build the TF-IDF index from a list of documents.

        Must be called before `search`.  Can be re-called to update the
        index when new documents are added (re-fits from scratch).

        Parameters
        ----------
        documents : list[str]
            Corpus to index.  Each element is one document string.

        Returns
        -------
        self
        """
        if not documents:
            self._is_fitted = True
            self._n_docs    = 0
            self._vocab     = {}
            self._idf       = np.array([])
            self._matrix    = np.zeros((0, 0))
            self._doc_norms = np.array([])
            return self

        tokenised = [tokenise(d) for d in documents]
        N         = len(documents)

        # ── Build vocabulary ──────────────────────────────────────────────
        df: dict[str, int] = {}
        for tokens in tokenised:
            for tok in set(tokens):     # count each token once per doc
                df[tok] = df.get(tok, 0) + 1

        # Filter by min_df and assign column indices
        vocab = {
            tok: idx
            for idx, (tok, freq) in enumerate(sorted(df.items()))
            if freq >= self.min_df
        }

        V = len(vocab)
        if V == 0:
            # All tokens below min_df; still fit but searches return nothing
            self._vocab     = {}
            self._idf       = np.array([])
            self._matrix    = np.zeros((N, 0))
            self._doc_norms = np.zeros(N)
            self._n_docs    = N
            self._is_fitted = True
            return self

        # ── Compute smooth IDF ────────────────────────────────────────────
        # idf(t) = log((1+N) / (1+df(t))) + 1   (sklearn sublinear_tf=False)
        idf = np.array([
            math.log((1 + N) / (1 + df[tok])) + 1.0
            for tok in sorted(vocab.keys())
        ])

        # ── Compute TF-IDF matrix  (N × V) ───────────────────────────────
        matrix = np.zeros((N, V), dtype=np.float32)
        for i, tokens in enumerate(tokenised):
            if not tokens:
                continue
            tf_raw = {}
            for tok in tokens:
                if tok in vocab:
                    tf_raw[tok] = tf_raw.get(tok, 0) + 1
            for tok, cnt in tf_raw.items():
                j            = vocab[tok]
                tf           = cnt / len(tokens)
                matrix[i, j] = tf * idf[j]

        # ── Compute L2 norms (for cosine similarity) ──────────────────────
        norms = np.linalg.norm(matrix, axis=1)
        # Replace zero norms with 1 to avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)

        self._vocab     = vocab
        self._idf       = idf
        self._matrix    = matrix
        self._doc_norms = norms
        self._n_docs    = N
        self._is_fitted = True
        return self

    def transform(self, text: str) -> np.ndarray:
        """
        Encode a single text string as a TF-IDF vector.

        Tokens not in the fitted vocabulary receive a score of 0.
        The returned vector has shape `(|V|,)`.

        Returns a zero vector if the retriever has an empty vocabulary.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        V = len(self._vocab)
        if V == 0:
            return np.zeros(0, dtype=np.float32)

        tokens = tokenise(text)
        if not tokens:
            return np.zeros(V, dtype=np.float32)

        vec    = np.zeros(V, dtype=np.float32)
        tf_raw: dict[str, int] = {}
        for tok in tokens:
            if tok in self._vocab:
                tf_raw[tok] = tf_raw.get(tok, 0) + 1

        for tok, cnt in tf_raw.items():
            j       = self._vocab[tok]
            tf      = cnt / len(tokens)
            vec[j]  = tf * self._idf[j]

        return vec

    def search(
        self,
        query:  str,
        top_k:  int = 3,
    ) -> list[tuple[int, float]]:
        """
        Return the indices and cosine-similarity scores of the top-k
        most relevant documents for `query`.

        Results are sorted descending by score (most relevant first).
        Returns an empty list if the retriever is empty or not fitted.

        Parameters
        ----------
        query : str
            The search string.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[tuple[int, float]]
            Each element is ``(document_index, cosine_similarity_score)``.
        """
        if not self._is_fitted or self._n_docs == 0 or len(self._vocab) == 0:
            return []

        query_vec = self.transform(query)
        if np.all(query_vec == 0):
            return []   # query has no vocabulary overlap → no matches

        # Cosine similarity: dot(matrix, query) / (doc_norms * query_norm)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scores = self._matrix.dot(query_vec) / (self._doc_norms * query_norm)

        # Rank by descending score; only return positive-score matches
        top_k = min(top_k, self._n_docs)
        ranked = np.argsort(scores)[::-1][:top_k]

        return [
            (int(idx), float(scores[idx]))
            for idx in ranked
            if scores[idx] > 0
        ]

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the fitted vocabulary."""
        return len(self._vocab)

    @property
    def n_docs(self) -> int:
        """Number of documents in the fitted index."""
        return self._n_docs

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        return (
            f"TFIDFRetriever("
            f"docs={self._n_docs}, "
            f"vocab={self.vocab_size}, "
            f"min_df={self.min_df})"
        )
