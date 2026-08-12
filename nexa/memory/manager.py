"""
nexa/memory/manager.py
=======================
MemoryManager — high-level interface for Nexa's memory system.

This is the single object the rest of the codebase interacts with.
It owns a MemoryStore and a TFIDFRetriever and keeps them in sync.

Typical usage
-------------
    manager = MemoryManager(persist_path=Path("data/memories.json"))

    # After each chat turn:
    manager.add(user_text="the cat sat", assistant_text="on the mat")

    # Before the next turn, retrieve relevant past context:
    hits = manager.search("cat", top_k=3)
    for memory, score in hits:
        print(f"[{score:.2f}] {memory.user_text!r}")

    # Format for prompt injection:
    context_block = manager.format_context(hits)

Independence
------------
Only uses MemoryStore + TFIDFRetriever (both pure Python/NumPy).
No external AI calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from nexa.memory.retriever import TFIDFRetriever
from nexa.memory.store import Memory, MemoryStore


# ===========================================================================
# MemoryManager
# ===========================================================================

class MemoryManager:
    """
    Coordinates the MemoryStore and TFIDFRetriever.

    Every time a memory is added, the retriever index is rebuilt from
    the current store contents.  For the memory counts expected in
    Phase 7 (< 1000 memories), this re-fit is fast (< 10 ms on CPU).

    Parameters
    ----------
    persist_path : Path | None
        If set, memories are persisted to this JSON file.
    max_memories : int
        Maximum total memories to keep (FIFO eviction).  0 = unlimited.
    min_df : int
        Minimum document frequency for a token to enter the TF-IDF vocab.
        Use 1 (default) so every word counts, even for small stores.
    session_default : str
        Default session ID for add() / search() when none is specified.

    Examples
    --------
    >>> mgr = MemoryManager(max_memories=50)
    >>> mgr.add("the cat sat", "on the mat")
    Memory(id=..., user='the cat sat', ...)
    >>> hits = mgr.search("cat")
    >>> mgr.format_context(hits)
    '[Past context]\\nUser: the cat sat → Nexa: on the mat\\n[End context]'
    """

    def __init__(
        self,
        persist_path:    Optional[Path] = None,
        max_memories:    int            = 200,
        min_df:          int            = 1,
        session_default: str            = "default",
    ) -> None:
        self._store     = MemoryStore(
            persist_path = persist_path,
            max_memories = max_memories,
        )
        self._retriever  = TFIDFRetriever(min_df=min_df)
        self._session    = session_default

        # Rebuild index from any memories loaded from disk
        if len(self._store) > 0:
            self._rebuild_index()

    # ── Write ─────────────────────────────────────────────────────────────

    def add(
        self,
        user_text:      str,
        assistant_text: str,
        session_id:     Optional[str]   = None,
        tags:           tuple[str, ...] = (),
    ) -> Memory:
        """
        Store a new user ↔ assistant exchange as a memory.

        After storing, the TF-IDF retriever is re-fitted so the new
        memory is immediately searchable.

        Parameters
        ----------
        user_text : str
            The user's message.
        assistant_text : str
            Nexa's reply.
        session_id : str | None
            Conversation session.  Defaults to `self.session_default`.
        tags : tuple[str, ...]
            Optional labels.

        Returns
        -------
        Memory
        """
        memory = self._store.add(
            user_text      = user_text,
            assistant_text = assistant_text,
            session_id     = session_id or self._session,
            tags           = tags,
        )
        self._rebuild_index()
        return memory

    def delete(self, memory_id: str) -> bool:
        """Remove a memory by ID and rebuild the retriever index."""
        deleted = self._store.delete(memory_id)
        if deleted:
            self._rebuild_index()
        return deleted

    def clear(self, session_id: Optional[str] = None) -> int:
        """
        Clear memories.

        Parameters
        ----------
        session_id : str | None
            If given, only clear memories for that session.
            If None, clear everything.

        Returns
        -------
        int
            Number of memories removed.
        """
        count = self._store.clear(session_id)
        if count:
            self._rebuild_index()
        return count

    # ── Read ──────────────────────────────────────────────────────────────

    def search(
        self,
        query:      str,
        top_k:      int           = 3,
        session_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """
        Find the most relevant memories for `query`.

        Uses TF-IDF cosine similarity.  If `session_id` is given, only
        memories from that session are considered.

        Parameters
        ----------
        query : str
            The search query (typically the latest user message).
        top_k : int
            Maximum number of memories to return.
        session_id : str | None
            Filter to a specific session.

        Returns
        -------
        list[tuple[Memory, float]]
            Each element is ``(memory, cosine_similarity_score)``.
            Sorted descending by score (most relevant first).
            Returns an empty list when no memories match.
        """
        all_memories = self._store.get_all(session_id)
        if not all_memories:
            return []

        # Re-fit retriever scoped to this session's memories
        if session_id and session_id != self._session:
            scoped = TFIDFRetriever(min_df=self._retriever.min_df)
            scoped.fit([m.combined_text for m in all_memories])
            hits = scoped.search(query, top_k=top_k)
        else:
            hits = self._retriever.search(query, top_k=top_k)

        # Map index → Memory
        all_store = self._store.get_all()
        results = []
        for idx, score in hits:
            if session_id:
                source = all_memories
            else:
                source = all_store
            if idx < len(source):
                results.append((source[idx], score))
        return results

    def get_recent(
        self,
        n:          int,
        session_id: Optional[str] = None,
    ) -> list[Memory]:
        """Return the N most recent memories (newest last)."""
        return self._store.get_recent(n, session_id)

    def get_all(self, session_id: Optional[str] = None) -> list[Memory]:
        """Return all stored memories, optionally filtered by session."""
        return self._store.get_all(session_id)

    # ── Formatting ────────────────────────────────────────────────────────

    def format_context(
        self,
        memories:  list[tuple[Memory, float]],
        max_chars: int = 300,
    ) -> str:
        """
        Format a list of (Memory, score) pairs as a context block for prompt injection.

        The output is prepended to the chat prompt so Nexa can see relevant
        past exchanges before generating its next reply.

        Parameters
        ----------
        memories : list[tuple[Memory, float]]
            Retrieved memories with their similarity scores.
        max_chars : int
            Soft cap on the total character count of the context block.
            Individual memory lines are truncated or omitted to stay under this.

        Returns
        -------
        str
            Empty string if no memories.  Otherwise a formatted block ending
            with a blank line, ready to be prepended to the chat prompt.

        Example output
        --------------
        [Past context]
        User: the cat sat → Nexa: on the mat and the rat...
        User: what about the rat? → Nexa: the rat ran fast
        [End context]
        """
        if not memories:
            return ""

        lines = []
        total = 0
        for memory, _score in memories:
            user_snip = memory.user_text[:80]
            asst_snip = memory.assistant_text[:80]
            line      = f"User: {user_snip} -> Nexa: {asst_snip}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        if not lines:
            return ""

        return "[Past context]\n" + "\n".join(lines) + "\n[End context]\n"

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def retriever(self) -> TFIDFRetriever:
        return self._retriever

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"memories={len(self)}, "
            f"vocab={self._retriever.vocab_size})"
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """Re-fit the TF-IDF retriever from the current store contents."""
        docs = [m.combined_text for m in self._store.get_all()]
        self._retriever.fit(docs)
