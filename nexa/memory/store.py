"""
nexa/memory/store.py
=====================
MemoryStore — persistent storage for conversation memories.

Each memory is a single conversation turn (user message + assistant reply).
The store persists to disk as JSON, so memories survive server restarts.

Design
------
- `Memory` dataclass: immutable record of one conversation turn.
- `MemoryStore`: ordered in-memory list + JSON persistence.
  Newest memory is always appended at the end (insertion order preserved).
  When `max_memories` is exceeded, the oldest memory is evicted (FIFO).

Independence
------------
Pure Python stdlib + dataclasses. No AI, no external APIs.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# ===========================================================================
# Memory dataclass
# ===========================================================================

@dataclass(frozen=True)
class Memory:
    """
    An immutable record of one user ↔ assistant exchange.

    Attributes
    ----------
    id : str
        Unique identifier (UUID4).
    user_text : str
        The user's message for this turn.
    assistant_text : str
        Nexa's reply for this turn.
    combined_text : str
        user_text + " " + assistant_text — used as the retrieval document.
    timestamp : float
        Unix timestamp of when this memory was created.
    session_id : str
        Conversation session this memory belongs to.
    tags : tuple[str, ...]
        Optional user-supplied labels (reserved for future use).
    """

    id:             str
    user_text:      str
    assistant_text: str
    combined_text:  str
    timestamp:      float
    session_id:     str
    tags:           tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        user_text:      str,
        assistant_text: str,
        session_id:     str = "default",
        tags:           tuple[str, ...] = (),
    ) -> "Memory":
        """Factory: build a Memory from a user/assistant turn pair."""
        combined = f"{user_text.strip()} {assistant_text.strip()}"
        return cls(
            id             = str(uuid.uuid4()),
            user_text      = user_text.strip(),
            assistant_text = assistant_text.strip(),
            combined_text  = combined,
            timestamp      = time.time(),
            session_id     = session_id,
            tags           = tags,
        )

    def age_seconds(self) -> float:
        """How many seconds ago this memory was created."""
        return time.time() - self.timestamp

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON persistence."""
        d = asdict(self)
        d["tags"] = list(d["tags"])   # tuple → list for JSON
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        """Deserialize from a plain dict."""
        d = dict(d)
        d["tags"] = tuple(d.get("tags", []))
        return cls(**d)

    def __repr__(self) -> str:
        age = self.age_seconds()
        age_str = f"{age:.0f}s ago"
        return (
            f"Memory(id={self.id[:8]}…, "
            f"user={self.user_text[:30]!r}, "
            f"age={age_str})"
        )


# ===========================================================================
# MemoryStore
# ===========================================================================

class MemoryStore:
    """
    In-memory list of `Memory` objects with optional JSON persistence.

    Memories are stored in insertion order (oldest first).
    When `max_memories` is set, the oldest are evicted automatically.

    Parameters
    ----------
    persist_path : Path | None
        If set, memories are loaded from this JSON file on init and
        saved automatically on every write (add / delete / clear).
    max_memories : int
        Maximum number of memories to keep.  Oldest are evicted when exceeded.
        0 = unlimited.

    Examples
    --------
    >>> store = MemoryStore(max_memories=100)
    >>> m = store.add("hello", "hi there", session_id="abc")
    >>> len(store)
    1
    >>> store.get(m.id).user_text
    'hello'
    """

    def __init__(
        self,
        persist_path:  Optional[Path] = None,
        max_memories:  int            = 200,
    ) -> None:
        self._memories:    list[Memory] = []
        self.persist_path: Optional[Path] = Path(persist_path) if persist_path else None
        self.max_memories: int = max_memories

        if self.persist_path and self.persist_path.exists():
            self._load()

    # ── Write operations ──────────────────────────────────────────────────

    def add(
        self,
        user_text:      str,
        assistant_text: str,
        session_id:     str          = "default",
        tags:           tuple[str, ...] = (),
    ) -> Memory:
        """
        Create and store a new memory.

        If the store is at capacity, the oldest memory is evicted first.

        Parameters
        ----------
        user_text : str
            The user's message.
        assistant_text : str
            Nexa's reply.
        session_id : str
            Conversation session identifier.
        tags : tuple[str, ...]
            Optional labels.

        Returns
        -------
        Memory
            The newly created memory.
        """
        memory = Memory.create(
            user_text      = user_text,
            assistant_text = assistant_text,
            session_id     = session_id,
            tags           = tags,
        )
        # Evict oldest if at capacity
        if self.max_memories > 0 and len(self._memories) >= self.max_memories:
            self._memories.pop(0)

        self._memories.append(memory)
        self._autosave()
        return memory

    def delete(self, memory_id: str) -> bool:
        """
        Remove a memory by ID.

        Returns True if found and deleted, False if not found.
        """
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.id != memory_id]
        deleted = len(self._memories) < before
        if deleted:
            self._autosave()
        return deleted

    def clear(self, session_id: Optional[str] = None) -> int:
        """
        Remove all memories, or only memories for a specific session.

        Returns the number of memories removed.
        """
        if session_id is None:
            count = len(self._memories)
            self._memories = []
        else:
            before = len(self._memories)
            self._memories = [m for m in self._memories if m.session_id != session_id]
            count = before - len(self._memories)

        if count:
            self._autosave()
        return count

    # ── Read operations ───────────────────────────────────────────────────

    def get(self, memory_id: str) -> Optional[Memory]:
        """Return the memory with the given ID, or None."""
        for m in self._memories:
            if m.id == memory_id:
                return m
        return None

    def get_all(self, session_id: Optional[str] = None) -> list[Memory]:
        """Return all memories, optionally filtered by session."""
        if session_id is None:
            return list(self._memories)
        return [m for m in self._memories if m.session_id == session_id]

    def get_recent(
        self,
        n:          int,
        session_id: Optional[str] = None,
    ) -> list[Memory]:
        """Return the N most recent memories (newest last)."""
        memories = self.get_all(session_id)
        return memories[-n:] if n > 0 else memories

    def __len__(self) -> int:
        return len(self._memories)

    def __iter__(self) -> Iterator[Memory]:
        return iter(self._memories)

    def __contains__(self, memory_id: str) -> bool:
        return any(m.id == memory_id for m in self._memories)

    def __repr__(self) -> str:
        return (
            f"MemoryStore("
            f"memories={len(self)}, "
            f"max={self.max_memories}, "
            f"persist={self.persist_path})"
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all memories to the JSON file (if persist_path is set)."""
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self._memories]
        tmp = self.persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.persist_path)    # atomic write

    def _load(self) -> None:
        """Load memories from the JSON file."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            self._memories = [Memory.from_dict(d) for d in raw]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupt file — start fresh
            self._memories = []

    def _autosave(self) -> None:
        """Save if persist_path is configured."""
        if self.persist_path:
            self.save()
