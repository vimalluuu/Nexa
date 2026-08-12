"""
nexa/memory/__init__.py
========================
Public API for Nexa's memory system (Phase 7).

The memory system lets Nexa retrieve relevant past conversation turns
and inject them into the prompt context, extending its effective memory
beyond the fixed transformer context window.

Architecture
------------
  MemoryManager     ← top-level interface (use this)
  ├── MemoryStore   ← ordered list of Memory objects + JSON persistence
  └── TFIDFRetriever ← pure-NumPy TF-IDF similarity search

Independence
------------
No pretrained AI model or external embedding API is used.
Retrieval is based on TF-IDF cosine similarity — a purely algorithmic,
weight-free technique.

Usage
-----
    from nexa.memory import MemoryManager

    mgr = MemoryManager(persist_path=Path("data/memories.json"))

    # Store a conversation turn
    mgr.add(user_text="the cat sat", assistant_text="on the mat")

    # Retrieve relevant memories before the next turn
    hits = mgr.search("cat sat on mat", top_k=3)
    context = mgr.format_context(hits)

    # context is a string ready to prepend to the chat prompt
"""

from nexa.memory.manager import MemoryManager
from nexa.memory.retriever import TFIDFRetriever, tokenise
from nexa.memory.store import Memory, MemoryStore

__all__ = [
    "Memory",
    "MemoryStore",
    "TFIDFRetriever",
    "tokenise",
    "MemoryManager",
]
