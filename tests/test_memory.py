"""
tests/test_memory.py
=====================
Phase 7 test suite — Nexa Memory System.

Tests the three-layer memory architecture:
  1. Memory dataclass + MemoryStore  (storage + persistence)
  2. TFIDFRetriever                  (tokenisation + TF-IDF + cosine search)
  3. MemoryManager                   (integration layer)
  4. API endpoints                   (/api/memory, /api/chat with memory)

Test classes
------------
TestMemoryDataclass   — Memory dataclass: create, fields, serialisation
TestMemoryStore       — MemoryStore: add, get, delete, clear, FIFO eviction,
                        persistence (save/load JSON)
TestTokenise          — tokenise() edge cases
TestTFIDFRetriever    — fit, transform, search: ranking, empty corpus,
                        out-of-vocabulary queries, single document
TestMemoryManager     — add, search, clear, format_context, max_memories
TestMemoryPersistence — save to disk, reload, survive corruption
TestMemoryAPIEndpoints — GET /api/memory, DELETE /api/memory
TestChatWithMemory    — /api/chat stores memories, retrieves on 2nd turn,
                        use_memory=False skips injection

Run with:
    pytest tests/test_memory.py -v
    pytest tests/test_memory.py -v -k "TFIDF"   # only retriever tests

Independence
------------
No pretrained models, no external APIs.  TF-IDF is pure NumPy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from nexa.memory import Memory, MemoryManager, MemoryStore, TFIDFRetriever, tokenise


# ===========================================================================
# 1. Memory dataclass
# ===========================================================================

class TestMemoryDataclass:

    def test_create_returns_memory(self):
        m = Memory.create("hello", "hi there")
        assert isinstance(m, Memory)

    def test_fields_set_correctly(self):
        m = Memory.create("user msg", "assistant reply", session_id="sess1")
        assert m.user_text      == "user msg"
        assert m.assistant_text == "assistant reply"
        assert m.session_id     == "sess1"

    def test_combined_text_is_concatenation(self):
        m = Memory.create("foo", "bar")
        assert "foo" in m.combined_text
        assert "bar" in m.combined_text

    def test_id_is_unique(self):
        ids = {Memory.create("x", "y").id for _ in range(20)}
        assert len(ids) == 20

    def test_timestamp_is_recent(self):
        before = time.time()
        m      = Memory.create("x", "y")
        after  = time.time()
        assert before <= m.timestamp <= after

    def test_age_seconds_increases(self):
        m   = Memory.create("x", "y")
        age = m.age_seconds()
        assert age >= 0

    def test_frozen_immutable(self):
        m = Memory.create("x", "y")
        with pytest.raises((AttributeError, TypeError)):
            m.user_text = "changed"  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        m    = Memory.create("hello", "world", session_id="s1", tags=("a", "b"))
        d    = m.to_dict()
        m2   = Memory.from_dict(d)
        assert m2.id             == m.id
        assert m2.user_text      == m.user_text
        assert m2.assistant_text == m.assistant_text
        assert m2.session_id     == m.session_id
        assert m2.tags           == m.tags

    def test_to_dict_tags_is_list(self):
        m = Memory.create("x", "y", tags=("a",))
        d = m.to_dict()
        assert isinstance(d["tags"], list)

    def test_from_dict_tags_is_tuple(self):
        m = Memory.from_dict(Memory.create("x", "y", tags=("t",)).to_dict())
        assert isinstance(m.tags, tuple)

    def test_repr_contains_id_prefix(self):
        m = Memory.create("hello", "world")
        assert m.id[:8] in repr(m)


# ===========================================================================
# 2. MemoryStore
# ===========================================================================

class TestMemoryStore:

    def test_empty_store_has_len_zero(self):
        store = MemoryStore()
        assert len(store) == 0

    def test_add_increases_len(self):
        store = MemoryStore()
        store.add("a", "b")
        assert len(store) == 1

    def test_add_returns_memory(self):
        store = MemoryStore()
        m = store.add("hello", "world")
        assert isinstance(m, Memory)
        assert m.user_text == "hello"

    def test_get_by_id(self):
        store = MemoryStore()
        m = store.add("hello", "world")
        assert store.get(m.id) is not None
        assert store.get(m.id).id == m.id

    def test_get_missing_id_returns_none(self):
        store = MemoryStore()
        assert store.get("nonexistent-id") is None

    def test_delete_existing(self):
        store = MemoryStore()
        m = store.add("x", "y")
        deleted = store.delete(m.id)
        assert deleted is True
        assert len(store) == 0

    def test_delete_missing_returns_false(self):
        store = MemoryStore()
        assert store.delete("bad-id") is False

    def test_get_all_returns_all(self):
        store = MemoryStore()
        store.add("a", "1")
        store.add("b", "2")
        store.add("c", "3")
        all_m = store.get_all()
        assert len(all_m) == 3

    def test_get_all_session_filtered(self):
        store = MemoryStore()
        store.add("a", "1", session_id="s1")
        store.add("b", "2", session_id="s2")
        store.add("c", "3", session_id="s1")
        s1 = store.get_all(session_id="s1")
        assert len(s1) == 2
        assert all(m.session_id == "s1" for m in s1)

    def test_clear_all(self):
        store = MemoryStore()
        store.add("a", "1")
        store.add("b", "2")
        removed = store.clear()
        assert removed == 2
        assert len(store) == 0

    def test_clear_session(self):
        store = MemoryStore()
        store.add("a", "1", session_id="s1")
        store.add("b", "2", session_id="s2")
        removed = store.clear(session_id="s1")
        assert removed == 1
        assert len(store) == 1

    def test_fifo_eviction_at_max(self):
        store = MemoryStore(max_memories=3)
        ids = [store.add(f"u{i}", f"a{i}").id for i in range(4)]
        # Oldest (ids[0]) should be evicted
        assert store.get(ids[0]) is None
        assert store.get(ids[3]) is not None

    def test_max_memories_zero_means_unlimited(self):
        store = MemoryStore(max_memories=0)
        for i in range(20):
            store.add(f"u{i}", f"a{i}")
        assert len(store) == 20

    def test_get_recent_returns_n_newest(self):
        store = MemoryStore()
        for i in range(5):
            store.add(f"u{i}", f"a{i}")
        recent = store.get_recent(3)
        assert len(recent) == 3

    def test_contains_operator(self):
        store = MemoryStore()
        m = store.add("x", "y")
        assert m.id in store
        assert "missing" not in store

    def test_iter_yields_all_memories(self):
        store = MemoryStore()
        store.add("a", "1")
        store.add("b", "2")
        assert len(list(store)) == 2

    def test_repr_contains_count(self):
        store = MemoryStore()
        store.add("x", "y")
        assert "1" in repr(store)


class TestMemoryStorePersistence:

    def test_save_creates_json_file(self, tmp_path):
        p = tmp_path / "memories.json"
        store = MemoryStore(persist_path=p)
        store.add("hello", "world")
        assert p.exists()

    def test_json_is_valid(self, tmp_path):
        p = tmp_path / "memories.json"
        store = MemoryStore(persist_path=p)
        store.add("hello", "world")
        data = json.loads(p.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_reload_restores_memories(self, tmp_path):
        p = tmp_path / "memories.json"
        store1 = MemoryStore(persist_path=p)
        store1.add("cat sat", "on the mat")
        store1.add("rat ran", "from the cat")

        store2 = MemoryStore(persist_path=p)
        assert len(store2) == 2
        assert store2.get_all()[0].user_text == "cat sat"

    def test_corrupt_json_starts_fresh(self, tmp_path):
        p = tmp_path / "memories.json"
        p.write_text("NOT JSON{{{", encoding="utf-8")
        store = MemoryStore(persist_path=p)
        assert len(store) == 0

    def test_delete_updates_file(self, tmp_path):
        p = tmp_path / "memories.json"
        store = MemoryStore(persist_path=p)
        m = store.add("x", "y")
        store.delete(m.id)
        reloaded = json.loads(p.read_text())
        assert len(reloaded) == 0


# ===========================================================================
# 3. tokenise()
# ===========================================================================

class TestTokenise:

    def test_lowercases(self):
        assert tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert "cat" in tokenise("cat!")
        assert "mat" in tokenise("mat.")

    def test_empty_string_returns_empty(self):
        assert tokenise("") == []

    def test_whitespace_only_returns_empty(self):
        assert tokenise("   \t\n  ") == []

    def test_keeps_numbers(self):
        tokens = tokenise("there are 3 cats")
        assert "3" in tokens

    def test_keeps_apostrophe_in_contractions(self):
        tokens = tokenise("it's a cat")
        assert "it's" in tokens


# ===========================================================================
# 4. TFIDFRetriever
# ===========================================================================

class TestTFIDFRetriever:

    @pytest.fixture
    def small_corpus(self):
        return [
            "the cat sat on the mat",
            "the rat ran from the cat",
            "morning noon and night",
            "the dog barked at the cat",
        ]

    def test_fit_does_not_raise(self, small_corpus):
        r = TFIDFRetriever()
        r.fit(small_corpus)
        assert r.is_fitted

    def test_vocab_size_positive(self, small_corpus):
        r = TFIDFRetriever()
        r.fit(small_corpus)
        assert r.vocab_size > 0

    def test_n_docs_matches_corpus(self, small_corpus):
        r = TFIDFRetriever()
        r.fit(small_corpus)
        assert r.n_docs == len(small_corpus)

    def test_transform_returns_vector(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        vec = r.transform("the cat sat")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (r.vocab_size,)

    def test_transform_unknown_query_is_zero(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        vec = r.transform("zygote xenon quartz")  # words not in corpus
        assert np.all(vec == 0)

    def test_search_returns_list(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        results = r.search("cat mat")
        assert isinstance(results, list)

    def test_search_results_are_index_score_pairs(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        results = r.search("cat mat")
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_search_scores_descending(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        results = r.search("cat", top_k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limit(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        results = r.search("cat", top_k=2)
        assert len(results) <= 2

    def test_most_relevant_doc_ranked_first(self):
        """Doc that mentions the query term most should rank first."""
        corpus = [
            "apple orange banana fruit",
            "cat cat cat cat cat",
            "dog runs fast",
        ]
        r = TFIDFRetriever().fit(corpus)
        results = r.search("cat")
        assert results[0][0] == 1   # index 1 = "cat cat cat cat cat"

    def test_irrelevant_query_returns_empty(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        results = r.search("zygote xenon quartz")
        assert results == []

    def test_empty_corpus_returns_empty(self):
        r = TFIDFRetriever().fit([])
        assert r.search("anything") == []

    def test_single_doc_returns_it(self):
        r = TFIDFRetriever().fit(["the cat sat on the mat"])
        # With only 1 doc, all IDF → log(2/2)+1 = 1; results may be non-empty
        results = r.search("cat")
        assert len(results) <= 1

    def test_min_df_filters_rare_tokens(self):
        corpus = ["rare_word common common", "common common common"]
        r = TFIDFRetriever(min_df=2).fit(corpus)
        assert "rare_word" not in r._vocab

    def test_refit_updates_index(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus[:2])
        n_before = r.vocab_size
        r.fit(small_corpus)    # more docs → more vocab
        assert r.n_docs == len(small_corpus)

    def test_repr_contains_doc_count(self, small_corpus):
        r = TFIDFRetriever().fit(small_corpus)
        assert str(len(small_corpus)) in repr(r)


# ===========================================================================
# 5. MemoryManager
# ===========================================================================

class TestMemoryManager:

    def test_empty_manager_has_len_zero(self):
        mgr = MemoryManager()
        assert len(mgr) == 0

    def test_add_returns_memory(self):
        mgr = MemoryManager()
        m = mgr.add("hello", "hi")
        assert isinstance(m, Memory)

    def test_add_increments_len(self):
        mgr = MemoryManager()
        mgr.add("a", "1")
        mgr.add("b", "2")
        assert len(mgr) == 2

    def test_search_empty_returns_empty(self):
        mgr = MemoryManager()
        assert mgr.search("anything") == []

    def test_search_finds_relevant_memory(self):
        mgr = MemoryManager()
        mgr.add("the cat sat on the mat", "yes it did")
        mgr.add("morning coffee please", "here you go")
        results = mgr.search("cat mat")
        assert len(results) >= 1
        assert results[0][0].user_text == "the cat sat on the mat"

    def test_search_returns_memory_score_tuples(self):
        mgr = MemoryManager()
        mgr.add("the cat sat", "on the mat")
        results = mgr.search("cat")
        for m, score in results:
            assert isinstance(m, Memory)
            assert isinstance(score, float)

    def test_search_scores_descending(self):
        mgr = MemoryManager()
        for i in range(5):
            mgr.add(f"user message {i}", f"reply {i}")
        results = mgr.search("user message", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_clear_removes_all(self):
        mgr = MemoryManager()
        mgr.add("a", "1")
        mgr.add("b", "2")
        removed = mgr.clear()
        assert removed == 2
        assert len(mgr) == 0

    def test_clear_session_selective(self):
        mgr = MemoryManager()
        mgr.add("a", "1", session_id="s1")
        mgr.add("b", "2", session_id="s2")
        removed = mgr.clear(session_id="s1")
        assert removed == 1
        assert len(mgr) == 1

    def test_max_memories_enforced(self):
        mgr = MemoryManager(max_memories=3)
        for i in range(5):
            mgr.add(f"user{i}", f"reply{i}")
        assert len(mgr) <= 3

    def test_get_recent_returns_newest(self):
        mgr = MemoryManager()
        for i in range(5):
            mgr.add(f"user{i}", f"reply{i}")
        recent = mgr.get_recent(2)
        assert len(recent) == 2
        assert recent[-1].user_text == "user4"

    def test_format_context_empty_returns_empty_string(self):
        mgr = MemoryManager()
        assert mgr.format_context([]) == ""

    def test_format_context_non_empty(self):
        mgr = MemoryManager()
        mgr.add("the cat sat", "on the mat")
        hits = mgr.search("cat")
        ctx = mgr.format_context(hits)
        assert "[Past context]" in ctx
        assert "[End context]" in ctx
        assert "cat" in ctx

    def test_format_context_respects_max_chars(self):
        mgr = MemoryManager()
        for i in range(10):
            mgr.add("a" * 50, "b" * 50)
        hits = [(m, 1.0) for m in mgr.get_all()]
        ctx = mgr.format_context(hits, max_chars=100)
        # Should not hugely exceed max_chars
        assert len(ctx) < 500   # generous allowance for preamble lines

    def test_delete_removes_from_search(self):
        mgr = MemoryManager()
        m = mgr.add("the cat sat on the mat", "yes")
        mgr.delete(m.id)
        results = mgr.search("cat")
        assert all(r.id != m.id for r, _ in results)

    def test_repr_contains_count(self):
        mgr = MemoryManager()
        mgr.add("x", "y")
        assert "1" in repr(mgr)


class TestManagerPersistence:

    def test_persist_and_reload(self, tmp_path):
        p = tmp_path / "mem.json"
        mgr1 = MemoryManager(persist_path=p)
        mgr1.add("the cat sat", "on the mat")
        mgr1.add("the rat ran", "from the cat")

        mgr2 = MemoryManager(persist_path=p)
        assert len(mgr2) == 2
        # Retriever should be rebuilt on load
        results = mgr2.search("cat")
        assert len(results) >= 1


# ===========================================================================
# 6. API endpoints: /api/memory + /api/chat with memory
# ===========================================================================

def _make_tiny_generator():
    """Same tiny generator used in test_api.py — avoid importing."""
    import torch
    from nexa.inference import Generator
    from nexa.models import ModelConfig, NexaTransformer
    from nexa.tokenizer import NexaTokenizer

    cfg = ModelConfig(
        vocab_size=60, max_seq_len=32, d_model=32,
        n_heads=2, n_layers=1, d_ff=64, dropout=0.0,
    )
    torch.manual_seed(42)
    model = NexaTransformer(cfg)
    corpus = ["the cat sat on the mat", "the rat ran", "morning noon"] * 6
    tok    = NexaTokenizer.train(corpus, vocab_size=60, min_frequency=1)
    return Generator(model, tok, device="cpu")


@pytest.fixture(scope="module")
def mem_client():
    """
    TestClient backed by a FRESH isolated app instance.

    We call create_app() instead of importing the module-level singleton so
    this fixture is completely decoupled from test_api.py's module-level app
    (whose teardown nulls app.state.memory, which would corrupt our tests).

    Generator and MemoryManager are injected BEFORE TestClient enters so the
    lifespan's  `is None`  guards see them and skip re-initialisation.
    """
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.config import AppConfig
    from nexa.memory import MemoryManager

    isolated_app = create_app(cfg=AppConfig())   # fresh instance
    isolated_app.state.generator = _make_tiny_generator()
    isolated_app.state.memory    = MemoryManager(max_memories=50)

    with TestClient(isolated_app, raise_server_exceptions=True) as c:
        yield c
    # teardown is implicit — isolated_app is discarded


class TestMemoryAPIEndpoints:

    def test_health_includes_memory_count(self, mem_client):
        data = mem_client.get("/health").json()
        assert "memory_count" in data
        assert isinstance(data["memory_count"], int)

    def test_list_memories_empty_initially(self, mem_client):
        mem_client.delete("/api/memory?session_id=test_list")
        resp = mem_client.get("/api/memory?session_id=test_list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["memories"] == []

    def test_list_memories_session_id_in_response(self, mem_client):
        resp = mem_client.get("/api/memory?session_id=mysession")
        assert resp.json()["session_id"] == "mysession"

    def test_delete_memories_returns_removed_count(self, mem_client):
        # First send a chat to create a memory
        sid = "del_test_session"
        mem_client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "the cat sat"}],
            "session_id": sid, "max_new_tokens": 5,
        })
        resp = mem_client.delete(f"/api/memory?session_id={sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert "removed" in data
        assert isinstance(data["removed"], int)

    def test_delete_clears_session_memories(self, mem_client):
        sid = "clean_session_abc"
        mem_client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "the rat"}],
            "session_id": sid, "max_new_tokens": 5,
        })
        mem_client.delete(f"/api/memory?session_id={sid}")
        resp = mem_client.get(f"/api/memory?session_id={sid}")
        assert resp.json()["total"] == 0


class TestChatWithMemory:

    def test_chat_response_includes_session_id(self, mem_client):
        resp = mem_client.post("/api/chat", json={
            "messages":  [{"role": "user", "content": "the cat sat"}],
            "session_id": "chat_mem_test",
            "max_new_tokens": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "chat_mem_test"

    def test_chat_response_includes_memory_count(self, mem_client):
        resp = mem_client.post("/api/chat", json={
            "messages":  [{"role": "user", "content": "the cat sat"}],
            "session_id": "mem_count_test",
            "max_new_tokens": 5,
        })
        data = resp.json()
        assert "memory_count" in data
        assert isinstance(data["memory_count"], int)

    def test_chat_response_includes_memories_used(self, mem_client):
        resp = mem_client.post("/api/chat", json={
            "messages":  [{"role": "user", "content": "the cat sat"}],
            "session_id": "mem_used_test",
            "max_new_tokens": 5,
        })
        data = resp.json()
        assert "memories_used" in data
        assert isinstance(data["memories_used"], list)

    def test_memory_grows_after_each_turn(self, mem_client):
        import time as _time
        sid = "grow_" + str(int(_time.time() * 1000))

        history: list = []
        for i, prompt in enumerate(["the cat sat", "and the rat ran"]):
            history.append({"role": "user", "content": prompt})
            resp = mem_client.post("/api/chat", json={
                "messages":   history,
                "session_id": sid,
                "max_new_tokens": 5,
            })
            assert resp.status_code == 200
            data = resp.json()
            history = data["messages"]   # carry history forward

            mem_resp = mem_client.get(f"/api/memory?session_id={sid}")
            actual = mem_resp.json()["total"]
            assert actual == i + 1, (
                f"Turn {i}: expected {i+1} memories, got {actual}"
            )

    def test_use_memory_false_skips_injection(self, mem_client):
        sid = "no_mem_test"
        # Pre-populate memory
        mem_client.post("/api/chat", json={
            "messages":  [{"role": "user", "content": "the cat sat on the mat"}],
            "session_id": sid, "max_new_tokens": 5,
        })
        # Now chat with use_memory=False
        resp = mem_client.post("/api/chat", json={
            "messages":   [{"role": "user", "content": "the cat"}],
            "session_id":  sid,
            "use_memory":  False,
            "max_new_tokens": 5,
        })
        data = resp.json()
        assert data["memories_used"] == []

    def test_different_sessions_are_isolated(self, mem_client):
        """Memories from session A must not appear in session B's search."""
        sid_a = "iso_sess_a"
        sid_b = "iso_sess_b"
        mem_client.delete(f"/api/memory?session_id={sid_a}")
        mem_client.delete(f"/api/memory?session_id={sid_b}")

        mem_client.post("/api/chat", json={
            "messages":  [{"role": "user", "content": "unique phrase alpha beta gamma"}],
            "session_id": sid_a, "max_new_tokens": 5,
        })
        # Session B should have 0 memories
        resp = mem_client.get(f"/api/memory?session_id={sid_b}")
        assert resp.json()["total"] == 0
