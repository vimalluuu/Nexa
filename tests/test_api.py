"""
tests/test_api.py
==================
Phase 6 test suite — FastAPI REST API for the Nexa Chat server.

Test strategy
-------------
We use FastAPI's TestClient (synchronous) with a tiny in-memory model
injected into app.state BEFORE the TestClient starts. This means:
  - No checkpoint files are required on disk.
  - Tests are fast (no file I/O, tiny model forward passes).
  - The app lifespan sees the pre-set generator and skips disk loading.

Test classes
------------
TestHealthEndpoint      — GET /health
TestConfigEndpoint      — GET /api/config
TestGenerateEndpoint    — POST /api/generate (success + error paths)
TestChatEndpoint        — POST /api/chat (single-turn, multi-turn, errors)
TestSchemaValidation    — Request schema validation (invalid inputs)
TestNoModel             — 503 behaviour when no model is loaded
TestChatHelpers         — _format_chat_prompt and _extract_reply logic
TestChatFlow            — Multi-turn conversational flow

Run with:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v -k "not Flow"  # skip slower flow tests

Independence
------------
All generation uses a tiny NexaTransformer built in-process.
No pretrained weights, no external APIs, no network calls.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexa.inference import Generator
from nexa.models import ModelConfig, NexaTransformer
from nexa.tokenizer import NexaTokenizer


# ===========================================================================
# Fixtures — shared test infrastructure
# ===========================================================================

def _make_tiny_generator() -> Generator:
    """
    Create a minimal Generator for fast API tests.
    Uses an untrained model — we only test API plumbing, not text quality.
    """
    cfg = ModelConfig(
        vocab_size  = 60,
        max_seq_len = 32,
        d_model     = 32,
        n_heads     = 2,
        n_layers    = 1,
        d_ff        = 64,
        dropout     = 0.0,
    )
    import torch
    torch.manual_seed(0)
    model = NexaTransformer(cfg)

    corpus = [
        "the cat sat on the mat",
        "the rat ran from the cat",
        "morning noon and night",
    ] * 6
    tokenizer = NexaTokenizer.train(corpus, vocab_size=60, min_frequency=1)
    return Generator(model, tokenizer, device="cpu")


@pytest.fixture(scope="module")
def tiny_generator() -> Generator:
    """Module-scoped: built once, shared across all test classes."""
    return _make_tiny_generator()


@pytest.fixture(scope="module")
def client(tiny_generator: Generator):
    """
    TestClient with a tiny generator pre-injected into app.state.

    The lifespan checks for an existing generator and skips disk loading,
    so tests run without any checkpoint files on disk.
    """
    from app.main import app
    # Pre-inject BEFORE TestClient starts the lifespan
    app.state.generator = tiny_generator
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.state.generator = None


@pytest.fixture(scope="module")
def client_no_model():
    """
    TestClient backed by a FRESH app with a non-existent checkpoint dir.
    The lifespan will try to load from that dir, fail to find files, and
    leave app.state.generator = None → all generate/chat calls return 503.
    """
    from pathlib import Path
    from app.main import create_app
    from app.config import AppConfig

    # Point to a directory that definitely does not exist
    no_model_cfg = AppConfig(
        checkpoint_dir = Path("__nonexistent_checkpoints_for_test__"),
        tokenizer_dir  = Path("__nonexistent_tokenizer_for_test__"),
    )
    isolated_app = create_app(cfg=no_model_cfg)
    with TestClient(isolated_app, raise_server_exceptions=True) as c:
        yield c


# ===========================================================================
# 1. GET /health
# ===========================================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, client):
        data = client.get("/health").json()
        assert "status"       in data
        assert "model_loaded" in data
        assert "version"      in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_vocab_size_positive(self, client):
        data = client.get("/health").json()
        assert isinstance(data.get("vocab_size"), int)
        assert data["vocab_size"] > 0

    def test_health_n_params_positive(self, client):
        data = client.get("/health").json()
        assert isinstance(data.get("n_params"), int)
        assert data["n_params"] > 0

    def test_health_version_string(self, client):
        data = client.get("/health").json()
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0


# ===========================================================================
# 2. GET /api/config
# ===========================================================================

class TestConfigEndpoint:
    """Tests for GET /api/config."""

    def test_config_returns_200(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200

    def test_config_has_all_sampling_fields(self, client):
        data = client.get("/api/config").json()
        for field in ("temperature", "greedy", "top_k", "top_p",
                      "repetition_penalty", "max_new_tokens"):
            assert field in data, f"Missing field: {field}"

    def test_config_temperature_in_range(self, client):
        data = client.get("/api/config").json()
        assert 0.0 < data["temperature"] <= 10.0

    def test_config_top_p_in_range(self, client):
        data = client.get("/api/config").json()
        assert 0.0 < data["top_p"] <= 1.0

    def test_config_repetition_penalty_positive(self, client):
        data = client.get("/api/config").json()
        assert data["repetition_penalty"] > 0.0

    def test_config_max_new_tokens_positive(self, client):
        data = client.get("/api/config").json()
        assert data["max_new_tokens"] >= 1


# ===========================================================================
# 3. POST /api/generate
# ===========================================================================

class TestGenerateEndpoint:
    """Tests for POST /api/generate."""

    def _post(self, client, **kwargs) -> dict:
        payload = {"prompt": "the cat", **kwargs}
        resp = client.post("/api/generate", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_generate_returns_200(self, client):
        resp = client.post("/api/generate", json={"prompt": "the cat"})
        assert resp.status_code == 200

    def test_generate_response_has_required_fields(self, client):
        data = self._post(client)
        for field in ("prompt", "generated_text", "full_text",
                      "tokens_generated", "stopped_by"):
            assert field in data, f"Missing field: {field}"

    def test_generate_prompt_preserved(self, client):
        data = self._post(client, prompt="the rat ran")
        assert data["prompt"] == "the rat ran"

    def test_generate_tokens_generated_positive(self, client):
        data = self._post(client, max_new_tokens=5)
        assert data["tokens_generated"] >= 1

    def test_generate_tokens_generated_le_max(self, client):
        data = self._post(client, max_new_tokens=3)
        assert data["tokens_generated"] <= 3

    def test_generate_stopped_by_is_valid(self, client):
        data = self._post(client, max_new_tokens=5)
        assert data["stopped_by"] in ("eos", "max_new_tokens", "max_seq_len")

    def test_generate_full_text_is_string(self, client):
        data = self._post(client)
        assert isinstance(data["full_text"], str)

    def test_generate_greedy_is_deterministic(self, client):
        """Two greedy calls on the same prompt must return the same text."""
        payload = {"prompt": "the cat sat", "greedy": True, "max_new_tokens": 10}
        r1 = client.post("/api/generate", json=payload).json()
        r2 = client.post("/api/generate", json=payload).json()
        assert r1["generated_text"] == r2["generated_text"]

    def test_generate_empty_prompt_rejected(self, client):
        resp = client.post("/api/generate", json={"prompt": ""})
        assert resp.status_code == 422   # Pydantic validation error

    def test_generate_missing_prompt_rejected(self, client):
        resp = client.post("/api/generate", json={"max_new_tokens": 5})
        assert resp.status_code == 422

    def test_generate_with_temperature(self, client):
        data = self._post(client, temperature=0.5, max_new_tokens=5)
        assert data["tokens_generated"] >= 1

    def test_generate_with_top_k(self, client):
        data = self._post(client, top_k=5, max_new_tokens=5)
        assert data["tokens_generated"] >= 1

    def test_generate_with_top_p(self, client):
        data = self._post(client, top_p=0.85, max_new_tokens=5)
        assert data["tokens_generated"] >= 1

    def test_generate_with_repetition_penalty(self, client):
        data = self._post(client, repetition_penalty=1.5, max_new_tokens=5)
        assert data["tokens_generated"] >= 1

    def test_generate_all_sampling_params(self, client):
        """All sampling params combined should work without error."""
        data = self._post(
            client,
            temperature=0.7, top_k=10, top_p=0.85,
            repetition_penalty=1.3, max_new_tokens=5,
        )
        assert data["tokens_generated"] >= 1


# ===========================================================================
# 4. POST /api/chat
# ===========================================================================

class TestChatEndpoint:
    """Tests for POST /api/chat."""

    def _post(self, client, messages, **kwargs) -> dict:
        payload = {"messages": messages, **kwargs}
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def _user_msg(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def test_chat_single_turn_returns_200(self, client):
        resp = client.post("/api/chat", json={
            "messages": [self._user_msg("the cat sat")],
        })
        assert resp.status_code == 200

    def test_chat_response_has_reply(self, client):
        data = self._post(client, [self._user_msg("the cat")])
        assert "reply" in data
        assert isinstance(data["reply"], str)

    def test_chat_response_has_messages(self, client):
        data = self._post(client, [self._user_msg("the cat")])
        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_chat_messages_includes_user_message(self, client):
        user_text = "the cat sat on the mat"
        data = self._post(client, [self._user_msg(user_text)])
        roles = [m["role"] for m in data["messages"]]
        assert "user" in roles

    def test_chat_messages_includes_assistant_reply(self, client):
        data = self._post(client, [self._user_msg("the cat")])
        roles = [m["role"] for m in data["messages"]]
        assert "assistant" in roles

    def test_chat_messages_count_grows(self, client):
        """After a single turn, returned messages should have at least 2 items."""
        data = self._post(client, [self._user_msg("the cat")])
        assert len(data["messages"]) >= 2

    def test_chat_last_message_is_assistant(self, client):
        data = self._post(client, [self._user_msg("the rat")])
        assert data["messages"][-1]["role"] == "assistant"

    def test_chat_multi_turn_history_grows(self, client):
        """Simulate two turns by re-submitting the full history."""
        msgs = [self._user_msg("the cat")]
        data1 = self._post(client, msgs, max_new_tokens=5)
        # Second turn: send the full history from turn 1
        data2 = self._post(client, data1["messages"] + [self._user_msg("and the rat")],
                           max_new_tokens=5)
        # History should have grown: 2 messages after turn 1, ≥4 after turn 2
        assert len(data2["messages"]) >= len(data1["messages"]) + 2

    def test_chat_empty_messages_rejected(self, client):
        resp = client.post("/api/chat", json={"messages": []})
        assert resp.status_code == 422

    def test_chat_invalid_role_rejected(self, client):
        resp = client.post("/api/chat", json={
            "messages": [{"role": "system", "content": "hello"}]
        })
        assert resp.status_code == 422

    def test_chat_with_sampling_params(self, client):
        data = self._post(
            client, [self._user_msg("the cat")],
            temperature=0.9, top_p=0.85, repetition_penalty=1.3, max_new_tokens=5,
        )
        assert data["reply"] is not None

    def test_chat_reply_is_nonempty(self, client):
        data = self._post(client, [self._user_msg("the cat sat")],
                          max_new_tokens=10)
        assert len(data["reply"]) > 0


# ===========================================================================
# 5. Schema validation
# ===========================================================================

class TestSchemaValidation:
    """Pydantic validation — invalid inputs must return HTTP 422."""

    def test_temperature_too_low(self, client):
        resp = client.post("/api/generate", json={"prompt": "test", "temperature": 0.0})
        assert resp.status_code == 422

    def test_temperature_too_high(self, client):
        resp = client.post("/api/generate", json={"prompt": "test", "temperature": 15.0})
        assert resp.status_code == 422

    def test_top_k_negative(self, client):
        resp = client.post("/api/generate", json={"prompt": "test", "top_k": -1})
        assert resp.status_code == 422

    def test_top_p_zero(self, client):
        resp = client.post("/api/generate", json={"prompt": "test", "top_p": 0.0})
        assert resp.status_code == 422

    def test_top_p_greater_than_one(self, client):
        resp = client.post("/api/generate", json={"prompt": "test", "top_p": 1.5})
        assert resp.status_code == 422

    def test_max_new_tokens_zero(self, client):
        resp = client.post("/api/generate", json={"prompt": "test", "max_new_tokens": 0})
        assert resp.status_code == 422

    def test_max_new_tokens_negative(self, client):
        resp = client.post("/api/generate", json={"prompt": "test", "max_new_tokens": -5})
        assert resp.status_code == 422

    def test_chat_role_must_be_user_or_assistant(self, client):
        resp = client.post("/api/chat", json={
            "messages": [{"role": "bot", "content": "hello"}]
        })
        assert resp.status_code == 422


# ===========================================================================
# 6. No model (offline mode → 503)
# ===========================================================================

class TestNoModel:
    """When no model is loaded, generate/chat must return 503."""

    def test_health_model_loaded_false(self, client_no_model):
        data = client_no_model.get("/health").json()
        assert data["model_loaded"] is False

    def test_health_still_returns_200_when_no_model(self, client_no_model):
        """Health check itself should succeed even without a model."""
        resp = client_no_model.get("/health")
        assert resp.status_code == 200

    def test_generate_503_when_no_model(self, client_no_model):
        resp = client_no_model.post("/api/generate", json={"prompt": "test"})
        assert resp.status_code == 503

    def test_chat_503_when_no_model(self, client_no_model):
        resp = client_no_model.post("/api/chat", json={
            "messages": [{"role": "user", "content": "test"}]
        })
        assert resp.status_code == 503

    def test_503_has_detail_message(self, client_no_model):
        resp = client_no_model.post("/api/generate", json={"prompt": "test"})
        data = resp.json()
        assert "detail" in data
        assert len(data["detail"]) > 0


# ===========================================================================
# 7. Chat helper functions
# ===========================================================================

class TestChatHelpers:
    """Unit tests for _format_chat_prompt and _extract_reply."""

    def test_format_single_user_message(self):
        from app.router import _format_chat_prompt
        from app.schemas import ChatMessage
        msgs = [ChatMessage(role="user", content="hello world")]
        result = _format_chat_prompt(msgs)
        assert "User: hello world" in result
        assert result.endswith("Nexa:")

    def test_format_alternating_messages(self):
        from app.router import _format_chat_prompt
        from app.schemas import ChatMessage
        msgs = [
            ChatMessage(role="user",      content="hi"),
            ChatMessage(role="assistant", content="hey"),
            ChatMessage(role="user",      content="bye"),
        ]
        result = _format_chat_prompt(msgs)
        assert "User: hi"   in result
        assert "Nexa: hey"  in result
        assert "User: bye"  in result
        assert result.endswith("Nexa:")

    def test_format_always_ends_with_nexa_prefix(self):
        from app.router import _format_chat_prompt
        from app.schemas import ChatMessage
        for n in range(1, 5):
            msgs = [ChatMessage(role="user", content=f"msg {i}") for i in range(n)]
            result = _format_chat_prompt(msgs)
            assert result.endswith("Nexa:"), f"Failed for n={n}: {result!r}"

    def test_extract_reply_strips_whitespace(self):
        from app.router import _extract_reply
        assert _extract_reply("  hello world  ") == "hello world"

    def test_extract_reply_stops_at_user_prefix(self):
        from app.router import _extract_reply
        text = "I am Nexa.\nUser: and what else?"
        result = _extract_reply(text)
        assert "User:" not in result
        assert "I am Nexa" in result

    def test_extract_reply_stops_at_nexa_prefix(self):
        from app.router import _extract_reply
        text = "hello there\nNexa: something else"
        result = _extract_reply(text)
        assert result == "hello there"

    def test_extract_reply_empty_becomes_no_response(self):
        from app.router import _extract_reply
        assert _extract_reply("") == "(no response)"
        assert _extract_reply("   ") == "(no response)"

    def test_extract_reply_clean_text_unchanged(self):
        from app.router import _extract_reply
        text = "the cat sat on the mat"
        assert _extract_reply(text) == text


# ===========================================================================
# 8. Multi-turn chat flow
# ===========================================================================

class TestChatFlow:
    """
    End-to-end multi-turn conversation flow tests.

    These simulate realistic usage: the client accumulates history
    and sends it with each request, exactly like the JavaScript frontend does.
    """

    def test_three_turn_conversation(self, client):
        """Simulate 3 back-and-forth turns."""
        history = []

        for prompt in ["the cat sat", "the rat ran", "morning noon"]:
            history.append({"role": "user", "content": prompt})
            resp = client.post("/api/chat", json={
                "messages":     history,
                "max_new_tokens": 5,
            })
            assert resp.status_code == 200, f"Turn failed on prompt: {prompt!r}"
            data = resp.json()
            history = data["messages"]   # use server's returned history

        # After 3 turns: 3 user + 3 assistant = 6 messages
        assert len(history) == 6
        # History should alternate user/assistant
        for i, msg in enumerate(history):
            expected = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected, (
                f"Message {i} should be {expected!r}, got {msg['role']!r}"
            )

    def test_history_content_preserved(self, client):
        """Earlier user messages must be preserved unchanged in subsequent turns."""
        first_msg = "the cat sat on the mat"
        history = [{"role": "user", "content": first_msg}]

        data1 = client.post("/api/chat", json={"messages": history, "max_new_tokens": 5}).json()
        data2 = client.post("/api/chat", json={
            "messages": data1["messages"] + [{"role": "user", "content": "and the rat"}],
            "max_new_tokens": 5,
        }).json()

        # First user message must still be in the history
        user_messages = [m["content"] for m in data2["messages"] if m["role"] == "user"]
        assert first_msg in user_messages

    def test_assistant_replies_are_strings(self, client):
        """All assistant messages in history must be non-None strings."""
        history = [{"role": "user", "content": "the cat"}]
        for _ in range(3):
            resp = client.post("/api/chat", json={"messages": history, "max_new_tokens": 5})
            data = resp.json()
            history = data["messages"] + [{"role": "user", "content": "and more"}]

        for msg in [m for m in history if m["role"] == "assistant"]:
            assert isinstance(msg["content"], str)
