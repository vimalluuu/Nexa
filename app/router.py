"""
app/router.py
==============
FastAPI APIRouter — all REST API endpoints.

Endpoint map
------------
GET    /health           → HealthResponse
GET    /api/config       → ConfigResponse (default sampling params)
POST   /api/generate     → GenerateResponse (raw text completion)
POST   /api/chat         → ChatResponse    (multi-turn + memory retrieval)
GET    /api/memory       → MemoryListResponse (list session memories)
DELETE /api/memory       → MemoryClearResponse (clear session memories)

Memory flow in /api/chat
------------------------
1. Client sends {messages, session_id, use_memory=True, ...}
2. MemoryManager.search(latest_user_message) → top-3 relevant past turns
3. format_context(hits) prepended to the chat prompt
4. NexaTransformer generates the reply
5. MemoryManager.add(user_text, reply) → stored for future retrieval
6. Response includes memories_used (snippets shown in UI) + memory_count

State storage
-------------
  app.state.generator   → Generator (model + tokenizer)
  app.state.memory      → MemoryManager (shared across all sessions)

Tests can pre-inject both before TestClient starts.

Independence
------------
All text generation uses Nexa's own trained weights.
Memory retrieval uses TF-IDF (pure NumPy) — no pretrained AI.
"""

from __future__ import annotations

from typing import Optional

import nexa
from fastapi import APIRouter, HTTPException, Request, status

from app.config import get_app_config
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MemoryClearResponse,
    MemoryItem,
    MemoryListResponse,
)
from nexa.inference import Generator, SamplingConfig
from nexa.memory import MemoryManager
from nexa.utils import get_logger

router = APIRouter()
log    = get_logger("nexa.app.router", log_to_file=False)

_MEMORY_TOP_K = 3   # max memories to inject per turn


# ===========================================================================
# Internal helpers
# ===========================================================================

def _require_generator(request: Request) -> Generator:
    """
    Extract the Generator from app.state.

    Raises HTTP 503 if the model was not loaded at startup.
    """
    gen: Optional[Generator] = getattr(request.app.state, "generator", None)
    if gen is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "Model not loaded. "
                "Run  python scripts/train.py --demo  first to train a model, "
                "then restart the server."
            ),
        )
    return gen


def _get_memory(request: Request) -> Optional[MemoryManager]:
    """Return the MemoryManager from app.state, or None if not initialised."""
    return getattr(request.app.state, "memory", None)


def _build_sampling_config(body) -> SamplingConfig:
    """Convert a SamplingParams (or subclass) Pydantic model → SamplingConfig."""
    return SamplingConfig(
        temperature        = body.temperature,
        greedy             = body.greedy,
        top_k              = body.top_k,
        top_p              = body.top_p,
        repetition_penalty = body.repetition_penalty,
        max_new_tokens     = body.max_new_tokens,
    )


def _format_chat_prompt(
    messages:       list[ChatMessage],
    memory_context: str = "",
) -> str:
    """
    Format a list of ChatMessages into a plain-text prompt for Nexa,
    optionally preceded by a memory context block.

    Since Nexa is a raw language model (not instruction-tuned), we format
    the conversation as a simple dialogue transcript and let the model
    continue from the "Nexa:" prefix.

    Example output (with memory):
        [Past context]
        User: the cat sat on the mat -> Nexa: and the rat ran away
        [End context]
        User: what happened to the rat?
        Nexa:
    """
    lines = []
    if memory_context:
        lines.append(memory_context.rstrip())
    for msg in messages:
        prefix = "User:" if msg.role == "user" else "Nexa:"
        lines.append(f"{prefix} {msg.content}")
    lines.append("Nexa:")   # prompt the model to generate the assistant reply
    return "\n".join(lines)


def _extract_reply(generated_text: str) -> str:
    """
    Extract just the Nexa reply from the generated text.

    Stop at the next "User:" or "Nexa:" to avoid the model writing the
    next user turn or a phantom second Nexa line.
    """
    reply = generated_text.strip()
    for stop_seq in ("User:", "Nexa:", "\nUser", "\nNexa", "[Past"):
        idx = reply.find(stop_seq)
        if idx > 0:
            reply = reply[:idx].strip()
    return reply or "(no response)"


# ===========================================================================
# Routes
# ===========================================================================

@router.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
    description    = "Returns server status, model info, and global memory count.",
)
async def health(request: Request) -> HealthResponse:
    """Server health + model status + memory size."""
    gen: Optional[Generator]     = getattr(request.app.state, "generator", None)
    mem: Optional[MemoryManager] = getattr(request.app.state, "memory",    None)
    return HealthResponse(
        status        = "ok",
        model_loaded  = gen is not None,
        vocab_size    = gen.tokenizer.vocab_size  if gen else None,
        n_params      = getattr(gen.model, "num_parameters", None) if gen else None,
        n_layers      = gen.model.config.n_layers if gen else None,
        memory_count  = len(mem) if mem is not None else 0,
        version       = nexa.__version__,
    )


@router.get(
    "/api/config",
    response_model = ConfigResponse,
    summary        = "Default sampling config",
    description    = "Returns the server's default sampling parameters.",
)
async def get_config(_request: Request) -> ConfigResponse:
    """Return the default sampling parameters (from AppConfig)."""
    cfg = get_app_config()
    return ConfigResponse(
        temperature        = cfg.temperature,
        greedy             = False,
        top_k              = cfg.top_k,
        top_p              = cfg.top_p,
        repetition_penalty = cfg.repetition_penalty,
        max_new_tokens     = cfg.max_new_tokens,
    )


@router.post(
    "/api/generate",
    response_model = GenerateResponse,
    summary        = "Raw text generation",
    description    = "Generate a text continuation from a prompt.",
)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    """Generate text from a prompt. Does not use or update memory."""
    gen    = _require_generator(request)
    config = _build_sampling_config(body)

    try:
        result = gen.generate(body.prompt, config)
    except Exception as exc:
        log.exception("Generation failed: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Generation error: {exc}",
        ) from exc

    return GenerateResponse(
        prompt           = result.prompt,
        generated_text   = result.generated_text,
        full_text        = result.full_text,
        tokens_generated = result.generated_tokens,
        stopped_by       = result.stopped_by,
    )


@router.post(
    "/api/chat",
    response_model = ChatResponse,
    summary        = "Multi-turn chat with memory",
    description    = (
        "Send a conversation history and get the next assistant reply. "
        "Relevant past memories are automatically retrieved and injected "
        "into the prompt (if use_memory=True). "
        "The reply is then stored as a new memory for future retrieval."
    ),
)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Generate the next assistant reply, with optional memory injection.

    Memory flow
    -----------
    1. Retrieve top-k memories most similar to the latest user message.
    2. Format them as a context block prepended to the chat prompt.
    3. Generate the reply.
    4. Store (user_message, reply) as a new memory.
    5. Return reply + updated history + memory metadata.
    """
    gen = _require_generator(request)
    mem = _get_memory(request)

    # ── 1. Retrieve relevant memories ────────────────────────────────────
    memory_context  = ""
    memories_used:  list[str] = []
    latest_user_msg = ""

    # Find the most recent user message (the query for retrieval)
    for msg in reversed(body.messages):
        if msg.role == "user":
            latest_user_msg = msg.content
            break

    if mem is not None and body.use_memory and latest_user_msg:
        hits = mem.search(
            query      = latest_user_msg,
            top_k      = _MEMORY_TOP_K,
            session_id = body.session_id,
        )
        if hits:
            memory_context = mem.format_context(hits)
            memories_used  = [
                f"{m.user_text[:40]!r} ({score:.2f})"
                for m, score in hits
            ]
            log.debug(
                "Memory: injected %d memories for session=%s",
                len(hits), body.session_id,
            )

    # ── 2. Build prompt with memory context ───────────────────────────────
    prompt = _format_chat_prompt(body.messages, memory_context)

    config = SamplingConfig(
        temperature        = body.temperature,
        greedy             = False,
        top_k              = body.top_k,
        top_p              = body.top_p,
        repetition_penalty = max(body.repetition_penalty, 1.1),
        max_new_tokens     = body.max_new_tokens,
    )

    # ── 3. Generate ───────────────────────────────────────────────────────
    try:
        result = gen.generate(prompt, config)
    except Exception as exc:
        log.exception("Chat generation failed: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Generation error: {exc}",
        ) from exc

    reply = _extract_reply(result.generated_text)

    # ── 4. Store this turn as a memory ───────────────────────────────────
    if mem is not None and latest_user_msg:
        mem.add(
            user_text      = latest_user_msg,
            assistant_text = reply,
            session_id     = body.session_id,
        )

    # ── 5. Return ─────────────────────────────────────────────────────────
    new_messages = list(body.messages) + [ChatMessage(role="assistant", content=reply)]

    return ChatResponse(
        reply          = reply,
        messages       = new_messages,
        session_id     = body.session_id,
        memory_count   = len(mem) if mem is not None else 0,
        memories_used  = memories_used,
    )


# ===========================================================================
# Memory management endpoints
# ===========================================================================

@router.get(
    "/api/memory",
    response_model = MemoryListResponse,
    summary        = "List memories",
    description    = "List all stored memories for a session.",
)
async def list_memories(
    request:    Request,
    session_id: str = "default",
) -> MemoryListResponse:
    """Return all memories for a session, newest last."""
    mem = _get_memory(request)
    if mem is None:
        return MemoryListResponse(session_id=session_id, memories=[], total=0)

    all_memories = mem.get_all(session_id=session_id)
    items = [
        MemoryItem(
            id             = m.id,
            user_text      = m.user_text,
            assistant_text = m.assistant_text,
            timestamp      = m.timestamp,
            session_id     = m.session_id,
        )
        for m in all_memories
    ]
    return MemoryListResponse(
        session_id = session_id,
        memories   = items,
        total      = len(items),
    )


@router.delete(
    "/api/memory",
    response_model = MemoryClearResponse,
    summary        = "Clear memories",
    description    = "Delete all memories for a session.",
)
async def clear_memories(
    request:    Request,
    session_id: str = "default",
) -> MemoryClearResponse:
    """Clear all memories for the given session."""
    mem = _get_memory(request)
    removed = mem.clear(session_id=session_id) if mem is not None else 0
    log.info("Cleared %d memories for session=%s", removed, session_id)
    return MemoryClearResponse(session_id=session_id, removed=removed)
