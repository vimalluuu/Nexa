"""
app/schemas.py
===============
Pydantic request/response schemas for the Nexa Chat API.

Every API endpoint has a typed input schema (what the client sends)
and a typed output schema (what the server returns). Pydantic validates
both automatically, so malformed requests are rejected before reaching
any model code.

Independence
------------
Pure Pydantic + Python stdlib. No AI model code here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# Sampling parameter mixin
# ===========================================================================

class SamplingParams(BaseModel):
    """
    Shared sampling parameters used in both /generate and /chat requests.

    These map 1-to-1 to SamplingConfig in nexa/inference/sampler.py.
    """
    temperature:        float = Field(default=0.8,  ge=0.01, le=10.0,
                                       description="Sampling temperature. < 1 = focused, > 1 = creative.")
    greedy:             bool  = Field(default=False,
                                       description="Greedy (argmax) decoding. Overrides temperature.")
    top_k:              int   = Field(default=0,    ge=0,
                                       description="Top-K filter. 0 = disabled.")
    top_p:              float = Field(default=0.9,  gt=0.0, le=1.0,
                                       description="Nucleus (top-p) filter. 1.0 = disabled.")
    repetition_penalty: float = Field(default=1.2,  gt=0.0,
                                       description="Repetition penalty. 1.0 = disabled.")
    max_new_tokens:     int   = Field(default=80,   ge=1, le=2000,
                                       description="Maximum tokens to generate.")


# ===========================================================================
# /api/generate
# ===========================================================================

class GenerateRequest(SamplingParams):
    """Request body for POST /api/generate."""
    prompt: str = Field(..., min_length=1, description="Text prompt to generate from.")

    model_config = {"json_schema_extra": {
        "example": {
            "prompt":             "the cat sat",
            "temperature":        0.8,
            "greedy":             False,
            "top_k":              0,
            "top_p":              0.9,
            "repetition_penalty": 1.2,
            "max_new_tokens":     40,
        }
    }}


class GenerateResponse(BaseModel):
    """Response body for POST /api/generate."""
    prompt:           str
    generated_text:   str
    full_text:        str
    tokens_generated: int
    stopped_by:       str

    model_config = {"json_schema_extra": {
        "example": {
            "prompt":           "the cat sat",
            "generated_text":   "on the mat",
            "full_text":        "the cat sat on the mat",
            "tokens_generated": 5,
            "stopped_by":       "max_new_tokens",
        }
    }}


# ===========================================================================
# /api/chat
# ===========================================================================

class ChatMessage(BaseModel):
    """
    A single message in the conversation history.

    role must be "user" or "assistant".
    """
    role:    str = Field(..., description="Message author: 'user' or 'assistant'.")
    content: str = Field(..., min_length=0, description="Message text.")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'.")
        return v


class ChatRequest(SamplingParams):
    """Request body for POST /api/chat."""
    messages: list[ChatMessage] = Field(
        ..., min_length=1,
        description="Conversation history (most recent message last)."
    )
    session_id:  str  = Field(
        default="default",
        description="Client-generated session ID. Memories are keyed per session."
    )
    use_memory:  bool = Field(
        default=True,
        description="If True, relevant past memories are injected into the prompt."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "messages":           [{"role": "user", "content": "the cat sat"}],
            "session_id":         "abc123",
            "use_memory":         True,
            "temperature":        0.8,
            "top_p":              0.9,
            "repetition_penalty": 1.2,
            "max_new_tokens":     40,
        }
    }}


class ChatResponse(BaseModel):
    """Response body for POST /api/chat."""
    reply:          str
    messages:       list[ChatMessage]
    session_id:     str
    memory_count:   int               = 0
    memories_used:  list[str]         = Field(default_factory=list,
                                              description="Snippets of memories injected into this prompt.")


# ===========================================================================
# /health  &  /api/config
# ===========================================================================

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status:        str            = "ok"
    model_loaded:  bool
    vocab_size:    Optional[int]  = None
    n_params:      Optional[int]  = None
    n_layers:      Optional[int]  = None
    memory_count:  int            = 0
    version:       str


# ===========================================================================
# /api/memory
# ===========================================================================

class MemoryItem(BaseModel):
    """A single memory as returned by the API."""
    id:             str
    user_text:      str
    assistant_text: str
    timestamp:      float
    session_id:     str


class MemoryListResponse(BaseModel):
    """Response body for GET /api/memory."""
    session_id:  str
    memories:    list[MemoryItem]
    total:       int


class MemoryClearResponse(BaseModel):
    """Response body for DELETE /api/memory."""
    session_id: str
    removed:    int


class ConfigResponse(BaseModel):
    """Response body for GET /api/config — reports default sampling parameters."""
    temperature:        float
    greedy:             bool
    top_k:              int
    top_p:              float
    repetition_penalty: float
    max_new_tokens:     int


# ===========================================================================
# Error
# ===========================================================================

class ErrorResponse(BaseModel):
    """Standard error response body."""
    error:   str
    detail:  Optional[str] = None
