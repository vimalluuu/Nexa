"""
app/main.py
============
FastAPI application factory for the Nexa Chat server.

Startup sequence
----------------
1. FastAPI app is created with a lifespan context manager.
2. On startup:
   a. Try to load Generator from checkpoints/.
   b. Initialise MemoryManager (with optional disk persistence).
3. Static files served at "/" (HTML/CSS/JS chat UI).
4. API router mounted at root path.

State in app.state
------------------
  app.state.generator  : Generator | None
  app.state.memory     : MemoryManager | None

Testing
-------
Tests pre-set app.state.generator and/or app.state.memory before TestClient
connects. The lifespan checks for existing values and skips loading/init,
enabling fully isolated API tests without touching checkpoint files.

Independence
------------
No external AI APIs. The only intelligence loaded is our own trained
NexaTransformer weights from checkpoints/nexa_final.pt.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import AppConfig, get_app_config
from app.router import router
from nexa.utils import get_logger

log = get_logger("nexa.app", log_to_file=False)

# Absolute path to the static files directory
_STATIC_DIR = Path(__file__).parent / "static"


# ===========================================================================
# Lifespan — model loading / cleanup
# ===========================================================================

def _make_lifespan(cfg: AppConfig):
    """
    Return a lifespan context manager that loads the Generator and
    initialises the MemoryManager on startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Startup ────────────────────────────────────────────────────────
        # Skip loading if already injected (test mode)
        if getattr(app.state, "generator", None) is None:
            if cfg.model_available:
                log.info("Loading Nexa model from %s ...", cfg.checkpoint_dir)
                try:
                    from nexa.inference import Generator
                    gen = Generator.from_checkpoint(
                        checkpoint_dir = cfg.checkpoint_dir,
                        tokenizer_dir  = cfg.tokenizer_dir,
                        device         = cfg.device,
                    )
                    app.state.generator = gen
                    log.info(
                        "Model ready: %s params, vocab=%d",
                        f"{gen.model.num_parameters:,}",
                        gen.tokenizer.vocab_size,
                    )
                except Exception as exc:
                    log.error("Failed to load model: %s", exc)
                    app.state.generator = None
            else:
                log.warning(
                    "No checkpoint found at %s. "
                    "Run  python scripts/train.py --demo  to train first. "
                    "Server running in offline mode (API returns 503).",
                    cfg.checkpoint_dir,
                )
                app.state.generator = None

        # Initialise MemoryManager (skip if test pre-injected one)
        if getattr(app.state, "memory", None) is None:
            from nexa.memory import MemoryManager
            persist_path = cfg.memory_persist_path  # may be None
            app.state.memory = MemoryManager(
                persist_path = persist_path,
                max_memories = cfg.max_memories,
            )
            log.info(
                "MemoryManager ready (persist=%s, max=%d)",
                persist_path, cfg.max_memories,
            )

        yield   # ← server is running here

        # ── Shutdown ───────────────────────────────────────────────────────
        app.state.generator = None
        app.state.memory    = None
        log.info("Nexa server shut down.")

    return lifespan


# ===========================================================================
# App factory
# ===========================================================================

def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    cfg : AppConfig, optional
        Server configuration. Defaults to the module-level AppConfig singleton.

    Returns
    -------
    FastAPI
        Fully configured application ready to be served by uvicorn.
    """
    cfg = cfg or get_app_config()

    app = FastAPI(
        title       = "Nexa Chat API",
        description = (
            "Local-only chat interface powered by NexaTransformer — "
            "an independent language model trained from scratch.\n\n"
            "No external AI APIs. No pretrained models. No cloud inference."
        ),
        version     = "0.5.0",
        lifespan    = _make_lifespan(cfg),
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    # Allow the browser to call the API even during local development
    # (e.g., if running the UI via a different port).
    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["GET", "POST", "OPTIONS"],
        allow_headers  = ["*"],
    )

    # ── API routes ────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Static files (chat UI) ────────────────────────────────────────────
    # Served at "/". The router's / routes take priority over static files
    # because FastAPI checks routes before StaticFiles.
    if _STATIC_DIR.exists():
        app.mount(
            "/",
            StaticFiles(directory=_STATIC_DIR, html=True),
            name="static",
        )
    else:
        log.warning("Static directory not found: %s (UI will not be served)", _STATIC_DIR)

    return app


# ===========================================================================
# Module-level app instance (used by uvicorn and tests)
# ===========================================================================

# This is the object uvicorn imports:  uvicorn app.main:app
app = create_app()
