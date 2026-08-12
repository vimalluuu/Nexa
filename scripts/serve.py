"""
scripts/serve.py
=================
CLI entry point: launch the Nexa Chat server.

This starts a uvicorn ASGI server that:
  - Loads the trained NexaTransformer from checkpoints/
  - Serves the FastAPI REST API at /health, /api/generate, /api/chat
  - Serves the static chat UI at /

Usage
-----
# Quick start (uses defaults: localhost:8000):
python scripts/serve.py

# Custom host/port:
python scripts/serve.py --host 0.0.0.0 --port 9000

# Development mode (auto-reload on file change):
python scripts/serve.py --reload

# Custom checkpoint/tokenizer paths:
python scripts/serve.py \\
    --checkpoint-dir path/to/checkpoints \\
    --tokenizer-dir  path/to/tokenizer

After starting, open your browser at:
    http://localhost:8000

The interactive API docs are at:
    http://localhost:8000/docs

Independence
------------
The server loads only Nexa's own trained weights.
No external AI APIs are called at runtime.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from app.config import AppConfig, set_app_config
from nexa.utils import get_logger

log = get_logger("nexa.serve", log_to_file=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Start the Nexa Chat server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                   help="Directory with nexa_final.pt and model_config.json.")
    p.add_argument("--tokenizer-dir",  type=Path, default=Path("data/processed/tokenizer"),
                   help="Directory with the trained BPE tokenizer.")
    p.add_argument("--device", default="auto",
                   help="Inference device: 'auto', 'cpu', 'cuda', or 'mps'.")

    # Server
    p.add_argument("--host",   default="127.0.0.1",
                   help="Network interface to bind to.")
    p.add_argument("--port",   type=int, default=8000,
                   help="Port number.")
    p.add_argument("--reload", action="store_true",
                   help="Auto-reload server on code changes (development only).")

    # Default sampling params
    g = p.add_argument_group("Default Sampling")
    g.add_argument("--temperature",        type=float, default=0.8)
    g.add_argument("--top-k",              type=int,   default=0)
    g.add_argument("--top-p",              type=float, default=0.9)
    g.add_argument("--repetition-penalty", type=float, default=1.2)
    g.add_argument("--max-new-tokens",     type=int,   default=80)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build and register the config
    cfg = AppConfig(
        checkpoint_dir      = args.checkpoint_dir,
        tokenizer_dir       = args.tokenizer_dir,
        device              = args.device,
        host                = args.host,
        port                = args.port,
        reload              = args.reload,
        temperature         = args.temperature,
        top_k               = args.top_k,
        top_p               = args.top_p,
        repetition_penalty  = args.repetition_penalty,
        max_new_tokens      = args.max_new_tokens,
    )
    set_app_config(cfg)

    # Print startup banner
    print()
    print("=" * 58)
    print("  Nexa Chat Server")
    print("=" * 58)
    print(f"  UI   : http://{cfg.host}:{cfg.port}")
    print(f"  Docs : http://{cfg.host}:{cfg.port}/docs")
    print(f"  Model: {cfg.checkpoint_dir}  (available={cfg.model_available})")
    print(f"  Device: {cfg.device}")
    print("=" * 58)

    if not cfg.model_available:
        print()
        print("  WARNING: No trained model found.")
        print("  Run the following first to train a demo model:")
        print("    python scripts/train.py --demo")
        print("  Then restart the server.")

    print()

    uvicorn.run(
        "app.main:app",
        host    = cfg.host,
        port    = cfg.port,
        reload  = cfg.reload,
        log_level = "info",
    )


if __name__ == "__main__":
    main()
