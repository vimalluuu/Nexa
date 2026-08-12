"""
nexa/utils/logger.py
=====================
Centralized logging utility for Nexa.

Concept — Why not just use print()?
-------------------------------------
`print()` has no concept of severity, timestamps, or destinations.
In an ML project we need:

    1. **Severity levels** — DEBUG for noisy details, INFO for milestones,
       WARNING for recoverable issues, ERROR for failures.
       This lets us mute low-level noise in production.

    2. **Timestamps** — every log line shows when it happened, making it
       easy to correlate training steps with wall-clock time.

    3. **Multiple destinations** — logs go to both the terminal (for live
       monitoring) AND a persistent file (for post-mortem analysis).

    4. **Pretty terminal output** — `rich` renders colored, formatted logs
       that are dramatically easier to scan than raw text.

Design
-------
We wrap Python's standard `logging` module (battle-tested, thread-safe,
widely understood) with a `rich` handler for terminal beauty, plus a
rotating file handler that caps log files at 10 MB each.

We expose a single public function: `get_logger(name)`.
Every module gets its own named logger:

    log = get_logger(__name__)       # e.g. "nexa.models.transformer"
    log.info("Model initialized")
    log.warning("Learning rate is unusually high")
    log.error("Out of memory — reduce batch size")

Usage
------
    from nexa.utils import get_logger
    log = get_logger(__name__)
    log.info("Hello from Nexa!")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from rich.logging import RichHandler
from rich.console import Console

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default log directory, relative to project root.
# Overridden in tests or when called with `log_dir`.
_DEFAULT_LOG_DIR = Path("logs")

# Format string for file logs (rich handles its own console format)
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Global registry so we never configure the same logger twice
_configured: set[str] = set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Return a named, configured logger.

    Parameters
    ----------
    name : str
        Logger name. Use ``__name__`` in every module so the hierarchy
        is automatically ``nexa.models.transformer``, etc.
    level : int
        Minimum severity to log. Defaults to ``logging.INFO``.
        Set to ``logging.DEBUG`` during development for verbose output.
    log_dir : Path, optional
        Directory for log files. Defaults to ``logs/`` in the project root.
    log_to_file : bool
        Whether to write logs to a rotating file. Disable in tests.

    Returns
    -------
    logging.Logger
        A fully configured logger instance.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("Training started")
    >>> log.warning("Validation loss increased")
    >>> log.error("Checkpoint save failed: %s", str(e))
    """
    logger = logging.getLogger(name)

    # Guard: only configure once even if get_logger() is called multiple times
    if name in _configured:
        return logger

    logger.setLevel(level)
    logger.propagate = False   # Prevent duplicate logs in the root logger

    # ------------------------------------------------------------------
    # Handler 1: Rich console handler
    # ------------------------------------------------------------------
    # `rich.logging.RichHandler` renders colorized, beautifully formatted
    # log lines with:
    #   - Level badges  (e.g. [INFO], [WARNING])
    #   - Clickable file+line links in supported terminals
    #   - Syntax-highlighted tracebacks
    # ------------------------------------------------------------------
    console = Console(stderr=True)   # Log to stderr, not stdout
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_path=True,
        markup=True,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # ------------------------------------------------------------------
    # Handler 2: Rotating file handler
    # ------------------------------------------------------------------
    # Writes plain-text logs to  logs/<name>.log
    # maxBytes=10 MB, keeps last 3 rotations → max 30 MB on disk per logger
    # ------------------------------------------------------------------
    if log_to_file:
        _log_dir = log_dir or _DEFAULT_LOG_DIR
        _log_dir.mkdir(parents=True, exist_ok=True)

        # Use the leaf name (e.g. "nexa.trainer" → "nexa.trainer.log")
        log_file = _log_dir / f"{name.replace('.', '_')}.log"

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(fmt=_FILE_FORMAT, datefmt=_DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    _configured.add(name)
    return logger


def set_global_level(level: int) -> None:
    """
    Change the log level for ALL configured Nexa loggers at once.

    Useful to switch between verbose (DEBUG) and quiet (WARNING) modes
    without restarting the process.

    Parameters
    ----------
    level : int
        A ``logging`` level constant, e.g. ``logging.DEBUG``.

    Examples
    --------
    >>> import logging
    >>> from nexa.utils.logger import set_global_level
    >>> set_global_level(logging.DEBUG)
    """
    for name in _configured:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
