"""
Nexa Phase 8.3 - Unicode and Whitespace Normalization
=====================================================
Deterministic, AI-free normalization functions.

Applies:
  - Unicode NFC normalization (canonical decomposition + composition)
  - Invalid/control character removal (with configurable exceptions)
  - Newline normalization (CRLF/CR → LF)
  - Whitespace normalization (collapse runs, strip trailing/leading)
  - Preserve case (do NOT lowercase)
  - Preserve punctuation (do NOT strip)

All functions are pure (no side effects) and deterministic.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Control character patterns
# ---------------------------------------------------------------------------
# Keep: tab (0x09), newline (0x0A), carriage return (0x0D)
# Remove: other C0 controls, C1 controls, BOM, soft-hyphens, zero-width chars
_REMOVE_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"   # C0 controls (except tab/LF/CR)
    r"\x80-\x9f"                            # C1 controls
    r"\ufeff"                               # BOM
    r"\u00ad"                               # soft hyphen
    r"\u200b-\u200f"                        # zero-width chars
    r"\u2028\u2029"                         # LS/PS
    r"\ufffe\uffff]",                       # non-characters
    re.UNICODE,
)

# Normalize multiple spaces/tabs into a single space (but NOT newlines)
_MULTI_SPACE_RE   = re.compile(r"[^\S\n]+")
# 3+ consecutive newlines → 2 newlines (preserve paragraph breaks)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Trailing whitespace on each line
_TRAILING_WS_RE   = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text: str) -> str:
    """Remove C0/C1 control chars, BOM, zero-width chars, etc."""
    return _REMOVE_CHARS_RE.sub("", text)


def normalize_newlines(text: str) -> str:
    """Normalize CRLF and lone CR to LF."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse intra-line whitespace runs, strip trailing whitespace per line,
    collapse 3+ blank lines to 2, strip leading/trailing whitespace overall.
    """
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Apply the full normalization pipeline in order:
      1. Unicode NFC
      2. CRLF normalization
      3. Control character removal
      4. Whitespace normalization
    """
    text = normalize_unicode(text)
    text = normalize_newlines(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    return text
