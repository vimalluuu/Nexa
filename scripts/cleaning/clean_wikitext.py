"""
Nexa Phase 8.3 - Wikitext Cleaner
===================================
Deterministic, AI-free conversion of MediaWiki wikitext to plain text.

Approach: multi-pass regex transformations in a defined order.
No external parsing library required — works on Python stdlib only.

Transformations applied (in order):
  1. HTML comments removed
  2. <ref> tags (inline and block) removed
  3. Templates removed (depth-tracking to handle nesting)
  4. Tables removed (depth-tracking)
  5. [[File:]] / [[Image:]] / [[Media:]] links removed
  6. [[Category:]] links removed
  7. Internal wikilinks resolved: [[target|label]] → label, [[target]] → target
  8. External links: [url label] → label, bare [url] → (removed)
  9. HTML tags stripped (keep inner text for span/div/p/etc.)
 10. Bold/italic markup removed (''' and '')
 11. Section headers: == Heading == → Heading
 12. List markers (* # : ;) removed (keep text)
 13. Magic words (__TOC__ etc.) removed
 14. Horizontal rules removed
 15. Residual markup cleanup

All transformations are deterministic and do not alter substantive prose.
"""

from __future__ import annotations

import re
from typing import Callable

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# HTML comments
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# <ref ...>...</ref> and self-closing <ref ... />
_REF_BLOCK_RE    = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF_RE     = re.compile(r"<ref[^/]*/\s*>", re.IGNORECASE)
_REFERENCES_TAG  = re.compile(r"<references\s*/?>", re.IGNORECASE)

# Nowiki — preserve content literally (remove tags, keep content)
_NOWIKI_RE       = re.compile(r"<nowiki>(.*?)</nowiki>", re.DOTALL | re.IGNORECASE)

# File / Image / Media wikilinks (before general wikilink handling)
_FILE_LINK_RE    = re.compile(
    r"\[\[\s*(?:File|Image|Media)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
    re.IGNORECASE,
)

# Category links
_CATEGORY_RE     = re.compile(r"\[\[\s*Category\s*:[^\]]*\]\]", re.IGNORECASE)

# External links: [url text] or [url]
_EXT_LINK_TEXT_RE = re.compile(r"\[https?://[^\s\[\]]+\s+([^\[\]]+?)\]")
_EXT_LINK_BARE_RE = re.compile(r"\[https?://[^\s\[\]]+\]")
_BARE_URL_RE      = re.compile(r"https?://\S+")

# Internal wikilinks [[target|label]] or [[target]]
_WIKI_LINK_LABELED_RE = re.compile(r"\[\[([^|\[\]]+)\|([^\[\]]+)\]\]")
_WIKI_LINK_PLAIN_RE   = re.compile(r"\[\[([^\[\]]+)\]\]")

# HTML tags — block-level (remove with content): script, style, math, etc.
_BLOCK_REMOVE_TAGS = re.compile(
    r"<(script|style|math|syntaxhighlight|source|timeline|gallery|poem|blockquote|"
    r"div\s+style=[^>]*display:\s*none[^>]*).*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# Generic HTML tags — strip tags, keep inner text
_HTML_TAG_RE      = re.compile(r"<[^>]+>")

# Bold/italic markup
_BOLD_ITALIC_RE   = re.compile(r"'{2,5}")

# Section headers == text == (1-6 levels)
_HEADER_RE        = re.compile(r"^={1,6}\s*(.*?)\s*={1,6}\s*$", re.MULTILINE)

# List markers at line start
_LIST_MARKER_RE   = re.compile(r"^[*#:;]+\s*", re.MULTILINE)

# Definition list term
_DT_RE            = re.compile(r"^;\s*", re.MULTILINE)

# Magic words
_MAGIC_WORD_RE    = re.compile(r"__[A-Z_]+__")

# Horizontal rules
_HR_RE            = re.compile(r"^-{4,}\s*$", re.MULTILINE)

# Indentation {{indent}} artifacts
_INDENT_RE        = re.compile(r"^:+\s*", re.MULTILINE)

# Residual {{ }} that weren't caught by depth removal (e.g. empty)
_LEFTOVER_BRACES  = re.compile(r"\{\{[^{}]*\}\}")

# Lines that are purely punctuation/symbols with no word chars
_JUNK_LINE_RE     = re.compile(r"^\W+$", re.MULTILINE)

# Multiple blank lines → double newline (paragraph boundary)
_MULTI_BLANK_RE   = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Depth-tracking template / table removal
# ---------------------------------------------------------------------------

def _remove_depth_matched(text: str, open_str: str, close_str: str) -> str:
    """
    Remove all spans delimited by open_str / close_str, handling nesting.
    e.g. open_str='{{', close_str='}}' removes nested templates.
    Uses a character-scan approach; O(n) in text length.
    """
    result = []
    i = 0
    n = len(text)
    olen = len(open_str)
    clen = len(close_str)
    depth = 0
    start = 0

    while i < n:
        if text[i:i + olen] == open_str:
            if depth == 0:
                result.append(text[start:i])
            depth += 1
            i += olen
        elif text[i:i + clen] == close_str and depth > 0:
            depth -= 1
            if depth == 0:
                # Optionally add a space where the template was
                result.append(" ")
                start = i + clen
            i += clen
        else:
            i += 1

    if depth == 0:
        result.append(text[start:])
    # If depth > 0, unclosed open — append what we have
    return "".join(result)


def _remove_tables(text: str) -> str:
    """Remove MediaWiki tables {| ... |}"""
    return _remove_depth_matched(text, "{|", "|}")


def _remove_templates(text: str) -> str:
    """Remove all {{ ... }} template invocations (nested)."""
    return _remove_depth_matched(text, "{{", "}}")


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_wikitext(wikitext: str) -> str:
    """
    Convert MediaWiki wikitext to plain prose text.

    Parameters
    ----------
    wikitext : str
        Raw wikitext as extracted from the MediaWiki XML dump.

    Returns
    -------
    str
        Plain text suitable for downstream normalization.
        Preserves prose sentences; removes markup, templates, tables, etc.
    """
    text = wikitext

    # 1. HTML comments
    text = _HTML_COMMENT_RE.sub("", text)

    # 2. <ref> blocks and self-closing refs
    text = _REF_BLOCK_RE.sub("", text)
    text = _REF_SELF_RE.sub("", text)
    text = _REFERENCES_TAG.sub("", text)

    # 3. <nowiki> — strip tags, keep literal content
    text = _NOWIKI_RE.sub(r"\1", text)

    # 4. Block-remove tags (script, style, math, etc.)
    text = _BLOCK_REMOVE_TAGS.sub("", text)

    # 5. Tables {| ... |} before templates (tables can contain templates)
    text = _remove_tables(text)

    # 6. Templates {{ ... }} — nested
    text = _remove_templates(text)

    # 7. File / Image / Media links
    text = _FILE_LINK_RE.sub("", text)

    # 8. Category links
    text = _CATEGORY_RE.sub("", text)

    # 9. External links: keep display text if present
    text = _EXT_LINK_TEXT_RE.sub(r"\1", text)
    text = _EXT_LINK_BARE_RE.sub("", text)

    # 10. Internal wikilinks: [[target|label]] → label
    #     Need multiple passes for possible remaining nested brackets
    for _ in range(3):
        text = _WIKI_LINK_LABELED_RE.sub(r"\2", text)
        text = _WIKI_LINK_PLAIN_RE.sub(r"\1", text)

    # 11. Strip remaining HTML tags (keep inner text)
    text = _HTML_TAG_RE.sub("", text)

    # 12. Bold/italic markup
    text = _BOLD_ITALIC_RE.sub("", text)

    # 13. Section headers → just the heading text on its own line
    text = _HEADER_RE.sub(r"\1", text)

    # 14. List markers
    text = _LIST_MARKER_RE.sub("", text)

    # 15. Magic words
    text = _MAGIC_WORD_RE.sub("", text)

    # 16. Horizontal rules
    text = _HR_RE.sub("", text)

    # 17. Residual leftover braces (any {{...}} not caught by depth removal)
    for _ in range(3):
        text = _LEFTOVER_BRACES.sub("", text)

    # 18. Collapse multiple blank lines
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text
