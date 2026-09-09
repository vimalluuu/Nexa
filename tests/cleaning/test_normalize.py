"""
Tests for scripts.cleaning.normalize
"""
import pytest
from scripts.cleaning.normalize import (
    normalize_unicode,
    normalize_newlines,
    remove_control_chars,
    normalize_whitespace,
    normalize_text,
)


class TestNormalizeUnicode:
    def test_nfc_normalization(self):
        # 'e' + combining acute → é (NFC)
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        assert normalize_unicode(decomposed) == composed

    def test_already_nfc_unchanged(self):
        s = "Hello, World! 123"
        assert normalize_unicode(s) == s

    def test_empty_string(self):
        assert normalize_unicode("") == ""

    def test_preserves_case(self):
        s = "ABC abc"
        assert normalize_unicode(s) == s

    def test_preserves_punctuation(self):
        s = "Hello, world! It's a test."
        assert normalize_unicode(s) == s


class TestNormalizeNewlines:
    def test_crlf_to_lf(self):
        assert normalize_newlines("a\r\nb") == "a\nb"

    def test_cr_to_lf(self):
        assert normalize_newlines("a\rb") == "a\nb"

    def test_lf_unchanged(self):
        assert normalize_newlines("a\nb") == "a\nb"

    def test_mixed_endings(self):
        result = normalize_newlines("a\r\nb\rc\nd")
        assert result == "a\nb\nc\nd"

    def test_empty_string(self):
        assert normalize_newlines("") == ""


class TestRemoveControlChars:
    def test_removes_null(self):
        assert remove_control_chars("a\x00b") == "ab"

    def test_removes_bell(self):
        assert remove_control_chars("a\x07b") == "ab"

    def test_removes_bom(self):
        assert remove_control_chars("\ufeffHello") == "Hello"

    def test_removes_zero_width(self):
        assert remove_control_chars("a\u200bb") == "ab"

    def test_keeps_tab(self):
        assert remove_control_chars("a\tb") == "a\tb"

    def test_keeps_newline(self):
        assert remove_control_chars("a\nb") == "a\nb"

    def test_keeps_printable_ascii(self):
        s = "Hello, World! 123"
        assert remove_control_chars(s) == s

    def test_removes_soft_hyphen(self):
        assert remove_control_chars("hyphen\u00adated") == "hyphenenated".replace("ne", "")
        # Precise test:
        result = remove_control_chars("soft\u00adhyphen")
        assert "\u00ad" not in result

    def test_empty_string(self):
        assert remove_control_chars("") == ""


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("a  b   c") == "a b c"

    def test_strips_trailing_spaces_per_line(self):
        result = normalize_whitespace("hello   \nworld   ")
        assert not any(line.endswith(" ") for line in result.splitlines())

    def test_collapses_excessive_blank_lines(self):
        result = normalize_whitespace("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strips_overall_whitespace(self):
        result = normalize_whitespace("  hello world  ")
        assert result == "hello world"

    def test_preserves_paragraph_breaks(self):
        result = normalize_whitespace("Para one.\n\nPara two.")
        assert "\n\n" in result

    def test_does_not_collapse_newlines_to_spaces(self):
        result = normalize_whitespace("line one\nline two")
        assert "\n" in result

    def test_preserves_punctuation(self):
        result = normalize_whitespace("Hello, world! It's a test.")
        assert "," in result and "!" in result and "'" in result

    def test_empty_string(self):
        assert normalize_whitespace("") == ""


class TestNormalizeText:
    def test_full_pipeline(self):
        raw = "  caf\u00e9\r\n\r\nsome text  \x00here  \n\n\n  "
        result = normalize_text(raw)
        assert "\r" not in result
        assert "\x00" not in result
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_deterministic(self):
        text = "Hello, world!\r\nThis is a test.\n\n\nFoo."
        assert normalize_text(text) == normalize_text(text)

    def test_preserves_case(self):
        result = normalize_text("UPPER lower Mixed")
        assert "UPPER" in result
        assert "lower" in result
        assert "Mixed" in result

    def test_does_not_strip_punctuation(self):
        result = normalize_text("Hello, world! It's a test.")
        assert "," in result
        assert "!" in result

    def test_unicode_preserved(self):
        result = normalize_text("Ångström naïve résumé")
        assert "Å" in result
        assert "ï" in result
        assert "é" in result
