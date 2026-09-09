"""
Tests for scripts.cleaning.filter_docs
"""
import pytest
from scripts.cleaning.filter_docs import FilterStats, filter_document, _is_binary_or_garbled


class TestFilterStats:
    def test_initial_state(self):
        s = FilterStats()
        assert s.total_input == 0
        assert s.kept == 0
        assert s.removed_total == 0

    def test_record_kept_updates_counters(self):
        s = FilterStats()
        s.record_kept("hello world")
        assert s.kept == 1
        assert s.total_chars == 11
        assert s.total_words == 2

    def test_avg_chars(self):
        s = FilterStats()
        s.record_kept("abcde")   # 5 chars
        s.record_kept("abcde")   # 5 chars
        assert s.avg_chars == 5.0

    def test_min_max_chars(self):
        s = FilterStats()
        s.record_kept("short")   # 5
        s.record_kept("a" * 100) # 100
        assert s.min_chars == 5
        assert s.max_chars == 100

    def test_removed_total(self):
        s = FilterStats()
        s.removed_empty = 2
        s.removed_too_short = 3
        s.removed_binary = 1
        assert s.removed_total == 6

    def test_to_dict_keys(self):
        s = FilterStats(dataset="test")
        d = s.to_dict()
        required_keys = [
            "dataset", "total_input", "kept", "removed_total",
            "removed_empty", "removed_too_short", "total_chars", "total_words",
        ]
        for k in required_keys:
            assert k in d


class TestIsBinaryOrGarbled:
    def test_clean_text_not_binary(self):
        assert not _is_binary_or_garbled("Hello, world! This is fine.")

    def test_empty_not_binary(self):
        assert not _is_binary_or_garbled("")

    def test_high_replacement_chars_is_binary(self):
        # Many replacement chars → garbled
        garbled = "\ufffd" * 50 + "a" * 10
        assert _is_binary_or_garbled(garbled)

    def test_mostly_printable_unicode_ok(self):
        # Normal Unicode text is fine
        assert not _is_binary_or_garbled("café naïve résumé Ångström")


class TestFilterDocument:
    def _long_text(self, n: int = 500) -> str:
        words = ["word"] * (n // 5)
        return " ".join(words)

    def test_keeps_valid_document(self):
        s = FilterStats()
        text = self._long_text(500)
        result = filter_document(text, s, min_chars=50, min_words=10)
        assert result is not None
        assert s.kept == 1
        assert s.removed_total == 0

    def test_rejects_empty(self):
        s = FilterStats()
        result = filter_document("", s)
        assert result is None
        assert s.removed_empty == 1
        assert s.kept == 0

    def test_rejects_whitespace_only(self):
        s = FilterStats()
        result = filter_document("   \n\t  ", s)
        assert result is None
        assert s.removed_empty == 1

    def test_rejects_too_short_chars(self):
        s = FilterStats()
        result = filter_document("hi there", s, min_chars=200, min_words=5)
        assert result is None
        assert s.removed_too_short == 1

    def test_rejects_too_short_words(self):
        s = FilterStats()
        # Many chars but few distinct words
        result = filter_document("a" * 300, s, min_chars=200, min_words=30)
        assert result is None
        assert s.removed_too_short == 1

    def test_rejects_too_long(self):
        s = FilterStats()
        huge = "word " * 1000  # ~5000 chars
        result = filter_document(huge, s, min_chars=10, min_words=5, max_chars=100)
        assert result is None
        assert s.removed_too_long == 1

    def test_rejects_binary_garbled(self):
        s = FilterStats()
        garbled = "\ufffd" * 200 + "hello"
        result = filter_document(garbled, s, min_chars=10, min_words=1)
        assert result is None
        assert s.removed_binary == 1

    def test_cumulative_stats(self):
        s = FilterStats()
        filter_document("", s)                               # empty
        filter_document("hi", s, min_chars=200)              # too short
        filter_document(self._long_text(500), s, min_chars=50, min_words=10)  # kept
        assert s.removed_empty == 1
        assert s.removed_too_short == 1
        assert s.kept == 1
        assert s.total_input == 3

    def test_total_input_always_counted(self):
        s = FilterStats()
        filter_document("", s)
        filter_document("short", s, min_chars=1000)
        assert s.total_input == 2

    def test_returns_stripped_text(self):
        s = FilterStats()
        text = "  " + "hello world " * 50 + "  "
        result = filter_document(text, s, min_chars=10, min_words=5)
        assert result is not None
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_thresholds_configurable(self):
        s1, s2 = FilterStats(), FilterStats()
        text = "This is a short text."
        # With high min_chars: rejected
        assert filter_document(text, s1, min_chars=1000, min_words=5) is None
        # With low min_chars: kept
        assert filter_document(text, s2, min_chars=1, min_words=1) is not None
