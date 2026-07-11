"""
tests/unit/test_fuzzy_aligner.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for modules/speech/fuzzy_aligner.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.speech.fuzzy_aligner import FuzzyAligner


SENTENCE = ["the", "quick", "brown", "fox", "jumps"]


class TestFuzzyAligner:

    def _aligner(self, threshold=75.0, lookahead=3) -> FuzzyAligner:
        a = FuzzyAligner(threshold=threshold, lookahead_window=lookahead)
        a.reset(SENTENCE)
        return a

    def test_exact_match_first_word(self):
        a = self._aligner()
        result = a.align("the")
        assert result is not None
        idx, word, score = result
        assert idx   == 0
        assert score == 100

    def test_exact_match_sequential(self):
        a = self._aligner()
        a.align("the")
        result = a.align("quick")
        assert result is not None
        assert result[0] == 1

    def test_no_match_below_threshold(self):
        a = self._aligner(threshold=75.0)
        # Completely unrelated word
        result = a.align("zzzzzz")
        assert result is None

    def test_case_insensitive(self):
        a = self._aligner()
        result = a.align("THE")
        assert result is not None
        assert result[0] == 0

    def test_punctuation_stripped(self):
        a = self._aligner()
        result = a.align("the,")
        assert result is not None
        assert result[0] == 0

    def test_already_matched_not_reused(self):
        a = self._aligner()
        a.align("the")
        # Second "the" should not re-match index 0
        # It should either match nothing or a future word
        result = a.align("the")
        if result is not None:
            assert result[0] != 0

    def test_lookahead_allows_skipping(self):
        a = self._aligner(lookahead=3)
        a.align("the")
        # Skip "quick" and match "brown" (within lookahead=3)
        result = a.align("brown")
        assert result is not None
        assert result[0] == 2

    def test_lookahead_prevents_far_skip(self):
        a = self._aligner(lookahead=1)
        a.align("the")
        # With lookahead=1, only "quick" (idx=1) is in range; "jumps" is idx=4
        result = a.align("jumps")
        # Should not match if it's outside lookahead
        if result is not None:
            assert result[0] <= 2

    def test_reset_clears_state(self):
        a = self._aligner()
        a.align("the")
        a.align("quick")
        a.reset(["one", "two", "three"])
        result = a.align("one")
        assert result is not None
        assert result[0] == 0

    def test_empty_sentence_returns_none(self):
        a = FuzzyAligner()
        a.reset([])
        assert a.align("hello") is None

    def test_full_sentence_sequential(self):
        a = self._aligner()
        expected_indices = list(range(len(SENTENCE)))
        matched = []
        for w in SENTENCE:
            r = a.align(w)
            if r:
                matched.append(r[0])
        assert matched == expected_indices
