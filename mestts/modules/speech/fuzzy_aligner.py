"""
modules/speech/fuzzy_aligner.py
─────────────────────────────────────────────────────────────────────────────
Lookahead fuzzy word alignment for ASR → sentence matching.

Maps ASR-recognized words to their positions in the expected sentence token
list using RapidFuzz Levenshtein similarity. A lookahead window allows
out-of-order matches within a limited range of the current position.

Architecture spec §Layer 3.2.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed; falling back to exact matching.")


class FuzzyAligner:
    """
    Stateful word aligner for one sentence.

    Maintains a current position pointer and a set of already-matched
    word indices. For each ASR word, searches within a lookahead window
    for the best-matching unmatched sentence word.

    Parameters
    ----------
    threshold : float
        Minimum fuzzy ratio (0–100) for a match to be accepted.
    lookahead_window : int
        Number of words ahead of current position to consider.
    """

    def __init__(
        self,
        threshold:        float = 75.0,
        lookahead_window: int   = 3,
    ) -> None:
        self._threshold  = threshold
        self._lookahead  = lookahead_window
        self._words:     List[str] = []
        self._matched:   Set[int]  = set()
        self._position:  int       = 0

    def reset(self, sentence_words: List[str]) -> None:
        """Reset aligner for a new sentence."""
        self._words    = [w.lower().strip(".,!?;:\"'") for w in sentence_words]
        self._matched  = set()
        self._position = 0

    def align(self, raw_word: str) -> Optional[Tuple[int, str, int]]:
        """
        Attempt to align `raw_word` to an unmatched sentence position.

        Parameters
        ----------
        raw_word : str
            ASR-recognized word (case-insensitive, strip punctuation).

        Returns
        -------
        (word_idx, matched_word, fuzz_score) or None if no match found.
        """
        if not self._words:
            return None

        query = raw_word.lower().strip(".,!?;:\"'")
        end   = min(len(self._words), self._position + self._lookahead + 1)

        best_score  = -1
        best_idx    = -1

        for idx in range(self._position, end):
            if idx in self._matched:
                continue
            candidate = self._words[idx]
            score     = self._score(query, candidate)
            if score > best_score:
                best_score = score
                best_idx   = idx

        if best_idx == -1 or best_score < self._threshold:
            return None

        self._matched.add(best_idx)
        if best_idx >= self._position:
            self._position = best_idx + 1

        return (best_idx, self._words[best_idx], int(best_score))

    @staticmethod
    def _score(a: str, b: str) -> float:
        """Compute fuzzy similarity (0–100) between two strings."""
        if _HAS_RAPIDFUZZ:
            return _fuzz.ratio(a, b)
        # Fallback: simple exact / startswith match
        if a == b:
            return 100.0
        if a.startswith(b) or b.startswith(a):
            return 80.0
        return 0.0
