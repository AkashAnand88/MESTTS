"""
tests/unit/test_temporal_metrics.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for alignment/temporal_metrics.py.
"""

import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alignment.temporal_metrics import SentenceMetrics, compute_sentence_metrics
from core.word_slot import (
    DataQuality,
    GazeRecord,
    SlotStatus,
    SpeechRecord,
    TemporalMetrics,
    TypingRecord,
    WordSlot,
)


def _make_slot(
    idx:          int,
    expected:     str,
    fixation_ns:  Optional[int] = None,
    speech_ns:    Optional[int] = None,
    typing_ns:    Optional[int] = None,
    fix_dur_ms:   float = 200.0,
    regression:   bool  = False,
    corrections:  int   = 0,
    typing_lat:   float = 300.0,
    excluded:     bool  = False,
) -> WordSlot:
    slot = WordSlot(word_index=idx, expected_word=expected)
    if excluded:
        slot.data_quality = DataQuality.EXCLUDED
        slot.status       = SlotStatus.TIMED_OUT
        return slot

    if fixation_ns is not None:
        slot.gaze_record = GazeRecord(
            word_index           = idx,
            fixation_onset_ns    = fixation_ns,
            fixation_duration_ms = fix_dur_ms,
            gaze_x_norm          = 0.5,
            gaze_y_norm          = 0.3,
            regression_flag      = regression,
            frame_id             = idx * 10,
        )
    if speech_ns is not None:
        slot.speech_record = SpeechRecord(
            word_index          = idx,
            speech_timestamp_ns = speech_ns,
            vosk_confidence     = 0.95,
            raw_word            = expected,
            matched_word        = expected,
            fuzzy_score         = 95,
        )
    if typing_ns is not None:
        slot.typing_record = TypingRecord(
            word_index          = idx,
            typing_timestamp_ns = typing_ns,
            latency_ms          = typing_lat,
            correction_count    = corrections,
            submitted_word      = expected,
            expected_word       = expected,
        )

    slot.compute_temporal_metrics()
    slot.resolve_status()
    return slot


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestComputeSentenceMetrics:

    def _make_full_sentence(self) -> List[WordSlot]:
        """5-word sentence with complete data."""
        base_ns = 1_000_000_000   # 1 second epoch
        slots = []
        for i in range(5):
            fix_ns    = base_ns + i * 500_000_000         # every 500ms
            speech_ns = fix_ns  + 800_000_000              # 800ms after fixation
            type_ns   = fix_ns  + 1_200_000_000            # 1200ms after fixation
            slots.append(_make_slot(
                idx         = i,
                expected    = f"word{i}",
                fixation_ns = fix_ns,
                speech_ns   = speech_ns,
                typing_ns   = type_ns,
                fix_dur_ms  = 200 + i * 20,
                corrections = i % 2,
            ))
        return slots

    def test_empty_slots_returns_zero_completeness(self):
        m = compute_sentence_metrics([], sentence_index=0)
        assert m.n_words == 0
        assert m.data_completeness == 0.0

    def test_completeness_with_all_valid(self):
        slots = self._make_full_sentence()
        m = compute_sentence_metrics(slots, sentence_index=1)
        assert m.n_words == 5
        assert m.n_valid_slots == 5
        assert m.data_completeness == 1.0

    def test_evs_speech_computed(self):
        slots = self._make_full_sentence()
        m = compute_sentence_metrics(slots, sentence_index=0)
        # EVS_speech = (speech_ns - fix_ns) / 1e6 = 800ms for all words
        assert m.mean_evs_speech_ms == pytest.approx(800.0, abs=1.0)
        assert m.std_evs_speech_ms  == pytest.approx(0.0,   abs=0.1)

    def test_evs_typing_computed(self):
        slots = self._make_full_sentence()
        m = compute_sentence_metrics(slots, sentence_index=0)
        assert m.mean_evs_typing_ms == pytest.approx(1200.0, abs=1.0)

    def test_stg_computed(self):
        slots = self._make_full_sentence()
        m = compute_sentence_metrics(slots, sentence_index=0)
        # STG = (typing_ns - speech_ns) / 1e6 = 400ms
        assert m.mean_stg_ms == pytest.approx(400.0, abs=1.0)

    def test_excluded_slots_do_not_count(self):
        slots = self._make_full_sentence()
        # Exclude 2 slots
        slots[0].data_quality = DataQuality.EXCLUDED
        slots[1].data_quality = DataQuality.EXCLUDED
        m = compute_sentence_metrics(slots, sentence_index=0)
        assert m.n_valid_slots == 3
        assert m.data_completeness == pytest.approx(3 / 5)

    def test_regression_count_and_rate(self):
        slots = self._make_full_sentence()
        # Mark 2 slots as regressions
        slots[1].gaze_record.regression_flag = True
        slots[3].gaze_record.regression_flag = True
        m = compute_sentence_metrics(slots, sentence_index=0)
        assert m.regression_count == 2
        assert m.regression_rate  == pytest.approx(2 / 5)

    def test_correction_rate(self):
        slots = self._make_full_sentence()
        # words 0,2,4 have 0 corrections; 1,3 have 1 each
        m = compute_sentence_metrics(slots, sentence_index=0)
        # mean corrections = (0+1+0+1+0) / 5 = 0.4
        assert m.mean_correction_rate == pytest.approx(0.4, abs=0.01)
        assert m.max_corrections == 1

    def test_inversion_rate_zero_when_positive_evs(self):
        # All EVS are positive → inversion rate should be 0
        slots = self._make_full_sentence()
        m = compute_sentence_metrics(slots, sentence_index=0)
        assert m.speech_inversion_rate == 0.0

    def test_inversion_rate_nonzero_when_negative_evs(self):
        base_ns = 1_000_000_000
        slots = []
        for i in range(4):
            fix_ns    = base_ns + i * 500_000_000
            # Speech BEFORE fixation (negative EVS)
            speech_ns = fix_ns - 200_000_000
            type_ns   = fix_ns + 500_000_000
            slots.append(_make_slot(i, f"w{i}", fix_ns, speech_ns, type_ns))
        m = compute_sentence_metrics(slots, sentence_index=0)
        assert m.speech_inversion_rate == 1.0

    def test_partial_data_no_speech(self):
        """Slots with only gaze + typing (no speech) still compute EVS_typing."""
        slots = []
        for i in range(3):
            fix_ns  = 1_000_000_000 + i * 300_000_000
            type_ns = fix_ns + 600_000_000
            slots.append(_make_slot(i, f"w{i}", fix_ns, None, type_ns))
        m = compute_sentence_metrics(slots, sentence_index=0)
        assert m.mean_evs_typing_ms is not None
        assert m.mean_evs_speech_ms is None   # No speech data

    def test_fixation_duration_stats(self):
        slots = self._make_full_sentence()
        m = compute_sentence_metrics(slots, sentence_index=0)
        # Durations: 200, 220, 240, 260, 280
        assert m.mean_fixation_duration_ms == pytest.approx(240.0, abs=0.1)
        assert m.std_fixation_duration_ms  > 0

    def test_sentence_index_preserved(self):
        slots = self._make_full_sentence()
        m = compute_sentence_metrics(slots, sentence_index=7)
        assert m.sentence_index == 7


class TestWordSlotMetrics:
    """Unit tests for WordSlot.compute_temporal_metrics()."""

    def test_evs_speech(self):
        slot = _make_slot(0, "the",
                          fixation_ns=1_000_000_000,
                          speech_ns  =1_800_000_000)
        assert slot.temporal_metrics.evs_speech_ms == pytest.approx(800.0)

    def test_evs_typing(self):
        slot = _make_slot(0, "the",
                          fixation_ns=1_000_000_000,
                          typing_ns  =2_200_000_000)
        assert slot.temporal_metrics.evs_typing_ms == pytest.approx(1200.0)

    def test_stg(self):
        slot = _make_slot(0, "the",
                          fixation_ns=1_000_000_000,
                          speech_ns  =1_500_000_000,
                          typing_ns  =2_000_000_000)
        assert slot.temporal_metrics.speech_typing_gap_ms == pytest.approx(500.0)

    def test_no_gaze_yields_no_metrics(self):
        slot = _make_slot(0, "the", speech_ns=1_000_000_000)
        assert slot.temporal_metrics.evs_speech_ms is None
        assert slot.temporal_metrics.evs_typing_ms is None

    def test_negative_evs_speech_is_valid(self):
        # Speech before fixation
        slot = _make_slot(0, "the",
                          fixation_ns=2_000_000_000,
                          speech_ns  =1_800_000_000)
        assert slot.temporal_metrics.evs_speech_ms == pytest.approx(-200.0)
