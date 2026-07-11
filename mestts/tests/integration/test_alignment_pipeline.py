"""
tests/integration/test_alignment_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for the alignment pipeline:
  WordSlot → SentenceMetrics → FeatureVector → RiskScore

These tests exercise the full data flow without hardware dependencies.
"""

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alignment.temporal_metrics import compute_sentence_metrics, SentenceMetrics
from analytics.feature_extractor import FeatureExtractor, FeatureVector
from analytics.risk_scorer import RiskBand, RiskScorer
from core.word_slot import (
    DataQuality, GazeRecord, SpeechRecord, TypingRecord, WordSlot
)


# ─── Fixture Builders ─────────────────────────────────────────────────────────

def _build_sentence(
    n_words:        int,
    base_evs_ms:    float = 800.0,
    regression_rate: float = 0.0,
    correction_rate: float = 0.0,
    inversion_rate:  float = 0.0,
    sentence_idx:   int   = 0,
) -> List[WordSlot]:
    """
    Build a synthetic sentence of `n_words` WordSlots with controlled metrics.

    Parameters
    ----------
    base_evs_ms : float
        EVS (speech lead) in milliseconds. Positive = eye leads voice.
    regression_rate : float
        Proportion of words with regression flag.
    correction_rate : float
        Mean backspace corrections per word.
    inversion_rate : float
        Proportion of words with negative EVS (voice leads eye).
    """
    slots = []
    base_ns = 1_000_000_000_000  # 1000 seconds

    for i in range(n_words):
        fix_ns = base_ns + i * 1_000_000_000  # 1 second per word

        # Apply inversion for some words
        if i < int(n_words * inversion_rate):
            speech_ns = fix_ns - abs(int(base_evs_ms * 1_000_000))  # voice leads
        else:
            speech_ns = fix_ns + int(base_evs_ms * 1_000_000)

        type_ns = speech_ns + 400_000_000  # 400ms after speech

        slot = WordSlot(word_index=i, expected_word=f"word{i}")
        slot.gaze_record = GazeRecord(
            word_index           = i,
            fixation_onset_ns    = fix_ns,
            fixation_duration_ms = 200.0 + i * 10,
            gaze_x_norm          = 0.4 + i * 0.05,
            gaze_y_norm          = 0.3,
            regression_flag      = (i < int(n_words * regression_rate)),
            frame_id             = i * 30,
        )
        slot.speech_record = SpeechRecord(
            word_index          = i,
            speech_timestamp_ns = speech_ns,
            vosk_confidence     = 0.92,
            raw_word            = f"word{i}",
            matched_word        = f"word{i}",
            fuzzy_score         = 95,
        )
        slot.typing_record = TypingRecord(
            word_index          = i,
            typing_timestamp_ns = type_ns,
            latency_ms          = 350.0,
            correction_count    = int(correction_rate),
            submitted_word      = f"word{i}",
            expected_word       = f"word{i}",
        )
        slot.compute_temporal_metrics()
        slot.resolve_status()
        slots.append(slot)

    return slots


def _default_scorer() -> RiskScorer:
    return RiskScorer(
        thresholds={
            "F01_evs_speech_std_ms":         120.0,
            "F04_speech_inversion_rate":      0.25,
            "F06_mean_fixation_duration_ms":  300.0,
            "F08_regression_rate":            0.20,
            "F12_mean_correction_rate":       0.30,
            "F14_stg_std_ms":                200.0,
            "F16_gaze_speech_corr":           0.30,
        },
        weights={
            "F01": 0.20,
            "F04": 0.15,
            "F06": 0.15,
            "F08": 0.15,
            "F12": 0.15,
            "F14": 0.10,
            "F16": 0.10,
        },
        risk_bands={
            "low_max":              0.20,
            "moderate_max":         0.45,
            "high_max":             0.70,
            "min_data_completeness": 0.60,
        },
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestAlignmentPipeline:

    def test_fluent_reader_produces_low_risk(self):
        """
        Simulate a fluent reader:
        - Low EVS variability (constant 800ms lead)
        - No regressions
        - No corrections
        - No inversions
        """
        sentences = [
            _build_sentence(10, base_evs_ms=800.0, sentence_idx=i+2)
            for i in range(8)
        ]
        all_metrics = [
            compute_sentence_metrics(s, sentence_index=i+2)
            for i, s in enumerate(sentences)
        ]

        fv = FeatureExtractor(practice_count=2).extract(all_metrics)
        result = _default_scorer().score(fv, data_completeness=1.0)

        assert result.risk_band in (RiskBand.LOW, RiskBand.MODERATE)
        assert result.risk_score < 0.50

    def test_struggling_reader_produces_high_risk(self):
        """
        Simulate a struggling reader:
        - High regression rate (50%)
        - High correction rate
        - High inversion rate (40%)
        - High EVS variability (varies strongly)
        """
        sentences = []
        for i in range(8):
            s = _build_sentence(
                10,
                base_evs_ms     = 200.0,
                regression_rate = 0.50,
                correction_rate = 1.0,
                inversion_rate  = 0.40,
                sentence_idx    = i + 2,
            )
            sentences.append(s)

        all_metrics = [
            compute_sentence_metrics(s, sentence_index=i+2)
            for i, s in enumerate(sentences)
        ]

        fv = FeatureExtractor(practice_count=2).extract(all_metrics)
        result = _default_scorer().score(fv, data_completeness=0.90)

        # At minimum: F04 (inversions), F08 (regressions), F12 (corrections) fire
        assert result.risk_score >= 0.30
        assert result.risk_band != RiskBand.LOW

    def test_practice_sentences_excluded(self):
        """Practice sentences must not affect the feature vector."""
        # 2 practice sentences with extreme data
        practice = [
            _build_sentence(10, regression_rate=0.90, sentence_idx=i)
            for i in range(2)
        ]
        # 6 scored sentences with clean data
        scored = [
            _build_sentence(10, regression_rate=0.0, sentence_idx=i+2)
            for i in range(6)
        ]

        all_sentences = practice + scored
        all_metrics = [
            compute_sentence_metrics(s, sentence_index=i)
            for i, s in enumerate(all_sentences)
        ]

        # With practice_count=2, only scored sentences contribute
        fv_with_practice_excluded = FeatureExtractor(practice_count=2).extract(all_metrics)
        fv_no_exclusion           = FeatureExtractor(practice_count=0).extract(all_metrics)

        # Regression rate should be near 0 when practice excluded
        assert (fv_with_practice_excluded.F08_regression_rate or 0.0) < 0.05
        # Without exclusion, practice contamination raises it
        assert (fv_no_exclusion.F08_regression_rate or 0.0) > 0.10

    def test_low_completeness_yields_inconclusive(self):
        sentences = [_build_sentence(10, sentence_idx=i+2) for i in range(5)]
        metrics = [compute_sentence_metrics(s, sentence_index=i+2)
                   for i, s in enumerate(sentences)]
        fv = FeatureExtractor(practice_count=2).extract(metrics)
        result = _default_scorer().score(fv, data_completeness=0.30)
        assert result.risk_band == RiskBand.INCONCLUSIVE

    def test_data_completeness_tracked_per_sentence(self):
        # 4 complete slots + 1 excluded → completeness = 0.8
        slots = _build_sentence(5, sentence_idx=2)
        slots[0].data_quality = DataQuality.EXCLUDED
        m = compute_sentence_metrics(slots, sentence_index=2)
        assert m.data_completeness == pytest.approx(0.8)

    def test_feature_vector_has_no_none_with_full_data(self):
        sentences = [_build_sentence(10, sentence_idx=i+2) for i in range(6)]
        metrics   = [compute_sentence_metrics(s, i+2) for i, s in enumerate(sentences)]
        fv        = FeatureExtractor(practice_count=2).extract(metrics)

        # Core EVS and regression features should be populated
        assert fv.F01_evs_speech_std_ms      is not None
        assert fv.F04_speech_inversion_rate  is not None
        assert fv.F08_regression_rate        is not None
        assert fv.F12_mean_correction_rate   is not None

    def test_n_missing_reflects_absent_features(self):
        # All fields None
        fv = FeatureVector()
        assert fv.n_missing() == 17  # all 17 fields None
