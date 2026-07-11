"""
alignment/temporal_metrics.py
─────────────────────────────────────────────────────────────────────────────
Sentence-level temporal metric aggregation.

Operates on the finalized list of WordSlots produced by the synchronizer's
flush() method. Computes all statistical aggregates defined in §6 of the
architecture specification.

All metrics exclude slots with data_quality == EXCLUDED. Slots with
DEGRADED quality are included but flagged in the output.

Mathematical definitions (from spec §6):
  F_i  = fixation_onset_ns (camera-latency-corrected)
  P_i  = speech_timestamp_ns
  Y_i  = typing_timestamp_ns

  EVS_speech_i       = (P_i − F_i) / 1e6          [ms]
  EVS_typing_i       = (Y_i − F_i) / 1e6          [ms]
  SpeechTypingGap_i  = (Y_i − P_i) / 1e6          [ms]

  All values already computed per-slot; this module aggregates them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.word_slot import DataQuality, WordSlot

logger = logging.getLogger(__name__)


@dataclass
class SentenceMetrics:
    """
    Aggregated temporal metrics for one sentence.
    All duration values in milliseconds. None = insufficient data.
    """
    sentence_index:    int
    n_words:           int
    n_valid_slots:     int
    data_completeness: float    # n_complete / n_words

    # EVS Speech
    mean_evs_speech_ms:  Optional[float] = None
    std_evs_speech_ms:   Optional[float] = None
    skew_evs_speech:     Optional[float] = None
    speech_inversion_rate: Optional[float] = None   # Pr(EVS_speech < 0)

    # EVS Typing
    mean_evs_typing_ms:  Optional[float] = None
    std_evs_typing_ms:   Optional[float] = None
    typing_inversion_rate: Optional[float] = None   # Pr(EVS_typing < 0)

    # Speech-Typing Gap
    mean_stg_ms:  Optional[float] = None
    std_stg_ms:   Optional[float] = None
    stg_negative_rate: Optional[float] = None       # Pr(STG < 0)

    # Fixation
    mean_fixation_duration_ms: Optional[float] = None
    std_fixation_duration_ms:  Optional[float] = None
    fixation_outlier_ratio:    Optional[float] = None   # max(FD) / mean(FD)
    regression_count:          int = 0
    regression_rate:           float = 0.0

    # Typing
    mean_typing_latency_ms:  Optional[float] = None
    std_typing_latency_ms:   Optional[float] = None
    mean_correction_rate:    Optional[float] = None   # corrections / word
    max_corrections:         int = 0

    # Cross-modal correlations (Pearson)
    corr_evs_speech_fixation_duration: Optional[float] = None
    corr_evs_typing_latency:           Optional[float] = None

    # Risk score contribution (filled by RiskScorer)
    sentence_risk_score: Optional[float] = None


def compute_sentence_metrics(
    slots:          List[WordSlot],
    sentence_index: int,
) -> SentenceMetrics:
    """
    Compute all sentence-level temporal metrics from finalized WordSlots.

    Parameters
    ----------
    slots : list[WordSlot]
        Ordered list from SentenceSynchronizer.flush().
    sentence_index : int

    Returns
    -------
    SentenceMetrics
    """
    n_words = len(slots)
    if n_words == 0:
        return SentenceMetrics(
            sentence_index=sentence_index,
            n_words=0,
            n_valid_slots=0,
            data_completeness=0.0,
        )

    # Filter valid slots (not EXCLUDED)
    valid_slots = [
        s for s in slots
        if s.data_quality != DataQuality.EXCLUDED
        and s.temporal_metrics is not None
    ]
    n_valid = len(valid_slots)
    completeness = n_valid / n_words

    # ── Extract per-word series ───────────────────────────────────────────────

    evs_speech   = _collect(valid_slots, lambda s: s.temporal_metrics.evs_speech_ms)
    evs_typing   = _collect(valid_slots, lambda s: s.temporal_metrics.evs_typing_ms)
    stg          = _collect(valid_slots, lambda s: s.temporal_metrics.speech_typing_gap_ms)
    fix_dur      = _collect(valid_slots, lambda s: s.gaze_record.fixation_duration_ms
                             if s.gaze_record else None)
    typing_lat   = _collect(valid_slots, lambda s: s.typing_record.latency_ms
                             if s.typing_record else None)
    corrections  = [s.typing_record.correction_count
                    for s in valid_slots if s.typing_record]
    regressions  = [s for s in valid_slots
                    if s.gaze_record and s.gaze_record.regression_flag]

    m = SentenceMetrics(
        sentence_index=sentence_index,
        n_words=n_words,
        n_valid_slots=n_valid,
        data_completeness=completeness,
    )

    # ── EVS Speech ───────────────────────────────────────────────────────────
    if evs_speech:
        arr = np.array(evs_speech)
        m.mean_evs_speech_ms     = float(np.mean(arr))
        m.std_evs_speech_ms      = float(np.std(arr))
        m.skew_evs_speech        = _skewness(arr)
        m.speech_inversion_rate  = float(np.mean(arr < 0))

    # ── EVS Typing ───────────────────────────────────────────────────────────
    if evs_typing:
        arr = np.array(evs_typing)
        m.mean_evs_typing_ms     = float(np.mean(arr))
        m.std_evs_typing_ms      = float(np.std(arr))
        m.typing_inversion_rate  = float(np.mean(arr < 0))

    # ── Speech-Typing Gap ────────────────────────────────────────────────────
    if stg:
        arr = np.array(stg)
        m.mean_stg_ms        = float(np.mean(arr))
        m.std_stg_ms         = float(np.std(arr))
        m.stg_negative_rate  = float(np.mean(arr < 0))

    # ── Fixation ─────────────────────────────────────────────────────────────
    if fix_dur:
        arr = np.array(fix_dur)
        m.mean_fixation_duration_ms = float(np.mean(arr))
        m.std_fixation_duration_ms  = float(np.std(arr))
        if m.mean_fixation_duration_ms > 0:
            m.fixation_outlier_ratio = float(np.max(arr)) / m.mean_fixation_duration_ms

    m.regression_count = len(regressions)
    m.regression_rate  = len(regressions) / n_words if n_words > 0 else 0.0

    # ── Typing ───────────────────────────────────────────────────────────────
    if typing_lat:
        arr = np.array(typing_lat)
        m.mean_typing_latency_ms = float(np.mean(arr))
        m.std_typing_latency_ms  = float(np.std(arr))

    if corrections:
        m.mean_correction_rate = sum(corrections) / len(corrections)
        m.max_corrections      = max(corrections)

    # ── Cross-modal Correlations ──────────────────────────────────────────────
    # corr(EVS_speech, fixation_duration)
    paired_es_fd = [
        (s.temporal_metrics.evs_speech_ms, s.gaze_record.fixation_duration_ms)
        for s in valid_slots
        if s.temporal_metrics.evs_speech_ms is not None
        and s.gaze_record is not None
        and s.gaze_record.fixation_duration_ms is not None
    ]
    if len(paired_es_fd) >= 3:
        xs = np.array([p[0] for p in paired_es_fd])
        ys = np.array([p[1] for p in paired_es_fd])
        m.corr_evs_speech_fixation_duration = _safe_pearsonr(xs, ys)

    # corr(EVS_typing, typing_latency)
    paired_et_tl = [
        (s.temporal_metrics.evs_typing_ms, s.typing_record.latency_ms)
        for s in valid_slots
        if s.temporal_metrics.evs_typing_ms is not None
        and s.typing_record is not None
    ]
    if len(paired_et_tl) >= 3:
        xs = np.array([p[0] for p in paired_et_tl])
        ys = np.array([p[1] for p in paired_et_tl])
        m.corr_evs_typing_latency = _safe_pearsonr(xs, ys)

    logger.debug(
        "Sentence %d metrics: completeness=%.2f evs_speech_std=%.1f "
        "evs_typing_std=%.1f regression_rate=%.2f",
        sentence_index, completeness,
        m.std_evs_speech_ms or -1,
        m.std_evs_typing_ms or -1,
        m.regression_rate,
    )

    return m


# ─── Utility Functions ────────────────────────────────────────────────────────

def _collect(slots: List[WordSlot], extractor) -> List[float]:
    """Extract non-None float values from slots via extractor function."""
    result = []
    for s in slots:
        try:
            v = extractor(s)
            if v is not None and math.isfinite(v):
                result.append(v)
        except Exception:
            pass
    return result


def _skewness(arr: np.ndarray) -> float:
    """Pearson moment skewness of an array."""
    if len(arr) < 3:
        return 0.0
    mean = np.mean(arr)
    std  = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Compute Pearson correlation. Return None if computation fails
    (e.g., zero variance in either array).
    """
    try:
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        r = float(np.corrcoef(x, y)[0, 1])
        return r if math.isfinite(r) else None
    except Exception:
        return None
