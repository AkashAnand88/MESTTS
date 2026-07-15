"""
analytics/feature_extractor.py
─────────────────────────────────────────────────────────────────────────────
FeatureExtractor — aggregates SentenceMetrics across a session into
the session-level FeatureVector used by RiskScorer.

Feature IDs F01–F17 correspond to the architecture spec §8.
Practice sentences (first N) are excluded from aggregation.

Architecture spec §Layer 5, §8.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional

import numpy as np

from alignment.temporal_metrics import SentenceMetrics

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """
    Session-level feature vector.

    All values are means across scored sentences. None = insufficient data.
    """
    # F01 — EVS Speech variability
    F01_evs_speech_std_ms:       Optional[float] = None

    # F02 — Mean EVS Speech
    F02_mean_evs_speech_ms:      Optional[float] = None

    # F03 — EVS skewness
    F03_evs_speech_skew:         Optional[float] = None

    # F04 — Speech inversion rate
    F04_speech_inversion_rate:   Optional[float] = None

    # F05 — EVS Typing variability
    F05_evs_typing_std_ms:       Optional[float] = None

    # F06 — Mean fixation duration
    F06_mean_fixation_duration_ms: Optional[float] = None

    # F07 — Fixation duration std (variability)
    F07_std_fixation_duration_ms:  Optional[float] = None

    # F08 — Regression rate
    F08_regression_rate:           Optional[float] = None

    # F09 — Regression count total
    F09_regression_count_total:    Optional[int]   = None

    # F10 — Fixation outlier ratio (max/mean)
    F10_fixation_outlier_ratio:    Optional[float] = None

    # F11 — Typing inversion rate
    F11_typing_inversion_rate:     Optional[float] = None

    # F12 — Mean correction rate
    F12_mean_correction_rate:      Optional[float] = None

    # F13 — Max corrections in any word
    F13_max_corrections:           Optional[int]   = None

    # F14 — Speech-Typing Gap std
    F14_stg_std_ms:                Optional[float] = None

    # F15 — STG negative rate
    F15_stg_negative_rate:         Optional[float] = None

    # F16 — Gaze-speech correlation (mean across sentences)
    F16_corr_evs_speech_fixation:  Optional[float] = None

    # F17 — EVS-typing correlation
    F17_corr_evs_typing_latency:   Optional[float] = None

    def n_missing(self) -> int:
        """Count of None feature values."""
        count = 0
        for f in fields(self):
            if getattr(self, f.name) is None:
                count += 1
        return count

    def to_dict(self) -> Dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Mean of non-None, finite values. Returns None if insufficient data."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _safe_sum_int(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


class FeatureExtractor:
    """
    Aggregates per-sentence SentenceMetrics into a session FeatureVector.

    Parameters
    ----------
    practice_count : int
        Number of practice sentences at the start of the session to skip.
    min_sentence_completeness : float
        Sentences with data_completeness below this are dropped from
        aggregation entirely, rather than silently pulled into the mean
        alongside fully-captured sentences. Prevents one poorly-captured
        sentence (e.g. trailing words lost to ASR latency) from smuggling
        outlier std/corr values into the session feature vector while
        the *session-level* completeness average still looks fine.
    """

    def __init__(
        self,
        practice_count: int = 0,
        min_sentence_completeness: float = 0.60,
    ) -> None:
        self._practice_count = practice_count
        self._min_sentence_completeness = min_sentence_completeness

    def extract(self, sentence_metrics: List[SentenceMetrics]) -> FeatureVector:
        """
        Compute the session-level feature vector.

        Parameters
        ----------
        sentence_metrics : list[SentenceMetrics]
            Ordered list, including practice sentences at indices [0, practice_count).

        Returns
        -------
        FeatureVector
        """
        eligible = [
            m for m in sentence_metrics
            if m.sentence_index >= self._practice_count
        ]

        scored = [
            m for m in eligible
            if m.data_completeness >= self._min_sentence_completeness
        ]

        n_dropped = len(eligible) - len(scored)
        if n_dropped:
            logger.warning(
                "FeatureExtractor: dropped %d/%d scored sentence(s) below "
                "min_sentence_completeness=%.2f (indices=%s) — excluded from "
                "feature aggregation to avoid outlier contamination.",
                n_dropped, len(eligible), self._min_sentence_completeness,
                [m.sentence_index for m in eligible
                 if m.data_completeness < self._min_sentence_completeness],
            )

        if not scored:
            logger.warning("FeatureExtractor: no scored sentences; returning empty vector.")
            return FeatureVector()

        fv = FeatureVector()

        fv.F01_evs_speech_std_ms       = _safe_mean([m.std_evs_speech_ms       for m in scored])
        fv.F02_mean_evs_speech_ms      = _safe_mean([m.mean_evs_speech_ms      for m in scored])
        fv.F03_evs_speech_skew         = _safe_mean([m.skew_evs_speech          for m in scored])
        fv.F04_speech_inversion_rate   = _safe_mean([m.speech_inversion_rate    for m in scored])
        fv.F05_evs_typing_std_ms       = _safe_mean([m.std_evs_typing_ms        for m in scored])
        fv.F06_mean_fixation_duration_ms = _safe_mean([m.mean_fixation_duration_ms for m in scored])
        fv.F07_std_fixation_duration_ms  = _safe_mean([m.std_fixation_duration_ms  for m in scored])
        fv.F08_regression_rate          = _safe_mean([m.regression_rate           for m in scored])
        fv.F09_regression_count_total   = _safe_sum_int([m.regression_count        for m in scored])
        fv.F10_fixation_outlier_ratio   = _safe_mean([m.fixation_outlier_ratio    for m in scored])
        fv.F11_typing_inversion_rate    = _safe_mean([m.typing_inversion_rate     for m in scored])
        fv.F12_mean_correction_rate     = _safe_mean([m.mean_correction_rate      for m in scored])
        fv.F13_max_corrections          = (
            max((m.max_corrections for m in scored), default=None)
        )
        fv.F14_stg_std_ms              = _safe_mean([m.std_stg_ms               for m in scored])
        fv.F15_stg_negative_rate       = _safe_mean([m.stg_negative_rate         for m in scored])
        fv.F16_corr_evs_speech_fixation = _safe_mean(
            [m.corr_evs_speech_fixation_duration for m in scored]
        )
        fv.F17_corr_evs_typing_latency  = _safe_mean(
            [m.corr_evs_typing_latency for m in scored]
        )

        logger.info(
            "FeatureExtractor: %d scored sentences, %d features missing.",
            len(scored), fv.n_missing(),
        )
        return fv
