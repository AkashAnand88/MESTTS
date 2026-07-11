"""
analytics/risk_scorer.py
─────────────────────────────────────────────────────────────────────────────
Rule-based risk scoring module.

Implements the weighted indicator scoring formula defined in architecture
spec §8. All logic is fully transparent and interpretable.

Formula (from spec §8.3):
  RS = Σ_k (w_k × I_k)

where I_k ∈ {0, 1} is a binary indicator function comparing feature F_k
against its configured threshold θ_k, and w_k is the feature weight.

MANDATORY DISCLAIMER:
  This system produces a behavioral risk indicator only. It does not
  constitute a clinical diagnosis of dyslexia or any other condition.
  Results must be interpreted by a qualified specialist in the context
  of a comprehensive assessment protocol.

Configuration:
  All thresholds and weights are read from experiment_config.yaml.
  They must be validated against normative population data before use
  in any research publication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from analytics.feature_extractor import FeatureVector

logger = logging.getLogger(__name__)

SYSTEM_DISCLAIMER = (
    "IMPORTANT: This system produces a behavioral risk indicator only. "
    "It does not constitute a clinical diagnosis of dyslexia or any other "
    "condition. Results must be interpreted by a qualified specialist in the "
    "context of a comprehensive assessment protocol."
)


class RiskBand:
    LOW          = "LOW"
    MODERATE     = "MODERATE"
    HIGH         = "HIGH"
    SEVERE_FLAG  = "SEVERE_FLAG"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class IndicatorResult:
    """Result of evaluating one feature against its threshold."""
    feature_id:   str
    feature_value: Optional[float]
    threshold:    float
    indicator:    int              # 0 or 1
    weight:       float
    contribution: float            # weight × indicator
    direction:    str              # 'above' | 'below' (which direction flags risk)
    description:  str


@dataclass
class RiskScoreResult:
    """Complete risk scoring output for one session."""
    risk_score:      float
    risk_band:       str
    indicators:      List[IndicatorResult]
    data_completeness: float
    n_features_missing: int
    disclaimer:      str = field(default=SYSTEM_DISCLAIMER)

    def to_dict(self) -> dict:
        return {
            "risk_score":          self.risk_score,
            "risk_band":           self.risk_band,
            "data_completeness":   self.data_completeness,
            "n_features_missing":  self.n_features_missing,
            "indicators": [
                {
                    "feature_id":    ind.feature_id,
                    "feature_value": ind.feature_value,
                    "threshold":     ind.threshold,
                    "indicator":     ind.indicator,
                    "weight":        ind.weight,
                    "contribution":  ind.contribution,
                    "description":   ind.description,
                }
                for ind in self.indicators
            ],
            "disclaimer": self.disclaimer,
        }


class RiskScorer:
    """
    Computes the session-level risk score from a FeatureVector.

    Parameters (from experiment_config.yaml thresholds and weights sections)
    ----------
    thresholds : dict
        Mapping from feature_id to numeric threshold.
    weights : dict
        Mapping from feature_id to float weight (must sum to 1.0).
    risk_bands : dict
        Keys: low_max, moderate_max, high_max, min_data_completeness.
    """

    def __init__(
        self,
        thresholds: Dict[str, float],
        weights:    Dict[str, float],
        risk_bands: Dict[str, float],
    ) -> None:
        self._thresholds  = thresholds
        self._weights     = weights
        self._bands       = risk_bands
        self._validate_weights()

    def score(
        self,
        feature_vector:    FeatureVector,
        data_completeness: float,
    ) -> RiskScoreResult:
        """
        Compute risk score from the session feature vector.

        Parameters
        ----------
        feature_vector : FeatureVector
        data_completeness : float
            Mean sentence-level data completeness across scored sentences.

        Returns
        -------
        RiskScoreResult
        """
        indicators: List[IndicatorResult] = []

        # ── F01: EVS Speech Variability ───────────────────────────────────────
        indicators.append(self._eval_above(
            "F01",
            feature_vector.F01_evs_speech_std_ms,
            self._thresholds["F01_evs_speech_std_ms"],
            self._weights.get("F01", 0.20),
            "Eye-voice span (speech) standard deviation",
        ))

        # ── F04: Speech Inversion Rate ────────────────────────────────────────
        indicators.append(self._eval_above(
            "F04",
            feature_vector.F04_speech_inversion_rate,
            self._thresholds["F04_speech_inversion_rate"],
            self._weights.get("F04", 0.15),
            "Proportion of words where speech led fixation (inversions)",
        ))

        # ── F06: Mean Fixation Duration ───────────────────────────────────────
        indicators.append(self._eval_above(
            "F06",
            feature_vector.F06_mean_fixation_duration_ms,
            self._thresholds["F06_mean_fixation_duration_ms"],
            self._weights.get("F06", 0.15),
            "Mean fixation duration (prolonged fixations indicate decoding difficulty)",
        ))

        # ── F08: Regression Rate ──────────────────────────────────────────────
        indicators.append(self._eval_above(
            "F08",
            feature_vector.F08_regression_rate,
            self._thresholds["F08_regression_rate"],
            self._weights.get("F08", 0.15),
            "Saccadic regression rate (re-reading frequency)",
        ))

        # ── F12: Mean Correction Rate ─────────────────────────────────────────
        indicators.append(self._eval_above(
            "F12",
            feature_vector.F12_mean_correction_rate,
            self._thresholds["F12_mean_correction_rate"],
            self._weights.get("F12", 0.15),
            "Mean typing correction rate per word (spelling difficulty proxy)",
        ))

        # ── F14: STG Variability ──────────────────────────────────────────────
        indicators.append(self._eval_above(
            "F14",
            feature_vector.F14_stg_std_ms,
            self._thresholds["F14_stg_std_ms"],
            self._weights.get("F14", 0.10),
            "Speech-typing gap standard deviation (cross-modal timing instability)",
        ))

        # ── F16: Gaze-Speech Correlation ──────────────────────────────────────
        # NOTE: Inverted — LOW correlation flags risk
        indicators.append(self._eval_below(
            "F16",
            feature_vector.F16_corr_evs_speech_fixation,
            self._thresholds["F16_gaze_speech_corr"],
            self._weights.get("F16", 0.10),
            "Gaze-speech coupling correlation (low = decoupled reading modalities)",
        ))

        # ── Compute total risk score ──────────────────────────────────────────
        rs = sum(ind.contribution for ind in indicators)
        rs = round(min(1.0, max(0.0, rs)), 4)

        # ── Assign risk band ──────────────────────────────────────────────────
        min_completeness = self._bands.get("min_data_completeness", 0.60)

        if data_completeness < min_completeness:
            risk_band = RiskBand.INCONCLUSIVE
            logger.warning(
                "Risk band INCONCLUSIVE: data_completeness=%.2f < threshold=%.2f",
                data_completeness, min_completeness,
            )
        elif rs < self._bands["low_max"]:
            risk_band = RiskBand.LOW
        elif rs < self._bands["moderate_max"]:
            risk_band = RiskBand.MODERATE
        elif rs < self._bands["high_max"]:
            risk_band = RiskBand.HIGH
        else:
            risk_band = RiskBand.SEVERE_FLAG

        n_missing = feature_vector.n_missing()

        logger.info(
            "Risk score: RS=%.4f band=%s completeness=%.2f n_missing=%d",
            rs, risk_band, data_completeness, n_missing,
        )

        return RiskScoreResult(
            risk_score=rs,
            risk_band=risk_band,
            indicators=indicators,
            data_completeness=data_completeness,
            n_features_missing=n_missing,
        )

    # ── Indicator Evaluators ─────────────────────────────────────────────────

    @staticmethod
    def _eval_above(
        feature_id:    str,
        value:         Optional[float],
        threshold:     float,
        weight:        float,
        description:   str,
    ) -> IndicatorResult:
        """Flag I=1 if value > threshold (risk increases above threshold)."""
        if value is None:
            indicator = 0
            logger.debug("Feature %s: missing — indicator=0 (conservative).", feature_id)
        else:
            indicator = 1 if value > threshold else 0

        return IndicatorResult(
            feature_id=feature_id,
            feature_value=value,
            threshold=threshold,
            indicator=indicator,
            weight=weight,
            contribution=weight * indicator,
            direction="above",
            description=description,
        )

    @staticmethod
    def _eval_below(
        feature_id:    str,
        value:         Optional[float],
        threshold:     float,
        weight:        float,
        description:   str,
    ) -> IndicatorResult:
        """Flag I=1 if value < threshold (risk increases below threshold)."""
        if value is None:
            indicator = 0
        else:
            indicator = 1 if value < threshold else 0

        return IndicatorResult(
            feature_id=feature_id,
            feature_value=value,
            threshold=threshold,
            indicator=indicator,
            weight=weight,
            contribution=weight * indicator,
            direction="below",
            description=description,
        )

    def _validate_weights(self) -> None:
        total = sum(self._weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Risk scorer weights sum to %.4f (expected 1.000). "
                "Normalize weights in experiment_config.yaml.",
                total,
            )
