"""
tests/unit/test_risk_scorer.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for analytics/risk_scorer.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics.feature_extractor import FeatureVector
from analytics.risk_scorer import RiskBand, RiskScorer


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


def _low_risk_fv() -> FeatureVector:
    """Feature vector that should produce a LOW risk score."""
    return FeatureVector(
        F01_evs_speech_std_ms       = 50.0,   # below 120 → I=0
        F04_speech_inversion_rate   = 0.05,   # below 0.25 → I=0
        F06_mean_fixation_duration_ms = 150.0, # below 300 → I=0
        F08_regression_rate         = 0.05,   # below 0.20 → I=0
        F12_mean_correction_rate    = 0.05,   # below 0.30 → I=0
        F14_stg_std_ms              = 80.0,   # below 200 → I=0
        F16_corr_evs_speech_fixation = 0.80,  # above 0.30 → I=0
    )


def _high_risk_fv() -> FeatureVector:
    """Feature vector that should produce a HIGH risk score."""
    return FeatureVector(
        F01_evs_speech_std_ms       = 250.0,  # above 120 → I=1
        F04_speech_inversion_rate   = 0.50,   # above 0.25 → I=1
        F06_mean_fixation_duration_ms = 450.0, # above 300 → I=1
        F08_regression_rate         = 0.40,   # above 0.20 → I=1
        F12_mean_correction_rate    = 0.60,   # above 0.30 → I=1
        F14_stg_std_ms              = 350.0,  # above 200 → I=1
        F16_corr_evs_speech_fixation = 0.10,  # below 0.30 → I=1
    )


class TestRiskScorer:

    def test_low_risk_band(self):
        scorer = _default_scorer()
        result = scorer.score(_low_risk_fv(), data_completeness=0.95)
        assert result.risk_score == 0.0
        assert result.risk_band  == RiskBand.LOW

    def test_high_risk_band(self):
        scorer = _default_scorer()
        result = scorer.score(_high_risk_fv(), data_completeness=0.95)
        assert result.risk_score == pytest.approx(1.0, abs=0.01)
        assert result.risk_band  in (RiskBand.HIGH, RiskBand.SEVERE_FLAG)

    def test_inconclusive_when_low_completeness(self):
        scorer = _default_scorer()
        result = scorer.score(_high_risk_fv(), data_completeness=0.40)
        assert result.risk_band == RiskBand.INCONCLUSIVE

    def test_all_indicators_present(self):
        scorer = _default_scorer()
        result = scorer.score(_low_risk_fv(), data_completeness=0.90)
        feature_ids = {ind.feature_id for ind in result.indicators}
        assert "F01" in feature_ids
        assert "F04" in feature_ids
        assert "F16" in feature_ids

    def test_missing_feature_treated_conservatively(self):
        """Missing feature should contribute 0 (not inflate risk)."""
        scorer = _default_scorer()
        fv = _high_risk_fv()
        fv.F01_evs_speech_std_ms = None  # Missing
        result_no_f01 = scorer.score(fv, data_completeness=0.90)
        result_full   = scorer.score(_high_risk_fv(), data_completeness=0.90)
        # Missing F01 should yield lower score (conservative)
        assert result_no_f01.risk_score < result_full.risk_score

    def test_disclaimer_present(self):
        scorer = _default_scorer()
        result = scorer.score(_low_risk_fv(), data_completeness=1.0)
        assert "behavioral risk indicator" in result.disclaimer
        assert "diagnosis" not in result.disclaimer.lower() or \
               "not" in result.disclaimer.lower()

    def test_weight_contributions_sum_to_risk_score(self):
        scorer = _default_scorer()
        result = scorer.score(_high_risk_fv(), data_completeness=1.0)
        total = sum(ind.contribution for ind in result.indicators)
        assert total == pytest.approx(result.risk_score, abs=0.001)

    def test_to_dict_includes_required_keys(self):
        scorer = _default_scorer()
        result = scorer.score(_low_risk_fv(), data_completeness=0.80)
        d = result.to_dict()
        assert "risk_score"        in d
        assert "risk_band"         in d
        assert "data_completeness" in d
        assert "indicators"        in d
        assert "disclaimer"        in d

    def test_moderate_risk_band(self):
        scorer = _default_scorer()
        fv = FeatureVector(
            F01_evs_speech_std_ms       = 150.0,  # above → I=1 (w=0.20)
            F04_speech_inversion_rate   = 0.10,   # below → I=0
            F06_mean_fixation_duration_ms = 250.0, # below → I=0
            F08_regression_rate         = 0.10,   # below → I=0
            F12_mean_correction_rate    = 0.10,   # below → I=0
            F14_stg_std_ms              = 100.0,  # below → I=0
            F16_corr_evs_speech_fixation = 0.60,  # above → I=0
        )
        result = scorer.score(fv, data_completeness=0.80)
        # Only F01 fires: RS = 0.20 → LOW/MODERATE boundary
        assert result.risk_score == pytest.approx(0.20, abs=0.01)

    def test_f16_inverted_direction(self):
        """F16 (correlation) should flag risk when value is BELOW threshold."""
        scorer = _default_scorer()
        fv_low_corr = _low_risk_fv()
        fv_low_corr.F16_corr_evs_speech_fixation = 0.10  # Below 0.30 → risk
        result = scorer.score(fv_low_corr, data_completeness=0.90)
        f16_ind = next(i for i in result.indicators if i.feature_id == "F16")
        assert f16_ind.indicator == 1
        assert f16_ind.direction  == "below"
