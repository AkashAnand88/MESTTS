"""
core/config.py
─────────────────────────────────────────────────────────────────────────────
Configuration loader for MESTTS.

Loads experiment_config.yaml and exposes typed attributes for all modules.
Unknown keys are accessible via _get() with a default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    raise ImportError("PyYAML required: pip install pyyaml")

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """
    Typed accessor for experiment_config.yaml.

    All parameters have documented defaults so the system can operate
    with a minimal config file during development / testing.
    """

    def __init__(self, config_path: str = "config/experiment_config.yaml") -> None:
        self._path = Path(config_path)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r") as f:
                self._data = yaml.safe_load(f) or {}
            logger.info("Config loaded from %s", self._path)
        else:
            logger.warning(
                "Config file not found: %s — using defaults.", self._path
            )

    def _get(self, *keys: str, default: Any = None) -> Any:
        """
        Hierarchical key lookup with default fallback.

        Example: self._get("vision", "target_fps", default=30)
        traverses self._data["vision"]["target_fps"].
        """
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, None)
            if node is None:
                return default
        return node

    # ── Protocol ─────────────────────────────────────────────────────────────

    @property
    def protocol_version(self) -> str:
        return self._get("protocol", "version", default="1.0")

    @property
    def protocol_path(self) -> str:
        return self._get("protocol", "sentence_file", default="config/sentence_protocol.json")

    @property
    def practice_sentences(self) -> int:
        return int(self._get("sentence_flow", "practice_sentences", default=2))

    # ── Vision ────────────────────────────────────────────────────────────────

    @property
    def ipcam_url(self) -> str:
        return self._get("vision", "ipcam_url", default="http://192.168.0.130:8080/video")

    @property
    def target_fps(self) -> int:
        return int(self._get("vision", "target_fps", default=30))

    @property
    def camera_latency_ns(self) -> int:
        return int(self._get("vision", "camera_latency_ms", default=50) * 1_000_000)

    @property
    def left_iris_indices(self) -> List[int]:
        return self._get("vision", "left_iris_indices", default=[468, 469, 470, 471])

    @property
    def right_iris_indices(self) -> List[int]:
        return self._get("vision", "right_iris_indices", default=[473, 474, 475, 476])

    @property
    def fixation_buffer_frames(self) -> int:
        return int(self._get("vision", "fixation_buffer_frames", default=3))

    @property
    def fixation_spatial_threshold(self) -> float:
        return float(self._get("vision", "fixation_spatial_threshold_px", default=40.0))

    @property
    def regression_delta(self) -> int:
        return int(self._get("vision", "regression_delta_words", default=1))

    @property
    def max_dropped_frame_pct(self) -> float:
        return float(self._get("vision", "max_dropped_frame_pct", default=0.20))

    @property
    def calibration_points(self) -> List[Tuple[float, float]]:
        raw = self._get(
            "vision", "calibration_points",
            default=[[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]],
        )
        return [tuple(p) for p in raw]

    @property
    def calibration_rmse_warn(self) -> float:
        return float(self._get("vision", "calibration_rmse_warn_px", default=50.0))

    # ── Speech ────────────────────────────────────────────────────────────────

    @property
    def vosk_model_path(self) -> str:
        return self._get("speech", "vosk_model_path", default="model/")

    @property
    def sample_rate(self) -> int:
        return int(self._get("speech", "sample_rate", default=16000))

    @property
    def chunk_size(self) -> int:
        return int(self._get("speech", "chunk_size", default=4096))

    @property
    def fuzzy_threshold(self) -> float:
        return float(self._get("speech", "fuzzy_threshold", default=75.0))

    @property
    def lookahead_window(self) -> int:
        return int(self._get("speech", "lookahead_window", default=3))

    # ── Timing & Session Flow ─────────────────────────────────────────────────

    @property
    def sentence_timeout_s(self) -> float:
        return float(self._get("sentence_flow", "timeout_s", default=60.0))

    @property
    def flush_grace_s(self) -> float:
        """
        Max seconds to wait, on sentence advance, for in-flight ASR
        results to land before flushing (and thereby excluding) any
        remaining pending speech slots. Keeps fast typers/readers from
        losing trailing-word speech data to the ASR pipeline's latency.
        """
        return float(self._get("sentence_flow", "flush_grace_s", default=0.8))

    # ── Scoring ───────────────────────────────────────────────────────────────

    @property
    def thresholds(self) -> Dict[str, float]:
        return self._get("scoring", "thresholds", default={
            "F01_evs_speech_std_ms":        120.0,
            "F04_speech_inversion_rate":     0.25,
            "F06_mean_fixation_duration_ms": 300.0,
            "F08_regression_rate":           0.20,
            "F12_mean_correction_rate":      0.30,
            "F14_stg_std_ms":               200.0,
            "F16_gaze_speech_corr":          0.30,
        })

    @property
    def weights(self) -> Dict[str, float]:
        return self._get("scoring", "weights", default={
            "F01": 0.20,
            "F04": 0.15,
            "F06": 0.15,
            "F08": 0.15,
            "F12": 0.15,
            "F14": 0.10,
            "F16": 0.10,
        })

    @property
    def risk_bands(self) -> Dict[str, float]:
        return self._get("scoring", "risk_bands", default={
            "low_max":              0.20,
            "moderate_max":         0.45,
            "high_max":             0.70,
            "min_data_completeness": 0.60,
        })

    # ── Persistence ───────────────────────────────────────────────────────────

    @property
    def db_path(self) -> str:
        return self._get("persistence", "db_path", default="data/sessions.db")

    @property
    def export_dir(self) -> str:
        return self._get("persistence", "export_dir", default="data/exports")

    @property
    def log_dir(self) -> str:
        return self._get("persistence", "log_dir", default="data/logs")

    @property
    def log_level(self) -> str:
        return self._get("persistence", "log_level", default="INFO")
