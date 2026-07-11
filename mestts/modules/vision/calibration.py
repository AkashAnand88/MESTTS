"""
modules/vision/calibration.py
─────────────────────────────────────────────────────────────────────────────
3-point gaze calibration using LinearRegression.

Maps raw iris pixel coordinates → screen pixel coordinates.
Calibration uses 3 reference points (left, center, right) along
the horizontal mid-line, each with multiple iris centroid samples.

Architecture spec §Layer 2.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a calibration fit attempt."""
    status: str          # 'ACCEPTED' | 'REJECTED'
    rmse: float
    n_samples: int


@dataclass
class CalibProfile:
    """Fitted calibration profile for one session."""
    session_id: str
    rmse: float
    valid: bool
    points: List[Dict]           # {'target': (sx,sy), 'mean_iris': (px,py)}
    model_x_coef: List[float]
    model_x_intercept: float
    model_y_coef: List[float]
    model_y_intercept: float

    def to_dict(self) -> dict:
        return {
            "session_id":        self.session_id,
            "rmse":              self.rmse,
            "valid":             self.valid,
            "points":            self.points,
            "model_x_coef":      self.model_x_coef,
            "model_x_intercept": self.model_x_intercept,
            "model_y_coef":      self.model_y_coef,
            "model_y_intercept": self.model_y_intercept,
        }


class GazeCalibrator:
    """
    3-point linear regression gaze calibrator.

    Usage:
        calibrator = GazeCalibrator(calibration_points)
        # During calibration sequence:
        calibrator.add_sample(point_idx=0, iris_x=412.3, iris_y=201.5)
        ...
        result = calibrator.fit()
        sx, sy = calibrator.map(iris_x, iris_y)

    Parameters
    ----------
    calibration_points : list of (norm_x, norm_y)
        Screen positions as normalized [0,1] fractions.
    screen_w, screen_h : int
        Screen dimensions in pixels.
    rmse_threshold : float
        RMSE (pixels) above which calibration is REJECTED.
    """

    def __init__(
        self,
        calibration_points:  List[Tuple[float, float]],
        screen_w:            int   = 1920,
        screen_h:            int   = 1080,
        rmse_threshold:      float = 50.0,
        session_id:          str   = "",
    ) -> None:
        self._points_norm    = calibration_points
        self._screen_w       = screen_w
        self._screen_h       = screen_h
        self._rmse_threshold = rmse_threshold
        self._session_id     = session_id

        # Raw iris sample collections per calibration point
        self._samples: List[List[Tuple[float, float]]] = [
            [] for _ in calibration_points
        ]

        self._model_x:  Optional[LinearRegression] = None
        self._model_y:  Optional[LinearRegression] = None
        self._profile:  Optional[CalibProfile]     = None
        self._fitted:   bool                       = False
        self._lock      = threading.Lock()

    # ── Sample collection ─────────────────────────────────────────────────────

    def add_sample(
        self,
        point_idx: int,
        iris_x: float,
        iris_y: float,
    ) -> None:
        """Add one iris centroid sample for calibration point `point_idx`."""
        with self._lock:
            if 0 <= point_idx < len(self._samples):
                self._samples[point_idx].append((iris_x, iris_y))

    def sample_count(self, point_idx: int) -> int:
        with self._lock:
            return len(self._samples[point_idx])

    def reset_samples(self) -> None:
        with self._lock:
            self._samples = [[] for _ in self._points_norm]
            self._fitted  = False
            self._profile = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self) -> CalibrationResult:
        """
        Fit LinearRegression models from collected samples.

        Requires at least 2 calibration points with ≥1 sample each.
        Returns CalibrationResult with status and RMSE.
        """
        with self._lock:
            # Build feature matrix and targets
            iris_xy: List[Tuple[float, float]] = []
            target_x: List[float] = []
            target_y: List[float] = []
            points_summary = []

            for idx, norm in enumerate(self._points_norm):
                sx = norm[0] * self._screen_w
                sy = norm[1] * self._screen_h
                samples = self._samples[idx]
                if not samples:
                    logger.warning("Calibration point %d has no samples.", idx)
                    continue
                mean_ix = float(np.mean([s[0] for s in samples]))
                mean_iy = float(np.mean([s[1] for s in samples]))
                iris_xy.append((mean_ix, mean_iy))
                target_x.append(sx)
                target_y.append(sy)
                points_summary.append({
                    "target": [sx, sy],
                    "mean_iris": [mean_ix, mean_iy],
                })

            n_points = len(iris_xy)
            if n_points < 2:
                raise ValueError(
                    f"Calibration requires ≥2 points with samples; got {n_points}."
                )

            X = np.array(iris_xy)    # shape (n, 2)
            yx = np.array(target_x)
            yy = np.array(target_y)

            self._model_x = LinearRegression().fit(X, yx)
            self._model_y = LinearRegression().fit(X, yy)

            # Compute RMSE on calibration points
            pred_x = self._model_x.predict(X)
            pred_y = self._model_y.predict(X)
            residuals = np.sqrt((pred_x - yx) ** 2 + (pred_y - yy) ** 2)
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

            valid  = rmse <= self._rmse_threshold
            status = "ACCEPTED" if valid else "REJECTED"

            self._profile = CalibProfile(
                session_id       = self._session_id,
                rmse             = rmse,
                valid            = valid,
                points           = points_summary,
                model_x_coef     = self._model_x.coef_.tolist(),
                model_x_intercept= float(self._model_x.intercept_),
                model_y_coef     = self._model_y.coef_.tolist(),
                model_y_intercept= float(self._model_y.intercept_),
            )
            self._fitted = True

        logger.info(
            "Calibration %s: RMSE=%.2fpx, %d points, threshold=%.0fpx.",
            status, rmse, n_points, self._rmse_threshold,
        )
        return CalibrationResult(status=status, rmse=rmse, n_samples=n_points)

    # ── Mapping ───────────────────────────────────────────────────────────────

    def map(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """
        Map raw iris coordinates to screen coordinates.

        Returns (screen_x, screen_y). Raises RuntimeError if not fitted.
        """
        if not self._fitted or self._model_x is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        X = np.array([[iris_x, iris_y]])
        sx = float(self._model_x.predict(X)[0])
        sy = float(self._model_y.predict(X)[0])
        # Clamp to screen bounds
        sx = max(0.0, min(float(self._screen_w), sx))
        sy = max(0.0, min(float(self._screen_h), sy))
        return sx, sy

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def profile(self) -> Optional[CalibProfile]:
        return self._profile

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save calibration profile to JSON."""
        if self._profile is None:
            logger.warning("No calibration profile to save.")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._profile.to_dict(), f, indent=2)
        logger.info("Calibration profile saved to %s.", path)

    def load(self, path: str) -> None:
        """Load a previously saved calibration profile."""
        data = json.loads(Path(path).read_text())
        coef_x  = np.array(data["model_x_coef"])
        inter_x = data["model_x_intercept"]
        coef_y  = np.array(data["model_y_coef"])
        inter_y = data["model_y_intercept"]

        self._model_x = LinearRegression()
        self._model_x.coef_      = coef_x
        self._model_x.intercept_ = inter_x

        self._model_y = LinearRegression()
        self._model_y.coef_      = coef_y
        self._model_y.intercept_ = inter_y

        self._profile = CalibProfile(**data)
        self._fitted  = True
        logger.info("Calibration profile loaded from %s (RMSE=%.2f).", path, data["rmse"])
