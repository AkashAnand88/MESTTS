"""
modules/vision/fixation_detector.py
─────────────────────────────────────────────────────────────────────────────
Rolling-buffer fixation detector using the I-DT (dispersion-threshold)
algorithm, plus backward-saccade regression detection.

Architecture spec §4.6.
"""

from __future__ import annotations

import math
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GazeSample:
    ts_ns:    int
    screen_x: float
    screen_y: float
    word_idx: Optional[int]
    frame_id: int


class FixationDetector:
    """
    I-DT fixation detector with regression detection.

    Maintains a rolling buffer of gaze samples. When the spatial
    dispersion of the buffer falls below `spatial_threshold`, the
    current dwell is classified as a fixation on the modal word slot.

    Regressions are detected as a backward jump of ≥ `regression_delta`
    word positions from the current fixation to a previous one.

    Parameters
    ----------
    buffer_frames : int
        Minimum number of consecutive frames to confirm a fixation.
    spatial_threshold : float
        Max dispersion in screen pixels for fixation classification.
    regression_delta : int
        Min backward word-index step to flag as regression.
    """

    def __init__(
        self,
        buffer_frames:     int   = 3,
        spatial_threshold: float = 40.0,
        regression_delta:  int   = 1,
    ) -> None:
        self._min_frames       = buffer_frames
        self._spatial_thresh   = spatial_threshold
        self._regression_delta = regression_delta

        self._buffer: Deque[GazeSample] = deque()
        self._last_fixation_word_idx: Optional[int] = None
        self._current_fixation_onset_ns: Optional[int] = None
        self._current_fixation_word: Optional[int] = None

    def reset(self) -> None:
        """Reset state between sentences."""
        self._buffer.clear()
        self._last_fixation_word_idx     = None
        self._current_fixation_onset_ns  = None
        self._current_fixation_word      = None

    def update(
        self,
        screen_x: float,
        screen_y: float,
        word_idx: Optional[int],
        ts_ns:    int,
        frame_id: int,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process one gaze sample.

        Returns
        -------
        fixation_events : list[dict]
            Newly completed fixation records.
        regression_events : list[dict]
            Newly detected regression records.
        """
        sample = GazeSample(
            ts_ns    = ts_ns,
            screen_x = screen_x,
            screen_y = screen_y,
            word_idx = word_idx,
            frame_id = frame_id,
        )
        self._buffer.append(sample)

        fixation_events   = []
        regression_events = []

        if len(self._buffer) < self._min_frames:
            return fixation_events, regression_events

        # Check dispersion
        xs  = np.array([s.screen_x for s in self._buffer])
        ys  = np.array([s.screen_y for s in self._buffer])
        disp = math.sqrt((float(xs.max()) - float(xs.min())) ** 2 +
                         (float(ys.max()) - float(ys.min())) ** 2)

        if disp <= self._spatial_thresh and word_idx is not None:
            # Within fixation
            if self._current_fixation_onset_ns is None:
                self._current_fixation_onset_ns = self._buffer[0].ts_ns
                self._current_fixation_word     = word_idx
        else:
            # Fixation broken — emit if one was active
            if self._current_fixation_onset_ns is not None:
                onset_ns   = self._current_fixation_onset_ns
                offset_ns  = self._buffer[-2].ts_ns if len(self._buffer) >= 2 else ts_ns
                duration_ms = (offset_ns - onset_ns) / 1_000_000.0
                fx_word    = self._current_fixation_word

                if duration_ms > 0:
                    fixation_events.append({
                        "word_idx":          fx_word,
                        "onset_ns":          onset_ns,
                        "offset_ns":         offset_ns,
                        "duration_ms":       duration_ms,
                        "centroid_x":        float(np.mean(xs[:-1])) if len(xs) > 1 else screen_x,
                        "centroid_y":        float(np.mean(ys[:-1])) if len(ys) > 1 else screen_y,
                        "sample_count":      len(self._buffer) - 1,
                        "regression_flag":   False,
                    })

                    # Regression check
                    if (
                        self._last_fixation_word_idx is not None
                        and fx_word is not None
                        and self._last_fixation_word_idx - fx_word >= self._regression_delta
                    ):
                        regression_events.append({
                            "from_word_idx": self._last_fixation_word_idx,
                            "to_word_idx":   fx_word,
                            "saccade_ts_ns": offset_ns,
                            "amplitude_px":  float(
                                abs(self._last_fixation_word_idx - fx_word)
                            ),
                        })
                        logger.debug(
                            "Regression: word %d → word %d.",
                            self._last_fixation_word_idx, fx_word,
                        )

                    self._last_fixation_word_idx = fx_word

            self._current_fixation_onset_ns = None
            self._current_fixation_word     = None
            self._buffer.clear()

        return fixation_events, regression_events
