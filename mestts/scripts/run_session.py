#!/usr/bin/env python3
"""
scripts/run_session.py
----------------------
MESTTS multi-sentence reading session.

Gaze tracking, calibration, and audio logic taken verbatim from the verified
working script (main_camera_word.py).  The MESTTS persistence, risk-scoring,
and feature-extraction layers are wired on top identically to before.

Controls
--------
  C      – run / re-run gaze calibration
  ENTER  – advance to next sentence after reading
  R      – reset current sentence data
  ESC    – quit

Usage
-----
  python scripts/run_session.py --participant P001
"""

from __future__ import annotations

import argparse, io, json, logging, os, queue, sys, threading, time, uuid, warnings
from collections import deque
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import cv2
import mediapipe as mp
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout, QScrollArea, QSizePolicy,
    QFrame, QTabWidget, QLineEdit,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QPen

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from core.config import ExperimentConfig
from core.clock import get_clock
from core.event_bus import get_bus, Event, EventType
from alignment.synchronizer import SynchronizerThread
from alignment.temporal_metrics import compute_sentence_metrics
from analytics.feature_extractor import FeatureExtractor
from analytics.risk_scorer import RiskScorer
from modules.typing.hook import TypingHook
from persistence.db_manager import DatabaseManager
from persistence.export import ResearchExporter
from persistence.session_logger import SessionLogger
from persistence.export_report_pdf import SessionReportExporter


# ── logging ───────────────────────────────────────────────────────────────────
def _setup_logging(log_dir: str, level: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(Path(log_dir) / f"session_{stamp}.log", encoding="utf-8")
    sh = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if sys.platform == "win32" else sys.stdout
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(threadName)s] %(name)s %(levelname)s - %(message)s",
        handlers=[fh, sh],
    )

logger = logging.getLogger("run_session")


# ══════════════════════════════════════════════════════════════════════════════
# GAZE TRACKER — verbatim port of DiagnosticGazeTracker from main_camera_word.py
# ══════════════════════════════════════════════════════════════════════════════
class GazeTracker:
    LEFT_IRIS        = 468
    RIGHT_IRIS       = 473
    LEFT_EYE_OUTER   = 33
    RIGHT_EYE_OUTER  = 362
    NOSE_TIP         = 1

    GAZE_BUFFER_SIZE = 7
    STABILITY_FRAMES = 3
    CALIB_DURATION   = 3.5          # seconds per calibration point
    MIN_SAMPLES      = 20
    MIN_SPAN         = 0.05

    def __init__(self, num_words: int):
        self.num_words = num_words
        self.gaze_buffer              = deque(maxlen=self.GAZE_BUFFER_SIZE)
        self.head_yaw_buffer          = deque(maxlen=30)
        self.inter_eye_distance_buffer= deque(maxlen=30)

        self.calibration_data: Dict[str, List[float]] = {
            "left": [], "center": [], "right": []
        }
        self._is_calibrated   = False
        self.calibration_span = 0.0
        self.calibration_quality = 0.0
        self.gaze_to_screen_slope     = None
        self.gaze_to_screen_intercept = None

        self.word_boundaries = np.linspace(-1.0, 1.0, num_words + 1)
        self.temporal_stability_counter = 0
        self._last_smoothed = 0.0

    # --- estimate_normalized_gaze copied verbatim ---
    def estimate_normalized_gaze(self, landmarks) -> Optional[float]:
        try:
            left_iris  = landmarks[self.LEFT_IRIS]
            right_iris = landmarks[self.RIGHT_IRIS]
            iris_x = (left_iris.x + right_iris.x) / 2.0

            left_outer  = landmarks[self.LEFT_EYE_OUTER]
            right_outer = landmarks[self.RIGHT_EYE_OUTER]

            eye_center_x    = (left_outer.x + right_outer.x) / 2.0
            inter_eye_distance = abs(right_outer.x - left_outer.x)

            self.inter_eye_distance_buffer.append(inter_eye_distance)

            if inter_eye_distance < 0.05 or inter_eye_distance > 0.4:
                return None

            nose = landmarks[self.NOSE_TIP]
            head_yaw = (nose.x - eye_center_x) / inter_eye_distance
            self.head_yaw_buffer.append(abs(head_yaw))

            raw_gaze       = iris_x - eye_center_x
            normalized_gaze = raw_gaze / inter_eye_distance

            if abs(normalized_gaze) > 2.0:
                return None

            self.gaze_buffer.append(normalized_gaze)

            if len(self.gaze_buffer) >= 5:
                recent   = list(self.gaze_buffer)[-5:]
                smoothed = float(np.median(recent))
                variance = np.var(recent)
                if variance < 0.02:
                    self.temporal_stability_counter += 1
                else:
                    self.temporal_stability_counter = 0
            else:
                smoothed = normalized_gaze

            self._last_smoothed = smoothed
            return smoothed
        except Exception:
            return None

    # --- add_calibration_sample copied verbatim ---
    def add_calibration_sample(self, point_name: str, gaze_value: float):
        if point_name in self.calibration_data and gaze_value is not None:
            self.calibration_data[point_name].append(gaze_value)

    # --- finalize_calibration copied verbatim ---
    def finalize_calibration(self) -> bool:
        counts = {p: len(self.calibration_data[p]) for p in ["left", "center", "right"]}
        if any(c < self.MIN_SAMPLES for c in counts.values()):
            logger.warning("Insufficient calibration samples: %s", counts)
            return False

        def robust_median(samples):
            if len(samples) == 0:
                return 0.0
            arr = np.array(samples)
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            filtered = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
            return np.median(filtered) if len(filtered) > 0 else np.median(arr)

        left_gaze   = robust_median(self.calibration_data["left"])
        center_gaze = robust_median(self.calibration_data["center"])
        right_gaze  = robust_median(self.calibration_data["right"])

        logger.info("Calibration raw: left=%.4f center=%.4f right=%.4f",
                    left_gaze, center_gaze, right_gaze)

        self.calibration_span = abs(right_gaze - left_gaze)

        if self.calibration_span < self.MIN_SPAN:
            logger.warning("Low calibration span: %.4f (ideally > 0.15). "
                           "Move phone closer or exaggerate eye movement. Continuing anyway.",
                           self.calibration_span)

        span_score    = min(1.0, self.calibration_span / 0.4)
        midpoint      = (left_gaze + right_gaze) / 2
        offset        = abs(center_gaze - midpoint)
        balance_score = (
            max(0.0, 1.0 - (offset / (self.calibration_span * 0.5)))
            if self.calibration_span > 0 else 0.0
        )
        self.calibration_quality = span_score * 0.6 + balance_score * 0.4

        if self.calibration_quality < 0.25:
            logger.warning("Poor calibration quality: %.1f%%",
                           self.calibration_quality * 100)

        gaze_points   = np.array([left_gaze, center_gaze, right_gaze])
        screen_points = np.array([-1.0, 0.0, 1.0])
        self.gaze_to_screen_slope, self.gaze_to_screen_intercept = np.polyfit(
            gaze_points, screen_points, deg=1
        )

        self._is_calibrated = True
        logger.info("Calibration OK — span=%.4f quality=%.1f%%",
                    self.calibration_span, self.calibration_quality * 100)
        return True

    # --- get_live_diagnostics copied verbatim ---
    def get_live_diagnostics(self) -> str:
        if len(self.head_yaw_buffer) == 0:
            return "Waiting for face..."
        yaw  = np.mean(list(self.head_yaw_buffer)[-10:])
        dist = np.mean(list(self.inter_eye_distance_buffer)[-10:])
        parts = []
        if yaw > 0.25:
            parts.append("HEAD MOVING")
        elif yaw > 0.15:
            parts.append("Slight head movement")
        else:
            parts.append("Head stable")
        if dist < 0.12:
            parts.append("TOO FAR")
        elif dist > 0.25:
            parts.append("TOO CLOSE")
        else:
            parts.append("Distance OK")
        return " | ".join(parts)

    # --- map_gaze_to_screen copied verbatim ---
    def map_gaze_to_screen(self, gaze) -> Optional[float]:
        if not self._is_calibrated or gaze is None:
            return None
        return float(np.clip(
            self.gaze_to_screen_slope * gaze + self.gaze_to_screen_intercept,
            -1.5, 1.5
        ))

    # --- get_word_index copied verbatim ---
    def get_word_index(self, screen_x: float) -> Optional[int]:
        if screen_x is None or not self._is_calibrated:
            return None
        screen_x = float(np.clip(screen_x, -1.0, 1.0))
        boundaries = self.word_boundaries
        for i in range(self.num_words):
            if boundaries[i] <= screen_x < boundaries[i + 1]:
                return i
        return self.num_words - 1

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def is_temporally_stable(self) -> bool:
        return self.temporal_stability_counter >= self.STABILITY_FRAMES

    def reset(self):
        self.calibration_data = {"left": [], "center": [], "right": []}
        self._is_calibrated   = False
        self.calibration_span = 0.0
        self.calibration_quality = 0.0
        self.gaze_to_screen_slope     = None
        self.gaze_to_screen_intercept = None
        self.gaze_buffer.clear()
        self.temporal_stability_counter = 0
        self._last_smoothed = 0.0

    def set_num_words(self, n: int): 
        self.num_words       = n
        self.word_boundaries = np.linspace(-1.0, 1.0, n + 1)


# ══════════════════════════════════════════════════════════════════════════════
# WORD FIXATION TRACKER — verbatim port of WordFixationTracker
# ══════════════════════════════════════════════════════════════════════════════
class WordFixationTracker:
    FIXATION_MIN_MS = 150

    def __init__(self, words: List[str]):
        self.words             = words
        self.current_word_index: Optional[int] = None
        self.fixation_start:     Optional[float] = None
        self.logged_words:       Set[int] = set()
        self.fixation_stable_time: float  = 0.0
        self.min_duration: float = self.FIXATION_MIN_MS / 1000.0
        # ── Regression tracking ───────────────────────────────────────────────
        # last_logged_index: word_index of the most recently emitted fixation.
        # -1 = no fixation logged yet this sentence (sentinel, never a regression).
        # A regression is detected when current_word_index < last_logged_index,
        # i.e. the eye moved BACKWARD to an earlier word.
        self.last_logged_index: int = -1

    def update(self, word_index: Optional[int],
               gaze: GazeTracker) -> Optional[Dict]:
        if not gaze.is_calibrated():
            return None
        now = time.time()

        if word_index != self.current_word_index:
            # ── Regression detection at transition ────────────────────────────
            # If the new word index is LOWER than the last logged fixation,
            # this is a backward saccade (regression). Clear the word from
            # logged_words so it can be re-logged with regression_flag=True.
            if (word_index is not None
                    and self.last_logged_index >= 0
                    and word_index < self.last_logged_index):
                self.logged_words.discard(word_index)  # allow re-logging
            self.current_word_index   = word_index
            self.fixation_start       = now
            self.fixation_stable_time = 0.0
            return None

        if gaze.is_temporally_stable():
            self.fixation_stable_time = now - self.fixation_start

        if (self.current_word_index is not None
                and self.fixation_start is not None
                and (now - self.fixation_start) >= self.min_duration
                and self.current_word_index not in self.logged_words):

            self.logged_words.add(self.current_word_index)

            # Regression: eye moved to a word with a LOWER index than the last
            # logged fixation — i.e. a backward saccade.
            is_regression = (
                self.last_logged_index >= 0
                and self.current_word_index < self.last_logged_index
            )
            self.last_logged_index = self.current_word_index

            word_text = (
                self.words[self.current_word_index]
                if self.current_word_index < len(self.words) else "?"
            )
            if is_regression:
                logger.info(
                    "Regression: [%d] '%s' ← was at [%d], duration=%.3fs",
                    self.current_word_index, word_text,
                    self.last_logged_index, now - self.fixation_start,
                )
            else:
                logger.info(
                    "Fixation: [%d] '%s' duration=%.3fs",
                    self.current_word_index, word_text, now - self.fixation_start,
                )
            return {
                "word_index":        self.current_word_index,
                "word":              word_text,
                "timestamp":         self.fixation_start,
                "duration":          now - self.fixation_start,
                "temporal_stability":self.fixation_stable_time,
                "quality":           gaze.calibration_quality,
                "regression":        is_regression,   # ← NEW
            }
        return None

    def reset(self, new_words: Optional[List[str]] = None):
        if new_words is not None:
            self.words = new_words
        self.current_word_index   = None
        self.fixation_start       = None
        self.logged_words.clear()
        self.fixation_stable_time = 0.0
        self.last_logged_index    = -1   # ← reset regression sentinel


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO PROCESSOR — verbatim port of AudioProcessor + _match logic
# ══════════════════════════════════════════════════════════════════════════════
class AudioProcessor:
    VOSK_LAG_MS      = 100
    FUZZY_THRESHOLD  = 0.70
    LOOKAHEAD_WORDS  = 5

    def __init__(self, cfg: ExperimentConfig, words: List[str],
                 model_path_override: Optional[str] = None,
                 bus=None, clock=None):
        self.sample_rate = cfg.sample_rate
        self.block_size  = cfg.chunk_size
        self.words       = words
        self.words_lower = [w.lower() for w in words]
        self.processed: Set[str] = set()
        self.lag_comp    = self.VOSK_LAG_MS / 1000.0

        # shared state (lock-protected, mirrors SystemState in original)
        self.lock              = threading.Lock()
        self.voice_log: Dict[str, float] = {}
        self.current_word_index = 0
        self.running = False
        self.bus   = bus    # EventBus reference for publishing speech events
        self.clock = clock  # SessionClock for session-relative timestamps
        self.sentence_start_t = 0.0  # fence against cross-sentence pollution

        self._audio_q: queue.Queue = queue.Queue()

        model_path = model_path_override if model_path_override else cfg.vosk_model_path
        if not Path(model_path).exists():
            logger.error("Vosk model not found at %s", model_path)
            self.rec = None
            return
        try:
            model    = Model(model_path)
            self.rec = KaldiRecognizer(model, self.sample_rate)
            self.rec.SetWords(True)
            logger.info("Vosk model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Vosk load error: %s", exc)
            self.rec = None

    def start(self):
        if self.rec is None:
            logger.warning("Audio not started (Vosk unavailable).")
            return
        self.running = True
        threading.Thread(target=self._run, daemon=True, name="AudioProc").start()

    def stop(self):
        self.running = False

    def _audio_cb(self, indata, frames, time_info, status):
        self._audio_q.put(bytes(indata))

    def _run(self):
        try:
            device_info = sd.query_devices(kind='input')
            logger.info("Audio input device: %s (sr=%s)",
                        device_info.get('name','?'), device_info.get('default_samplerate','?'))
        except Exception as e:
            logger.warning("Could not query audio device: %s", e)
        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="int16",
                channels=1,
                callback=self._audio_cb,
            ):
                logger.info("Audio stream active.")
                while self.running:
                    try:
                        data = self._audio_q.get(timeout=0.5)
                        if not self.running:
                            break
                        self._process(data)
                    except queue.Empty:
                        continue
        except Exception as exc:
            logger.error("Audio error: %s", exc)

    # --- _process copied verbatim ---
    def _process(self, data):
        if self.rec.AcceptWaveform(data):
            result = json.loads(self.rec.Result())
            text = result.get("text", "")
            if text:
                logger.info("Vosk final: '%s'", text)
                self._match(text, time.time(), is_final=True)
        else:
            partial = json.loads(self.rec.PartialResult())
            text = partial.get("partial", "")
            if text:
                logger.info("Vosk partial: '%s'", text)
                self._match(text, time.time(), is_final=False)

    # --- _match copied verbatim ---
    def _match(self, text, t, is_final: bool = False):
        estimated_time = t - self.lag_comp
        # Ignore speech that was captured before the current sentence started.
        # This prevents previous-sentence Vosk partials from polluting new sentence.
        with self.lock:
            fence = self.sentence_start_t
        if estimated_time < fence - 1.5:  # 1.5s grace for Vosk processing lag
            return
        for spoken in text.lower().split():
            clean = "".join(c for c in spoken if c.isalnum())
            if not clean or clean in self.processed:
                continue
            with self.lock:
                idx = self.current_word_index
                # On final result: scan ALL remaining words (not just lookahead window)
                # so Vosk corrections can still match words it initially garbled
                scan_end = len(self.words) if is_final else min(idx + self.LOOKAHEAD_WORDS, len(self.words))
                for i in range(idx, scan_end):
                    ratio = SequenceMatcher(None, clean, self.words_lower[i]).ratio()
                    target = self.words_lower[i]
                    # Length-aware matching: short words require exact match to
                    # avoid false positives like "she"→"the", "and"→"around"
                    min_len = min(len(clean), len(target))
                    if min_len < 2:
                        is_match = (clean == target)
                    elif min_len <= 3:
                        is_match = (clean == target)   # exact only for short words
                    elif (target.startswith(clean) or clean.startswith(target)) and min_len >= 4:
                        is_match = True                # prefix match: "news"→"newspaper"
                    elif min_len <= 5:
                        is_match = (ratio >= 0.80)     # stricter for medium words
                    else:
                        is_match = (ratio >= self.FUZZY_THRESHOLD)  # 0.65 for long words
                    if is_match:
                        word = self.words[i]
                        if word not in self.voice_log:
                            self.voice_log[word]   = estimated_time
                            self.current_word_index = i + 1
                            self.processed.add(clean)
                            logger.info("Speech matched: '%s' (via '%s' ratio=%.2f) @ %.3fs",
                                        word, spoken, ratio, estimated_time)
                            # Publish to bus so synchronizer records it in the WordSlot
                            if self.bus is not None:
                                # Use session-relative clock if available,
                                # applying the same Vosk lag compensation in ns.
                                if self.clock is not None and self.clock.started:
                                    _lag_ns = int(self.lag_comp * 1e9)
                                    _ts_ns  = max(0, self.clock.now_ns() - _lag_ns)
                                else:
                                    _ts_ns = int(estimated_time * 1e9)
                                self.bus.publish(Event(
                                    event_type   = EventType.SPEECH_WORD,
                                    timestamp_ns = _ts_ns,
                                    payload      = {
                                        "word_idx":             i,
                                        "speech_timestamp_ns":  _ts_ns,
                                        "vosk_confidence":      ratio,
                                        "word_raw":             spoken,
                                        "matched_word":         word,
                                        "fuzzy_score":          ratio,
                                    },
                                ))
                        break

    def reset_sentence(self, new_words: List[str]):
        with self.lock:
            self.words              = new_words
            self.words_lower        = [w.lower() for w in new_words]
            self.voice_log.clear()
            self.current_word_index = 0
            self.processed.clear()
            self.sentence_start_t   = time.time()  # fence: ignore older speech
        # Drain pending audio queue to discard buffered audio from previous sentence
        drained = 0
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
                drained += 1
            except Exception:
                break
        if drained:
            logger.debug("Drained %d stale audio chunks on sentence reset.", drained)
        # Reset Vosk's internal audio accumulation state so it stops
        # transcribing previous-sentence context into the new sentence.
        if self.rec is not None:
            try:
                self.rec.Reset()
                # Feed a small block of silence so Vosk's decoder settles
                # before real audio for the next sentence arrives.
                silence = bytes(self.block_size * 2)  # int16 = 2 bytes/sample
                self.rec.AcceptWaveform(silence)
                logger.debug("Vosk recognizer reset for new sentence.")
            except Exception as e:
                logger.warning("Vosk Reset() failed: %s", e)



# ══════════════════════════════════════════════════════════════════════════════
# TYPING INPUT — QLineEdit with explicit key routing (no event propagation)
# ══════════════════════════════════════════════════════════════════════════════
class _TypingInput(QLineEdit):
    """Routes every key directly to a named session handler. Nothing propagates.

    SPACE        → insert space (textChanged fires → word submitted), consumed
    ENTER/Return → _handle_advance()
    C            → _handle_calibrate()
    R            → _handle_reset()
    Escape       → _handle_quit()
    anything else→ normal QLineEdit behaviour
    """
    def __init__(self, session_window, parent=None):
        super().__init__(parent)
        self._s = session_window

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            super().keyPressEvent(event)   # inserts space → textChanged fires
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._s._handle_advance()
            event.accept()
            return
        if key == Qt.Key_0:
            self._s._handle_calibrate()
            event.accept()
            return
        if key == Qt.Key_2:
            self._s._handle_reset()
            event.accept()
            return
        if key == Qt.Key_Escape:
            self._s._handle_quit()
            event.accept()
            return
        super().keyPressEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION DOT — copied verbatim from main_camera_word.py
# ══════════════════════════════════════════════════════════════════════════════
class CalibrationDot(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(70, 70)
        self.progress = 0.0
        self.active   = False

    def set_progress(self, p):
        self.progress = p
        self.update()

    def set_active(self, a):
        self.active = a
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.active:
            if self.progress > 0:
                painter.setPen(QPen(QColor(46, 204, 113), 5))
                painter.setBrush(Qt.NoBrush)
                painter.drawArc(5, 5, 60, 60, 90 * 16, -int(self.progress * 3.6 * 16))
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(17, 17, 36, 36)
            painter.setBrush(QColor(231, 76, 60))
            painter.drawEllipse(25, 25, 20, 20)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(31, 31, 8, 8)
        else:
            painter.setPen(QPen(QColor(149, 165, 166), 2))
            painter.setBrush(QColor(189, 195, 199, 100))
            painter.drawEllipse(23, 23, 24, 24)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD — shown after session completes
# Displays risk score, key indicators, and multi-variable visualizations
# ══════════════════════════════════════════════════════════════════════════════
class ResultsDashboard(QWidget):
    """
    Post-session analytics window.

    Shows:
      Tab 1 — Summary: Risk score card, key scalar indicators
               (EVS span, calibration quality, data completeness, feature vector)
      Tab 2 — Per-sentence chart: fixation coverage, speech coverage,
               data completeness, sentence duration across sentences
      Tab 3 — Feature vector heatmap / radar: all extracted features
    """

    DARK_BG   = "#1a1a1a"
    CARD_BG   = "#2d2d2d"
    ACCENT    = "#00d2d3"
    GREEN     = "#00b894"
    YELLOW    = "#fdcb6e"
    RED       = "#ff6348"
    BLUE      = "#74b9ff"
    TEXT      = "#f1f2f6"
    MPL_STYLE = {
        "axes.facecolor":  "#2d2d2d",
        "figure.facecolor":"#1a1a1a",
        "axes.edgecolor":  "#555555",
        "axes.labelcolor": "#f1f2f6",
        "xtick.color":     "#f1f2f6",
        "ytick.color":     "#f1f2f6",
        "text.color":      "#f1f2f6",
        "grid.color":      "#444444",
        "grid.linewidth":  0.5,
    }

    def __init__(self, participant_id: str, session_id: str,
                 risk, fv, sentence_metrics_list: list,
                 completeness: float, gaze_span: float, gaze_quality: float,
                 typing_summary: dict = None):
        super().__init__()
        self.participant_id       = participant_id
        self.session_id           = session_id
        self.risk                 = risk
        self.fv                   = fv          # feature vector dict/object
        self.sentence_metrics_list= sentence_metrics_list
        self.completeness         = completeness
        self.gaze_span            = gaze_span
        self.gaze_quality         = gaze_quality
        self.typing_summary       = typing_summary or {}  # per-sentence typing stats

        plt.rcParams.update(self.MPL_STYLE)

        self.setWindowTitle("MESTTS — Session Results")
        self.setGeometry(100, 80, 1300, 820)
        self.setStyleSheet(f"background-color: {self.DARK_BG};")
        self._build_ui()
        # Store data needed for PDF export
        self._pdf_exporter   = None   # set via set_exporter()
        self._export_dir     = ""

    def set_exporter(self, export_dir: str):
        from persistence.export_report_pdf import SessionReportExporter
        self._pdf_exporter = SessionReportExporter(export_dir)
        self._export_dir   = export_dir

    def _export_pdf(self):
            if self._pdf_exporter is None:
                from persistence.export_report_pdf import SessionReportExporter
                self._pdf_exporter = SessionReportExporter("data/exports")
            self._export_btn.setText("Exporting…")
            self._export_btn.setEnabled(False)
            try:
                path = self._pdf_exporter.export(
                    participant_id        = self.participant_id,
                    session_id            = self.session_id,
                    risk                  = self.risk,
                    fv                    = self.fv,
                    sentence_metrics_list = self.sentence_metrics_list,
                    completeness          = self.completeness,
                    gaze_span             = self.gaze_span,
                    gaze_quality          = self.gaze_quality,
                    typing_summary        = self.typing_summary,
                )
                self._export_btn.setText(f"Saved: {path.name}")
                self._export_btn.setStyleSheet(
                    "background-color: #00b894; color: #1a1a1a;"
                    "padding: 10px 30px; border-radius: 6px;"
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._export_btn.setText(f"Export failed: {exc}")
                self._export_btn.setStyleSheet(
                    "background-color: #ff6348; color: white;"
                    "padding: 10px 30px; border-radius: 6px;"
                )
            finally:
                self._export_btn.setEnabled(True)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _risk_color(self) -> str:
        band = getattr(self.risk, "risk_band", "").lower()
        if "low"       in band: return self.GREEN
        if "moderate"  in band: return self.YELLOW
        if "high"      in band: return self.RED
        return self.ACCENT

    def _card(self, label: str, value: str, color: str = None) -> QWidget:
        """Small metric card widget."""
        color = color or self.ACCENT
        w = QFrame()
        w.setStyleSheet(
            f"background-color: {self.CARD_BG}; border-radius: 10px;"
            f"border: 1px solid {color};"
        )
        vl = QVBoxLayout(w)
        vl.setContentsMargins(14, 10, 14, 10)
        vl.setSpacing(4)
        lbl = QLabel(label)
        lbl.setFont(QFont("Arial", 9))
        lbl.setStyleSheet(f"color: #aaaaaa; border: none;")
        lbl.setAlignment(Qt.AlignCenter)
        val = QLabel(value)
        val.setFont(QFont("Arial", 16, QFont.Bold))
        val.setStyleSheet(f"color: {color}; border: none;")
        val.setAlignment(Qt.AlignCenter)
        vl.addWidget(lbl)
        vl.addWidget(val)
        return w

    # ── UI build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # ── header ───────────────────────────────────────────────────────────
        hdr = QLabel(
            f"SESSION RESULTS  |  Participant: {self.participant_id}"
            f"  |  ID: {self.session_id[:8]}"
        )
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setFont(QFont("Arial", 13, QFont.Bold))
        hdr.setStyleSheet(
            f"color: {self.ACCENT}; background-color: {self.CARD_BG};"
            "padding: 12px; border-radius: 8px;"
        )
        root.addWidget(hdr)

        # ── tabs ─────────────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane  {{ background: {self.DARK_BG}; border: none; }}
            QTabBar::tab      {{ background: {self.CARD_BG}; color: #aaa;
                                 padding: 8px 20px; border-radius: 4px; }}
            QTabBar::tab:selected {{ background: {self.ACCENT}; color: #000; font-weight: bold; }}
        """)
        tabs.addTab(self._build_interpretation_tab(), "🧠  Interpretation")
        tabs.addTab(self._build_summary_tab(),         "📊  Summary")
        tabs.addTab(self._build_sentence_tab(),        "📈  Per-Sentence")
        tabs.addTab(self._build_typing_tab(),          "⌨️  Typing")
        tabs.addTab(self._build_feature_tab(),         "🔬  Feature Vector")
        root.addWidget(tabs)

        # ── close button ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
 
        self._export_btn = QPushButton("Export PDF Report")
        self._export_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self._export_btn.setStyleSheet(
            "background-color: #00b894; color: #1a1a1a;"
            "padding: 10px 30px; border-radius: 6px;"
        )
        self._export_btn.clicked.connect(self._export_pdf)
        btn_row.addWidget(self._export_btn)
 
        btn = QPushButton("Close  (ESC)")
        btn.setFont(QFont("Arial", 11, QFont.Bold))
        btn.setStyleSheet(
            "background-color: #636e72; color: white;"
            "padding: 10px 30px; border-radius: 6px;"
        )
        btn.clicked.connect(self.close)
        btn_row.addWidget(btn)
        root.addLayout(btn_row)


    # ── Tab 0: Plain-language Interpretation ─────────────────────────────────
    def _interpret_risk(self) -> Tuple[str, str, str]:
        """Returns (headline, body, color) for the overall risk narrative."""
        band  = getattr(self.risk, "risk_band", "").lower()
        score = getattr(self.risk, "risk_score", 0.0)
        if "low" in band:
            headline = "Low likelihood of dyslexia-related reading difficulties detected."
            body = (
                f"The participant's reading behaviour across all sentences fell largely within "
                f"typical ranges (risk score {score:.3f}). Eye movement patterns, speech timing, "
                f"and gaze-speech coordination were consistent with fluent reading. "
                f"No significant markers of phonological decoding difficulty were observed."
            )
            color = self.GREEN
        elif "moderate" in band:
            headline = "Some markers of reading difficulty were detected — further assessment recommended."
            body = (
                f"The participant's score of {score:.3f} places them in the moderate-risk band. "
                f"This means certain reading behaviours — such as irregular fixation timing, "
                f"gaze-speech misalignment, or variable reading pace — appeared more often "
                f"than typical. This is not a diagnosis, but suggests a follow-up clinical "
                f"assessment would be worthwhile."
            )
            color = self.YELLOW
        elif "high" in band:
            headline = "Significant markers of reading difficulty detected — clinical follow-up strongly advised."
            body = (
                f"The participant's score of {score:.3f} is in the high-risk band. Multiple "
                f"behavioural indicators — including atypical fixation patterns, gaze-speech "
                f"timing mismatches, and/or frequent correction attempts — were present across "
                f"sentences. A qualified specialist should evaluate these results in the context "
                f"of a full assessment."
            )
            color = self.RED
        else:
            headline = f"Risk band: {band or 'Unknown'}  (score: {score:.3f})"
            body     = "Insufficient data to provide a detailed interpretation."
            color    = self.ACCENT
        return headline, body, color

    def _interpret_evs_span(self) -> Tuple[str, str, str]:
        span = self.gaze_span
        if span > 0.15:
            return (
                "EVS Gaze Span — GOOD",
                f"Span of {span:.4f} indicates the eye-tracker captured a wide, reliable sweep "
                f"across the sentence. The calibration covered left-to-right movement well, "
                f"making gaze-to-word mapping accurate.",
                self.GREEN
            )
        elif span > 0.10:
            return (
                "EVS Gaze Span — ACCEPTABLE",
                f"Span of {span:.4f} is workable but on the lower side. Minor inaccuracies in "
                f"which word the participant was looking at are possible. Results should be "
                f"interpreted with slight caution.",
                self.YELLOW
            )
        else:
            return (
                "EVS Gaze Span — LOW",
                f"Span of {span:.4f} is below the recommended threshold (0.10). The tracker "
                f"could not clearly distinguish gaze position across the full sentence width. "
                f"Consider re-running calibration with more exaggerated left/right eye movement, "
                f"or moving the camera closer.",
                self.RED
            )

    def _interpret_calib_quality(self) -> Tuple[str, str, str]:
        q = self.gaze_quality
        if q >= 0.70:
            return (
                "Calibration Quality — EXCELLENT",
                f"{q:.0%} quality means the three calibration points (left, center, right) were "
                f"well-distributed and consistent. Gaze data is reliable.",
                self.GREEN
            )
        elif q >= 0.40:
            return (
                "Calibration Quality — MODERATE",
                f"{q:.0%} quality suggests the calibration was acceptable but not ideal — "
                f"possibly due to slight head movement or asymmetric gaze during calibration. "
                f"Word-level mapping may have minor errors.",
                self.YELLOW
            )
        else:
            return (
                "Calibration Quality — POOR",
                f"{q:.0%} quality is below acceptable levels. The calibration points may not have "
                f"been captured cleanly. This can inflate or deflate fixation counts and reduce "
                f"the reliability of all gaze-based features. Re-calibration is strongly advised "
                f"before trusting these results.",
                self.RED
            )

    def _interpret_completeness(self) -> Tuple[str, str, str]:
        c = self.completeness
        if c >= 0.80:
            return (
                "Data Completeness — HIGH",
                f"{c:.0%} of word slots had both gaze and speech data successfully captured. "
                f"The session data is highly reliable and the risk score is based on a full picture "
                f"of the participant's reading.",
                self.GREEN
            )
        elif c >= 0.60:
            return (
                "Data Completeness — ADEQUATE",
                f"{c:.0%} completeness means most words were captured, but some slots had only "
                f"partial data (gaze without speech, or vice versa). The risk score is still valid "
                f"but may miss some nuance.",
                self.YELLOW
            )
        else:
            return (
                "Data Completeness — LOW",
                f"{c:.0%} completeness is below the minimum threshold (60%). A large portion of "
                f"word slots lacked matched gaze+speech data, which means the risk score may not "
                f"accurately reflect the participant's reading profile. Consider repeating the session.",
                self.RED
            )

    def _interpret_features(self) -> List[Tuple[str, str, str]]:
        """Returns list of (label, explanation, color) for key feature vector entries."""
        fv = self.fv
        items = []

        def _g(name, default=0.0):
            if hasattr(fv, name):
                return getattr(fv, name) or default
            if isinstance(fv, dict):
                return fv.get(name, default) or default
            return default

        # F02 — Mean EVS (gaze leads speech)
        f02 = _g("F02_mean_evs_speech_ms")
        if f02 > 0:
            evs_interp = (
                f"The eye typically looked ahead of spoken words by ~{f02:.0f} ms. "
                f"This is normal — fluent readers pre-scan upcoming words. "
                f"A larger span suggests confident preview; near-zero or negative may indicate decoding difficulty."
            )
            evs_color = self.GREEN if 400 < f02 < 2500 else self.YELLOW
        else:
            evs_interp = "Could not compute mean EVS — insufficient matched gaze+speech pairs."
            evs_color  = self.RED
        items.append(("Eye-Voice Span (F02)", evs_interp, evs_color))

        # F01 — EVS std
        f01 = _g("F01_evs_speech_std_ms")
        if f01 < 300:
            std_interp = f"EVS variability was low ({f01:.0f} ms std). The participant read at a steady, consistent pace."
            std_color  = self.GREEN
        elif f01 < 600:
            std_interp = f"EVS variability was moderate ({f01:.0f} ms std). Some inconsistency in reading pace — common in borderline cases."
            std_color  = self.YELLOW
        else:
            std_interp = (
                f"EVS variability was high ({f01:.0f} ms std). Erratic pacing — the participant's gaze "
                f"and speech were poorly synchronised across words. This is a notable dyslexia marker."
            )
            std_color  = self.RED
        items.append(("EVS Consistency (F01)", std_interp, std_color))

        # F06 — Mean fixation duration
        f06 = _g("F06_mean_fixation_duration_ms")
        if f06 < 120:
            fix_interp = f"Mean fixation of {f06:.0f} ms is very short — fast, skimming-style reading."
            fix_color  = self.YELLOW
        elif f06 < 280:
            fix_interp = (
                f"Mean fixation of {f06:.0f} ms is in the typical range (150–280 ms). "
                f"The participant paused a normal amount of time on each word."
            )
            fix_color  = self.GREEN
        else:
            fix_interp = (
                f"Mean fixation of {f06:.0f} ms is longer than typical. Extended fixations "
                f"suggest the participant spent more time decoding individual words — a potential "
                f"sign of word-level reading difficulty."
            )
            fix_color  = self.RED
        items.append(("Mean Fixation Duration (F06)", fix_interp, fix_color))

        # F08 — Regression rate
        f08 = _g("F08_regression_rate")
        if f08 < 0.05:
            reg_interp = f"Regression rate was very low ({f08:.1%}). The participant rarely re-read words — forward reading flow was smooth."
            reg_color  = self.GREEN
        elif f08 < 0.20:
            reg_interp = f"Regression rate of {f08:.1%} is mild. Some re-reading occurred, which can be normal in careful readers."
            reg_color  = self.YELLOW
        else:
            reg_interp = (
                f"Regression rate of {f08:.1%} is elevated. Frequent backward eye movements "
                f"suggest the participant often lost their place or had difficulty decoding, "
                f"and needed to re-read — a key indicator in dyslexia screening."
            )
            reg_color  = self.RED
        items.append(("Regression Rate (F08)", reg_interp, reg_color))

        # F10 — Fixation outlier ratio
        f10 = _g("F10_fixation_outlier_ratio")
        if f10 < 0.3:
            out_interp = f"Fixation outlier ratio of {f10:.3f} is low — most fixations were of normal duration. Consistent reading behaviour."
            out_color  = self.GREEN
        elif f10 < 0.7:
            out_interp = f"Fixation outlier ratio of {f10:.3f} is moderate — some fixations were unusually short or long, indicating variable attention or decoding."
            out_color  = self.YELLOW
        else:
            out_interp = (
                f"Fixation outlier ratio of {f10:.3f} is high. A large proportion of fixations "
                f"were atypically short or long, pointing to inconsistent visual processing across words."
            )
            out_color  = self.RED
        items.append(("Fixation Outlier Ratio (F10)", out_interp, out_color))

        # F16 — Gaze-speech correlation
        f16 = _g("F16_corr_evs_speech_fixation")
        if f16 > 0.4:
            cor_interp = (
                f"Gaze-speech correlation of {f16:.3f} is strong. Eye movements and speech were "
                f"well-coordinated — the participant looked at words close to when they spoke them."
            )
            cor_color  = self.GREEN
        elif f16 > 0.1:
            cor_interp = f"Gaze-speech correlation of {f16:.3f} is weak. Eye and voice were loosely linked — moderate coordination difficulty."
            cor_color  = self.YELLOW
        else:
            cor_interp = (
                f"Gaze-speech correlation of {f16:.3f} is very low or negative. The participant's "
                f"eye movements and speech were poorly synchronised — they often spoke words "
                f"without clearly fixating on them, or vice versa."
            )
            cor_color  = self.RED
        items.append(("Gaze-Speech Correlation (F16)", cor_interp, cor_color))

        return items

    def _build_interpretation_tab(self) -> QWidget:
        outer  = QWidget()
        outer_vl = QVBoxLayout(outer)
        outer_vl.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        outer_vl.addWidget(scroll)

        content = QWidget()
        vl = QVBoxLayout(content)
        vl.setContentsMargins(18, 18, 18, 18)
        vl.setSpacing(14)
        scroll.setWidget(content)

        def _section_title(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setFont(QFont("Arial", 12, QFont.Bold))
            lbl.setStyleSheet(f"color: {self.ACCENT}; padding: 4px 0;")
            return lbl

        def _interp_block(title: str, body: str, color: str) -> QFrame:
            f = QFrame()
            f.setStyleSheet(
                f"background-color: {self.CARD_BG}; border-radius: 10px;"
                f"border-left: 4px solid {color};"
            )
            fl = QVBoxLayout(f)
            fl.setContentsMargins(16, 12, 16, 12)
            fl.setSpacing(6)

            t = QLabel(title)
            t.setFont(QFont("Arial", 10, QFont.Bold))
            t.setStyleSheet(f"color: {color}; border: none;")
            fl.addWidget(t)

            b = QLabel(body)
            b.setFont(QFont("Arial", 10))
            b.setStyleSheet(f"color: {self.TEXT}; border: none;")
            b.setWordWrap(True)
            fl.addWidget(b)
            return f

        # ── Section 1: Overall Verdict ────────────────────────────────────────
        vl.addWidget(_section_title("Overall Assessment"))
        headline, body, color = self._interpret_risk()
        hero = QFrame()
        hero.setStyleSheet(
            f"background-color: {self.CARD_BG}; border-radius: 12px;"
            f"border: 2px solid {color};"
        )
        hero_vl = QVBoxLayout(hero)
        hero_vl.setContentsMargins(20, 16, 20, 16)
        hero_vl.setSpacing(8)
        hl = QLabel(headline)
        hl.setFont(QFont("Arial", 12, QFont.Bold))
        hl.setStyleSheet(f"color: {color}; border: none;")
        hl.setWordWrap(True)
        hero_vl.addWidget(hl)
        bl = QLabel(body)
        bl.setFont(QFont("Arial", 10))
        bl.setStyleSheet(f"color: {self.TEXT}; border: none;")
        bl.setWordWrap(True)
        hero_vl.addWidget(bl)
        vl.addWidget(hero)

        # ── Section 2: Data Quality ───────────────────────────────────────────
        vl.addWidget(_section_title("Data Quality"))
        dq_note = QLabel(
            "Before interpreting clinical indicators, it is important to understand how reliable "
            "the session data was. Poor calibration or low completeness can reduce confidence in the risk score."
        )
        dq_note.setFont(QFont("Arial", 9))
        dq_note.setStyleSheet("color: #888888;")
        dq_note.setWordWrap(True)
        vl.addWidget(dq_note)

        for title, body, color in [
            self._interpret_evs_span(),
            self._interpret_calib_quality(),
            self._interpret_completeness(),
        ]:
            vl.addWidget(_interp_block(title, body, color))

        # ── Section 3: Key Reading Behaviour Indicators ───────────────────────
        vl.addWidget(_section_title("Key Reading Behaviour Indicators"))
        rb_note = QLabel(
            "These are the primary behavioural signals that contribute to the risk score. "
            "Each is derived from the synchronisation of eye movements and speech during reading."
        )
        rb_note.setFont(QFont("Arial", 9))
        rb_note.setStyleSheet("color: #888888;")
        rb_note.setWordWrap(True)
        vl.addWidget(rb_note)

        for title, body, color in self._interpret_features():
            vl.addWidget(_interp_block(title, body, color))

        # ── Section 4: What to Do Next ────────────────────────────────────────
        vl.addWidget(_section_title("What to Do Next"))
        band = getattr(self.risk, "risk_band", "").lower()
        if "low" in band:
            next_steps = (
                "• No immediate action required based on these results.\n"
                "• If the participant or parent/guardian has ongoing concerns, a routine reading "
                "assessment with a specialist is always a valid step.\n"
                "• Consider repeating the session in 3–6 months if difficulties re-emerge."
            )
        elif "moderate" in band:
            next_steps = (
                "• Share these results with a qualified educational psychologist or specialist "
                "reading teacher.\n"
                "• A formal psychoeducational or phonological assessment is recommended to clarify "
                "whether intervention is needed.\n"
                "• Interim classroom supports (extended time, audio aids) may be worth exploring "
                "while awaiting formal assessment."
            )
        else:
            next_steps = (
                "• These results warrant prompt referral to a qualified specialist — educational "
                "psychologist, speech-language therapist, or dyslexia specialist.\n"
                "• Do not rely on this tool alone for diagnosis — a comprehensive assessment "
                "covering phonological awareness, working memory, and reading fluency is essential.\n"
                "• Early structured literacy intervention significantly improves outcomes."
            )
        ns_frame = QFrame()
        ns_frame.setStyleSheet(
            f"background-color: {self.CARD_BG}; border-radius: 10px;"
            f"border-left: 4px solid {self.ACCENT};"
        )
        ns_vl = QVBoxLayout(ns_frame)
        ns_vl.setContentsMargins(16, 12, 16, 12)
        ns_lbl = QLabel(next_steps)
        ns_lbl.setFont(QFont("Arial", 10))
        ns_lbl.setStyleSheet(f"color: {self.TEXT}; border: none;")
        ns_lbl.setWordWrap(True)
        ns_vl.addWidget(ns_lbl)
        vl.addWidget(ns_frame)

        # ── Disclaimer ────────────────────────────────────────────────────────
        disc = QLabel(getattr(self.risk, "disclaimer", ""))
        disc.setFont(QFont("Arial", 8))
        disc.setStyleSheet("color: #666666;")
        disc.setWordWrap(True)
        disc.setAlignment(Qt.AlignCenter)
        vl.addWidget(disc)
        vl.addStretch()

        return outer

    # ── Tab 1: Summary ────────────────────────────────────────────────────────
    def _build_summary_tab(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setSpacing(16)

        # Risk score hero card
        rc = self._risk_color()
        score_card = QFrame()
        score_card.setStyleSheet(
            f"background-color: {self.CARD_BG}; border-radius: 12px;"
            f"border: 2px solid {rc};"
        )
        sc_layout = QVBoxLayout(score_card)
        sc_layout.setContentsMargins(20, 16, 20, 16)

        risk_title = QLabel("DYSLEXIA RISK ASSESSMENT")
        risk_title.setFont(QFont("Arial", 10))
        risk_title.setStyleSheet(f"color: #aaaaaa; border: none;")
        risk_title.setAlignment(Qt.AlignCenter)

        risk_band_lbl = QLabel(
            getattr(self.risk, "risk_band", "Unknown").upper()
        )
        risk_band_lbl.setFont(QFont("Arial", 32, QFont.Bold))
        risk_band_lbl.setStyleSheet(f"color: {rc}; border: none;")
        risk_band_lbl.setAlignment(Qt.AlignCenter)

        risk_score_lbl = QLabel(
            f"Risk Score: {getattr(self.risk, 'risk_score', 0.0):.4f}"
        )
        risk_score_lbl.setFont(QFont("Courier", 14))
        risk_score_lbl.setStyleSheet(f"color: {self.TEXT}; border: none;")
        risk_score_lbl.setAlignment(Qt.AlignCenter)

        disclaimer = getattr(self.risk, "disclaimer", "")
        disc_lbl = QLabel(disclaimer)
        disc_lbl.setFont(QFont("Arial", 8))
        disc_lbl.setStyleSheet("color: #888888; border: none;")
        disc_lbl.setAlignment(Qt.AlignCenter)
        disc_lbl.setWordWrap(True)

        sc_layout.addWidget(risk_title)
        sc_layout.addWidget(risk_band_lbl)
        sc_layout.addWidget(risk_score_lbl)
        sc_layout.addWidget(disc_lbl)
        vl.addWidget(score_card)

        # Key indicator cards row
        cards_row = QHBoxLayout()
        span_color = (
            self.GREEN  if self.gaze_span > 0.15 else
            self.YELLOW if self.gaze_span > 0.10 else self.RED
        )
        qual_color = (
            self.GREEN  if self.gaze_quality >= 0.70 else
            self.YELLOW if self.gaze_quality >= 0.40 else self.RED
        )
        comp_color = (
            self.GREEN  if self.completeness >= 0.80 else
            self.YELLOW if self.completeness >= 0.60 else self.RED
        )
        n_scored = sum(
            1 for m in self.sentence_metrics_list
            if m.sentence_index >= 0
        )
        cards_row.addWidget(self._card("EVS Gaze Span",
            f"{self.gaze_span:.4f}", span_color))
        cards_row.addWidget(self._card("Calibration Quality",
            f"{self.gaze_quality:.1%}", qual_color))
        cards_row.addWidget(self._card("Data Completeness",
            f"{self.completeness:.1%}", comp_color))
        cards_row.addWidget(self._card("Sentences Scored",
            str(n_scored), self.ACCENT))
        vl.addLayout(cards_row)

        # Feature vector key stats (if fv exposes a dict or has __dict__)
        fv_dict = {}
        if hasattr(self.fv, "__dict__"):
            fv_dict = {k: v for k, v in self.fv.__dict__.items()
                       if isinstance(v, (int, float)) and not k.startswith("_")}
        elif isinstance(self.fv, dict):
            fv_dict = {k: v for k, v in self.fv.items()
                       if isinstance(v, (int, float))}

        if fv_dict:
            fv_title = QLabel("Extracted Feature Vector")
            fv_title.setFont(QFont("Arial", 11, QFont.Bold))
            fv_title.setStyleSheet(f"color: {self.ACCENT};")
            vl.addWidget(fv_title)

            grid = QGridLayout()
            grid.setSpacing(6)
            for col_i, (key, val) in enumerate(sorted(fv_dict.items())):
                r, c = divmod(col_i, 4)
                color = self.BLUE
                grid.addWidget(self._card(key, f"{val:.4f}" if isinstance(val, float) else str(val), color), r, c)
            vl.addLayout(grid)

        vl.addStretch()
        return w

    # ── Tab 2: Per-sentence charts ────────────────────────────────────────────
    def _build_sentence_tab(self) -> QWidget:
        w   = QWidget()
        vl  = QVBoxLayout(w)

        metrics = self.sentence_metrics_list
        if not metrics:
            vl.addWidget(QLabel("No sentence metrics available."))
            return w

        sent_labels = [f"S{m.sentence_index+1}" for m in metrics]
        n = len(metrics)

        def _safe(m, attr, default=0.0):
            v = getattr(m, attr, default)
            return float(v) if v is not None else default

        # ── Correct attribute names from SentenceMetrics dataclass ────────────
        # fixation coverage = fraction of words with fixation data
        fix_cov      = [_safe(m, "n_valid_slots") / max(_safe(m, "n_words", 1), 1)
                        for m in metrics]
        # speech coverage proxy = slots where EVS speech was computed
        speech_cov   = [min(1.0, _safe(m, "n_valid_slots") / max(_safe(m, "n_words", 1), 1))
                        if _safe(m, "mean_evs_speech_ms") != 0.0
                        else _safe(m, "n_valid_slots") / max(_safe(m, "n_words", 1), 1)
                        for m in metrics]
        completeness = [_safe(m, "data_completeness") for m in metrics]
        mean_evs     = [_safe(m, "mean_evs_speech_ms") for m in metrics]   # replaces missing duration
        mean_fix_dur = [_safe(m, "mean_fixation_duration_ms") for m in metrics]
        regression   = [_safe(m, "regression_rate")    for m in metrics]
        std_fix_dur  = [_safe(m, "std_fixation_duration_ms")  for m in metrics]
        mean_stg     = [_safe(m, "mean_stg_ms")        for m in metrics]

        fig = Figure(figsize=(12, 9), tight_layout=True)
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
        x     = np.arange(n)
        bar_w = 0.35

        # ── Plot 1: Fixation & Speech coverage (now using real fields) ─────────
        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(x - bar_w/2, fix_cov,    bar_w, label="Fixation Coverage",
                color="#74b9ff", alpha=0.85)
        ax1.bar(x + bar_w/2, speech_cov, bar_w, label="Speech Coverage",
                color="#00b894", alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(sent_labels, fontsize=8)
        ax1.set_ylim(0, 1.15)
        ax1.axhline(1.0, color="#555", linestyle="--", linewidth=0.8)
        ax1.set_title("Fixation & Speech Coverage per Sentence", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Coverage (0–1)")
        ax1.legend(fontsize=8); ax1.grid(axis="y")

        # ── Plot 2: Data completeness ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x, completeness, marker="o", color="#fdcb6e", linewidth=2)
        ax2.fill_between(x, completeness, alpha=0.2, color="#fdcb6e")
        ax2.axhline(0.6, color=self.RED, linestyle="--", linewidth=1, label="Min threshold (0.6)")
        ax2.set_xticks(x); ax2.set_xticklabels(sent_labels, fontsize=8)
        ax2.set_ylim(0, 1.15)
        ax2.set_title("Data Completeness per Sentence", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Completeness"); ax2.legend(fontsize=8); ax2.grid(axis="y")

        # ── Plot 3: Mean EVS Speech (replaces missing duration_ns) ────────────
        ax3 = fig.add_subplot(gs[1, 1])
        colors3 = ["#00b894" if v > 0 else "#e17055" for v in mean_evs]
        ax3.bar(x, mean_evs, color=colors3, alpha=0.85)
        ax3.axhline(0, color="#888", linewidth=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(sent_labels, fontsize=8)
        ax3.set_title("Mean Eye-Voice Span per Sentence (ms)", fontsize=10, fontweight="bold")
        ax3.set_ylabel("EVS (ms)"); ax3.grid(axis="y")

        # ── Plot 4: Mean fixation duration ────────────────────────────────────
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(x, mean_fix_dur, marker="s", color="#fd79a8", linewidth=2)
        ax4.fill_between(x, mean_fix_dur, alpha=0.2, color="#fd79a8")
        # shaded error band ± std
        lo = [max(0, m - s) for m, s in zip(mean_fix_dur, std_fix_dur)]
        hi = [m + s          for m, s in zip(mean_fix_dur, std_fix_dur)]
        ax4.fill_between(x, lo, hi, alpha=0.12, color="#fd79a8")
        ax4.set_xticks(x); ax4.set_xticklabels(sent_labels, fontsize=8)
        ax4.set_title("Mean Fixation Duration ± Std (ms)", fontsize=10, fontweight="bold")
        ax4.set_ylabel("ms"); ax4.grid(axis="y")

        # ── Plot 5: Regression rate ───────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, 1])
        reg_colors = ["#ff6348" if r > 0.1 else "#fdcb6e" if r > 0.02 else "#00b894"
                      for r in regression]
        ax5.bar(x, regression, color=reg_colors, alpha=0.85)
        ax5.set_xticks(x); ax5.set_xticklabels(sent_labels, fontsize=8)
        ax5.set_title("Regression Rate per Sentence", fontsize=10, fontweight="bold")
        ax5.set_ylabel("Rate (0–1)"); ax5.grid(axis="y")

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll = QScrollArea()
        scroll.setWidget(canvas)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        vl.addWidget(scroll)
        return w

    # ── Tab: Typing analysis ──────────────────────────────────────────────────
    def _build_typing_tab(self) -> QWidget:
        """Per-sentence typing metrics: coverage, corrections, STG, EVS typing."""
        w  = QWidget()
        vl = QVBoxLayout(w)

        if not self.typing_summary:
            lbl = QLabel(
                "No typing data recorded for this session.\n"
                "Typing is recorded when the participant types each word into the "
                "text box during the session."
            )
            lbl.setFont(QFont("Arial", 11))
            lbl.setStyleSheet("color: #888888; padding: 20px;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            vl.addWidget(lbl)
            return w

        metrics     = self.sentence_metrics_list
        n           = len(metrics)
        sent_labels = [f"S{m.sentence_index+1}" for m in metrics]
        x = np.arange(n)

        def _ts(i, key, default=0.0):
            return float(self.typing_summary.get(i, {}).get(key, default))

        def _ms(m, attr):
            v = getattr(m, attr, None)
            return float(v) if v is not None else 0.0

        words_typed  = [_ts(i, "words_typed")    for i in range(n)]
        total_words  = [_ts(i, "total_words", 1) for i in range(n)]
        typed_pct    = [wt / max(tot, 1) for wt, tot in zip(words_typed, total_words)]
        corrections  = [_ts(i, "corrections")    for i in range(n)]
        corr_rate    = [c / max(wt, 1) for c, wt in zip(corrections, words_typed)]
        mean_stg     = [_ms(m, "mean_stg_ms")       for m in metrics]
        std_stg      = [_ms(m, "std_stg_ms")        for m in metrics]
        mean_evs_t   = [_ms(m, "mean_evs_typing_ms") for m in metrics]
        mean_corr_r  = [_ms(m, "mean_correction_rate") for m in metrics]

        fig = Figure(figsize=(13, 9))
        fig.patch.set_facecolor("#1a1a1a")
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38,
                                left=0.07, right=0.97, top=0.95, bottom=0.07)
        bar_w = 0.35

        def _style_ax(ax):
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="#f1f2f6", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#555")

        # ── Plot 1: Words typed vs total ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        _style_ax(ax1)
        ax1.bar(x - bar_w/2, total_words, bar_w, label="Total Words",
                color="#636e72", alpha=0.75)
        ax1.bar(x + bar_w/2, words_typed, bar_w, label="Words Typed",
                color="#a29bfe", alpha=0.90)
        ax1.set_xticks(x); ax1.set_xticklabels(sent_labels, fontsize=8, color="#f1f2f6")
        ax1.set_title("Words Typed vs Total per Sentence",
                      fontsize=10, fontweight="bold", color="#f1f2f6")
        ax1.set_ylabel("Word Count", color="#f1f2f6")
        ax1.legend(fontsize=8, facecolor="#2d2d2d", labelcolor="#f1f2f6")
        ax1.grid(axis="y", alpha=0.3)

        # ── Plot 2: Typing coverage ───────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        _style_ax(ax2)
        colors2 = ["#00b894" if p >= 0.8 else "#fdcb6e" if p >= 0.5 else "#ff6348"
                   for p in typed_pct]
        ax2.bar(x, typed_pct, color=colors2, alpha=0.85)
        ax2.axhline(1.0, color="#555", linestyle="--", linewidth=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(sent_labels, fontsize=8, color="#f1f2f6")
        ax2.set_ylim(0, 1.15)
        ax2.set_title("Typing Coverage per Sentence",
                      fontsize=10, fontweight="bold", color="#f1f2f6")
        ax2.set_ylabel("Fraction Typed", color="#f1f2f6")
        ax2.grid(axis="y", alpha=0.3)

        # ── Plot 3: Correction rate from SentenceMetrics ──────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        _style_ax(ax3)
        corr_colors = ["#ff6348" if r > 0.3 else "#fdcb6e" if r > 0.05 else "#00b894"
                       for r in mean_corr_r]
        ax3.bar(x, mean_corr_r, color=corr_colors, alpha=0.85)
        ax3.set_xticks(x); ax3.set_xticklabels(sent_labels, fontsize=8, color="#f1f2f6")
        ax3.set_title("Correction Rate per Sentence (backspaces / word)",
                      fontsize=10, fontweight="bold", color="#f1f2f6")
        ax3.set_ylabel("Rate", color="#f1f2f6")
        ax3.grid(axis="y", alpha=0.3)

        # ── Plot 4: Speech-Typing Gap (STG) ───────────────────────────────────
        ax4 = fig.add_subplot(gs[2, 0])
        _style_ax(ax4)
        stg_colors = ["#00b894" if v > 0 else "#e17055" for v in mean_stg]
        ax4.bar(x, mean_stg, color=stg_colors, alpha=0.85)
        lo = [m - s for m, s in zip(mean_stg, std_stg)]
        hi = [m + s for m, s in zip(mean_stg, std_stg)]
        ax4.errorbar(x, mean_stg, yerr=std_stg, fmt="none",
                     ecolor="#f1f2f6", capsize=3, linewidth=1.2, alpha=0.6)
        ax4.axhline(0, color="#888", linewidth=0.8)
        ax4.set_xticks(x); ax4.set_xticklabels(sent_labels, fontsize=8, color="#f1f2f6")
        ax4.set_title("Speech–Typing Gap per Sentence (ms)\n"
                      "(+) = spoke before typed  |  (–) = typed before spoke",
                      fontsize=9, fontweight="bold", color="#f1f2f6")
        ax4.set_ylabel("STG (ms)", color="#f1f2f6")
        ax4.grid(axis="y", alpha=0.3)

        # ── Plot 5: EVS Typing (eye leads typing) ─────────────────────────────
        ax5 = fig.add_subplot(gs[2, 1])
        _style_ax(ax5)
        evst_colors = ["#a29bfe" if v > 0 else "#e17055" for v in mean_evs_t]
        ax5.bar(x, mean_evs_t, color=evst_colors, alpha=0.85)
        ax5.axhline(0, color="#888", linewidth=0.8)
        ax5.set_xticks(x); ax5.set_xticklabels(sent_labels, fontsize=8, color="#f1f2f6")
        ax5.set_title("Mean Eye-Typing Span per Sentence (ms)\n"
                      "(+) = eyes ahead of typing",
                      fontsize=9, fontweight="bold", color="#f1f2f6")
        ax5.set_ylabel("EVS Typing (ms)", color="#f1f2f6")
        ax5.grid(axis="y", alpha=0.3)

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(canvas)

        # ── Summary cards ──────────────────────────────────────────────────────
        total_typed   = sum(int(w) for w in words_typed)
        total_all     = sum(int(w) for w in total_words)
        total_corr    = sum(int(c) for c in corrections)
        overall_cov   = total_typed / max(total_all, 1)
        mean_stg_all  = float(np.mean([v for v in mean_stg if v != 0])) if any(v != 0 for v in mean_stg) else 0.0
        mean_evst_all = float(np.mean([v for v in mean_evs_t if v != 0])) if any(v != 0 for v in mean_evs_t) else 0.0

        cov_col  = self.GREEN if overall_cov >= 0.8 else self.YELLOW if overall_cov >= 0.5 else self.RED
        corr_col = self.GREEN if total_corr == 0 else self.YELLOW if total_corr < 10 else self.RED
        stg_col  = self.GREEN if mean_stg_all > 0 else self.YELLOW if mean_stg_all > -500 else self.RED
        evst_col = self.GREEN if mean_evst_all > 0 else self.YELLOW

        cards_row = QHBoxLayout()
        cards_row.addWidget(self._card("Words Typed", f"{total_typed} / {total_all}", cov_col))
        cards_row.addWidget(self._card("Typing Coverage", f"{overall_cov:.0%}", cov_col))
        cards_row.addWidget(self._card("Total Corrections", str(total_corr), corr_col))
        cards_row.addWidget(self._card("Mean STG",
                            f"{mean_stg_all:.0f} ms" if mean_stg_all != 0 else "N/A", stg_col))
        cards_row.addWidget(self._card("Mean EVS Typing",
                            f"{mean_evst_all:.0f} ms" if mean_evst_all != 0 else "N/A", evst_col))
        vl.addLayout(cards_row)
        return w

    # ── Tab 3: Feature vector — split by scale + fixed heatmap ──────────────
    def _build_feature_tab(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)

        fv_dict = {}
        if hasattr(self.fv, "__dict__"):
            fv_dict = {k: v for k, v in self.fv.__dict__.items()
                       if isinstance(v, (int, float)) and not k.startswith("_")}
        elif isinstance(self.fv, dict):
            fv_dict = {k: v for k, v in self.fv.items()
                       if isinstance(v, (int, float))}

        if not fv_dict:
            vl.addWidget(QLabel("Feature vector not available or empty."))
            return w

        # Split features by scale so small values aren't crushed by ms values
        MS_FEATURES   = {"F01", "F02", "F05", "F06", "F07", "F14"}  # large ms scale
        keys_all   = sorted(fv_dict.keys())
        keys_ms    = [k for k in keys_all if any(k.startswith(p) for p in MS_FEATURES)]
        keys_norm  = [k for k in keys_all if k not in keys_ms]
        vals_ms    = [fv_dict[k] for k in keys_ms]
        vals_norm  = [fv_dict[k] for k in keys_norm]

        # ── Figure 1: Two side-by-side bar charts (ms scale / normalised scale) ──
        fig1_h = max(5, max(len(keys_ms), len(keys_norm)) * 0.42 + 1.5)
        fig1 = Figure(figsize=(13, fig1_h))
        fig1.patch.set_facecolor("#1a1a1a")
        gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.55, left=0.22, right=0.97,
                                top=0.92, bottom=0.06)

        def _barh(ax, keys, values, title, xlabel):
            colors = ["#00b894" if v >= 0 else "#e17055" for v in values]
            # show value 0 as a tiny stub so the label is still visible
            display_vals = [v if abs(v) > 1e-9 else 1e-9 for v in values]
            bars = ax.barh(range(len(keys)), display_vals, color=colors, alpha=0.88)
            # annotate actual value at end of bar
            for bar, val in zip(bars, values):
                xpos = bar.get_width()
                align = "left" if xpos >= 0 else "right"
                offset = max(abs(xpos) * 0.02, abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01)
                ax.text(xpos + (offset if xpos >= 0 else -offset),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3g}", va="center", ha=align,
                        fontsize=7, color="#f1f2f6")
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels(keys, fontsize=8)
            ax.axvline(0, color="#888", linewidth=0.7)
            ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.grid(axis="x", alpha=0.4)

        if keys_ms:
            ax_ms = fig1.add_subplot(gs1[0])
            _barh(ax_ms, keys_ms, vals_ms,
                  "Timing Features (ms scale)", "ms")
        if keys_norm:
            ax_norm = fig1.add_subplot(gs1[1])
            _barh(ax_norm, keys_norm, vals_norm,
                  "Rate / Ratio Features (0–1 scale)", "Value")

        canvas1 = FigureCanvas(fig1)
        canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        canvas1.setMinimumHeight(int(fig1_h * 72))
        vl.addWidget(canvas1)

        # ── Figure 2: Per-feature heatmap with full labels ─────────────────────
        n_feat = len(keys_all)
        vals_all = [fv_dict[k] for k in keys_all]

        # Normalise each feature independently for colour mapping
        # (avoids ms features dominating the 0-1 scale)
        def _minmax(arr):
            mn, mx = min(arr), max(arr)
            span = mx - mn
            if span < 1e-12:
                return [0.5] * len(arr)
            return [(v - mn) / span for v in arr]

        norm_vals = _minmax(vals_all)

        fig2_w = max(10, n_feat * 0.72)
        fig2 = Figure(figsize=(fig2_w, 2.8))
        fig2.patch.set_facecolor("#1a1a1a")
        ax_hm = fig2.add_subplot(111)
        ax_hm.set_facecolor("#2d2d2d")
        im = ax_hm.imshow(
            np.array(norm_vals).reshape(1, -1),
            aspect="auto", cmap="RdYlGn", vmin=0, vmax=1
        )
        ax_hm.set_xticks(range(n_feat))
        ax_hm.set_xticklabels(keys_all, rotation=55, ha="right",
                               fontsize=7.5, color="#f1f2f6")
        ax_hm.set_yticks([])
        ax_hm.set_title("Feature Heatmap — per-feature min→max normalised",
                         fontsize=10, fontweight="bold", color="#f1f2f6")
        fig2.colorbar(im, ax=ax_hm, orientation="horizontal",
                      pad=0.35, fraction=0.08, label="low → high (per feature)")
        fig2.subplots_adjust(bottom=0.42, left=0.02, right=0.98, top=0.88)

        canvas2 = FigureCanvas(fig2)
        canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        canvas2.setMinimumHeight(220)

        scroll = QScrollArea()
        inner = QWidget()
        inner_vl = QVBoxLayout(inner)
        inner_vl.setContentsMargins(0, 0, 0, 0)
        inner_vl.addWidget(canvas2)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("background: transparent; border: none;")
        vl.addWidget(scroll)
        return w

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION WINDOW
# UI structure mirrors ReadingScreen from main_camera_word.py.
# Sentence cycling + MESTTS persistence wired on top.
# ══════════════════════════════════════════════════════════════════════════════
def _avg_indicator(risk_result, prefixes: Tuple[str, ...]) -> float:
    """
    Average the `contribution` of every RiskScoreResult indicator whose
    feature_id starts with one of `prefixes`. Used to derive a rough
    per-modality confidence score for the frontend export
    (avg_eye_conf / avg_type_conf / avg_audio_conf).

    NOTE: this is a stand-in, not a validated per-modality confidence
    metric — RiskScorer only produces session-level indicator
    contributions today, grouped here by feature family:
      F06/F08 -> eye tracking, F01 -> typing, F04 -> speech/audio.
    If you need a real per-modality score, that belongs in
    analytics/risk_scorer.py, not here.
    """
    vals = [ind.contribution for ind in risk_result.indicators
            if ind.feature_id.startswith(prefixes)]
    return sum(vals) / len(vals) if vals else 0.0


class SessionWindow(QWidget):

    CALIB_DURATION = GazeTracker.CALIB_DURATION
    CALIB_POINTS   = ["left", "center", "right"]

    def __init__(self, cfg: ExperimentConfig, participant_id: str,
                 sentences: List[Dict], audio: AudioProcessor):
        super().__init__()
        self.cfg            = cfg
        self.participant_id = participant_id
        self.sentences      = sentences
        self.audio          = audio
        self.session_id     = str(uuid.uuid4())
        self.clock          = get_clock()
        self.bus            = get_bus()

        # Calibration state (mirrors ReadingScreen)
        self.calibrating           = False
        self.calibration_stage     = 0
        self.calibration_start_time: Optional[float] = None

        # Session / sentence state
        self.session_started     = False
        self._sentence_loaded_at = 0.0      # debounce: stop ENTER ghost-firing
        self.sentence_index   = 0
        self.current_words:   List[str] = []
        self.fixation_log:    List[Dict] = []
        self.typed_words:     List[str]  = []   # words submitted via typing
        self._typing_summary: dict = {}         # {sentence_idx: {words_typed,..}}
        self.sentence_start_ns = 0
        self.sentence_metrics_list = []

        # Gaze + fixation
        first_words = sentences[0]["text"].split()
        self.gaze    = GazeTracker(num_words=len(first_words))
        self.fixation = WordFixationTracker(words=first_words)

        # MESTTS infrastructure
        self.db        = DatabaseManager(cfg.db_path)
        self.sess_log  = SessionLogger(self.db, cfg.export_dir)
        self.exporter  = ResearchExporter(self.db, cfg.export_dir)

        self.synchronizer = SynchronizerThread(
            bus=self.bus,
            on_all_words_complete=lambda: None,
        )
        self.synchronizer.start()

        self.typing_hook = TypingHook(bus=self.bus)
        self.typing_hook.start()

        # Camera (same fallback logic as setup_camera() in original)
        url = cfg.ipcam_url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            logger.warning("IPCam failed (%s), falling back to webcam 0.", url)
            self.cap = cv2.VideoCapture(0)

        mp_fm = mp.solutions.face_mesh
        self.face_mesh = mp_fm.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        self._build_ui()

        # QTimer loop — mirrors setup_timer() / update_loop() in original
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_loop)
        self.timer.start(30)

    # ── UI (mirrors setup_ui from ReadingScreen) ──────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("MESTTS Reading Session")
        self.setGeometry(50, 50, 1900, 700)
        self.setStyleSheet("background-color: #1a1a1a;")

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(30, 30, 30, 30)

        banner = QLabel(
            f"MESTTS  |  Participant: {self.participant_id}"
            f"  |  Session: {self.session_id[:8]}"
        )
        banner.setAlignment(Qt.AlignCenter)
        banner.setFont(QFont("Arial", 11, QFont.Bold))
        banner.setStyleSheet(
            "color: #00d2d3; padding: 12px; "
            "background-color: #2d2d2d; border-radius: 6px;"
        )
        layout.addWidget(banner)

        self.diag_lbl = QLabel("Waiting...")
        self.diag_lbl.setAlignment(Qt.AlignCenter)
        self.diag_lbl.setFont(QFont("Courier", 10, QFont.Bold))
        self.diag_lbl.setStyleSheet(
            "color: #feca57; padding: 10px; "
            "background-color: #2d2d2d; border-radius: 5px;"
        )
        layout.addWidget(self.diag_lbl)

        self.status_lbl = QLabel("Press C to calibrate, then ENTER to start")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_lbl.setStyleSheet(
            "color: #f1f2f6; padding: 15px; "
            "background-color: #2d2d2d; border-radius: 8px;"
        )
        layout.addWidget(self.status_lbl)

        self.progress_lbl = QLabel(
            "Prop phone at eye level | Look EXAGGERATED left/right during calibration"
        )
        self.progress_lbl.setAlignment(Qt.AlignCenter)
        self.progress_lbl.setFont(QFont("Arial", 10))
        self.progress_lbl.setStyleSheet("color: #7f8c8d; padding: 8px;")
        layout.addWidget(self.progress_lbl)

        # Calibration dots
        self.calib_widget = QWidget()
        calib_layout = QGridLayout()
        calib_layout.setContentsMargins(80, 15, 80, 15)
        self.dot_left   = CalibrationDot()
        self.dot_center = CalibrationDot()
        self.dot_right  = CalibrationDot()
        calib_layout.addWidget(self.dot_left,   0, 0, Qt.AlignLeft   | Qt.AlignVCenter)
        calib_layout.addWidget(self.dot_center, 0, 1, Qt.AlignCenter)
        calib_layout.addWidget(self.dot_right,  0, 2, Qt.AlignRight  | Qt.AlignVCenter)
        self.calib_widget.setLayout(calib_layout)
        self.calib_widget.hide()
        layout.addWidget(self.calib_widget)

        # Sentence counter
        self.sentence_num_lbl = QLabel("")
        self.sentence_num_lbl.setAlignment(Qt.AlignCenter)
        self.sentence_num_lbl.setFont(QFont("Arial", 11))
        self.sentence_num_lbl.setStyleSheet("color: #7f8c8d; padding: 4px;")
        layout.addWidget(self.sentence_num_lbl)

        # Word display area
        self.word_container = QWidget()
        self.word_container.setStyleSheet(
            "background-color: #2d2d2d; border-radius: 10px; padding: 18px;"
        )
        self.word_layout = QHBoxLayout()
        self.word_layout.setSpacing(12)
        self.word_container.setLayout(self.word_layout)
        self.word_labels: List[QLabel] = []

        layout.addStretch()
        layout.addWidget(self.word_container)
        layout.addStretch()

        self.gaze_word_lbl = QLabel("Gaze: —")
        self.gaze_word_lbl.setAlignment(Qt.AlignCenter)
        self.gaze_word_lbl.setFont(QFont("Courier", 11, QFont.Bold))
        self.gaze_word_lbl.setStyleSheet(
            "color: #74b9ff; padding: 8px; "
            "background-color: #2d2d2d; border-radius: 5px;"
        )
        layout.addWidget(self.gaze_word_lbl)

        # ── Typing panel ──────────────────────────────────────────────────────
        typing_panel = QWidget()
        typing_panel.setStyleSheet(
            "background-color: #2d2d2d; border-radius: 8px; padding: 4px;"
        )
        tp_layout = QHBoxLayout()
        tp_layout.setContentsMargins(12, 8, 12, 8)

        type_hint = QLabel("Type:")
        type_hint.setFont(QFont("Arial", 12, QFont.Bold))
        type_hint.setStyleSheet("color: #b2bec3; min-width: 50px;")
        type_hint.setFocusPolicy(Qt.NoFocus)
        tp_layout.addWidget(type_hint)

        self.type_input = _TypingInput(self)
        self.type_input.setFont(QFont("Courier", 16, QFont.Bold))
        self.type_input.setPlaceholderText("Type each word and press SPACE...")
        self.type_input.setStyleSheet(
            "background-color: #1a1a1a; color: #fdcb6e; "
            "border: 2px solid #636e72; border-radius: 6px; "
            "padding: 6px 10px; selection-background-color: #74b9ff;"
        )
        self.type_input.textChanged.connect(self._on_type_changed)
        tp_layout.addWidget(self.type_input, stretch=1)

        self.type_target_lbl = QLabel("")
        self.type_target_lbl.setFont(QFont("Arial", 11))
        self.type_target_lbl.setStyleSheet(
            "color: #636e72; min-width: 180px; padding-left: 8px;"
        )
        self.type_target_lbl.setFocusPolicy(Qt.NoFocus)
        tp_layout.addWidget(self.type_target_lbl)

        typing_panel.setLayout(tp_layout)
        layout.addWidget(typing_panel)

        instructions = QLabel(
            "C = Calibrate  |  ENTER = Next sentence  |  R = Reset  |  ESC = Quit\n"
            "White/border = gaze+speech+typed  |  Green = gaze+speech  |  Blue = gaze  "
            "|  Purple = typed+spoken  |  Pink = typed  |  Teal = fixated+spoken  |  Yellow = spoken"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setFont(QFont("Arial", 9))
        instructions.setStyleSheet(
            "background-color: #2d2d2d; color: #ecf0f1; padding: 10px; border-radius: 5px;"
        )
        instructions.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(instructions)

        reset_btn = QPushButton("Reset Sentence (R)")
        reset_btn.setFont(QFont("Arial", 10))
        reset_btn.setStyleSheet(
            "background-color: #e74c3c; color: white; "
            "padding: 10px 20px; border-radius: 5px;"
        )
        reset_btn.setFocusPolicy(Qt.NoFocus)
        reset_btn.setAutoDefault(False)
        reset_btn.setDefault(False)
        reset_btn.clicked.connect(self._reset_sentence)
        layout.addWidget(reset_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setFocusPolicy(Qt.StrongFocus)

    def _rebuild_word_labels(self, words: List[str]):
        for lbl in self.word_labels:
            self.word_layout.removeWidget(lbl)
            lbl.deleteLater()
        self.word_labels.clear()
        for word in words:
            lbl = QLabel(word)
            lbl.setFont(QFont("Arial", 26, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #f1f2f6; padding: 8px;")
            self.word_layout.addWidget(lbl)
            self.word_labels.append(lbl)

    # ── Named action handlers — one function per user action ─────────────────
    # Both keyPressEvent (window focus) and _TypingInput call these directly.
    # All guards live here; callers contain no logic.

    def _handle_calibrate(self):
        if not self.calibrating:
            self._start_calibration()

    def _handle_advance(self):
        """Start session or advance sentence. 600 ms debounce stops the
        key-up of the ENTER that loaded a sentence from ghost-firing again."""
        if self.calibrating:
            return
        if not self.gaze.is_calibrated():
            return
        if not self.session_started:
            self._start_session()
            return
        # Debounce: ignore rapid ENTER within 600 ms of sentence load
        if time.monotonic() - self._sentence_loaded_at < 0.6:
            return
        self._advance_sentence()

    def _handle_reset(self):
        if self.session_started:
            self._reset_sentence()

    def _handle_quit(self):
        self.close()

    def keyPressEvent(self, event):
        """Fires only when main window has focus (typing box NOT focused)."""
        key = event.key()
        if key == Qt.Key_0:
            self._handle_calibrate()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._handle_advance()
        elif key == Qt.Key_1:
            # SPACE starts session pre-calibration. During session the typing
            # box owns SPACE so this branch is only reached before session start.
            if not self.calibrating and self.gaze.is_calibrated():
                if not self.session_started:
                    self._start_session()
        elif key == Qt.Key_2:
            self._handle_reset()
        elif key == Qt.Key_Escape:
            self._handle_quit()

    # ── Calibration (mirrors start_calibration / start_calibration_point /
    #                 update_calibration_progress / finalize_calibration) ──────

    def _start_calibration(self):
        self.gaze.reset()
        self.calibrating            = True
        self.calibration_stage      = 0
        self.calib_widget.show()
        self._start_calibration_point(0)

    def _start_calibration_point(self, stage):
        self.calibration_stage      = stage
        self.calibration_start_time = time.time()
        self.dot_left.set_active(stage == 0)
        self.dot_center.set_active(stage == 1)
        self.dot_right.set_active(stage == 2)
        for dot in (self.dot_left, self.dot_center, self.dot_right):
            dot.set_progress(0)
        names = {0: "FAR LEFT", 1: "CENTER", 2: "FAR RIGHT"}
        self.status_lbl.setText(f"Look {names[stage]} - EXAGGERATED!")
        self.status_lbl.setStyleSheet(
            "color: #fdcb6e; padding: 15px; "
            "background-color: #2d2d2d; border-radius: 8px; font-weight: bold;"
        )

    def _update_calibration_progress(self):
        # mirrors update_calibration_progress verbatim
        if not self.calibrating or not self.calibration_start_time:
            return
        elapsed  = time.time() - self.calibration_start_time
        progress = min(100, (elapsed / self.CALIB_DURATION) * 100)

        dots = [self.dot_left, self.dot_center, self.dot_right]
        dots[self.calibration_stage].set_progress(progress)

        point_name = self.CALIB_POINTS[self.calibration_stage]
        # Use the smoothed gaze value (median-filtered) for better calibration quality
        self.gaze.add_calibration_sample(point_name, self.gaze._last_smoothed)

        n = len(self.gaze.calibration_data[point_name])
        self.progress_lbl.setText(f"Samples: {n} | {int(progress)}%")

        if elapsed >= self.CALIB_DURATION:
            if self.calibration_stage < 2:
                self._start_calibration_point(self.calibration_stage + 1)
            else:
                self._finalize_calibration()

    def _finalize_calibration(self):
        # mirrors finalize_calibration verbatim
        success = self.gaze.finalize_calibration()
        self.calibrating = False
        self.calib_widget.hide()

        if success:
            span    = self.gaze.calibration_span
            quality = self.gaze.calibration_quality
            if span > 0.15:
                msg, color = f"EXCELLENT (span: {span:.3f})", "#00b894"
            elif span > 0.10:
                msg, color = f"GOOD (span: {span:.3f})", "#fdcb6e"
            else:
                msg, color = f"MARGINAL (span: {span:.3f}) — recalibrate", "#ff6348"
            n = len(self.current_words) if self.current_words else "?"
            self.status_lbl.setText(
                f"{msg}  Quality: {quality:.0%} | {n} words | Press ENTER to start"
            )
            self.status_lbl.setStyleSheet(
                f"color: {color}; padding: 15px; "
                "background-color: #2d2d2d; border-radius: 8px; font-weight: bold;"
            )
        else:
            self.status_lbl.setText(
                "Calibration FAILED — look more exaggeratedly L/R, then press C again"
            )
            self.status_lbl.setStyleSheet(
                "color: #ff6348; padding: 15px; "
                "background-color: #2d2d2d; border-radius: 8px;"
            )

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def _start_session(self):
        self.session_started     = True
        self._sentence_loaded_at = time.monotonic()  # pre-stamp debounce
        start_ns = self.clock.start()
        logger.info("Session %s started. Epoch: %d", self.session_id, start_ns)

        self.sess_log.begin_session(
            session_id        = self.session_id,
            participant_id    = self.participant_id,
            start_ns          = start_ns,
            calibration_rmse  = 0.0,          # normalized gaze; no pixel RMSE
            camera_latency_ms = self.cfg.camera_latency_ns / 1_000_000,
            fuzzy_threshold   = self.cfg.fuzzy_threshold,
            lookahead_window  = self.cfg.lookahead_window,
            n_sentences       = len(self.sentences),
            protocol_version  = self.cfg.protocol_version,
        )
        self._load_sentence(0)

    def _load_sentence(self, idx: int):
        sent  = self.sentences[idx]
        words = sent["text"].split()

        self.sentence_index  = idx
        self.current_words   = words
        self.fixation_log    = []
        self.typed_words     = []

        # update gaze word-boundary mapping for this sentence
        self.gaze.set_num_words(len(words))
        self.fixation.reset(new_words=words)
        self.audio.reset_sentence(words)
        self.synchronizer.activate_sentence(words)

        self.sentence_start_ns = self.clock.now_ns()
        self._rebuild_word_labels(words)

        n = len(self.sentences)
        practice = idx < self.cfg.practice_sentences
        tag = " [PRACTICE]" if practice else ""
        self.sentence_num_lbl.setText(f"Sentence {idx + 1} / {n}{tag}")
        self.status_lbl.setText("Read aloud and type each word — press ENTER when done")
        self.status_lbl.setStyleSheet(
            "color: #f1f2f6; padding: 15px; "
            "background-color: #2d2d2d; border-radius: 8px;"
        )
        # Reset typing panel and give it focus
        self.type_input.clear()
        if words:
            self.type_target_lbl.setText(f"\u2192 word 1: '{words[0]}'")
            QTimer.singleShot(50, self.type_input.setFocus)
        else:
            self.type_target_lbl.setText("")
        self.typing_hook.activate_sentence(
            sentence_words    = words,
            sentence_start_ns = self.sentence_start_ns,
        )
        self._sentence_loaded_at = time.monotonic()  # stamp for debounce
        logger.info("Sentence %d/%d%s: '%s'", idx + 1, n, tag, sent["text"])

    def _advance_sentence(self):
        end_ns = self.clock.now_ns()
        idx    = self.sentence_index
        sent   = self.sentences[idx]

        # Save per-sentence typing stats (filled properly after metrics computed)
        self._typing_summary[idx] = {
            "words_typed": len(self.typed_words),
            "total_words": len(self.current_words),
            "corrections": 0,    # updated below after compute_sentence_metrics
            "mean_iki_ms": 0.0,  # updated below from metrics
        }

        self.typing_hook.deactivate_sentence()

        # Grace-wait: if the participant advanced quickly, trailing ASR
        # results for the last word(s) may still be in flight. Poll briefly
        # rather than flushing immediately, so we don't manufacture
        # "uncaptured audio" purely from a race between typing/advance and
        # the speech recognizer's latency. Bounded — never blocks the UI
        # for more than cfg.flush_grace_s.
        grace_deadline = time.monotonic() + self.cfg.flush_grace_s
        poll_interval  = 0.05
        while (self.synchronizer.pending_speech_count() > 0
               and time.monotonic() < grace_deadline):
            # Runs on the Qt main thread — pump the event loop instead of a
            # blocking sleep so the UI doesn't freeze during the wait.
            QApplication.processEvents()
            time.sleep(poll_interval)

        slots = self.synchronizer.flush_sentence()

        # Re-admit gaze-only partial slots so fixation features are computed.
        # The synchronizer marks PENDING/PARTIAL as TIMED_OUT + EXCLUDED, but
        # slots with a gaze_record carry valid fixation duration data that
        # contributes to F06/F07/F08 even without a speech match.
        from core.word_slot import DataQuality, SlotStatus
        for slot in slots:
            if (slot.data_quality == DataQuality.EXCLUDED
                    and slot.gaze_record is not None
                    and slot.speech_record is None):
                slot.data_quality    = DataQuality.DEGRADED   # partial but usable
                slot.exclusion_reason = None

        metrics = compute_sentence_metrics(slots, sentence_index=idx)
        self.sentence_metrics_list.append(metrics)

        # Back-fill typing stats now that metrics are available
        if idx in self._typing_summary:
            n_valid = max(metrics.n_valid_slots, 1)
            corr_rate = metrics.mean_correction_rate or 0.0
            self._typing_summary[idx]["corrections"] = int(round(corr_rate * n_valid))
            self._typing_summary[idx]["mean_iki_ms"] = (
                float(metrics.std_evs_typing_ms) if metrics.std_evs_typing_ms is not None else 0.0
            )

        self.sess_log.log_sentence(
            sentence_id    = str(uuid.uuid4()),
            sentence_index = idx,
            sentence_text  = sent["text"],
            sentence_type  = sent.get("type", "scored"),
            start_ns       = self.sentence_start_ns,
            end_ns         = end_ns,
            slots          = slots,
            metrics        = metrics,
        )

        next_idx = idx + 1
        if next_idx >= len(self.sentences):
            self._close_session()
        else:
            self._load_sentence(next_idx)

    def _close_session(self):
        end_ns    = self.clock.now_ns()
        n_prac       = self.cfg.practice_sentences
        min_complete = self.cfg.risk_bands.get("min_data_completeness", 0.60)

        extractor = FeatureExtractor(
            practice_count=n_prac,
            min_sentence_completeness=min_complete,
        )
        fv        = extractor.extract(self.sentence_metrics_list)

        scorer    = RiskScorer(
            thresholds=self.cfg.thresholds,
            weights   =self.cfg.weights,
            risk_bands=self.cfg.risk_bands,
        )
        scored = [m for m in self.sentence_metrics_list if m.sentence_index >= n_prac]
        completeness = (
            sum(m.data_completeness for m in scored) / len(scored)
            if scored else 0.0
        )
        # A fine session-wide average can still hide one badly-captured
        # sentence (e.g. 1.0, 1.0, 0.5 averages to 0.83, well above the
        # min threshold, even though that 0.5 sentence's speech data was
        # mostly lost). Gate on the worst sentence too, not just the mean —
        # the FeatureExtractor already excluded it from aggregation above,
        # so this just makes sure the result is honestly labelled.
        worst_complete = min((m.data_completeness for m in scored), default=1.0)
        effective_completeness = min(completeness, worst_complete)
        risk = scorer.score(fv, data_completeness=effective_completeness)

        self.sess_log.log_risk_score(risk, fv)
        self.sess_log.finalize_session(
            end_ns     = end_ns,
            risk_score = risk.risk_score,
            risk_band  = risk.risk_band,
            validity   = completeness >= 0.6,
            full_export= {
                "session_id":     self.session_id,
                "participant_id": self.participant_id,
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "risk_band":      risk.risk_band,
                "avg_fusion_score": risk.risk_score,
                "avg_eye_conf":   _avg_indicator(risk, ("F06", "F08")),
                "avg_type_conf":  _avg_indicator(risk, ("F01",)),
                "avg_audio_conf": _avg_indicator(risk, ("F04",)),
                # Per-sentence eye/type/audio confidence and a per-sentence
                # risk_label don't exist yet (RiskScorer only scores at
                # session level) — final_score/risk_label below are
                # placeholders (data_completeness / session risk_band)
                # so the /results page doesn't crash on missing keys.
                # Replace with real per-sentence scoring if the demo needs it.
                "per_sentence": [
                    {
                        "sentence_index": m.sentence_index,
                        "final_score":    m.data_completeness,
                        "risk_label":     risk.risk_band,
                        "eye_conf":       m.mean_fixation_duration_ms or 0.0,
                        "type_conf":      m.mean_evs_typing_ms or 0.0,
                        "audio_conf":     m.mean_evs_speech_ms or 0.0,
                    }
                    for m in self.sentence_metrics_list
                ],
            },
        )
        self.exporter.export_session_summary()
        self.exporter.export_word_records()
        self.exporter.export_sentence_records()

        print("\n" + "=" * 60)
        print(f"SESSION COMPLETE  —  {self.session_id[:8]}")
        print(f"Risk Score   : {risk.risk_score:.4f}")
        print(f"Risk Band    : {risk.risk_band}")
        print(f"Completeness : {completeness:.0%}")
        print(f"EVS Gaze Span: {self.gaze.calibration_span:.4f}")
        print(f"Calib Quality: {self.gaze.calibration_quality:.1%}")
        print(f"\n{risk.disclaimer}")
        print("=" * 60 + "\n")

        self.status_lbl.setText(
            f"SESSION COMPLETE  |  Risk: {risk.risk_band}  "
            f"({risk.risk_score:.3f})  |  Loading results dashboard..."
        )
        self.status_lbl.setStyleSheet(
            "color: #00b894; padding: 15px; "
            "background-color: #2d2d2d; border-radius: 8px; font-weight: bold;"
        )

        # ── Launch Results Dashboard ──────────────────────────────────────────
        self._results_dashboard = ResultsDashboard(
            participant_id        = self.participant_id,
            session_id            = self.session_id,
            risk                  = risk,
            fv                    = fv,
            sentence_metrics_list = self.sentence_metrics_list,
            completeness          = completeness,
            gaze_span             = self.gaze.calibration_span,
            gaze_quality          = self.gaze.calibration_quality,
            typing_summary        = self._typing_summary,
        )
        self._results_dashboard.set_exporter(self.cfg.export_dir)
        self._results_dashboard.show()
        self._results_dashboard.raise_()

    def _reset_sentence(self):
        if not self.session_started:
            return
        self.fixation_log = []
        self.typed_words  = []
        self.type_input.clear()
        if self.current_words:
            self.type_target_lbl.setText(f"\u2192 word 1: '{self.current_words[0]}'")
            QTimer.singleShot(50, self.type_input.setFocus)
        self.fixation.reset()
        self.audio.reset_sentence(self.current_words)
        self.status_lbl.setText("Sentence reset — press ENTER when done reading")

    # ── Typing input handler ──────────────────────────────────────────────────

    def _on_type_changed(self, text: str):
        """Fires on every keystroke. SPACE submits the current word."""
        if not self.session_started or not self.current_words:
            return
        if " " in text:
            submitted = text.strip()
            if submitted:
                self.typed_words.append(submitted.lower())
            self.type_input.clear()
            next_idx = len(self.typed_words)
            if next_idx < len(self.current_words):
                self.type_target_lbl.setText(
                    f"→ word {next_idx + 1}: '{self.current_words[next_idx]}'"
                )
            else:
                self.type_target_lbl.setText("✓ all words typed")

    # ── QTimer camera loop (mirrors update_loop from ReadingScreen) ───────────

    def _update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        current_word_index = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            gaze_val  = self.gaze.estimate_normalized_gaze(landmarks)
            self.diag_lbl.setText(self.gaze.get_live_diagnostics())

            if gaze_val is not None:
                if self.calibrating:
                    self._update_calibration_progress()
                elif self.gaze.is_calibrated() and self.session_started:
                    screen_x = self.gaze.map_gaze_to_screen(gaze_val)
                    if screen_x is not None:
                        word_idx = self.gaze.get_word_index(screen_x)
                        current_word_index = word_idx

                        fix = self.fixation.update(word_idx, self.gaze)
                        if fix is not None:
                            self.fixation_log.append(fix)
                            # Publish to bus so synchronizer records it in the WordSlot
                            self.bus.publish(Event(
                                event_type   = EventType.GAZE_FIXATION,
                                timestamp_ns = self.clock.now_ns(),  # session-relative
                                payload      = {
                                    "word_idx":        fix["word_index"],
                                    "duration_ms":     fix["duration"] * 1000.0,
                                    "centroid_x":      0.0,
                                    "centroid_y":      0.0,
                                    "regression_flag": fix.get("regression", False),
                                    "frame_id":        0,
                                },
                            ))

        # gaze indicator
        if current_word_index is not None and current_word_index < len(self.current_words):
            self.gaze_word_lbl.setText(
                f"Gaze: [{current_word_index}] '{self.current_words[current_word_index]}'"
            )
        else:
            self.gaze_word_lbl.setText("Gaze: —")

        self._update_word_colors(current_word_index)

    # Word colours: 3-channel (gaze / speech / typing)
    def _update_word_colors(self, gazed_word_index: Optional[int]):
        with self.audio.lock:
            spoken_words = set(self.audio.voice_log.keys())
        fixated_indices = {f["word_index"] for f in self.fixation_log}
        typed_indices   = set(range(len(self.typed_words)))   # words typed so far

        for idx, lbl in enumerate(self.word_labels):
            word       = self.current_words[idx] if idx < len(self.current_words) else ""
            is_gazed   = (idx == gazed_word_index)
            is_spoken  = word in spoken_words
            is_fixated = idx in fixated_indices
            is_typed   = idx in typed_indices

            if is_gazed and is_spoken and is_typed:
                # All three channels — white on green border (full confidence)
                lbl.setStyleSheet(
                    "color: #ffffff; padding: 8px; font-weight: bold; "
                    "background-color: rgba(0, 184, 148, 0.35); border-radius: 6px; "
                    "border: 2px solid #00b894;"
                )
            elif is_gazed and is_spoken:
                lbl.setStyleSheet(
                    "color: #00b894; padding: 8px; font-weight: bold; "
                    "background-color: rgba(0, 184, 148, 0.25); border-radius: 6px;"
                )
            elif is_gazed:
                lbl.setStyleSheet(
                    "color: #74b9ff; padding: 8px; font-weight: bold; "
                    "background-color: rgba(116, 185, 255, 0.25); border-radius: 6px; "
                    "border: 2px solid #74b9ff;"
                )
            elif is_typed and is_spoken:
                lbl.setStyleSheet("color: #a29bfe; padding: 8px; font-weight: bold;")  # purple
            elif is_typed:
                lbl.setStyleSheet("color: #fd79a8; padding: 8px; font-weight: bold;")  # pink
            elif is_fixated and is_spoken:
                lbl.setStyleSheet("color: #55efc4; padding: 8px; font-weight: bold;")
            elif is_spoken:
                lbl.setStyleSheet("color: #fdcb6e; padding: 8px; font-weight: bold;")
            elif is_fixated:
                lbl.setStyleSheet("color: #81ecec; padding: 8px;")
            else:
                lbl.setStyleSheet("color: #f1f2f6; padding: 8px;")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.timer.stop()
        self.audio.stop()
        self.typing_hook.stop()
        self.synchronizer.stop()
        self.sess_log.stop()
        self.db.close()
        self.cap.release()
        self.face_mesh.close()
        self.clock.reset()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MESTTS Session")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--config", default="config/experiment_config.yaml")
    parser.add_argument("--model", default=None,
                        help="Path to Vosk model directory (overrides config)")
    args = parser.parse_args()

    cfg = ExperimentConfig(args.config)
    _setup_logging(cfg.log_dir, cfg.log_level)
    logger.info("Starting session for participant %s", args.participant)

    sentences = json.loads(
        Path(cfg.protocol_path).read_text()
    )["sentences"]

    # Bus must exist before AudioProcessor so speech events can be published
    from core.event_bus import get_bus as _get_bus
    _bus = _get_bus()
    audio = AudioProcessor(cfg, words=sentences[0]["text"].split(),
                           model_path_override=args.model, bus=_bus,
                           clock=get_clock())
    audio.start()

    app    = QApplication(sys.argv)
    window = SessionWindow(cfg, args.participant, sentences, audio)
    window.show()
    window.setFocus()
    window.activateWindow()
    app.exec_()


if __name__ == "__main__":
    main()
