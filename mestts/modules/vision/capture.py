"""
modules/vision/capture.py
─────────────────────────────────────────────────────────────────────────────
VisionCaptureThread — IPCam frame grab, MediaPipe iris extraction,
fixation detection, and word-slot assignment.

Runs at ~30 fps in a daemon thread. Publishes GAZE_FIXATION and
GAZE_REGRESSION events to the EventBus when fixation boundaries are crossed.

Architecture spec §Layer 1.1, §Layer 3.1, §Layer 4.6.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Tuple

from core.clock import get_clock
from core.event_bus import Event, EventBus, EventType
from modules.vision.calibration import GazeCalibrator
from modules.vision.fixation_detector import FixationDetector

logger = logging.getLogger(__name__)

# MediaPipe iris landmark indices (4 peripheral per eye, excluding pupil center)
LEFT_IRIS_DEFAULT  = [468, 469, 470, 471]
RIGHT_IRIS_DEFAULT = [473, 474, 475, 476]


def _iris_centroid(
    landmarks,
    indices: List[int],
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float]:
    """Compute mean (x, y) of iris landmarks in pixel space."""
    pts = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in indices]
    cx  = sum(p[0] for p in pts) / len(pts)
    cy  = sum(p[1] for p in pts) / len(pts)
    return cx, cy


class VisionCaptureThread(threading.Thread):
    """
    Daemon thread: captures IPCam frames, extracts iris landmarks via
    MediaPipe FaceMesh, applies calibration, runs FixationDetector, and
    publishes gaze events to the EventBus.

    Parameters
    ----------
    ipcam_url : str
        OpenCV-compatible camera URL (RTSP or HTTP MJPEG).
    calibrator : GazeCalibrator
    bus : EventBus
    camera_latency_ns : int
        Estimated pipeline latency in nanoseconds (applied as correction).
    target_fps : int
    buffer_frames : int
        Number of consecutive frames required to confirm a fixation.
    spatial_threshold : float
        Maximum spatial dispersion (screen pixels) for fixation detection.
    regression_delta : int
        Minimum word-index step backward to count as regression.
    left_iris_indices, right_iris_indices : list[int]
        MediaPipe FaceMesh landmark indices.
    max_dropped_pct : float
        Session-end warning threshold for dropped frame rate.
    """

    def __init__(
        self,
        ipcam_url:          str,
        calibrator:         GazeCalibrator,
        bus:                EventBus,
        camera_latency_ns:  int   = 50_000_000,
        target_fps:         int   = 30,
        buffer_frames:      int   = 3,
        spatial_threshold:  float = 40.0,
        regression_delta:   int   = 1,
        left_iris_indices:  Optional[List[int]] = None,
        right_iris_indices: Optional[List[int]] = None,
        max_dropped_pct:    float = 0.20,
    ) -> None:
        super().__init__(name="VisionCapture", daemon=True)
        self._url              = ipcam_url
        self._calibrator       = calibrator
        self._bus              = bus
        self._latency_ns       = camera_latency_ns
        self._target_fps       = target_fps
        self._max_dropped_pct  = max_dropped_pct
        self._clock            = get_clock()

        self._left_iris  = left_iris_indices  or LEFT_IRIS_DEFAULT
        self._right_iris = right_iris_indices or RIGHT_IRIS_DEFAULT

        self._fixation_detector = FixationDetector(
            buffer_frames      = buffer_frames,
            spatial_threshold  = spatial_threshold,
            regression_delta   = regression_delta,
        )

        # Sentence state
        self._active          = False
        self._n_words:        int = 0
        self._word_bboxes:    List[Tuple[int,int,int,int]] = []  # (x1,y1,x2,y2)
        self._sentence_lock   = threading.Lock()

        # Calibration sample collection mode
        # Set _calib_collecting=True and read from _calib_queue
        self._calib_collecting = False
        self._calib_queue: List[Tuple[float,float]] = []
        self._calib_lock   = threading.Lock()

        # Metrics
        self._frame_count  = 0
        self._dropped_count = 0
        self._stop_event   = threading.Event()

    # ── Thread Lifecycle ──────────────────────────────────────────────────────

    def stop(self) -> None:
        """Request thread shutdown."""
        self._stop_event.set()

    def activate_sentence(
        self,
        n_words:    int,
        word_bboxes: Optional[List[Tuple[int,int,int,int]]] = None,
    ) -> None:
        """Prepare for a new sentence. Must be called from orchestrator thread."""
        with self._sentence_lock:
            self._n_words     = n_words
            self._word_bboxes = word_bboxes or []
            self._active      = True
        self._fixation_detector.reset()
        logger.debug("VisionCapture: sentence activated (%d words).", n_words)

    def deactivate_sentence(self) -> None:
        """Stop recording for the current sentence."""
        with self._sentence_lock:
            self._active = False
        logger.debug("VisionCapture: sentence deactivated.")

    def set_word_bboxes(self, bboxes: List[Tuple[int,int,int,int]]) -> None:
        """Update word bounding boxes (called from display thread after render)."""
        with self._sentence_lock:
            self._word_bboxes = bboxes

    def start_calib_collection(self) -> None:
        """Enable calibration sample collection mode."""
        with self._calib_lock:
            self._calib_queue.clear()
            self._calib_collecting = True

    def stop_calib_collection(self) -> Optional[Tuple[float, float]]:
        """
        Disable calibration collection and return mean iris centroid,
        or None if no samples were collected.
        """
        with self._calib_lock:
            self._calib_collecting = False
            samples = list(self._calib_queue)
            self._calib_queue.clear()
        if not samples:
            return None
        mean_x = sum(s[0] for s in samples) / len(samples)
        mean_y = sum(s[1] for s in samples) / len(samples)
        logger.debug("Calib collection: %d samples, mean=(%.1f, %.1f).", len(samples), mean_x, mean_y)
        return mean_x, mean_y

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            logger.error("opencv-python and mediapipe are required for vision capture.")
            return

        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            logger.error("Cannot open camera: %s", self._url)
            self._bus.publish(Event(
                event_type=EventType.SYSTEM_ERROR,
                timestamp_ns=self._clock.now_ns(),
                payload={"source": "VisionCapture", "error": "Cannot open camera"},
            ))
            return

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh    = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
        )

        frame_interval = 1.0 / self._target_fps
        logger.info("VisionCaptureThread started (url=%s, fps=%d).", self._url, self._target_fps)

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            self._frame_count += 1

            ret, frame = cap.read()
            if not ret:
                self._dropped_count += 1
                logger.debug("VisionCapture: frame drop #%d.", self._dropped_count)
                time.sleep(frame_interval)
                continue

            raw_ts_ns = self._clock.now_ns() - self._latency_ns

            h, w = frame.shape[:2]
            rgb   = frame[:, :, ::-1]  # BGR → RGB

            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                time.sleep(max(0, frame_interval - (time.monotonic() - loop_start)))
                continue

            lm  = results.multi_face_landmarks[0].landmark
            lx, ly = _iris_centroid(lm, self._left_iris,  w, h)
            rx, ry = _iris_centroid(lm, self._right_iris, w, h)
            mean_ix = (lx + rx) / 2.0
            mean_iy = (ly + ry) / 2.0

            # Feed calibration queue if in collection mode
            with self._calib_lock:
                if self._calib_collecting:
                    self._calib_queue.append((mean_ix, mean_iy))

            # Calibrate
            if self._calibrator.is_fitted:
                try:
                    sx, sy = self._calibrator.map(mean_ix, mean_iy)
                except Exception:
                    sx, sy = mean_ix, mean_iy
            else:
                sx, sy = mean_ix, mean_iy

            # Slot assignment
            word_idx = self._assign_slot(sx, sy)

            # Sentence active?
            with self._sentence_lock:
                is_active = self._active

            if is_active:
                fixation_events, regression_events = self._fixation_detector.update(
                    screen_x  = sx,
                    screen_y  = sy,
                    word_idx  = word_idx,
                    ts_ns     = raw_ts_ns,
                    frame_id  = self._frame_count,
                )

                for fe in fixation_events:
                    self._bus.publish(Event(
                        event_type   = EventType.GAZE_FIXATION,
                        timestamp_ns = fe["onset_ns"],
                        payload      = fe,
                    ))

                for re in regression_events:
                    self._bus.publish(Event(
                        event_type   = EventType.GAZE_REGRESSION,
                        timestamp_ns = re["saccade_ts_ns"],
                        payload      = re,
                    ))

            elapsed = time.monotonic() - loop_start
            sleep_t = max(0, frame_interval - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()
        face_mesh.close()

        drop_pct = self._dropped_count / max(1, self._frame_count)
        logger.info(
            "VisionCaptureThread stopped. Frames=%d Dropped=%d (%.1f%%).",
            self._frame_count, self._dropped_count, drop_pct * 100,
        )
        if drop_pct > self._max_dropped_pct:
            logger.warning("Excessive frame drop rate: %.1f%%.", drop_pct * 100)

    def _assign_slot(self, sx: float, sy: float) -> Optional[int]:
        """Point-in-rectangle test against word bounding boxes."""
        with self._sentence_lock:
            bboxes = self._word_bboxes

        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                return idx
        return None