"""
alignment/synchronizer.py
─────────────────────────────────────────────────────────────────────────────
SynchronizerThread — per-sentence word-slot join table.

Subscribes to GAZE_FIXATION, SPEECH_WORD, and TYPING_WORD events from
the EventBus and populates the WordSlot join table for the active sentence.

When all slots reach COMPLETE status (or the sentence times out), calls
`on_all_words_complete` to signal the FSM.

flush_sentence() returns the finalized WordSlot list with temporal metrics
computed for each slot.

Architecture spec §Layer 4.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

from core.clock import get_clock
from core.event_bus import Event, EventBus, EventType
from core.word_slot import (
    DataQuality,
    GazeRecord,
    SlotStatus,
    SpeechRecord,
    TypingRecord,
    WordSlot,
)

logger = logging.getLogger(__name__)


class SynchronizerThread(threading.Thread):
    """
    Manages the WordSlot join table for each sentence.

    Parameters
    ----------
    bus : EventBus
    on_all_words_complete : callable
        Called (with no arguments) when all slots reach COMPLETE.
    """

    def __init__(
        self,
        bus:                     EventBus,
        on_all_words_complete:   Callable[[], None],
    ) -> None:
        super().__init__(name="Synchronizer", daemon=True)
        self._bus                   = bus
        self._on_all_words_complete = on_all_words_complete
        self._clock                 = get_clock()

        self._slots:          List[WordSlot] = []
        self._n_words:        int            = 0
        self._active:         bool           = False
        self._complete_fired: bool           = False
        self._lock           = threading.Lock()
        self._stop_event     = threading.Event()

    # ── Thread Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Subscribe to bus events and start thread."""
        self._bus.subscribe(EventType.GAZE_FIXATION, self._on_gaze_fixation)
        self._bus.subscribe(EventType.GAZE_REGRESSION, self._on_gaze_regression)
        self._bus.subscribe(EventType.SPEECH_WORD,    self._on_speech_word)
        self._bus.subscribe(EventType.TYPING_WORD,    self._on_typing_word)
        super().start()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        """Thread body — just keeps thread alive; events are dispatched via callbacks."""
        self._stop_event.wait()
        logger.info("SynchronizerThread stopped.")

    # ── Sentence Control ──────────────────────────────────────────────────────

    def activate_sentence(self, words: List[str]) -> None:
        """Initialize a fresh WordSlot table for the given sentence."""
        with self._lock:
            self._n_words  = len(words)
            self._slots    = [
                WordSlot(word_index=i, expected_word=w)
                for i, w in enumerate(words)
            ]
            self._active          = True
            self._complete_fired  = False
        logger.debug("Synchronizer: activated for %d words.", len(words))

    def pending_speech_count(self) -> int:
        """
        Number of active slots that have gaze/typing but are still
        waiting on a speech_record from the ASR pipeline.

        Callers (e.g. sentence-advance handlers) can poll this for a
        short grace window before calling flush_sentence(), so trailing
        ASR results aren't lost just because the participant advanced
        quickly. Cheap, lock-protected snapshot — safe to poll.
        """
        with self._lock:
            if not self._active:
                return 0
            return sum(
                1 for s in self._slots
                if s.speech_record is None
                and (s.gaze_record is not None or s.typing_record is not None)
            )

    def flush_sentence(self) -> List[WordSlot]:
        """
        Finalize and return all WordSlots.

        Computes temporal metrics for each slot, marks any still-PENDING
        or PARTIAL slots as TIMED_OUT.
        """
        with self._lock:
            self._active = False
            slots = list(self._slots)

        for slot in slots:
            # Resolve final status
            status = slot.resolve_status()
            if status in (SlotStatus.PENDING, SlotStatus.PARTIAL):
                slot.status = SlotStatus.TIMED_OUT
                slot.data_quality = DataQuality.EXCLUDED
                slot.exclusion_reason = "Sentence closed before slot completed."

            # Compute temporal metrics
            slot.compute_temporal_metrics()

        logger.info(
            "Synchronizer flush: %d slots, %d complete.",
            len(slots),
            sum(1 for s in slots if s.status == SlotStatus.COMPLETE),
        )
        return slots

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _on_gaze_fixation(self, event: Event) -> None:
        p = event.payload
        word_idx = p.get("word_idx")
        if word_idx is None:
            return

        with self._lock:
            if not self._active or word_idx >= len(self._slots):
                return
            slot = self._slots[word_idx]

            if slot.gaze_record is not None:
                return  # First fixation wins

            slot.gaze_record = GazeRecord(
                word_index           = word_idx,
                fixation_onset_ns    = event.timestamp_ns,
                fixation_duration_ms = p.get("duration_ms", 0.0),
                gaze_x_norm          = p.get("centroid_x", 0.0),
                gaze_y_norm          = p.get("centroid_y", 0.0),
                regression_flag      = p.get("regression_flag", False),
                frame_id             = p.get("frame_id", 0),
                raw_onset_ns         = event.timestamp_ns,
            )

            if p.get("regression_flag"):
                slot.data_quality = DataQuality.DEGRADED

        self._check_completion()

    def _on_gaze_regression(self, event: Event) -> None:
        p = event.payload
        to_idx = p.get("to_word_idx")
        if to_idx is None:
            return
        with self._lock:
            if not self._active or to_idx >= len(self._slots):
                return
            slot = self._slots[to_idx]
            if slot.gaze_record is not None:
                slot.gaze_record.regression_flag = True
                slot.data_quality = DataQuality.DEGRADED

    def _on_speech_word(self, event: Event) -> None:
        p = event.payload
        word_idx = p.get("word_idx")
        if word_idx is None:
            return

        with self._lock:
            if not self._active or word_idx >= len(self._slots):
                return
            slot = self._slots[word_idx]
            if slot.speech_record is not None:
                return

            conf = p.get("vosk_confidence", 1.0)
            slot.speech_record = SpeechRecord(
                word_index          = word_idx,
                speech_timestamp_ns = p.get("speech_timestamp_ns", event.timestamp_ns),
                vosk_confidence     = conf,
                raw_word            = p.get("word_raw", ""),
                matched_word        = p.get("matched_word", ""),
                fuzzy_score         = p.get("fuzzy_score", 0),
            )
            if conf < 0.5:
                slot.data_quality = DataQuality.DEGRADED

        self._check_completion()

    def _on_typing_word(self, event: Event) -> None:
        p = event.payload
        word_idx = p.get("word_idx")
        if word_idx is None:
            return

        with self._lock:
            if not self._active or word_idx >= len(self._slots):
                return
            slot = self._slots[word_idx]
            if slot.typing_record is not None:
                return

            slot.typing_record = TypingRecord(
                word_index          = word_idx,
                typing_timestamp_ns = p.get("typing_timestamp_ns", event.timestamp_ns),
                latency_ms          = p.get("latency_ms", 0.0),
                correction_count    = p.get("correction_count", 0),
                submitted_word      = p.get("submitted_word", ""),
                expected_word       = p.get("expected_word", ""),
            )

        self._check_completion()

    def _check_completion(self) -> None:
        """Fire on_all_words_complete if all slots are now COMPLETE."""
        with self._lock:
            if self._complete_fired or not self._active:
                return
            all_done = all(
                s.resolve_status() == SlotStatus.COMPLETE
                for s in self._slots
            )
            if not all_done:
                return
            self._complete_fired = True

        logger.info("All word slots complete — signaling FSM.")
        try:
            self._on_all_words_complete()
        except Exception as exc:
            logger.exception("on_all_words_complete callback error: %s", exc)
