"""
modules/typing/hook.py
─────────────────────────────────────────────────────────────────────────────
TypingHook — pynput keyboard listener, per-word character buffer,
word submission on SPACE/ENTER, IKI tracking, backspace counting.

Architecture spec §Layer 1.3, §Layer 3.3.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from core.clock import get_clock
from core.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)

try:
    from pynput import keyboard as _kb
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False
    logger.warning("pynput not installed; TypingHook will be a no-op.")


class TypingHook(threading.Thread):
    """
    Keyboard listener thread.

    Accumulates characters into a per-word buffer. On SPACE or ENTER,
    emits a TYPING_WORD event to the EventBus with the submitted word,
    latency, backspace count, and IKI list.

    Parameters
    ----------
    bus : EventBus
    """

    def __init__(self, bus: EventBus) -> None:
        super().__init__(name="TypingHook", daemon=True)
        self._bus    = bus
        self._clock  = get_clock()

        # Sentence state
        self._active:            bool       = False
        self._sentence_words:    List[str]  = []
        self._sentence_start_ns: int        = 0
        self._word_index:        int        = 0

        # Per-word state
        self._buffer:            List[str]  = []
        self._first_key_ns:      Optional[int] = None
        self._backspace_count:   int        = 0
        self._iki_list:          List[float] = []
        self._last_key_ns:       Optional[int] = None

        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._listener   = None

    # ── Sentence Control ──────────────────────────────────────────────────────

    def activate_sentence(
        self,
        sentence_words:    List[str],
        sentence_start_ns: int,
    ) -> None:
        """Prepare for a new sentence."""
        with self._lock:
            self._sentence_words    = list(sentence_words)
            self._sentence_start_ns = sentence_start_ns
            self._word_index        = 0
            self._active            = True
            self._reset_word_buffer()
        logger.debug(
            "TypingHook: sentence activated (%d words).", len(sentence_words)
        )

    def deactivate_sentence(self) -> None:
        with self._lock:
            self._active = False

    def _reset_word_buffer(self) -> None:
        self._buffer          = []
        self._first_key_ns    = None
        self._backspace_count = 0
        self._iki_list        = []
        self._last_key_ns     = None

    # ── Thread Lifecycle ──────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()
        if self._listener is not None:
            self._listener.stop()

    def start(self) -> None:
        """Start the thread (starts pynput listener)."""
        super().start()

    def run(self) -> None:
        if not _HAS_PYNPUT:
            logger.warning("TypingHook: pynput unavailable; blocking until stop.")
            self._stop_event.wait()
            return

        with _kb.Listener(
            on_press   = self._on_press,
            on_release = None,
        ) as listener:
            self._listener = listener
            self._stop_event.wait()
        logger.info("TypingHook stopped.")

    # ── Key Handler ───────────────────────────────────────────────────────────

    def _on_press(self, key) -> None:
        ts_ns = self._clock.now_ns()

        with self._lock:
            if not self._active:
                return

            # IKI
            if self._last_key_ns is not None:
                iki_ms = (ts_ns - self._last_key_ns) / 1_000_000.0
                self._iki_list.append(iki_ms)
            self._last_key_ns = ts_ns

            # Characterize key
            try:
                char = key.char
            except AttributeError:
                char = None

            if key in (_kb.Key.space, _kb.Key.enter) if _HAS_PYNPUT else False:
                self._submit_word(ts_ns)
                return

            if key == (_kb.Key.backspace if _HAS_PYNPUT else None):
                if self._buffer:
                    self._buffer.pop()
                self._backspace_count += 1
                return

            if char and char.isprintable():
                if self._first_key_ns is None:
                    self._first_key_ns = ts_ns
                self._buffer.append(char)

    def _submit_word(self, ts_ns: int) -> None:
        """Flush buffer as a completed word event."""
        submitted = "".join(self._buffer).strip()
        if not submitted:
            return

        idx          = self._word_index
        words        = self._sentence_words
        target       = words[idx] if idx < len(words) else ""
        first_key_ns = self._first_key_ns or self._sentence_start_ns
        latency_ms   = (ts_ns - first_key_ns) / 1_000_000.0

        self._bus.publish(Event(
            event_type   = EventType.TYPING_WORD,
            timestamp_ns = ts_ns,
            payload={
                "word_idx":          idx,
                "submitted_word":    submitted,
                "expected_word":     target,
                "typing_timestamp_ns": ts_ns,
                "latency_ms":        latency_ms,
                "correction_count":  self._backspace_count,
                "iki_list":          list(self._iki_list),
            },
        ))
        logger.debug(
            "TYPING: idx=%d submitted='%s' target='%s' latency=%.1fms bs=%d",
            idx, submitted, target, latency_ms, self._backspace_count,
        )

        self._word_index += 1
        self._reset_word_buffer()
