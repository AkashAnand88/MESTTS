"""
core/state_machine.py
─────────────────────────────────────────────────────────────────────────────
Session / sentence finite-state machine (FSM).

Implements the state machine defined in the architecture specification §5.
All state transitions are logged with timestamps and optionally fire
registered callbacks.

States
------
IDLE → CALIBRATING → CALIBRATED → SESSION_START →
SENTENCE_LOAD → SENTENCE_ACTIVE → SENTENCE_CLOSE →
SESSION_CLOSE | ERROR

Transitions are driven by FSMEvent signals sent from the orchestrator
and hardware event handlers.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SystemState(Enum):
    IDLE             = auto()
    CALIBRATING      = auto()
    CALIBRATED       = auto()
    SESSION_START    = auto()
    SENTENCE_LOAD    = auto()
    SENTENCE_ACTIVE  = auto()
    AWAITING_TRIGGER = auto()
    SENTENCE_CLOSE   = auto()
    SESSION_CLOSE    = auto()
    ERROR            = auto()


class FSMEvent(Enum):
    START_CALIBRATION    = auto()
    CALIBRATION_COMPLETE = auto()
    CALIBRATION_FAILED   = auto()
    SESSION_STARTED      = auto()
    SENTENCE_LOADED      = auto()
    WORDS_PROCESSED      = auto()   # All words received by synchronizer
    PHYSICAL_TRIGGER     = auto()   # Operator presses pedal/key
    SENTENCE_PROCESSED   = auto()   # TAE + FEL + RSE complete
    MORE_SENTENCES       = auto()
    NO_MORE_SENTENCES    = auto()
    TIMEOUT              = auto()
    ERROR                = auto()


# Transition table: (current_state, event) → next_state
_TRANSITIONS: Dict[Tuple[SystemState, FSMEvent], SystemState] = {
    (SystemState.IDLE,             FSMEvent.START_CALIBRATION):    SystemState.CALIBRATING,
    (SystemState.CALIBRATING,      FSMEvent.CALIBRATION_COMPLETE): SystemState.CALIBRATED,
    (SystemState.CALIBRATING,      FSMEvent.CALIBRATION_FAILED):   SystemState.CALIBRATING,
    (SystemState.CALIBRATED,       FSMEvent.SESSION_STARTED):      SystemState.SESSION_START,
    (SystemState.SESSION_START,    FSMEvent.SESSION_STARTED):      SystemState.SENTENCE_LOAD,
    (SystemState.SENTENCE_LOAD,    FSMEvent.SENTENCE_LOADED):      SystemState.SENTENCE_ACTIVE,
    (SystemState.SENTENCE_ACTIVE,  FSMEvent.WORDS_PROCESSED):      SystemState.AWAITING_TRIGGER,
    (SystemState.SENTENCE_ACTIVE,  FSMEvent.TIMEOUT):              SystemState.AWAITING_TRIGGER,
    (SystemState.AWAITING_TRIGGER, FSMEvent.PHYSICAL_TRIGGER):     SystemState.SENTENCE_CLOSE,
    (SystemState.SENTENCE_CLOSE,   FSMEvent.SENTENCE_PROCESSED):   SystemState.SENTENCE_LOAD,
    (SystemState.SENTENCE_CLOSE,   FSMEvent.MORE_SENTENCES):       SystemState.SENTENCE_LOAD,
    (SystemState.SENTENCE_CLOSE,   FSMEvent.NO_MORE_SENTENCES):    SystemState.SESSION_CLOSE,
    # Error transitions from any state
    (SystemState.SENTENCE_ACTIVE,  FSMEvent.ERROR):                SystemState.ERROR,
    (SystemState.SENTENCE_CLOSE,   FSMEvent.ERROR):                SystemState.ERROR,
}

# Callback type: called with (previous_state, **kwargs)
StateCallback = Callable[..., None]


class StateMachine:
    """
    Session FSM.

    Parameters
    ----------
    total_sentences : int
    sentence_timeout_s : float
        Seconds before auto-advance to AWAITING_TRIGGER.
    """

    def __init__(
        self,
        total_sentences:    int,
        sentence_timeout_s: float = 60.0,
    ) -> None:
        self._state             = SystemState.IDLE
        self._total             = total_sentences
        self._sentence_index    = 0
        self._timeout_s         = sentence_timeout_s
        self._callbacks: Dict[SystemState, List[StateCallback]] = {}
        self._history: List[Tuple[float, SystemState, SystemState, str]] = []
        self._lock              = threading.Lock()
        self._words_ready       = False
        self._trigger_received  = False
        self._timeout_timer: Optional[threading.Timer] = None

        logger.info("FSM initialized. Total sentences: %d.", total_sentences)

    # ── Public Interface ──────────────────────────────────────────────────────

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def sentence_index(self) -> int:
        return self._sentence_index

    def register_callback(self, state: SystemState, callback: StateCallback) -> None:
        """Register a callback to fire when FSM enters the given state."""
        self._callbacks.setdefault(state, []).append(callback)

    def send(self, event: FSMEvent, **kwargs) -> bool:
        """
        Send an event to the FSM.

        Returns True if transition was performed, False if ignored.
        """
        with self._lock:
            key = (self._state, event)
            next_state = _TRANSITIONS.get(key)

            if next_state is None:
                logger.debug(
                    "FSM: No transition from %s on %s — ignored.",
                    self._state.name, event.name,
                )
                return False

            previous = self._state
            self._state = next_state
            self._history.append((time.monotonic(), previous, next_state, event.name))

            logger.info(
                "FSM: %s —[%s]→ %s",
                previous.name, event.name, next_state.name,
            )

        # Fire callbacks outside lock
        for cb in self._callbacks.get(next_state, []):
            try:
                cb(previous, **kwargs)
            except Exception as exc:
                logger.exception("FSM callback error on %s: %s", next_state.name, exc)

        return True

    def signal_words_processed(self) -> None:
        """Called by SentenceSynchronizer when all word slots are complete."""
        with self._lock:
            self._words_ready = True
            if self._state == SystemState.SENTENCE_ACTIVE:
                self._cancel_timeout()
        self.send(FSMEvent.WORDS_PROCESSED)

    def signal_physical_trigger(self) -> None:
        """Called by display when operator presses the advance key."""
        with self._lock:
            self._trigger_received = True
        if self._state == SystemState.AWAITING_TRIGGER:
            self.send(FSMEvent.PHYSICAL_TRIGGER)
        # If words not yet ready, wait — physical trigger queued
        elif self._state == SystemState.SENTENCE_ACTIVE:
            logger.debug("Trigger received before words processed; will fire on words_processed.")

    def decide_next_after_close(self) -> None:
        """
        After sentence processing completes, determine whether to
        load the next sentence or close the session.
        """
        with self._lock:
            more = self._sentence_index + 1 < self._total

        if more:
            with self._lock:
                self._sentence_index += 1
                self._words_ready      = False
                self._trigger_received = False
            self.send(FSMEvent.MORE_SENTENCES)
        else:
            self.send(FSMEvent.NO_MORE_SENTENCES)

    def start_sentence_timeout(self) -> None:
        """Start a watchdog timer; auto-sends TIMEOUT if sentence stalls."""
        self._cancel_timeout()
        self._timeout_timer = threading.Timer(self._timeout_s, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()
        logger.debug("Sentence timeout timer started (%.1f s).", self._timeout_s)

    def _cancel_timeout(self) -> None:
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _on_timeout(self) -> None:
        logger.warning(
            "Sentence %d timed out after %.1f s — advancing to AWAITING_TRIGGER.",
            self._sentence_index, self._timeout_s,
        )
        self.send(FSMEvent.TIMEOUT)

    def history(self) -> List[Tuple[float, SystemState, SystemState, str]]:
        """Return list of (monotonic_ts, from_state, to_state, event_name)."""
        return list(self._history)
