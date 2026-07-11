"""
core/event_bus.py
─────────────────────────────────────────────────────────────────────────────
Thread-safe publish/subscribe message broker (EventBus).

All acquisition modules publish typed events to the bus. The synchronizer
and any other subscriber consume them. This decouples producers from
consumers and enables modular thread architecture.

Event types are defined in EventType. Each event carries a monotonic
session timestamp, an event type, and a payload dict.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(Enum):
    # Vision events
    GAZE_FIXATION      = auto()   # A fixation on a word slot was detected
    GAZE_REGRESSION    = auto()   # A backward saccade was detected
    GAZE_FRAME         = auto()   # Raw frame event (debug/logging only)

    # Speech events
    SPEECH_WORD        = auto()   # A word was recognized and aligned

    # Typing events
    TYPING_WORD        = auto()   # A word was submitted (SPACE/ENTER)
    TYPING_KEYPRESS    = auto()   # Individual key (for IKI logging)

    # Session control events
    SENTENCE_ACTIVATED = auto()   # A new sentence is now active
    SENTENCE_CLOSED    = auto()   # Sentence recording is complete
    SESSION_STARTED    = auto()
    SESSION_CLOSED     = auto()

    # Error / system events
    SYSTEM_ERROR       = auto()
    CALIBRATION_DONE   = auto()


@dataclass
class Event:
    """Single event published on the bus."""
    event_type: EventType
    timestamp_ns: int                  # session-epoch nanoseconds
    payload: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Event({self.event_type.name}, ts={self.timestamp_ns}, payload={self.payload})"


# Type alias for subscriber callbacks
Subscriber = Callable[[Event], None]


class EventBus:
    """
    Thread-safe publish/subscribe event broker.

    Subscribers register callbacks for specific event types.
    Published events are dispatched synchronously in the publisher's thread,
    then also pushed to a raw event queue for persistence.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Subscriber]] = {}
        self._raw_queue: Queue[Event] = Queue(maxsize=10_000)
        self._error_events: List[Event] = []
        self._lock = threading.RLock()

    def subscribe(self, event_type: EventType, callback: Subscriber) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)
        logger.debug("Subscribed %s to %s.", callback, event_type.name)

    def unsubscribe(self, event_type: EventType, callback: Subscriber) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, event: Event) -> None:
        """
        Publish an event to all registered subscribers.

        Callbacks are invoked synchronously in the publisher's thread.
        Exceptions in callbacks are caught and logged to avoid killing
        the publisher thread.
        """
        if event.event_type == EventType.SYSTEM_ERROR:
            with self._lock:
                self._error_events.append(event)

        # Push to raw queue (non-blocking; drop if full)
        try:
            self._raw_queue.put_nowait(event)
        except Exception:
            logger.warning("Event queue full; dropping event %s.", event.event_type.name)

        # Dispatch to subscribers
        with self._lock:
            callbacks = list(self._subscribers.get(event.event_type, []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.exception(
                    "Subscriber %s raised exception on event %s: %s",
                    cb, event.event_type.name, exc,
                )

    def drain(self, timeout: float = 0.0) -> List[Event]:
        """
        Drain all events currently in the raw queue.

        Parameters
        ----------
        timeout : float
            Seconds to wait for the first event. 0 = non-blocking.

        Returns
        -------
        list[Event]
        """
        events: List[Event] = []
        try:
            first = self._raw_queue.get(timeout=timeout) if timeout > 0 else self._raw_queue.get_nowait()
            events.append(first)
        except Empty:
            return events

        while True:
            try:
                events.append(self._raw_queue.get_nowait())
            except Empty:
                break
        return events

    def drain_errors(self) -> List[Event]:
        """Return and clear accumulated SYSTEM_ERROR events."""
        with self._lock:
            errs = list(self._error_events)
            self._error_events.clear()
        return errs

    def flush_raw_queue(self) -> List[Event]:
        """Drain the entire raw queue. Used at sentence close for persistence."""
        events: List[Event] = []
        while True:
            try:
                events.append(self._raw_queue.get_nowait())
            except Empty:
                break
        return events

    def clear(self) -> None:
        """Clear subscribers and queues (call at session reset)."""
        with self._lock:
            self._subscribers.clear()
            self._error_events.clear()
        while not self._raw_queue.empty():
            try:
                self._raw_queue.get_nowait()
            except Empty:
                break


# ── Singleton ─────────────────────────────────────────────────────────────────

_bus_instance: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_bus() -> EventBus:
    """Return the global EventBus singleton."""
    global _bus_instance
    with _bus_lock:
        if _bus_instance is None:
            _bus_instance = EventBus()
    return _bus_instance
