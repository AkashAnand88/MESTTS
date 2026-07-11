"""
core/clock.py
─────────────────────────────────────────────────────────────────────────────
Monotonic session clock (singleton).

Provides nanosecond-resolution monotonic timestamps anchored to a session
epoch. All modules in MESTTS use this clock exclusively to ensure temporal
consistency across threads.

Design:
  - Session epoch (epoch_ns) is recorded at session start as both
    monotonic_ns() and datetime.utcnow() for post-hoc wall-clock mapping.
  - All per-event timestamps are expressed as offsets from epoch_ns in ns.
  - Thread-safe: now_ns() can be called from any thread concurrently.
"""

from __future__ import annotations

import datetime
import threading
import time
from typing import Optional


class SessionClock:
    """
    Monotonic session clock.

    Usage:
        clock = get_clock()
        epoch_ns = clock.start()        # call once at session start
        ts_ns    = clock.now_ns()       # call anywhere during session
        clock.reset()                   # call at session end
    """

    def __init__(self) -> None:
        self._epoch_ns: int = 0
        self._epoch_wall: Optional[datetime.datetime] = None
        self._started: bool = False
        self._lock = threading.Lock()

    def start(self) -> int:
        """
        Record the session epoch. Returns epoch_ns.

        Must be called once before now_ns(). Calling again resets the epoch.
        """
        with self._lock:
            self._epoch_ns   = time.monotonic_ns()
            self._epoch_wall = datetime.datetime.utcnow()
            self._started    = True
        return self._epoch_ns

    @property
    def epoch_ns(self) -> int:
        """Monotonic epoch timestamp in nanoseconds."""
        return self._epoch_ns

    @property
    def epoch_wall(self) -> Optional[datetime.datetime]:
        """UTC wall-clock time corresponding to epoch_ns."""
        return self._epoch_wall

    @property
    def started(self) -> bool:
        return self._started

    def now_ns(self) -> int:
        """
        Return current monotonic time as nanoseconds since session epoch.

        Returns raw monotonic_ns() if clock not yet started.
        """
        if not self._started:
            return time.monotonic_ns()
        return time.monotonic_ns() - self._epoch_ns

    def elapsed_ms(self) -> float:
        """Milliseconds since session epoch."""
        return self.now_ns() / 1_000_000.0

    def to_wall_clock(self, session_ns: int) -> Optional[datetime.datetime]:
        """Convert session-epoch nanosecond offset to UTC wall-clock datetime."""
        if self._epoch_wall is None:
            return None
        delta = datetime.timedelta(microseconds=session_ns / 1_000.0)
        return self._epoch_wall + delta

    def reset(self) -> None:
        """Reset clock (call at session end or between sessions)."""
        with self._lock:
            self._epoch_ns   = 0
            self._epoch_wall = None
            self._started    = False


def raw_now_ns() -> int:
    """Return raw monotonic_ns() without session offset. Use for pre-epoch timestamps."""
    return time.monotonic_ns()


# ── Singleton ─────────────────────────────────────────────────────────────────

_clock_instance: Optional[SessionClock] = None
_clock_lock = threading.Lock()


def get_clock() -> SessionClock:
    """Return the global SessionClock singleton."""
    global _clock_instance
    with _clock_lock:
        if _clock_instance is None:
            _clock_instance = SessionClock()
    return _clock_instance
