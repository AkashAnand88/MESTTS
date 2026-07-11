"""
display/sentence_renderer.py
-----------------------------------------------------------------------------
SentenceRenderer - Tkinter fullscreen stimulus display.

Design rule: ZERO blocking calls on the main thread after mainloop() starts.
No time.sleep(), no root.update() inside callbacks. All sequencing is done
via root.after() chains.

show_*() methods are safe to call from ANY thread (they schedule via after()).
wait_for_enter_then(cb) registers a key binding and calls cb when triggered.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional, Tuple

from core.clock import get_clock

logger = logging.getLogger(__name__)

try:
    import tkinter as tk
    from tkinter import font as tkfont
    _HAS_TK = True
except ImportError:
    _HAS_TK = False
    logger.warning("tkinter not available; SentenceRenderer is a no-op.")

FONT_FAMILY  = "Verdana"
FONT_SIZE    = 32
WORD_SPACING = 20


class SentenceRenderer:
    """
    Tkinter stimulus display.

    Must be created on the main thread. run_mainloop() blocks until the
    window is closed. All calibration/session logic runs via on_ready callback
    which is scheduled with root.after() so it fires inside the event loop.
    """

    def __init__(
        self,
        on_trigger: Callable[[int], None],
        screen_w:   int = 0,
        screen_h:   int = 0,
    ) -> None:
        self._on_trigger = on_trigger
        self._screen_w   = screen_w
        self._screen_h   = screen_h
        self._clock      = get_clock()

        self._root:   Optional["tk.Tk"]       = None
        self._canvas: Optional["tk.Canvas"]   = None
        self._font:   Optional["tkfont.Font"] = None

        self._word_bboxes:  List[Tuple[int, int, int, int]] = []
        self._current_text: str  = ""
        self._lock               = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Create window (does not start event loop)."""
        if not _HAS_TK:
            return
        self._root = tk.Tk()
        self._root.title("MESTTS")
        self._root.configure(bg="white")
        self._root.attributes("-fullscreen", True)

        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._screen_w = sw
        self._screen_h = sh

        self._canvas = tk.Canvas(
            self._root, width=sw, height=sh, bg="white", cursor="none"
        )
        self._canvas.pack(fill="both", expand=True)
        self._font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE)

        self._root.bind("<Escape>", lambda e: self._root.destroy())

        logger.info("SentenceRenderer initialized (%dx%d).", sw, sh)

    def run_mainloop(self, on_ready: Optional[Callable] = None) -> None:
        """
        Start Tkinter event loop (blocks until window closed).
        on_ready is called ~300ms after startup, inside the event loop.
        """
        if not _HAS_TK or self._root is None:
            if on_ready:
                on_ready()
            return
        if on_ready:
            self._root.after(300, on_ready)
        self._root.mainloop()

    def close(self) -> None:
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    # ── Display (thread-safe via after) ───────────────────────────────────────

    def display_sentence(self, text: str) -> int:
        onset_ns = self._clock.now_ns()
        if not _HAS_TK or self._canvas is None:
            logger.info("SentenceRenderer (no-op): '%s'", text)
            return onset_ns

        def _render():
            words  = text.split()
            bboxes = self._compute_bboxes(words)
            self._canvas.delete("all")
            for word, (x1, y1, x2, y2) in zip(words, bboxes):
                self._canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2,
                    text=word, font=self._font, fill="black", anchor="center",
                )
            with self._lock:
                self._word_bboxes  = bboxes
                self._current_text = text

        self._root.after(0, _render)
        return onset_ns

    def show_message(self, message: str, color: str = "gray") -> None:
        if not _HAS_TK or self._canvas is None:
            logger.info("SentenceRenderer msg: %s", message)
            return

        def _msg():
            self._canvas.delete("all")
            self._canvas.create_text(
                self._screen_w // 2, self._screen_h // 2,
                text=message, font=(FONT_FAMILY, 24),
                fill=color, anchor="center", justify="center",
                width=self._screen_w - 160,
            )

        self._root.after(0, _msg)

    def show_calibration_dot(self, norm_x: float, norm_y: float, radius: int = 22) -> None:
        if not _HAS_TK or self._canvas is None:
            return
        px = int(norm_x * self._screen_w)
        py = int(norm_y * self._screen_h)

        def _dot():
            self._canvas.delete("all")
            self._canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="red", outline="darkred", width=3,
            )
            self._canvas.create_oval(
                px - 5, py - 5, px + 5, py + 5,
                fill="white", outline="white",
            )

        self._root.after(0, _dot)

    def get_word_bboxes(self) -> List[Tuple[int, int, int, int]]:
        with self._lock:
            return list(self._word_bboxes)

    # ── Calibration sequencing (pure after-chain, zero blocking) ──────────────

    def after_ms(self, delay_ms: int, callback: Callable) -> None:
        """Schedule callback after delay_ms. Safe from any thread."""
        if self._root:
            self._root.after(delay_ms, callback)
        else:
            callback()

    def wait_for_enter_then(self, callback: Callable, timeout_ms: int = 120_000) -> None:
        """
        Wait for ENTER/SPACE/click then call callback — all non-blocking.

        Binds <Return>, <space>, <F5>, <KP_Enter>, and <Button-1> (mouse click)
        so the user can advance by any means. Sets up a timeout fallback.
        Also adds a visible "Click here or press ENTER" prompt at the bottom
        of the screen so the user always has a clear affordance.
        """
        if not _HAS_TK or self._root is None:
            callback()
            return

        fired = [False]

        def _fire(event=None):
            if fired[0]:
                return
            fired[0] = True
            # Remove all temporary bindings
            for seq in ("<Return>", "<space>", "<F5>", "<KP_Enter>", "<Button-1>"):
                try:
                    self._root.unbind(seq)
                except Exception:
                    pass
            # Re-bind session advance keys for sentence flow
            for seq in ("<Return>", "<space>", "<F5>"):
                self._root.bind(seq, self._on_key_trigger)
            callback()

        # Add a "click to continue" prompt at bottom of screen
        def _add_prompt():
            self._canvas.create_text(
                self._screen_w // 2,
                self._screen_h - 60,
                text="Press ENTER, SPACE, or click anywhere to continue",
                font=(FONT_FAMILY, 14),
                fill="#888888",
                anchor="center",
                tags="prompt",
            )

        self._root.after(0, _add_prompt)

        # Bind all input methods
        for seq in ("<Return>", "<space>", "<F5>", "<KP_Enter>"):
            self._root.bind(seq, _fire)
        self._canvas.bind("<Button-1>", _fire)

        # Make sure window has focus
        self._root.lift()
        self._root.focus_force()
        self._canvas.focus_set()

        # Timeout fallback
        if timeout_ms > 0:
            self._root.after(timeout_ms, lambda: _fire(None))

    # ── Session trigger (sentence advance) ────────────────────────────────────

    def bind_session_trigger(self) -> None:
        """Enable ENTER/SPACE/F5 to fire on_trigger. Called at sentence start."""
        if not _HAS_TK or self._root is None:
            return
        for seq in ("<Return>", "<space>", "<F5>"):
            self._root.bind(seq, self._on_key_trigger)

    def unbind_session_trigger(self) -> None:
        if not _HAS_TK or self._root is None:
            return
        for seq in ("<Return>", "<space>", "<F5>"):
            self._root.unbind(seq)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_bboxes(self, words: List[str]) -> List[Tuple[int, int, int, int]]:
        if not words or self._font is None:
            return []
        widths  = [self._font.measure(w) for w in words]
        total_w = sum(widths) + WORD_SPACING * (len(words) - 1)
        start_x = (self._screen_w - total_w) // 2
        line_h  = self._font.metrics("linespace")
        cy      = int(self._screen_h * 0.35)
        y1, y2  = cy - line_h // 2, cy + line_h // 2
        bboxes  = []
        x = start_x
        for w_px in widths:
            bboxes.append((x, y1, x + w_px, y2))
            x += w_px + WORD_SPACING
        return bboxes

    def _on_key_trigger(self, event=None) -> None:
        ts_ns = self._clock.now_ns()
        logger.debug("Trigger: %s", event.keysym if event else "?")
        try:
            self._on_trigger(ts_ns)
        except Exception as exc:
            logger.exception("on_trigger error: %s", exc)