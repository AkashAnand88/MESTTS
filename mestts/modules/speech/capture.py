"""
modules/speech/capture.py
─────────────────────────────────────────────────────────────────────────────
SpeechCaptureThread — PyAudio microphone capture + Vosk ASR worker.

Architecture spec §Layer 1.2, §Layer 3.2.

Two cooperating threads:
  AudioCaptureThread — PyAudio callback → audio_queue
  ASRWorkerThread    — reads audio_queue → Vosk → aligns → publishes

Both are managed by SpeechCaptureThread (the public interface).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import List, Optional

from core.clock import get_clock
from core.event_bus import Event, EventBus, EventType
from modules.speech.fuzzy_aligner import FuzzyAligner

logger = logging.getLogger(__name__)


class SpeechCaptureThread(threading.Thread):
    """
    Combined audio capture and ASR thread.

    Parameters
    ----------
    model_path : str
        Path to Vosk offline model directory.
    bus : EventBus
    fuzzy_threshold : float
        Minimum RapidFuzz ratio (0–100) for word alignment.
    lookahead_window : int
        Number of future sentence words to consider during alignment.
    sample_rate : int
        Audio sample rate in Hz.
    chunk_size : int
        PyAudio chunk size in frames.
    """

    def __init__(
        self,
        model_path:      str,
        bus:             EventBus,
        fuzzy_threshold: float = 75.0,
        lookahead_window: int  = 3,
        sample_rate:     int   = 16000,
        chunk_size:      int   = 4096,
    ) -> None:
        super().__init__(name="SpeechCapture", daemon=True)
        self._model_path      = model_path
        self._bus             = bus
        self._sample_rate     = sample_rate
        self._chunk_size      = chunk_size
        self._clock           = get_clock()

        self._aligner = FuzzyAligner(
            threshold       = fuzzy_threshold,
            lookahead_window = lookahead_window,
        )

        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._sentence_words: List[str]       = []
        self._sentence_active: bool           = False
        self._sentence_start_mono_ns: int     = 0
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()

    # ── Sentence Control ──────────────────────────────────────────────────────

    def activate_sentence(self, sentence_words: List[str]) -> None:
        """Prepare for a new sentence."""
        with self._lock:
            self._sentence_words  = list(sentence_words)
            self._sentence_start_mono_ns = self._clock.now_ns()
            self._sentence_active = True
        self._aligner.reset(sentence_words)
        logger.debug("SpeechCapture: sentence activated (%d words).", len(sentence_words))

    def deactivate_sentence(self) -> None:
        with self._lock:
            self._sentence_active = False

    # ── Thread Lifecycle ──────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        # Import here to avoid hard dependency at module load time
        try:
            import pyaudio
            from vosk import KaldiRecognizer, Model, SetLogLevel
            SetLogLevel(-1)
        except ImportError:
            logger.error("pyaudio and vosk are required for speech capture.")
            self._bus.publish(Event(
                event_type=EventType.SYSTEM_ERROR,
                timestamp_ns=self._clock.now_ns(),
                payload={"source": "SpeechCapture", "error": "Missing pyaudio/vosk"},
            ))
            return

        try:
            model = Model(self._model_path)
        except Exception as exc:
            logger.error("Cannot load Vosk model from %s: %s", self._model_path, exc)
            self._bus.publish(Event(
                event_type=EventType.SYSTEM_ERROR,
                timestamp_ns=self._clock.now_ns(),
                payload={"source": "SpeechCapture", "error": str(exc)},
            ))
            return

        rec  = KaldiRecognizer(model, self._sample_rate)
        rec.SetWords(True)

        pa   = pyaudio.PyAudio()
        stream = pa.open(
            rate            = self._sample_rate,
            channels        = 1,
            format          = pyaudio.paInt16,
            input           = True,
            frames_per_buffer = self._chunk_size,
            stream_callback = self._audio_callback,
        )
        stream.start_stream()
        logger.info("SpeechCaptureThread started.")

        # ASR worker loop
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if rec.AcceptWaveform(chunk):
                result_str = rec.Result()
                self._process_result(json.loads(result_str))
            # Partial results discarded (only finals carry word timestamps)

        stream.stop_stream()
        stream.close()
        pa.terminate()
        logger.info("SpeechCaptureThread stopped.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — push audio chunk to queue."""
        import pyaudio
        try:
            self._audio_queue.put_nowait(in_data)
        except queue.Full:
            pass  # Drop oldest; latency preferred over blocking
        return (None, pyaudio.paContinue)

    def _process_result(self, result: dict) -> None:
        """
        Process Vosk final result JSON.

        Each word in result['result'] carries start/end times relative
        to utterance start; we convert these to session-epoch nanoseconds.
        """
        with self._lock:
            if not self._sentence_active:
                return
            sentence_words = list(self._sentence_words)
            utterance_start_ns = self._sentence_start_mono_ns

        word_results = result.get("result", [])
        if not word_results:
            return

        for wr in word_results:
            raw_word  = wr.get("word", "")
            start_sec = wr.get("start", 0.0)
            end_sec   = wr.get("end", 0.0)
            conf      = float(wr.get("conf", 0.0))

            # Convert Vosk relative time to session-epoch nanoseconds
            speech_ts_ns = utterance_start_ns + int(start_sec * 1_000_000_000)

            # Align to sentence word list
            match = self._aligner.align(raw_word)
            if match is None:
                logger.debug(
                    "SpeechCapture: '%s' could not be aligned (skipped).", raw_word
                )
                continue

            word_idx, matched_word, fuzz_score = match

            self._bus.publish(Event(
                event_type   = EventType.SPEECH_WORD,
                timestamp_ns = speech_ts_ns,
                payload={
                    "word_idx":          word_idx,
                    "word_raw":          raw_word,
                    "matched_word":      matched_word,
                    "speech_timestamp_ns": speech_ts_ns,
                    "vosk_confidence":   conf,
                    "fuzzy_score":       fuzz_score,
                },
            ))
            logger.debug(
                "SPEECH: idx=%d raw='%s' matched='%s' score=%d ts=%d",
                word_idx, raw_word, matched_word, fuzz_score, speech_ts_ns,
            )
