"""
core/word_slot.py
─────────────────────────────────────────────────────────────────────────────
Defines the canonical per-word synchronization record that the alignment
layer assembles from the three independent acquisition modules.

Each sentence is decomposed into N WordSlots, one per token. Records from
the vision, speech, and typing modules are inserted into their respective
fields as they arrive. Temporal metrics are computed at sentence close when
all available records have been joined.

Slot completion semantics (from the architecture spec):
  COMPLETE  — gaze + (speech OR typing) both present
  PARTIAL   — gaze only, or modality records without gaze
  TIMED_OUT — sentence closed before slot reached COMPLETE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ─── Enumeration Types ────────────────────────────────────────────────────────

class SlotStatus(Enum):
    PENDING   = auto()   # No records yet
    PARTIAL   = auto()   # Some but not enough for metric computation
    COMPLETE  = auto()   # Sufficient records for all defined metrics
    TIMED_OUT = auto()   # Sentence closed; slot incomplete


class DataQuality(Enum):
    HIGH      = auto()   # Complete data, no anomalies
    DEGRADED  = auto()   # Present but flagged (dropped frames, low confidence)
    EXCLUDED  = auto()   # Cannot be used in scoring


# ─── Sub-Records ─────────────────────────────────────────────────────────────

@dataclass
class GazeRecord:
    """
    Vision module output for a single word fixation event.

    All timestamps are nanoseconds since session epoch, camera-latency-
    corrected (F_i = fixation_onset_raw_ns - Δ_cam_ns).
    """
    word_index: int
    fixation_onset_ns: int              # Camera-latency-corrected onset
    fixation_duration_ms: float
    gaze_x_norm: float                  # Normalized [0.0, 1.0] screen coords
    gaze_y_norm: float
    regression_flag: bool
    frame_id: int
    raw_onset_ns: int = 0               # Pre-correction value (audit trail)


@dataclass
class SpeechRecord:
    """
    Speech module output for a single recognized and aligned word.

    speech_timestamp_ns represents the Vosk-reported word start time
    converted to session-epoch nanoseconds.
    """
    word_index: int
    speech_timestamp_ns: int
    vosk_confidence: float              # [0.0, 1.0]
    raw_word: str                       # As recognized by Vosk
    matched_word: str                   # After fuzzy alignment to stimulus
    fuzzy_score: int                    # 0–100


@dataclass
class TypingRecord:
    """
    Typing module output for a single submitted word.

    typing_timestamp_ns is the monotonic_ns() value at the moment the
    delimiter key is pressed (word submission event).
    """
    word_index: int
    typing_timestamp_ns: int
    latency_ms: float                   # Time from estimated word display onset
    correction_count: int               # Backspace presses before submission
    submitted_word: str
    expected_word: str


# ─── Temporal Metric Container ───────────────────────────────────────────────

@dataclass
class TemporalMetrics:
    """
    Derived temporal measurements computed at sentence close.
    All values in milliseconds. None indicates insufficient data.

    Definitions (from architecture spec §6):
      EVS_speech  = (P_i − F_i) / 1e6
      EVS_typing  = (Y_i − F_i) / 1e6
      STG         = (Y_i − P_i) / 1e6
    """
    evs_speech_ms: Optional[float] = None
    evs_typing_ms: Optional[float] = None
    speech_typing_gap_ms: Optional[float] = None

    def is_complete(self) -> bool:
        """True if at least one EVS value is available."""
        return self.evs_speech_ms is not None or self.evs_typing_ms is not None


# ─── WordSlot ────────────────────────────────────────────────────────────────

@dataclass
class WordSlot:
    """
    Central synchronization unit for one word position in a sentence.

    Thread-safety note: insertion of sub-records must be done under the
    SentenceSynchronizer lock, not directly on this object.
    """
    word_index: int
    expected_word: str

    gaze_record:   Optional[GazeRecord]   = field(default=None)
    speech_record: Optional[SpeechRecord] = field(default=None)
    typing_record: Optional[TypingRecord] = field(default=None)

    temporal_metrics: TemporalMetrics = field(default_factory=TemporalMetrics)

    status:       SlotStatus  = SlotStatus.PENDING
    data_quality: DataQuality = DataQuality.HIGH
    exclusion_reason: Optional[str] = None

    # ── Status Resolution ────────────────────────────────────────────────────

    def resolve_status(self) -> SlotStatus:
        """
        Evaluate current record population and assign the correct SlotStatus.

        Completion rule (from spec §3.4):
          COMPLETE if gaze present AND (speech OR typing) present.
        """
        has_gaze    = self.gaze_record is not None
        has_speech  = self.speech_record is not None
        has_typing  = self.typing_record is not None

        if has_gaze and (has_speech or has_typing):
            self.status = SlotStatus.COMPLETE
        elif has_gaze or has_speech or has_typing:
            self.status = SlotStatus.PARTIAL
        else:
            self.status = SlotStatus.PENDING

        return self.status

    # ── Metric Computation ───────────────────────────────────────────────────

    def compute_temporal_metrics(self) -> TemporalMetrics:
        """
        Compute EVS_speech, EVS_typing, and SpeechTypingGap from
        available sub-records.

        Returns the populated TemporalMetrics object. Unavailable metrics
        remain None (propagated null-safety pattern).
        """
        m = TemporalMetrics()

        if self.gaze_record is None:
            self.temporal_metrics = m
            return m

        F_ns = self.gaze_record.fixation_onset_ns  # Camera-latency-corrected

        if self.speech_record is not None:
            P_ns = self.speech_record.speech_timestamp_ns
            m.evs_speech_ms = (P_ns - F_ns) / 1_000_000.0

        if self.typing_record is not None:
            Y_ns = self.typing_record.typing_timestamp_ns
            m.evs_typing_ms = (Y_ns - F_ns) / 1_000_000.0

        if self.speech_record is not None and self.typing_record is not None:
            P_ns = self.speech_record.speech_timestamp_ns
            Y_ns = self.typing_record.typing_timestamp_ns
            m.speech_typing_gap_ms = (Y_ns - P_ns) / 1_000_000.0

        self.temporal_metrics = m
        return m

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary suitable for DB insertion."""
        d: dict = {
            "word_index":    self.word_index,
            "expected_word": self.expected_word,
            "slot_status":   self.status.name,
            "data_quality":  self.data_quality.name,
            "exclusion_reason": self.exclusion_reason,
        }

        if self.gaze_record:
            g = self.gaze_record
            d.update({
                "fixation_onset_ns":    g.fixation_onset_ns,
                "fixation_duration_ms": g.fixation_duration_ms,
                "gaze_x_norm":          g.gaze_x_norm,
                "gaze_y_norm":          g.gaze_y_norm,
                "regression_flag":      g.regression_flag,
            })
        else:
            d.update({
                "fixation_onset_ns": None, "fixation_duration_ms": None,
                "gaze_x_norm": None, "gaze_y_norm": None,
                "regression_flag": None,
            })

        if self.speech_record:
            s = self.speech_record
            d.update({
                "speech_timestamp_ns": s.speech_timestamp_ns,
                "vosk_confidence":     s.vosk_confidence,
                "matched_word":        s.matched_word,
                "fuzzy_score":         s.fuzzy_score,
            })
        else:
            d.update({
                "speech_timestamp_ns": None, "vosk_confidence": None,
                "matched_word": None, "fuzzy_score": None,
            })

        if self.typing_record:
            t = self.typing_record
            d.update({
                "typing_timestamp_ns": t.typing_timestamp_ns,
                "typing_latency_ms":   t.latency_ms,
                "correction_count":    t.correction_count,
                "submitted_word":      t.submitted_word,
            })
        else:
            d.update({
                "typing_timestamp_ns": None, "typing_latency_ms": None,
                "correction_count": None, "submitted_word": None,
            })

        m = self.temporal_metrics
        d.update({
            "evs_speech_ms":        m.evs_speech_ms,
            "evs_typing_ms":        m.evs_typing_ms,
            "speech_typing_gap_ms": m.speech_typing_gap_ms,
        })

        return d
