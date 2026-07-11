"""
persistence/session_logger.py
─────────────────────────────────────────────────────────────────────────────
SessionLogger — async-write session data to SQLite + JSONL.

Provides non-blocking logging by dispatching DB writes through a
background worker thread queue. JSONL events are flushed immediately
(append-only) for lossless raw event capture.

Architecture spec §Layer 7.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from alignment.temporal_metrics import SentenceMetrics
from analytics.risk_scorer import RiskScoreResult
from analytics.feature_extractor import FeatureVector
from core.word_slot import WordSlot
from persistence.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SessionLogger(threading.Thread):
    """
    Async session logger.

    DB writes are queued and executed by a background thread.
    JSONL writes are synchronous (append-only, fast I/O).

    Parameters
    ----------
    db : DatabaseManager
    export_dir : str
    """

    def __init__(self, db: DatabaseManager, export_dir: str) -> None:
        super().__init__(name="SessionLogger", daemon=True)
        self._db         = db
        self._export_dir = Path(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)

        self._write_queue: queue.Queue[Optional[Dict]] = queue.Queue()
        self._session_id: str  = ""
        self._jsonl_path: Optional[Path] = None
        self._jsonl_lock = threading.Lock()

        self.start()

    # ── Session Begin/End ─────────────────────────────────────────────────────

    def begin_session(
        self,
        session_id:       str,
        participant_id:   str,
        start_ns:         int,
        calibration_rmse: float,
        camera_latency_ms: float,
        fuzzy_threshold:  float,
        lookahead_window: int,
        n_sentences:      int,
        protocol_version: str,
    ) -> None:
        self._session_id = session_id
        self._jsonl_path = self._export_dir / f"{session_id}.jsonl"

        epoch_wall = time.time()
        epoch_mono = start_ns / 1_000_000_000.0

        self._enqueue({
            "__op": "insert_session",
            "session_id":       session_id,
            "participant_id":   participant_id,
            "start_epoch_ts":   epoch_wall,
            "start_mono_ts":    epoch_mono,
            "calib_rmse":       calibration_rmse,
            "calib_valid":      int(calibration_rmse > 0),
            "camera_latency_ms": camera_latency_ms,
            "fuzzy_threshold":  fuzzy_threshold,
            "lookahead_window": lookahead_window,
            "n_sentences":      n_sentences,
            "protocol_version": protocol_version,
        })
        self._append_jsonl({"event_type": "SESSION_START", "session_id": session_id,
                             "ts": epoch_wall})
        logger.info("SessionLogger: session %s opened.", session_id)

    def finalize_session(
        self,
        end_ns:     int,
        risk_score: float,
        risk_band:  str,
        validity:   bool,
        full_export: Dict,
    ) -> None:
        end_s = end_ns / 1_000_000_000.0
        self._enqueue({
            "__op":       "update_session_close",
            "session_id": self._session_id,
            "end_mono_ts": end_s,
            "risk_score": risk_score,
            "risk_band":  risk_band,
            "valid":      validity,
        })
        # Write full JSON export
        export_path = self._export_dir / f"{self._session_id}_export.json"
        export_path.write_text(json.dumps(full_export, indent=2))
        logger.info("SessionLogger: full export written to %s.", export_path)
        self._append_jsonl({"event_type": "SESSION_CLOSE", "ts": end_s,
                             "risk_score": risk_score, "risk_band": risk_band})

    # ── Sentence Logging ──────────────────────────────────────────────────────

    def log_sentence(
        self,
        sentence_id:    str,
        sentence_index: int,
        sentence_text:  str,
        sentence_type:  str,
        start_ns:       int,
        end_ns:         int,
        slots:          List[WordSlot],
        metrics:        SentenceMetrics,
    ) -> None:
        onset_s    = start_ns / 1_000_000_000.0
        end_s      = end_ns   / 1_000_000_000.0
        duration_ms = (end_ns - start_ns) / 1_000_000.0

        self._enqueue({
            "__op":              "insert_sentence",
            "sentence_id":       sentence_id,
            "session_id":        self._session_id,
            "sentence_index":    sentence_index,
            "sentence_type":     sentence_type,
            "text":              sentence_text,
            "word_count":        metrics.n_words,
            "onset_ts":          onset_s,
            "end_ts":            end_s,
            "duration_ms":       duration_ms,
            "data_completeness": metrics.data_completeness,
            "mean_evs_speech_ms": metrics.mean_evs_speech_ms,
            "std_evs_speech_ms":  metrics.std_evs_speech_ms,
            "regression_rate":    metrics.regression_rate,
            "regression_count":   metrics.regression_count,
        })

        # Word slots
        slot_dicts = [s.to_dict() for s in slots]
        self._enqueue({
            "__op":       "insert_word_slots",
            "sentence_id": sentence_id,
            "slots":       slot_dicts,
        })

        self._append_jsonl({
            "event_type":     "SENTENCE_CLOSE",
            "sentence_index": sentence_index,
            "ts":             end_s,
            "completeness":   metrics.data_completeness,
            "regression_rate": metrics.regression_rate,
        })

    # ── Risk Score Logging ────────────────────────────────────────────────────

    def log_risk_score(
        self,
        result: RiskScoreResult,
        fv:     FeatureVector,
    ) -> None:
        self._enqueue({
            "__op":              "insert_risk_score",
            "rs_id":             str(uuid.uuid4()),
            "session_id":        self._session_id,
            "risk_score":        result.risk_score,
            "risk_band":         result.risk_band,
            "data_completeness": result.data_completeness,
            "n_features_missing": result.n_features_missing,
            "indicators_json":   json.dumps(result.to_dict()["indicators"]),
            "feature_vector_json": json.dumps(fv.to_dict()),
            "disclaimer":        result.disclaimer,
        })

    # ── Worker Thread ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Flush queue and stop worker thread."""
        self._write_queue.put(None)  # sentinel
        self.join(timeout=10.0)

    def run(self) -> None:
        logger.info("SessionLogger worker started.")
        while True:
            item = self._write_queue.get()
            if item is None:
                break
            try:
                self._dispatch(item)
            except Exception as exc:
                logger.exception("SessionLogger write error: %s", exc)
        logger.info("SessionLogger worker stopped.")

    def _dispatch(self, item: Dict) -> None:
        op = item.pop("__op")
        if op == "insert_session":
            self._db.insert_session(item)
        elif op == "update_session_close":
            sid = item.pop("session_id")
            self._db.update_session_close(
                session_id  = sid,
                end_mono_ts = item["end_mono_ts"],
                risk_score  = item["risk_score"],
                risk_band   = item["risk_band"],
                valid       = item["valid"],
            )
        elif op == "insert_sentence":
            self._db.insert_sentence(item)
        elif op == "insert_word_slots":
            self._db.insert_word_slots(item["sentence_id"], item["slots"])
        elif op == "insert_risk_score":
            self._db.insert_risk_score(item)
        else:
            logger.warning("SessionLogger: unknown op '%s'.", op)

    def _enqueue(self, item: Dict) -> None:
        try:
            self._write_queue.put_nowait(item)
        except queue.Full:
            logger.warning("SessionLogger: write queue full; blocking.")
            self._write_queue.put(item)

    def _append_jsonl(self, record: Dict) -> None:
        if self._jsonl_path is None:
            return
        with self._jsonl_lock:
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")
