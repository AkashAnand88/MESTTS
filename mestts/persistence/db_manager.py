"""
persistence/db_manager.py
─────────────────────────────────────────────────────────────────────────────
SQLite database manager. Creates the schema on first use and exposes
typed insert methods for all session data entities.

Architecture spec §7.1.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    participant_id   TEXT NOT NULL,
    start_epoch_ts   REAL,
    start_mono_ts    REAL,
    calib_rmse       REAL,
    calib_valid      INTEGER,
    camera_latency_ms REAL,
    fuzzy_threshold  REAL,
    lookahead_window INTEGER,
    n_sentences      INTEGER,
    protocol_version TEXT,
    risk_score       REAL,
    risk_band        TEXT,
    valid            INTEGER,
    end_mono_ts      REAL,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS sentences (
    sentence_id      TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(session_id),
    sentence_index   INTEGER,
    sentence_type    TEXT,
    text             TEXT,
    word_count       INTEGER,
    onset_ts         REAL,
    end_ts           REAL,
    duration_ms      REAL,
    data_completeness REAL,
    mean_evs_speech_ms REAL,
    std_evs_speech_ms  REAL,
    regression_rate    REAL,
    regression_count   INTEGER,
    risk_score         REAL
);

CREATE TABLE IF NOT EXISTS word_slots (
    slot_id              TEXT PRIMARY KEY,
    sentence_id          TEXT NOT NULL REFERENCES sentences(sentence_id),
    word_index           INTEGER,
    expected_word        TEXT,
    slot_status          TEXT,
    data_quality         TEXT,
    exclusion_reason     TEXT,
    fixation_onset_ns    INTEGER,
    fixation_duration_ms REAL,
    gaze_x_norm          REAL,
    gaze_y_norm          REAL,
    regression_flag      INTEGER,
    speech_timestamp_ns  INTEGER,
    vosk_confidence      REAL,
    matched_word         TEXT,
    fuzzy_score          INTEGER,
    typing_timestamp_ns  INTEGER,
    typing_latency_ms    REAL,
    correction_count     INTEGER,
    submitted_word       TEXT,
    evs_speech_ms        REAL,
    evs_typing_ms        REAL,
    speech_typing_gap_ms REAL
);

CREATE TABLE IF NOT EXISTS risk_scores (
    rs_id              TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL REFERENCES sessions(session_id),
    risk_score         REAL,
    risk_band          TEXT,
    data_completeness  REAL,
    n_features_missing INTEGER,
    indicators_json    TEXT,
    feature_vector_json TEXT,
    disclaimer         TEXT
);
"""


class DatabaseManager:
    """
    SQLite persistence layer.

    Thread-safe via a per-connection threading.Lock. All writes are
    committed immediately (autocommit-style) for simplicity.

    Parameters
    ----------
    db_path : str
        Path to SQLite database file.
    """

    def __init__(self, db_path: str = "data/sessions.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn  = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock  = threading.Lock()
        self._create_schema()
        logger.info("DatabaseManager: connected to %s.", self._path)

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("DatabaseManager: connection closed.")

    # ── Session ───────────────────────────────────────────────────────────────

    def insert_session(self, data: Dict[str, Any]) -> None:
        sql = """
            INSERT OR REPLACE INTO sessions
            (session_id, participant_id, start_epoch_ts, start_mono_ts,
             calib_rmse, calib_valid, camera_latency_ms, fuzzy_threshold,
             lookahead_window, n_sentences, protocol_version)
            VALUES
            (:session_id, :participant_id, :start_epoch_ts, :start_mono_ts,
             :calib_rmse, :calib_valid, :camera_latency_ms, :fuzzy_threshold,
             :lookahead_window, :n_sentences, :protocol_version)
        """
        self._execute(sql, data)

    def update_session_close(
        self,
        session_id: str,
        end_mono_ts: float,
        risk_score:  float,
        risk_band:   str,
        valid:       bool,
    ) -> None:
        sql = """
            UPDATE sessions
            SET end_mono_ts=:end, risk_score=:rs, risk_band=:rb, valid=:v
            WHERE session_id=:sid
        """
        self._execute(sql, {
            "end": end_mono_ts,
            "rs":  risk_score,
            "rb":  risk_band,
            "v":   int(valid),
            "sid": session_id,
        })

    # ── Sentence ──────────────────────────────────────────────────────────────

    def insert_sentence(self, data: Dict[str, Any]) -> None:
        sql = """
            INSERT OR REPLACE INTO sentences
            (sentence_id, session_id, sentence_index, sentence_type, text,
             word_count, onset_ts, end_ts, duration_ms, data_completeness,
             mean_evs_speech_ms, std_evs_speech_ms, regression_rate,
             regression_count)
            VALUES
            (:sentence_id, :session_id, :sentence_index, :sentence_type, :text,
             :word_count, :onset_ts, :end_ts, :duration_ms, :data_completeness,
             :mean_evs_speech_ms, :std_evs_speech_ms, :regression_rate,
             :regression_count)
        """
        self._execute(sql, data)

    # ── Word Slots ────────────────────────────────────────────────────────────

    def insert_word_slots(self, sentence_id: str, slots: List[Dict]) -> None:
        sql = """
            INSERT OR REPLACE INTO word_slots
            (slot_id, sentence_id, word_index, expected_word, slot_status,
             data_quality, exclusion_reason, fixation_onset_ns,
             fixation_duration_ms, gaze_x_norm, gaze_y_norm, regression_flag,
             speech_timestamp_ns, vosk_confidence, matched_word, fuzzy_score,
             typing_timestamp_ns, typing_latency_ms, correction_count,
             submitted_word, evs_speech_ms, evs_typing_ms, speech_typing_gap_ms)
            VALUES
            (:slot_id, :sentence_id, :word_index, :expected_word, :slot_status,
             :data_quality, :exclusion_reason, :fixation_onset_ns,
             :fixation_duration_ms, :gaze_x_norm, :gaze_y_norm, :regression_flag,
             :speech_timestamp_ns, :vosk_confidence, :matched_word, :fuzzy_score,
             :typing_timestamp_ns, :typing_latency_ms, :correction_count,
             :submitted_word, :evs_speech_ms, :evs_typing_ms, :speech_typing_gap_ms)
        """
        import uuid
        rows = []
        for s in slots:
            row = dict(s)
            row["slot_id"]     = str(uuid.uuid4())
            row["sentence_id"] = sentence_id
            rows.append(row)

        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    # ── Risk Score ────────────────────────────────────────────────────────────

    def insert_risk_score(self, data: Dict[str, Any]) -> None:
        sql = """
            INSERT OR REPLACE INTO risk_scores
            (rs_id, session_id, risk_score, risk_band, data_completeness,
             n_features_missing, indicators_json, feature_vector_json, disclaimer)
            VALUES
            (:rs_id, :session_id, :risk_score, :risk_band, :data_completeness,
             :n_features_missing, :indicators_json, :feature_vector_json, :disclaimer)
        """
        self._execute(sql, data)

    # ── Query ─────────────────────────────────────────────────────────────────

    def fetch_session(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def fetch_sentences(self, session_id: str) -> List[Dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sentences WHERE session_id=? ORDER BY sentence_index",
                (session_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_word_slots(self, sentence_id: str) -> List[Dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM word_slots WHERE sentence_id=? ORDER BY word_index",
                (sentence_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _execute(self, sql: str, params: Dict) -> None:
        with self._lock:
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error("DB error: %s | SQL: %s", exc, sql[:80])
                raise
