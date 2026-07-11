"""
persistence/export.py
─────────────────────────────────────────────────────────────────────────────
ResearchExporter — generates flat CSV exports from the SQLite database
for downstream statistical analysis.

Exports:
  session_summary.csv  — one row per session
  sentence_records.csv — one row per sentence
  word_records.csv     — one row per word slot

Architecture spec §Layer 7.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from persistence.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class ResearchExporter:
    """
    Generates CSV exports from the SQLite database.

    Parameters
    ----------
    db : DatabaseManager
    export_dir : str
    """

    def __init__(self, db: DatabaseManager, export_dir: str) -> None:
        self._db  = db
        self._dir = Path(export_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def export_session_summary(self) -> Path:
        """Export all sessions to session_summary.csv."""
        with self._db._lock:
            cur = self._db._conn.execute(
                "SELECT * FROM sessions ORDER BY start_epoch_ts"
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        path = self._dir / "session_summary.csv"
        self._write_csv(path, cols, rows)
        logger.info("Exported session summary (%d rows) → %s.", len(rows), path)
        return path

    def export_sentence_records(self) -> Path:
        """Export all sentence records to sentence_records.csv."""
        with self._db._lock:
            cur = self._db._conn.execute(
                """
                SELECT s.session_id, p.participant_id, s.*
                FROM sentences s
                JOIN sessions p ON s.session_id = p.session_id
                ORDER BY p.start_epoch_ts, s.sentence_index
                """
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        path = self._dir / "sentence_records.csv"
        self._write_csv(path, cols, rows)
        logger.info("Exported sentence records (%d rows) → %s.", len(rows), path)
        return path

    def export_word_records(self) -> Path:
        """Export all word slot records to word_records.csv."""
        with self._db._lock:
            cur = self._db._conn.execute(
                """
                SELECT ws.*, s.session_id, s.sentence_index, s.text
                FROM word_slots ws
                JOIN sentences s ON ws.sentence_id = s.sentence_id
                ORDER BY s.session_id, s.sentence_index, ws.word_index
                """
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        path = self._dir / "word_records.csv"
        self._write_csv(path, cols, rows)
        logger.info("Exported word records (%d rows) → %s.", len(rows), path)
        return path

    def export_feature_vectors(self) -> Path:
        """Export feature vectors (unnested from JSON) to feature_vectors.csv."""
        with self._db._lock:
            cur = self._db._conn.execute(
                """
                SELECT rs.session_id, rs.risk_score, rs.risk_band,
                       rs.data_completeness, rs.n_features_missing,
                       rs.feature_vector_json
                FROM risk_scores rs
                """
            )
            rows_raw = cur.fetchall()

        flattened = []
        cols_set  = ["session_id", "risk_score", "risk_band",
                     "data_completeness", "n_features_missing"]
        fv_cols: List[str] = []

        for row in rows_raw:
            sid, rs, rb, dc, nm, fv_json = row
            fv = json.loads(fv_json) if fv_json else {}
            if not fv_cols:
                fv_cols = list(fv.keys())
            flat = {
                "session_id":         sid,
                "risk_score":         rs,
                "risk_band":          rb,
                "data_completeness":  dc,
                "n_features_missing": nm,
            }
            flat.update(fv)
            flattened.append(flat)

        path = self._dir / "feature_vectors.csv"
        self._write_csv(path, cols_set + fv_cols, flattened)
        logger.info("Exported feature vectors (%d rows) → %s.", len(flattened), path)
        return path

    @staticmethod
    def _write_csv(path: Path, columns: List[str], rows: List[Dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
