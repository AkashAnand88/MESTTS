"""
analytics/analytics_dashboard.py
─────────────────────────────────────────────────────────────────────────────
MESTTS — Participant Analytics Dashboard

Reads directly from the MESTTS SQLite database (data/sessions.db).

Features
--------
  - Loads every session and its feature vector from the DB
  - Saves a master dataset as  data/exports/participant_dataset.json
                            and  data/exports/participant_dataset.csv
  - Tab 1 — Session Log        : full scrollable table of every session
  - Tab 2 — Feature Trends     : line graph of any F01–F17 feature over
                                  sessions, one line per participant
  - Tab 3 — Risk Distribution  : doughnut chart of LOW/MOD/HIGH/SEVERE
  - Tab 4 — Feature Impact     : bar chart — how often each scored feature
                                  triggers its threshold
  - Tab 5 — Comparative Analysis:
      • Radar chart        — mean feature profile, normalised to threshold
      • Grouped bar chart  — side-by-side means ± std per participant
      • Risk trajectory    — risk score per session on a shared axis
      • Summary table      — mean ± std, Δ from group mean (red / green)

Usage
-----
  python analytics/analytics_dashboard.py
  python analytics/analytics_dashboard.py --db data/sessions.db
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFrame, QCheckBox,
    QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PARTICIPANT_COLORS = [
    "#185FA5", "#0F6E56", "#D85A30", "#993556",
    "#639922", "#BA7517", "#534AB7", "#888780",
    "#2E8B8B", "#A0522D", "#6A5ACD", "#2F7A2F",
]

BAND_COLORS = {
    "LOW":          "#639922",
    "MODERATE":     "#BA7517",
    "HIGH":         "#E24B4A",
    "SEVERE_FLAG":  "#D85A30",
    "INCONCLUSIVE": "#888780",
}
BAND_DISPLAY = {
    "LOW":          "Low",
    "MODERATE":     "Moderate",
    "HIGH":         "High",
    "SEVERE_FLAG":  "Severe",
    "INCONCLUSIVE": "Inconclusive",
}

FEATURES: List = [
    # (db_key,                         label,                          unit,   direction, threshold)
    ("F01_evs_speech_std_ms",          "F01 — EVS speech std",         "ms",   "above",   120.0),
    ("F02_mean_evs_speech_ms",         "F02 — Mean EVS speech",        "ms",   None,      None),
    ("F03_evs_speech_skew",            "F03 — EVS skewness",           "",     None,      None),
    ("F04_speech_inversion_rate",      "F04 — Speech inversion rate",  "rate", "above",   0.25),
    ("F05_evs_typing_std_ms",          "F05 — EVS typing std",         "ms",   None,      None),
    ("F06_mean_fixation_duration_ms",  "F06 — Mean fixation dur.",     "ms",   "above",   300.0),
    ("F07_std_fixation_duration_ms",   "F07 — Fixation dur. std",      "ms",   None,      None),
    ("F08_regression_rate",            "F08 — Regression rate",        "rate", "above",   0.20),
    ("F09_regression_count_total",     "F09 — Total regressions",      "n",    None,      None),
    ("F10_fixation_outlier_ratio",     "F10 — Fixation outlier ratio", "",     None,      None),
    ("F11_typing_inversion_rate",      "F11 — Typing inversion rate",  "rate", None,      None),
    ("F12_mean_correction_rate",       "F12 — Correction rate",        "rate", "above",   0.30),
    ("F13_max_corrections",            "F13 — Max corrections",        "n",    None,      None),
    ("F14_stg_std_ms",                 "F14 — STG std",                "ms",   "above",   200.0),
    ("F15_stg_negative_rate",          "F15 — STG negative rate",      "rate", None,      None),
    ("F16_corr_evs_speech_fixation",   "F16 — Gaze-speech corr.",      "",     "below",   0.30),
    ("F17_corr_evs_typing_latency",    "F17 — EVS-typing corr.",       "",     None,      None),
]

SCORED_FEATURES = [f for f in FEATURES if f[3] is not None]

MPL_STYLE = {
    "axes.facecolor":   "#1e1e2e",
    "figure.facecolor": "#181825",
    "axes.edgecolor":   "#44475a",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#bac2de",
    "ytick.color":      "#bac2de",
    "text.color":       "#cdd6f4",
    "grid.color":       "#313244",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#1e1e2e",
    "legend.edgecolor": "#44475a",
}
plt.rcParams.update(MPL_STYLE)

_BTN_STYLE = (
    "background:#313244; color:#cdd6f4; border:1px solid #45475a;"
    "padding:5px 14px; border-radius:5px; font-size:11px;"
    "font-family:'Courier New';"
)
_COMBO_STYLE = (
    "background:#313244; color:#cdd6f4; border:1px solid #45475a;"
    "padding:3px 8px; font-size:11px; font-family:'Courier New';"
)
_TABLE_STYLE = """
    QTableWidget {
        background-color: #1e1e2e;
        alternate-background-color: #181825;
        color: #cdd6f4;
        gridline-color: #313244;
        font-size: 11px;
        font-family: 'Courier New';
    }
    QHeaderView::section {
        background-color: #313244;
        color: #89b4fa;
        padding: 5px;
        border: none;
        font-size: 10px;
        font-weight: bold;
    }
    QTableWidget::item:selected { background-color: #45475a; }
"""
_INNER_TAB_STYLE = """
    QTabWidget::pane  { background:#1e1e2e; border:1px solid #313244; }
    QTabBar::tab      { background:#313244; color:#bac2de;
                        padding:5px 12px; font-size:10px;
                        font-family:'Courier New'; border-radius:3px;
                        min-width:110px; }
    QTabBar::tab:selected { background:#45475a; color:#cdd6f4; font-weight:bold; }
    QTabBar::tab:hover    { background:#3d3f53; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# Database I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(db_path: str) -> List[Dict]:
    """Pull every session + its feature vector from the MESTTS database."""
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT
            s.session_id, s.participant_id,
            s.start_epoch_ts,
            s.risk_score  AS session_risk_score,
            s.risk_band   AS session_risk_band,
            s.valid,
            rs.risk_score, rs.risk_band,
            rs.data_completeness, rs.n_features_missing,
            rs.feature_vector_json
        FROM sessions s
        LEFT JOIN risk_scores rs ON rs.session_id = s.session_id
        ORDER BY s.participant_id, s.start_epoch_ts
    """)
    rows = cur.fetchall()
    conn.close()

    records = []
    for r in rows:
        rec = dict(r)
        fv_json = rec.pop("feature_vector_json", None)
        rec.update(json.loads(fv_json) if fv_json else {})
        rec["risk_score"] = rec.get("risk_score") or rec.get("session_risk_score") or 0.0
        rec["risk_band"]  = rec.get("risk_band")  or rec.get("session_risk_band")  or "INCONCLUSIVE"
        ts = rec.get("start_epoch_ts")
        rec["date"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
        records.append(rec)
    return records


def save_dataset(records: List[Dict], export_dir: str) -> None:
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(export_dir) / "participant_dataset.json"
    json_path.write_text(json.dumps(records, indent=2, default=str))
    if not records:
        return
    csv_path = Path(export_dir) / "participant_dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"[dataset] Saved → {json_path}")
    print(f"[dataset] Saved → {csv_path}")


def export_to_path(records: List[Dict], path: str, fmt: str) -> None:
    if fmt == "json":
        Path(path).write_text(json.dumps(records, indent=2, default=str))
    else:
        if not records:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _color_for_band(band: str) -> QColor:
    return QColor(BAND_COLORS.get((band or "").upper(), "#888780"))


def _styled_label(text: str, size: int = 11, bold: bool = False,
                  color: str = "#cdd6f4") -> QLabel:
    lbl = QLabel(text)
    font = QFont("Courier New", size)
    font.setBold(bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color};")
    return lbl


def _pid_color(pid: str, all_pids: List[str]) -> str:
    idx = all_pids.index(pid) if pid in all_pids else 0
    return PARTICIPANT_COLORS[idx % len(PARTICIPANT_COLORS)]


def _flat_axes(axes_raw, rows: int, cols: int):
    """Return a flat Python list of Axes regardless of subplots() return shape."""
    return list(np.array(axes_raw).flatten())


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Session Log
# ══════════════════════════════════════════════════════════════════════════════

class SessionLogTab(QWidget):

    _BASE_COLS = [
        ("participant_id",    "Participant",  110),
        ("date",              "Date / Time",  148),
        ("risk_score",        "Risk Score",    95),
        ("risk_band",         "Band",          118),
        ("data_completeness", "Completeness",  108),
    ]
    # all 17 feature columns added dynamically from FEATURES list
    COLS = _BASE_COLS + [
        (fk,
         (lbl.split("\u2014")[0].strip() + " " + lbl.split("\u2014")[1].strip()
          if "\u2014" in lbl else
          lbl.split("\u2013")[0].strip() + " " + lbl.split("\u2013")[1].strip()
          if "\u2013" in lbl else lbl),
         108)
        for fk, lbl, *_ in FEATURES
    ]

    def __init__(self, records: List[Dict], parent=None):
        super().__init__(parent)
        self._records = records
        self._build()

    def _build(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(6)

        self._info_lbl = _styled_label(
            f"  {len(self._records)} sessions loaded from database",
            size=10, color="#6c7086"
        )
        vl.addWidget(self._info_lbl)

        self._table = QTableWidget()
        self._table.setColumnCount(len(self.COLS))
        self._table.setHorizontalHeaderLabels([c[1] for c in self.COLS])
        self._table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self._table.setStyleSheet(_TABLE_STYLE)

        hh = self._table.horizontalHeader()
        for col_i, (_, _, width) in enumerate(self.COLS):
            if col_i < len(self.COLS) - 1:
                hh.setSectionResizeMode(col_i, QHeaderView.Interactive)
                self._table.setColumnWidth(col_i, width)
            else:
                hh.setSectionResizeMode(col_i, QHeaderView.Stretch)

        self._populate()
        vl.addWidget(self._table)

    def _populate(self):
        self._table.setRowCount(len(self._records))
        for row_i, rec in enumerate(self._records):
            for col_i, (key, _, _) in enumerate(self.COLS):
                val = rec.get(key)
                if val is None:
                    text = "—"
                elif isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if key == "risk_band":
                    band = str(val).upper()
                    item.setForeground(_color_for_band(band))
                    item.setText(BAND_DISPLAY.get(band, str(val)))
                self._table.setItem(row_i, col_i, item)

    def refresh(self, records: List[Dict]):
        self._records = records
        self._info_lbl.setText(f"  {len(records)} sessions loaded from database")
        self._populate()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Feature Trends
# ══════════════════════════════════════════════════════════════════════════════

class FeatureTrendTab(QWidget):

    def __init__(self, records: List[Dict], parent=None):
        super().__init__(parent)
        self._records  = records
        self._all_pids: List[str] = []
        self._p_checks: Dict[str, QCheckBox] = {}
        self._build()

    def _build(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(8)

        self._ctrl = QHBoxLayout()
        self._ctrl.addWidget(_styled_label("Feature:", size=10, color="#89b4fa"))

        self._feature_combo = QComboBox()
        self._feature_combo.setStyleSheet(_COMBO_STYLE)
        for _, lbl, *_ in FEATURES:
            self._feature_combo.addItem(lbl)
        self._feature_combo.currentIndexChanged.connect(self._refresh_chart)
        self._ctrl.addWidget(self._feature_combo)

        self._ctrl.addSpacing(20)
        self._ctrl.addWidget(_styled_label("Participants:", size=10, color="#89b4fa"))

        self._rebuild_checkboxes(
            sorted(set(r["participant_id"] for r in self._records))
        )
        self._ctrl.addStretch()
        vl.addLayout(self._ctrl)

        self._fig = Figure(figsize=(10, 5))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self._canvas)
        self._refresh_chart()

    def _rebuild_checkboxes(self, pids: List[str]):
        for cb in self._p_checks.values():
            self._ctrl.removeWidget(cb)
            cb.deleteLater()
        self._p_checks.clear()
        self._all_pids = list(pids)
        # Insert before the stretch; find its position
        stretch_pos = self._ctrl.count()  # stretch is always last
        for i, pid in enumerate(pids):
            cb = QCheckBox(pid)
            color = PARTICIPANT_COLORS[i % len(PARTICIPANT_COLORS)]
            cb.setStyleSheet(
                f"color:{color}; font-size:11px; font-family:'Courier New';"
            )
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_chart)
            self._p_checks[pid] = cb
            self._ctrl.insertWidget(stretch_pos + i, cb)

    def _refresh_chart(self):
        feat_key, feat_lbl, feat_unit, direction, threshold = \
            FEATURES[self._feature_combo.currentIndex()]
        selected = [pid for pid, cb in self._p_checks.items() if cb.isChecked()]

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(MPL_STYLE["axes.facecolor"])

        any_data = False
        for i, pid in enumerate(selected):
            pid_sessions = sorted(
                [r for r in self._records if r["participant_id"] == pid],
                key=lambda r: r.get("start_epoch_ts") or 0,
            )
            vals, x_pts = [], []
            for j, r in enumerate(pid_sessions):
                v = r.get(feat_key)
                if v is not None:
                    vals.append(float(v))
                    x_pts.append(j + 1)
            if not vals:
                continue
            color = PARTICIPANT_COLORS[i % len(PARTICIPANT_COLORS)]
            ax.plot(x_pts, vals, marker="o", markersize=5,
                    linestyle="--" if i % 2 else "-", linewidth=1.8,
                    color=color, label=pid)
            any_data = True

        if threshold is not None:
            ax.axhline(threshold, color="#f38ba8", linewidth=1.0,
                       linestyle=":", label=f"Threshold ({threshold})")

        if any_data:
            ax.set_xlabel("Session #", fontsize=10)
            ax.set_ylabel(f"{feat_lbl} ({feat_unit})" if feat_unit else feat_lbl,
                          fontsize=10)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=9, loc="upper right")
        else:
            ax.text(0.5, 0.5, "No data for selected participants / feature",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#6c7086", fontsize=11)

        ax.set_title(f"{feat_lbl} — per participant across sessions",
                     fontsize=11, pad=10)
        self._fig.tight_layout()
        self._canvas.draw()

    def refresh(self, records: List[Dict]):
        self._records = records
        pids_new = sorted(set(r["participant_id"] for r in records))
        if pids_new != self._all_pids:
            self._rebuild_checkboxes(pids_new)
        self._refresh_chart()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Risk Distribution
# ══════════════════════════════════════════════════════════════════════════════

class RiskDistributionTab(QWidget):

    def __init__(self, records: List[Dict], parent=None):
        super().__init__(parent)
        self._records = records
        self._build()

    def _build(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(_styled_label("Show:", size=10, color="#89b4fa"))
        self._scope_combo = QComboBox()
        self._scope_combo.setStyleSheet(_COMBO_STYLE)
        self._scope_combo.addItems(["All participants", "Per participant"])
        self._scope_combo.currentIndexChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._scope_combo)
        ctrl.addStretch()
        vl.addLayout(ctrl)

        self._fig = Figure(figsize=(10, 5))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self._canvas)
        self._refresh_chart()

    def _refresh_chart(self):
        self._fig.clear()
        if self._scope_combo.currentIndex() == 0:
            self._draw_overall()
        else:
            self._draw_per_participant()
        self._canvas.draw()

    def _draw_overall(self):
        counts  = Counter((r.get("risk_band") or "INCONCLUSIVE").upper()
                          for r in self._records)
        labels  = list(counts.keys())
        sizes   = list(counts.values())
        colors  = [BAND_COLORS.get(l, "#888780") for l in labels]
        display = [BAND_DISPLAY.get(l, l) for l in labels]

        ax = self._fig.add_subplot(111)
        ax.set_facecolor(MPL_STYLE["figure.facecolor"])
        if not sizes:
            ax.text(0.5, 0.5, "No sessions in database",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#6c7086")
            return

        wedges, _, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct="%1.1f%%", pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="#181825", linewidth=1.5),
            startangle=90,
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("#1e1e2e")
        ax.legend(wedges, [f"{d}  ({s})" for d, s in zip(display, sizes)],
                  loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
        ax.set_title(f"Risk band distribution — {len(self._records)} sessions",
                     fontsize=11, pad=12)
        self._fig.tight_layout()

    def _draw_per_participant(self):
        pids = sorted(set(r["participant_id"] for r in self._records))
        n = len(pids)
        if n == 0:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            return

        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        flat = _flat_axes(self._fig.subplots(rows, cols), rows, cols)

        for idx, pid in enumerate(pids):
            ax = flat[idx]
            ax.set_facecolor(MPL_STYLE["figure.facecolor"])
            recs   = [r for r in self._records if r["participant_id"] == pid]
            counts = Counter((r.get("risk_band") or "INCONCLUSIVE").upper()
                             for r in recs)
            labels = list(counts.keys())
            sizes  = list(counts.values())
            colors = [BAND_COLORS.get(l, "#888780") for l in labels]
            ax.pie(sizes, colors=colors, autopct="%1.0f%%", pctdistance=0.70,
                   wedgeprops=dict(width=0.5, edgecolor="#181825", linewidth=1),
                   startangle=90)
            ax.set_title(f"{pid}  ({len(recs)} sessions)", fontsize=9, pad=6)

        for idx in range(n, rows * cols):
            flat[idx].set_visible(False)

        self._fig.suptitle("Risk distribution per participant", fontsize=11, y=1.01)
        self._fig.tight_layout()

    def refresh(self, records: List[Dict]):
        self._records = records
        self._refresh_chart()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Feature Impact
# ══════════════════════════════════════════════════════════════════════════════

class FeatureImpactTab(QWidget):

    def __init__(self, records: List[Dict], parent=None):
        super().__init__(parent)
        self._records = records
        self._build()

    def _build(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(_styled_label(
            "How often each scored feature triggers its threshold across all "
            "sessions — the features actually driving the risk score.",
            size=9, color="#6c7086"
        ))
        ctrl.addStretch()
        vl.addLayout(ctrl)

        self._fig = Figure(figsize=(10, 5))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self._canvas)
        self._refresh_chart()

    def _refresh_chart(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(MPL_STYLE["axes.facecolor"])

        if not self._records:
            ax.text(0.5, 0.5, "No sessions in database",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#6c7086")
            self._canvas.draw()
            return

        n = len(self._records)
        rates, labels = [], []
        for feat_key, feat_lbl, _, direction, threshold in SCORED_FEATURES:
            triggered = total = 0
            for r in self._records:
                val = r.get(feat_key)
                if val is None:
                    continue
                total += 1
                v = float(val)
                if (direction == "above" and v > threshold) or \
                   (direction == "below" and v < threshold):
                    triggered += 1
            rates.append((triggered / total * 100) if total else 0)
            labels.append(feat_lbl.split("—")[0].strip())

        bar_colors = ["#E24B4A" if r >= 60 else "#BA7517" if r >= 30 else "#639922"
                      for r in rates]
        bars = ax.bar(labels, rates, color=bar_colors,
                      edgecolor="#181825", linewidth=0.8)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{rate:.0f}%", ha="center", va="bottom",
                    fontsize=8, color="#cdd6f4")

        ax.set_ylim(0, 115)
        ax.set_ylabel("Sessions triggering threshold (%)", fontsize=10)
        ax.set_xlabel("Feature", fontsize=10)
        ax.set_title(
            f"Feature trigger rate across {n} sessions"
            "  |  Red ≥60%  ·  Orange 30–59%  ·  Green <30%",
            fontsize=10, pad=10,
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.4)
        ax.axhline(60, color="#f38ba8", linewidth=0.8, linestyle=":")
        ax.axhline(30, color="#f9e2af", linewidth=0.8, linestyle=":")
        self._fig.tight_layout()
        self._canvas.draw()

    def refresh(self, records: List[Dict]):
        self._records = records
        self._refresh_chart()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Comparative Analysis
# ══════════════════════════════════════════════════════════════════════════════

class ComparativeAnalysisTab(QWidget):
    """
    Four sub-panels inside a nested QTabWidget:
      1. Radar chart       — mean feature profile normalised to threshold = 1.0
      2. Grouped bar chart — side-by-side means ± std of every scored feature
      3. Risk trajectory   — risk score per session on a shared axis with band shading
      4. Summary table     — mean ± std, Δ from group mean coloured red / green
    """

    _SF = SCORED_FEATURES

    def __init__(self, records: List[Dict], parent=None):
        super().__init__(parent)
        self._records  = records
        self._all_pids: List[str] = []
        self._p_checks: Dict[str, QCheckBox] = {}
        self._build()

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(8)

        # controls bar
        self._ctrl = QHBoxLayout()
        self._ctrl.addWidget(
            _styled_label("Compare participants:", size=10, color="#89b4fa")
        )
        self._rebuild_checkboxes(
            sorted(set(r["participant_id"] for r in self._records))
        )
        self._ctrl.addStretch()

        export_btn = QPushButton("Export comparison CSV")
        export_btn.setStyleSheet(
            "background:#313244; color:#cdd6f4; border:1px solid #45475a;"
            "padding:4px 12px; border-radius:5px; font-size:10px;"
            "font-family:'Courier New';"
        )
        export_btn.clicked.connect(self._export_comparison)
        self._ctrl.addWidget(export_btn)
        vl.addLayout(self._ctrl)

        # inner tabs
        self._inner_tabs = QTabWidget()
        self._inner_tabs.setUsesScrollButtons(True)
        self._inner_tabs.setStyleSheet(_INNER_TAB_STYLE)

        # 1 — Radar
        self._radar_fig    = Figure(figsize=(8, 5))
        self._radar_canvas = FigureCanvas(self._radar_fig)
        self._radar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._inner_tabs.addTab(self._radar_canvas, "Radar — feature profiles")

        # 2 — Grouped bar
        self._gbar_fig    = Figure(figsize=(11, 5))
        self._gbar_canvas = FigureCanvas(self._gbar_fig)
        self._gbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._inner_tabs.addTab(self._gbar_canvas, "Feature comparison (bar)")

        # 3 — Trajectory
        self._traj_fig    = Figure(figsize=(10, 5))
        self._traj_canvas = FigureCanvas(self._traj_fig)
        self._traj_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._inner_tabs.addTab(self._traj_canvas, "Risk score trajectory")

        # 4 — Summary table
        self._tbl_widget = QWidget()
        tbl_vl = QVBoxLayout(self._tbl_widget)
        tbl_vl.setContentsMargins(0, 0, 0, 0)
        tbl_vl.addWidget(_styled_label(
            "  Mean ± std per participant.  "
            "Green = better than group mean.  Red = worse than group mean.",
            size=9, color="#6c7086"
        ))
        self._comp_table = QTableWidget()
        self._comp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._comp_table.setAlternatingRowColors(True)
        self._comp_table.verticalHeader().setVisible(False)
        self._comp_table.setStyleSheet(_TABLE_STYLE)
        tbl_vl.addWidget(self._comp_table)
        self._inner_tabs.addTab(self._tbl_widget, "Summary table (Δ from mean)")

        vl.addWidget(self._inner_tabs)
        self._refresh_all()

    # ── checkbox management ───────────────────────────────────────────────────

    def _rebuild_checkboxes(self, pids: List[str]):
        for cb in self._p_checks.values():
            self._ctrl.removeWidget(cb)
            cb.deleteLater()
        self._p_checks.clear()
        self._all_pids = list(pids)
        # insert after the "Compare participants:" label (index 0)
        for i, pid in enumerate(pids):
            cb = QCheckBox(pid)
            color = PARTICIPANT_COLORS[i % len(PARTICIPANT_COLORS)]
            cb.setStyleSheet(
                f"color:{color}; font-size:11px; font-family:'Courier New';"
            )
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_all)
            self._p_checks[pid] = cb
            self._ctrl.insertWidget(1 + i, cb)

    # ── statistics helpers ────────────────────────────────────────────────────

    def _selected_pids(self) -> List[str]:
        return [pid for pid, cb in self._p_checks.items() if cb.isChecked()]

    def _pid_color(self, pid: str) -> str:
        return _pid_color(pid, self._all_pids)

    def _participant_means(self, pid: str) -> Dict[str, float]:
        recs = [r for r in self._records if r["participant_id"] == pid]
        out  = {}
        for fk, *_ in self._SF:
            vals = [float(r[fk]) for r in recs if r.get(fk) is not None]
            out[fk] = float(np.mean(vals)) if vals else 0.0
        return out

    def _participant_std(self, pid: str) -> Dict[str, float]:
        recs = [r for r in self._records if r["participant_id"] == pid]
        out  = {}
        for fk, *_ in self._SF:
            vals = [float(r[fk]) for r in recs if r.get(fk) is not None]
            out[fk] = float(np.std(vals)) if len(vals) > 1 else 0.0
        return out

    # ── panel 1 — radar ───────────────────────────────────────────────────────

    def _draw_radar(self):
        self._radar_fig.clear()
        pids = self._selected_pids()
        if not pids:
            return

        labels  = [lbl.split("—")[1].strip() if "—" in lbl else lbl
                   for _, lbl, *_ in self._SF]
        n_feat  = len(self._SF)
        angles  = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
        angles += angles[:1]

        ax = self._radar_fig.add_subplot(111, polar=True)
        ax.set_facecolor(MPL_STYLE["axes.facecolor"])
        ax.spines["polar"].set_color("#44475a")

        ax.plot(angles, [1.0] * (n_feat + 1),
                color="#f38ba8", linewidth=0.8, linestyle=":", zorder=1)
        ax.fill(angles, [1.0] * (n_feat + 1), color="#f38ba8", alpha=0.05)

        for pid in pids:
            means = self._participant_means(pid)
            norm_vals = []
            for fk, _, _, direction, thresh in self._SF:
                v = means.get(fk, 0.0)
                if thresh and thresh != 0:
                    norm = ((thresh - v) / thresh) if direction == "below" \
                           else (v / thresh)
                else:
                    norm = 0.0
                norm_vals.append(max(0.0, norm))
            norm_vals += norm_vals[:1]

            color = self._pid_color(pid)
            ax.plot(angles, norm_vals, color=color, linewidth=1.8,
                    label=pid, zorder=3)
            ax.fill(angles, norm_vals, color=color, alpha=0.12, zorder=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8, color="#bac2de")
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.set_yticklabels(["0.5×", "threshold", "1.5×"], size=7, color="#6c7086")
        ax.tick_params(colors="#bac2de")
        ax.grid(color="#313244", linewidth=0.6)
        ax.set_title("Feature profile (normalised — threshold = 1.0)",
                     size=10, pad=14, color="#cdd6f4")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                  fontsize=9, framealpha=0.3)
        self._radar_fig.tight_layout()
        self._radar_canvas.draw()

    # ── panel 2 — grouped bar ────────────────────────────────────────────────

    def _draw_grouped_bar(self):
        self._gbar_fig.clear()
        pids = self._selected_pids()
        if not pids:
            return

        ax = self._gbar_fig.add_subplot(111)
        ax.set_facecolor(MPL_STYLE["axes.facecolor"])

        n_feat = len(self._SF)
        n_pid  = len(pids)
        x      = np.arange(n_feat)
        width  = 0.8 / max(n_pid, 1)
        feat_labels = [lbl.split("—")[0].strip() for _, lbl, *_ in self._SF]

        for i, pid in enumerate(pids):
            means  = self._participant_means(pid)
            stds   = self._participant_std(pid)
            vals   = [means.get(fk, 0.0) for fk, *_ in self._SF]
            errs   = [stds.get(fk, 0.0)  for fk, *_ in self._SF]
            offset = (i - n_pid / 2 + 0.5) * width
            color  = self._pid_color(pid)
            ax.bar(x + offset, vals, width * 0.9, label=pid,
                   color=color, alpha=0.82, edgecolor="#181825", linewidth=0.6)
            ax.errorbar(x + offset, vals, yerr=errs,
                        fmt="none", ecolor="#cdd6f4", capsize=2,
                        linewidth=0.8, alpha=0.5)

        for j, (fk, _, _, _, thresh) in enumerate(self._SF):
            if thresh is not None:
                ax.hlines(thresh, j - 0.45, j + 0.45,
                          colors="#f38ba8", linewidth=1.0, linestyles=":")

        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean value (± std)", fontsize=9)
        ax.set_title("Feature means per participant  |  dotted line = threshold",
                     fontsize=10, pad=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.4)
        self._gbar_fig.tight_layout()
        self._gbar_canvas.draw()

    # ── panel 3 — risk trajectory ─────────────────────────────────────────────

    def _draw_trajectory(self):
        self._traj_fig.clear()
        pids = self._selected_pids()
        if not pids:
            return

        ax = self._traj_fig.add_subplot(111)
        ax.set_facecolor(MPL_STYLE["axes.facecolor"])

        ax.axhspan(0.00, 0.20, color="#639922", alpha=0.07, zorder=0)
        ax.axhspan(0.20, 0.45, color="#BA7517", alpha=0.07, zorder=0)
        ax.axhspan(0.45, 0.70, color="#E24B4A", alpha=0.07, zorder=0)
        ax.axhspan(0.70, 1.00, color="#D85A30", alpha=0.10, zorder=0)
        for lbl_txt, y in [("Low", 0.10), ("Moderate", 0.325),
                            ("High", 0.575), ("Severe", 0.85)]:
            ax.text(0.01, y, lbl_txt, transform=ax.get_yaxis_transform(),
                    fontsize=7, color="#6c7086", va="center")

        for i, pid in enumerate(pids):
            pid_recs = sorted(
                [r for r in self._records if r["participant_id"] == pid],
                key=lambda r: r.get("start_epoch_ts") or 0,
            )
            scores = [r.get("risk_score") or 0.0 for r in pid_recs]
            x_pts  = list(range(1, len(scores) + 1))
            color  = self._pid_color(pid)
            ax.plot(x_pts, scores, marker="o", markersize=5,
                    linestyle="--" if i % 2 else "-", linewidth=1.8,
                    color=color, label=pid, zorder=3)
            if scores:
                band = (pid_recs[-1].get("risk_band") or "").upper()
                ax.annotate(
                    BAND_DISPLAY.get(band, band),
                    xy=(x_pts[-1], scores[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=color, va="center",
                )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Risk score (0–1)", fontsize=9)
        ax.set_xlabel("Session #", fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_title("Risk score trajectory — all participants", fontsize=10, pad=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.4)
        self._traj_fig.tight_layout()
        self._traj_canvas.draw()

    # ── panel 4 — summary table ───────────────────────────────────────────────

    def _draw_table(self):
        pids = self._selected_pids()
        if not pids:
            self._comp_table.setRowCount(0)
            return

        # Use ALL 17 features in the comparison table (not just scored 7)
        all_features = FEATURES
        col_headers = ["Feature"] + pids + ["Group mean", "Threshold"]
        self._comp_table.setColumnCount(len(col_headers))
        self._comp_table.setHorizontalHeaderLabels(col_headers)
        self._comp_table.setRowCount(len(all_features))

        # compute means/stds for ALL features
        pid_means_all: Dict[str, Dict[str, float]] = {}
        pid_stds_all:  Dict[str, Dict[str, float]] = {}
        for pid in pids:
            recs = [r for r in self._records if r["participant_id"] == pid]
            m, s = {}, {}
            for fk, *_ in all_features:
                vals = [float(r[fk]) for r in recs if r.get(fk) is not None]
                m[fk] = float(np.mean(vals)) if vals else None
                s[fk] = float(np.std(vals))  if len(vals) > 1 else 0.0
            pid_means_all[pid] = m
            pid_stds_all[pid]  = s

        group_means: Dict[str, float] = {}
        for fk, *_ in all_features:
            all_vals = [float(r[fk]) for r in self._records
                        if r["participant_id"] in pids and r.get(fk) is not None]
            group_means[fk] = float(np.mean(all_vals)) if all_vals else None

        for row_i, (fk, feat_lbl, unit, direction, thresh) in enumerate(all_features):
            short = feat_lbl.split("—")[1].strip() if "—" in feat_lbl else feat_lbl
            name_item = QTableWidgetItem(short)
            name_item.setForeground(QColor("#89b4fa"))
            self._comp_table.setItem(row_i, 0, name_item)

            gm = group_means.get(fk)
            for col_i, pid in enumerate(pids, start=1):
                mean_v = pid_means_all[pid].get(fk)
                std_v  = pid_stds_all[pid].get(fk, 0.0)

                if mean_v is None:
                    text = "—"
                    item = QTableWidgetItem(text)
                    item.setForeground(QColor("#45475a"))
                else:
                    if unit == "ms":
                        text = f"{mean_v:.1f} ± {std_v:.1f} ms"
                    elif unit == "n":
                        text = f"{mean_v:.1f} ± {std_v:.1f}"
                    else:
                        text = f"{mean_v:.4f} ± {std_v:.4f}"
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)

                    # colour only if this feature has a direction defined
                    if direction is not None and gm is not None and gm != 0:
                        delta = mean_v - gm
                        worse = (delta > 0) if direction == "above" else (delta < 0)
                        if abs(delta) > abs(gm) * 0.05:
                            item.setForeground(
                                QColor("#f38ba8") if worse else QColor("#a6e3a1")
                            )
                        else:
                            item.setForeground(QColor("#cdd6f4"))
                    else:
                        item.setForeground(QColor("#cdd6f4"))

                item.setTextAlignment(Qt.AlignCenter)
                self._comp_table.setItem(row_i, col_i, item)

            # group mean column
            if gm is not None:
                gm_text = f"{gm:.1f} ms" if unit == "ms" else f"{gm:.4f}"
            else:
                gm_text = "—"
            gm_item = QTableWidgetItem(gm_text)
            gm_item.setTextAlignment(Qt.AlignCenter)
            gm_item.setForeground(QColor("#585b70"))
            self._comp_table.setItem(row_i, len(pids) + 1, gm_item)

            # threshold column
            thresh_text = (
                f"{thresh} ({direction})" if thresh is not None else "—"
            )
            thresh_item = QTableWidgetItem(thresh_text)
            thresh_item.setTextAlignment(Qt.AlignCenter)
            thresh_item.setForeground(
                QColor("#f38ba8") if thresh is not None else QColor("#45475a")
            )
            self._comp_table.setItem(row_i, len(pids) + 2, thresh_item)

        hh = self._comp_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        # participant mean columns — stretch
        for c in range(1, len(pids) + 1):
            hh.setSectionResizeMode(c, QHeaderView.Stretch)
        # group mean + threshold — fixed content width
        hh.setSectionResizeMode(len(pids) + 1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(pids) + 2, QHeaderView.ResizeToContents)

    # ── orchestrate ───────────────────────────────────────────────────────────

    def _refresh_all(self):
        self._draw_radar()
        self._draw_grouped_bar()
        self._draw_trajectory()
        self._draw_table()

    def refresh(self, records: List[Dict]):
        self._records = records
        pids_new = sorted(set(r["participant_id"] for r in records))
        if pids_new != self._all_pids:
            self._rebuild_checkboxes(pids_new)
        self._refresh_all()

    # ── export ────────────────────────────────────────────────────────────────

    def _export_comparison(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save comparison CSV",
            "mestts_comparison.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        pids      = self._selected_pids()
        pid_means = {pid: self._participant_means(pid) for pid in pids}
        pid_stds  = {pid: self._participant_std(pid)   for pid in pids}
        pid_means_exp: Dict[str, Dict[str, float]] = {}
        pid_stds_exp:  Dict[str, Dict[str, float]] = {}
        for pid in pids:
            recs = [r for r in self._records if r["participant_id"] == pid]
            m2, s2 = {}, {}
            for fk, *_ in FEATURES:
                vals = [float(r[fk]) for r in recs if r.get(fk) is not None]
                m2[fk] = round(float(np.mean(vals)), 4) if vals else None
                s2[fk] = round(float(np.std(vals)),  4) if len(vals) > 1 else 0.0
            pid_means_exp[pid] = m2
            pid_stds_exp[pid]  = s2
        rows = []
        for fk, feat_lbl, unit, direction, thresh in FEATURES:
            row: Dict = {"feature": feat_lbl, "threshold": thresh,
                         "direction": direction if direction else "info_only"}
            for pid in pids:
                row[f"{pid}_mean"] = pid_means_exp[pid].get(fk)
                row[f"{pid}_std"]  = pid_stds_exp[pid].get(fk, 0.0)
            rows.append(row)
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        QMessageBox.information(self, "Export complete",
                                f"Comparison data exported to:\n{path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════════

class AnalyticsDashboard(QWidget):

    def __init__(self, db_path: str, export_dir: str):
        super().__init__()
        self._db_path    = db_path
        self._export_dir = export_dir
        self._records: List[Dict] = []
        self._build_ui()
        self._load_and_refresh()

    def _build_ui(self):
        self.setWindowTitle("MESTTS — Participant Analytics Dashboard")
        self.setGeometry(60, 60, 1600, 860)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("background-color: #181825; color: #cdd6f4;")

        vl = QVBoxLayout(self)
        vl.setContentsMargins(16, 16, 16, 16)
        vl.setSpacing(10)

        # header bar
        header = QFrame()
        header.setStyleSheet(
            "background:#1e1e2e; border-radius:8px; border:1px solid #313244;"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 10, 16, 10)
        hl.addWidget(_styled_label(
            "MESTTS  —  Participant Analytics Dashboard",
            size=13, bold=True, color="#89b4fa"
        ))
        hl.addStretch()
        self._db_lbl = _styled_label("", size=9, color="#6c7086")
        hl.addWidget(self._db_lbl)
        for label, slot in [
            ("Refresh DB",  self._load_and_refresh),
            ("Export JSON", lambda: self._export("json")),
            ("Export CSV",  lambda: self._export("csv")),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(_BTN_STYLE)
            btn.clicked.connect(slot)
            hl.addWidget(btn)
        vl.addWidget(header)

        # metric cards
        self._cards_row = QHBoxLayout()
        vl.addLayout(self._cards_row)

        # tabs
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setStyleSheet("""
            QTabWidget::pane  { background:#1e1e2e; border:1px solid #313244; }
            QTabBar::tab      { background:#313244; color:#bac2de;
                                padding:6px 14px; font-size:11px;
                                font-family:'Courier New'; border-radius:4px;
                                min-width:130px; }
            QTabBar::tab:selected { background:#45475a; color:#cdd6f4;
                                    font-weight:bold; }
            QTabBar::tab:hover    { background:#3d3f53; }
            QTabBar::scroller     { width:24px; }
            QTabBar QToolButton   { background:#313244; color:#cdd6f4;
                                    border:1px solid #45475a; border-radius:3px; }
        """)
        vl.addWidget(self._tabs)

        self._log_tab   = None
        self._trend_tab = None
        self._pie_tab   = None
        self._bar_tab   = None
        self._comp_tab  = None

    # ── metric cards ──────────────────────────────────────────────────────────

    def _rebuild_cards(self):
        while self._cards_row.count():
            item = self._cards_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        r         = self._records
        n         = len(r)
        pids      = len(set(x["participant_id"] for x in r))
        avg_risk  = (sum(x.get("risk_score") or 0 for x in r) / n) if n else 0
        high_flag = sum(1 for x in r
                        if (x.get("risk_band") or "").upper()
                        in ("HIGH", "SEVERE_FLAG"))
        avg_comp  = (sum(x.get("data_completeness") or 0 for x in r) / n) if n else 0

        for label, value, sub in [
            ("Participants",     str(pids),          "in dataset"),
            ("Sessions",         str(n),             "total logged"),
            ("Avg risk score",   f"{avg_risk:.3f}",  "across sessions"),
            ("High / Severe",    str(high_flag),     "sessions flagged"),
            ("Avg completeness", f"{avg_comp:.1%}",  "mean data quality"),
        ]:
            card = QFrame()
            card.setStyleSheet(
                "background:#1e1e2e; border-radius:6px; border:1px solid #313244;"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 10, 14, 10)
            cl.setSpacing(2)
            cl.addWidget(_styled_label(label, size=9,  color="#6c7086"))
            cl.addWidget(_styled_label(value, size=18, bold=True, color="#cdd6f4"))
            cl.addWidget(_styled_label(sub,   size=9,  color="#585b70"))
            self._cards_row.addWidget(card)

    # ── load / refresh ────────────────────────────────────────────────────────

    def _load_and_refresh(self):
        self._records = load_dataset(self._db_path)
        save_dataset(self._records, self._export_dir)
        self._db_lbl.setText(
            f"DB: {self._db_path}  |  {len(self._records)} sessions"
        )
        self._rebuild_cards()

        if self._log_tab is None:
            self._log_tab   = SessionLogTab(self._records)
            self._trend_tab = FeatureTrendTab(self._records)
            self._pie_tab   = RiskDistributionTab(self._records)
            self._bar_tab   = FeatureImpactTab(self._records)
            self._comp_tab  = ComparativeAnalysisTab(self._records)
            self._tabs.addTab(self._log_tab,   "Session Log")
            self._tabs.addTab(self._trend_tab, "Feature Trends")
            self._tabs.addTab(self._pie_tab,   "Risk Distribution")
            self._tabs.addTab(self._bar_tab,   "Feature Impact")
            self._tabs.addTab(self._comp_tab,  "Comparative Analysis")
        else:
            self._log_tab.refresh(self._records)
            self._trend_tab.refresh(self._records)
            self._pie_tab.refresh(self._records)
            self._bar_tab.refresh(self._records)
            self._comp_tab.refresh(self._records)

    # ── export ────────────────────────────────────────────────────────────────

    def _export(self, fmt: str):
        ext     = "JSON files (*.json)" if fmt == "json" else "CSV files (*.csv)"
        default = f"mestts_dataset.{fmt}"
        path, _ = QFileDialog.getSaveFileName(self, "Save dataset", default, ext)
        if not path:
            return
        try:
            export_to_path(self._records, path, fmt)
            QMessageBox.information(
                self, "Export complete",
                f"Dataset exported to:\n{path}\n\n"
                f"{len(self._records)} sessions written.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MESTTS Analytics Dashboard")
    parser.add_argument("--db",         default="data/sessions.db",
                        help="Path to sessions.db (default: data/sessions.db)")
    parser.add_argument("--export-dir", default="data/exports",
                        help="Directory for auto-exported dataset files")
    args = parser.parse_args()

    if not Path(args.db).exists():
        alt = Path(__file__).parent.parent / args.db
        if alt.exists():
            args.db = str(alt)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AnalyticsDashboard(db_path=args.db, export_dir=args.export_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()