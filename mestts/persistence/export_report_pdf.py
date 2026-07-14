"""
persistence/export_report_pdf.py
─────────────────────────────────────────────────────────────────────────────
SessionReportExporter — generates a multi-page PDF session report.

This module was missing from the delivered project (scripts/run_session.py
imports it and calls SessionReportExporter(...).export(...), but no such
file shipped). It has been reconstructed here rather than invented from
scratch: every section, threshold, and interpretation string below is
copied directly from scripts/run_session.py's ResultsDashboard class
(_interpret_risk, _interpret_evs_span, _interpret_calib_quality,
_interpret_completeness, _interpret_features, _build_summary_tab,
_build_sentence_tab, _build_typing_tab, _build_feature_tab), so the PDF a
participant/clinician receives matches the on-screen results window
exactly. It intentionally duplicates (rather than imports) that logic —
importing from scripts.run_session here would create a circular import,
since run_session.py is the module that imports THIS file.

Sections (mirrors ResultsDashboard tabs 0–3):
  Page 1 — Summary: risk badge, key scalar indicators, abbreviated
           feature vector grid.
  Page 2 — Interpretation: overall assessment, data quality notes,
           key reading-behaviour indicators, next steps, disclaimer.
  Page 3 — Per-Sentence Analysis: fixation/speech coverage, completeness,
           EVS, fixation duration, regression rate — one chart image.
  Page 4 — Typing Analysis: typing coverage/correction/STG/EVS-typing
           chart image + summary cards.
  Page 5 — Feature Vector: full F01–F17 table with descriptions.
  Page 6 — Feature Heatmap: per-feature min→max normalised chart.

Architecture spec §Layer 7 (mirrors persistence/export.py's CSV exporter,
but produces a human-readable PDF instead of flat CSVs).
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — no Qt event loop needed for PDF export
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import numpy as np

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER

logger = logging.getLogger(__name__)

# ── Colors — copied from ResultsDashboard so the PDF matches the on-screen
#    dashboard's palette exactly ──────────────────────────────────────────────
GREEN  = rl_colors.HexColor("#00b894")
YELLOW = rl_colors.HexColor("#fdcb6e")
RED    = rl_colors.HexColor("#ff6348")
ACCENT = rl_colors.HexColor("#00d2d3")
BLUE   = rl_colors.HexColor("#74b9ff")
TEXT   = rl_colors.HexColor("#1a1a1a")
MUTED  = rl_colors.HexColor("#888888")
CARD_BG = rl_colors.HexColor("#f2f2f2")

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "F01_evs_speech_std_ms":        "EVS Speech variability (std, ms)",
    "F02_mean_evs_speech_ms":       "Mean EVS Speech (ms)",
    "F03_evs_speech_skew":          "EVS Speech skewness",
    "F04_speech_inversion_rate":    "Speech inversion rate (voice leads eye)",
    "F05_evs_typing_std_ms":        "EVS Typing variability (std, ms)",
    "F06_mean_fixation_duration_ms":"Mean fixation duration (ms)",
    "F07_std_fixation_duration_ms": "Fixation duration std (ms)",
    "F08_regression_rate":          "Regression rate (re-reads / word)",
    "F09_regression_count_total":   "Total regression count",
    "F10_fixation_outlier_ratio":   "Fixation outlier ratio (max/mean)",
    "F11_typing_inversion_rate":    "Typing inversion rate",
    "F12_mean_correction_rate":     "Mean correction rate (backspaces/word)",
    "F13_max_corrections":          "Max corrections in any single word",
    "F14_stg_std_ms":               "Speech-Typing Gap std (ms)",
    "F15_stg_negative_rate":        "STG negative rate",
    "F16_corr_evs_speech_fixation": "Gaze-speech correlation",
    "F17_corr_evs_typing_latency":  "EVS-typing latency correlation",
}

HEADER_FOOTER_TEXT = (
    "MESTTS \u2014 Behavioral Risk Indicator Only. Not a clinical diagnosis. "
    "Results must be interpreted by a qualified specialist."
)


def _get(obj, name, default=0.0):
    """Read an attribute from a dataclass-like object OR a dict, same as
    ResultsDashboard's local _g()/_safe() helpers."""
    if hasattr(obj, name):
        v = getattr(obj, name)
        return v if v is not None else default
    if isinstance(obj, dict):
        v = obj.get(name, default)
        return v if v is not None else default
    return default


# ── Interpretation logic — copied verbatim (as free functions) from
#    ResultsDashboard._interpret_* in scripts/run_session.py ─────────────────
def _interpret_risk(risk):
    band  = getattr(risk, "risk_band", "").lower()
    score = getattr(risk, "risk_score", 0.0)
    if "low" in band:
        headline = "Low likelihood of dyslexia-related reading difficulties detected."
        body = (
            f"The participant's reading behaviour across all sentences fell largely within "
            f"typical ranges (risk score {score:.3f}). Eye movement patterns, speech timing, "
            f"and gaze-speech coordination were consistent with fluent reading. "
            f"No significant markers of phonological decoding difficulty were observed."
        )
        color = GREEN
    elif "moderate" in band:
        headline = "Some markers of reading difficulty were detected \u2014 further assessment recommended."
        body = (
            f"The participant's score of {score:.3f} places them in the moderate-risk band. "
            f"This means certain reading behaviours \u2014 such as irregular fixation timing, "
            f"gaze-speech misalignment, or variable reading pace \u2014 appeared more often "
            f"than typical. This is not a diagnosis, but suggests a follow-up clinical "
            f"assessment would be worthwhile."
        )
        color = YELLOW
    elif "high" in band:
        headline = "Significant markers of reading difficulty detected \u2014 clinical follow-up strongly advised."
        body = (
            f"The participant's score of {score:.3f} is in the high-risk band. Multiple "
            f"behavioural indicators \u2014 including atypical fixation patterns, gaze-speech "
            f"timing mismatches, and/or frequent correction attempts \u2014 were present across "
            f"sentences. A qualified specialist should evaluate these results in the context "
            f"of a full assessment."
        )
        color = RED
    else:
        headline = f"Risk band: {band or 'Unknown'}  (score: {score:.3f})"
        body     = "Insufficient data to provide a detailed interpretation."
        color    = ACCENT
    return headline, body, color


def _interpret_evs_span(gaze_span):
    span = gaze_span
    if span > 0.15:
        return ("EVS Gaze Span \u2014 GOOD",
                f"Span of {span:.4f} indicates the eye-tracker captured a wide, reliable sweep "
                f"across the sentence. The calibration covered left-to-right movement well, "
                f"making gaze-to-word mapping accurate.", GREEN)
    elif span > 0.10:
        return ("EVS Gaze Span \u2014 ACCEPTABLE",
                f"Span of {span:.4f} is workable but on the lower side. Minor inaccuracies in "
                f"which word the participant was looking at are possible. Results should be "
                f"interpreted with slight caution.", YELLOW)
    else:
        return ("EVS Gaze Span \u2014 LOW",
                f"Span of {span:.4f} is below the recommended threshold (0.10). The tracker "
                f"could not clearly distinguish gaze position across the full sentence width. "
                f"Consider re-running calibration with more exaggerated left/right eye movement, "
                f"or moving the camera closer.", RED)


def _interpret_calib_quality(gaze_quality):
    q = gaze_quality
    if q >= 0.70:
        return ("Calibration Quality \u2014 EXCELLENT",
                f"{q:.0%} quality means the three calibration points (left, center, right) were "
                f"well-distributed and consistent. Gaze data is reliable.", GREEN)
    elif q >= 0.40:
        return ("Calibration Quality \u2014 MODERATE",
                f"{q:.0%} quality suggests the calibration was acceptable but not ideal \u2014 "
                f"possibly due to slight head movement or asymmetric gaze during calibration. "
                f"Word-level mapping may have minor errors.", YELLOW)
    else:
        return ("Calibration Quality \u2014 POOR",
                f"{q:.0%} quality is below acceptable levels. The calibration points may not have "
                f"been captured cleanly. This can inflate or deflate fixation counts and reduce "
                f"the reliability of all gaze-based features. Re-calibration is strongly advised "
                f"before trusting these results.", RED)


def _interpret_completeness(completeness):
    c = completeness
    if c >= 0.80:
        return ("Data Completeness \u2014 HIGH",
                f"{c:.0%} of word slots had both gaze and speech data successfully captured. "
                f"The session data is highly reliable and the risk score is based on a full picture "
                f"of the participant's reading.", GREEN)
    elif c >= 0.60:
        return ("Data Completeness \u2014 ADEQUATE",
                f"{c:.0%} completeness means most words were captured, but some slots had only "
                f"partial data (gaze without speech, or vice versa). The risk score is still valid "
                f"but may miss some nuance.", YELLOW)
    else:
        return ("Data Completeness \u2014 LOW",
                f"{c:.0%} completeness is below the minimum threshold (60%). A large portion of "
                f"word slots lacked matched gaze+speech data, which means the risk score may not "
                f"accurately reflect the participant's reading profile. Consider repeating the session.",
                RED)


def _interpret_features(fv):
    """Returns list of (label, explanation, color) for key feature vector entries."""
    items = []

    f02 = _get(fv, "F02_mean_evs_speech_ms")
    if f02 > 0:
        evs_interp = (
            f"The eye typically looked ahead of spoken words by ~{f02:.0f} ms. "
            f"This is normal \u2014 fluent readers pre-scan upcoming words. "
            f"A larger span suggests confident preview; near-zero or negative may indicate decoding difficulty."
        )
        evs_color = GREEN if 400 < f02 < 2500 else YELLOW
    else:
        evs_interp = "Could not compute mean EVS \u2014 insufficient matched gaze+speech pairs."
        evs_color  = RED
    items.append(("Eye-Voice Span (F02)", evs_interp, evs_color))

    f01 = _get(fv, "F01_evs_speech_std_ms")
    if f01 < 300:
        std_interp = f"EVS variability was low ({f01:.0f} ms std). The participant read at a steady, consistent pace."
        std_color  = GREEN
    elif f01 < 600:
        std_interp = f"EVS variability was moderate ({f01:.0f} ms std). Some inconsistency in reading pace \u2014 common in borderline cases."
        std_color  = YELLOW
    else:
        std_interp = (
            f"EVS variability was high ({f01:.0f} ms std). Erratic pacing \u2014 the participant's gaze "
            f"and speech were poorly synchronised across words. This is a notable dyslexia marker."
        )
        std_color  = RED
    items.append(("EVS Consistency (F01)", std_interp, std_color))

    f06 = _get(fv, "F06_mean_fixation_duration_ms")
    if f06 < 120:
        fix_interp = f"Mean fixation of {f06:.0f} ms is very short \u2014 fast, skimming-style reading."
        fix_color  = YELLOW
    elif f06 < 280:
        fix_interp = (
            f"Mean fixation of {f06:.0f} ms is in the typical range (150\u2013280 ms). "
            f"The participant paused a normal amount of time on each word."
        )
        fix_color  = GREEN
    else:
        fix_interp = (
            f"Mean fixation of {f06:.0f} ms is long \u2014 the participant lingered on words "
            f"longer than typical, which can indicate decoding effort."
        )
        fix_color  = RED
    items.append(("Mean Fixation Duration (F06)", fix_interp, fix_color))

    f08 = _get(fv, "F08_regression_rate")
    if f08 < 0.05:
        reg_interp = f"Regression rate of {f08:.1%} is low. Very little re-reading occurred."
        reg_color  = GREEN
    elif f08 < 0.15:
        reg_interp = f"Regression rate of {f08:.1%} is mild. Some re-reading occurred."
        reg_color  = YELLOW
    else:
        reg_interp = (
            f"Regression rate of {f08:.1%} is elevated. Frequent re-reading/backtracking "
            f"is a common marker of reading difficulty."
        )
        reg_color  = RED
    items.append(("Regression Rate (F08)", reg_interp, reg_color))

    f16 = _get(fv, "F16_corr_evs_speech_fixation")
    if abs(f16) < 0.1:
        cor_interp = f"Gaze-speech correlation of {f16:.2f} is weak \u2014 timing between eye and voice was inconsistent."
        cor_color  = YELLOW
    else:
        cor_interp = f"Gaze-speech correlation of {f16:.2f} indicates a measurable relationship between fixation timing and speech onset."
        cor_color  = GREEN
    items.append(("Gaze-Speech Correlation (F16)", cor_interp, cor_color))

    return items


def _next_steps_text(risk) -> str:
    band = getattr(risk, "risk_band", "").lower()
    if "low" in band:
        return (
            "\u2022 No immediate action required based on these results.\n"
            "\u2022 If the participant or parent/guardian has ongoing concerns, a routine reading "
            "assessment with a specialist is always a valid step.\n"
            "\u2022 Consider repeating the session in 3\u20136 months if difficulties re-emerge."
        )
    elif "moderate" in band:
        return (
            "\u2022 Share these results with a qualified educational psychologist or specialist "
            "reading teacher.\n"
            "\u2022 A formal psychoeducational or phonological assessment is recommended to clarify "
            "whether intervention is needed.\n"
            "\u2022 Interim classroom supports (extended time, audio aids) may be worth exploring "
            "while awaiting formal assessment."
        )
    else:
        return (
            "\u2022 These results warrant prompt referral to a qualified specialist \u2014 educational "
            "psychologist, speech-language therapist, or dyslexia specialist.\n"
            "\u2022 Do not rely on this tool alone for diagnosis \u2014 a comprehensive assessment "
            "covering phonological awareness, working memory, and reading fluency is essential.\n"
            "\u2022 Early structured literacy intervention significantly improves outcomes."
        )


class SessionReportExporter:
    """
    Generates a multi-page PDF session report, mirroring the on-screen
    ResultsDashboard (scripts/run_session.py) exactly.

    Parameters
    ----------
    export_dir : str
    """

    def __init__(self, export_dir: str) -> None:
        self._dir = Path(export_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(
            name="Small", fontSize=8, textColor=MUTED, leading=10))
        self._styles.add(ParagraphStyle(
            name="Center", alignment=TA_CENTER))

    # ── public API ───────────────────────────────────────────────────────────
    def export(
        self,
        participant_id: str,
        session_id: str,
        risk,
        fv,
        sentence_metrics_list: list,
        completeness: float,
        gaze_span: float,
        gaze_quality: float,
        typing_summary: Optional[dict] = None,
    ) -> Path:
        typing_summary = typing_summary or {}
        timestamp = datetime.now()
        out_path = self._dir / (
            f"{session_id}_report_{timestamp:%Y%m%d_%H%M%S}.pdf"
        )

        doc = SimpleDocTemplate(
            str(out_path), pagesize=LETTER,
            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        )

        story: List = []
        story += self._page_summary(
            participant_id, session_id, timestamp, risk, fv,
            sentence_metrics_list, completeness, gaze_span, gaze_quality,
        )
        story.append(PageBreak())
        story += self._page_interpretation(
            risk, fv, gaze_span, gaze_quality, completeness,
        )
        story.append(PageBreak())
        story += self._page_sentence_analysis(sentence_metrics_list)
        story.append(PageBreak())
        story += self._page_typing_analysis(sentence_metrics_list, typing_summary)
        story.append(PageBreak())
        story += self._page_feature_table(fv)
        story.append(PageBreak())
        story += self._page_feature_heatmap(fv)

        doc.build(
            story,
            onFirstPage=self._header_footer,
            onLaterPages=self._header_footer,
        )
        logger.info("SessionReportExporter: report written to %s", out_path)
        return out_path

    # ── header/footer ────────────────────────────────────────────────────────
    def _header_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.6 * inch, LETTER[1] - 0.4 * inch, HEADER_FOOTER_TEXT)
        canvas.restoreState()

    # ── Page 1: Summary ─────────────────────────────────────────────────────
    def _page_summary(self, participant_id, session_id, timestamp, risk, fv,
                       sentence_metrics_list, completeness, gaze_span, gaze_quality):
        s = self._styles
        story: List = [
            Spacer(1, 0.15 * inch),
            Paragraph("MESTTS SESSION REPORT", ParagraphStyle(
                "Title", parent=s["Title"], fontSize=18, alignment=TA_CENTER)),
            Paragraph("Multimodal Eye\u2013Speech\u2013Typing Temporal Synchronization System",
                      ParagraphStyle("Sub", parent=s["Normal"], alignment=TA_CENTER,
                                     textColor=MUTED, fontSize=9)),
            Spacer(1, 0.2 * inch),
        ]

        n_scored = sum(1 for m in sentence_metrics_list if m.sentence_index >= 0)
        meta_table = Table([
            ["Participant ID", "Session ID", "Date / Time", "Sentences Scored"],
            [str(participant_id), str(session_id)[:16], timestamp.strftime("%Y-%m-%d %H:%M"),
             str(n_scored)],
        ], colWidths=[1.6 * inch] * 4)
        meta_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.25 * inch))

        rc = _interpret_risk(risk)[2]
        band = getattr(risk, "risk_band", "Unknown").upper()
        score = getattr(risk, "risk_score", 0.0)
        badge = Table([
            ["DYSLEXIA RISK ASSESSMENT"],
            [band],
            [f"Risk Score: {score:.4f}"],
            [getattr(risk, "disclaimer", "")],
        ], colWidths=[6.8 * inch])
        badge.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1.2, rc),
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (0, 0), 9),
            ("TEXTCOLOR", (0, 0), (0, 0), MUTED),
            ("FONTSIZE", (0, 1), (0, 1), 22),
            ("FONTNAME", (0, 1), (0, 1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, 1), rc),
            ("FONTSIZE", (0, 2), (0, 2), 12),
            ("FONTSIZE", (0, 3), (0, 3), 7),
            ("TEXTCOLOR", (0, 3), (0, 3), MUTED),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(badge)
        story.append(Spacer(1, 0.2 * inch))

        span_color = GREEN if gaze_span > 0.15 else YELLOW if gaze_span > 0.10 else RED
        qual_color = GREEN if gaze_quality >= 0.70 else YELLOW if gaze_quality >= 0.40 else RED
        comp_color = GREEN if completeness >= 0.80 else YELLOW if completeness >= 0.60 else RED

        cards = Table([
            ["EVS Gaze Span", "Calibration Quality", "Data Completeness", "Sentences Scored"],
            [f"{gaze_span:.4f}", f"{gaze_quality:.1%}", f"{completeness:.1%}", str(n_scored)],
        ], colWidths=[1.7 * inch] * 4)
        cards.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("FONTSIZE", (0, 1), (-1, 1), 13),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, 1), span_color),
            ("TEXTCOLOR", (1, 1), (1, 1), qual_color),
            ("TEXTCOLOR", (2, 1), (2, 1), comp_color),
            ("TEXTCOLOR", (3, 1), (3, 1), ACCENT),
        ]))
        story.append(cards)
        story.append(Spacer(1, 0.2 * inch))

        fv_dict = self._fv_to_dict(fv)
        if fv_dict:
            story.append(Paragraph("Extracted Feature Vector", ParagraphStyle(
                "H2", parent=s["Heading2"], textColor=ACCENT, fontSize=11)))
            rows, row = [], []
            for i, (k, v) in enumerate(sorted(fv_dict.items())):
                label = k if len(k) <= 22 else k[:20] + "."
                row.append(f"{label}\n{v:.4f}" if isinstance(v, float) else f"{label}\n{v}")
                if len(row) == 4:
                    rows.append(row); row = []
            if row:
                while len(row) < 4:
                    row.append("")
                rows.append(row)
            fv_table = Table(rows, colWidths=[1.7 * inch] * 4)
            fv_table.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.5, rl_colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, rl_colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(fv_table)

        return story

    # ── Page 2: Interpretation ───────────────────────────────────────────────
    def _page_interpretation(self, risk, fv, gaze_span, gaze_quality, completeness):
        s = self._styles
        body_style = ParagraphStyle("Body", parent=s["Normal"], fontSize=9.5, leading=13)
        story: List = [Paragraph("Interpretation", ParagraphStyle(
            "H1", parent=s["Heading1"], fontSize=15))]

        def _block(title, body, color):
            t = Table([[Paragraph(f"<b><font color='{color.hexval()}'>{title}</font></b>",
                                   body_style)],
                       [Paragraph(body, body_style)]], colWidths=[6.8 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]))
            return t

        story.append(Paragraph("Overall Assessment", ParagraphStyle(
            "H2", parent=s["Heading2"], textColor=ACCENT, fontSize=11)))
        headline, body, color = _interpret_risk(risk)
        story.append(_block(headline, body, color))
        story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("Data Quality", ParagraphStyle(
            "H2", parent=s["Heading2"], textColor=ACCENT, fontSize=11)))
        for title, b, color in [
            _interpret_evs_span(gaze_span),
            _interpret_calib_quality(gaze_quality),
            _interpret_completeness(completeness),
        ]:
            story.append(_block(title, b, color))
            story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("Key Reading Behaviour Indicators", ParagraphStyle(
            "H2", parent=s["Heading2"], textColor=ACCENT, fontSize=11)))
        for title, b, color in _interpret_features(fv):
            story.append(_block(title, b, color))
            story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("What to Do Next", ParagraphStyle(
            "H2", parent=s["Heading2"], textColor=ACCENT, fontSize=11)))
        story.append(_block("Recommended follow-up", _next_steps_text(risk), ACCENT))
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph(getattr(risk, "disclaimer", ""), self._styles["Small"]))
        return story

    # ── Page 3: Per-Sentence Analysis (chart image) ─────────────────────────
    def _page_sentence_analysis(self, metrics):
        s = self._styles
        story: List = [Paragraph("Per-Sentence Analysis", ParagraphStyle(
            "H1", parent=s["Heading1"], fontSize=15))]
        if not metrics:
            story.append(Paragraph("No sentence metrics available.", s["Normal"]))
            return story

        def _safe(m, attr, default=0.0):
            v = getattr(m, attr, default)
            return float(v) if v is not None else default

        sent_labels = [f"S{m.sentence_index + 1}" for m in metrics]
        n = len(metrics)
        fix_cov = [_safe(m, "n_valid_slots") / max(_safe(m, "n_words", 1), 1) for m in metrics]
        speech_cov = fix_cov  # same proxy used in ResultsDashboard
        completeness = [_safe(m, "data_completeness") for m in metrics]
        mean_evs = [_safe(m, "mean_evs_speech_ms") for m in metrics]
        mean_fix_dur = [_safe(m, "mean_fixation_duration_ms") for m in metrics]
        regression = [_safe(m, "regression_rate") for m in metrics]
        std_fix_dur = [_safe(m, "std_fixation_duration_ms") for m in metrics]

        fig = Figure(figsize=(9.5, 7.2), tight_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
        x = np.arange(n)
        bar_w = 0.35

        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(x - bar_w / 2, fix_cov, bar_w, label="Fixation Coverage", color="#74b9ff", alpha=0.85)
        ax1.bar(x + bar_w / 2, speech_cov, bar_w, label="Speech Coverage", color="#00b894", alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(sent_labels, fontsize=8)
        ax1.set_ylim(0, 1.15)
        ax1.set_title("Fixation & Speech Coverage per Sentence", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=7); ax1.grid(axis="y")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x, completeness, marker="o", color="#fdcb6e", linewidth=2)
        ax2.fill_between(x, completeness, alpha=0.2, color="#fdcb6e")
        ax2.axhline(0.6, color="#ff6348", linestyle="--", linewidth=1, label="Min (0.6)")
        ax2.set_xticks(x); ax2.set_xticklabels(sent_labels, fontsize=8)
        ax2.set_ylim(0, 1.15)
        ax2.set_title("Data Completeness per Sentence", fontsize=9, fontweight="bold")
        ax2.legend(fontsize=7); ax2.grid(axis="y")

        ax3 = fig.add_subplot(gs[1, 1])
        colors3 = ["#00b894" if v > 0 else "#e17055" for v in mean_evs]
        ax3.bar(x, mean_evs, color=colors3, alpha=0.85)
        ax3.axhline(0, color="#888", linewidth=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(sent_labels, fontsize=8)
        ax3.set_title("Mean EVS per Sentence (ms)", fontsize=9, fontweight="bold")
        ax3.grid(axis="y")

        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(x, mean_fix_dur, marker="s", color="#fd79a8", linewidth=2)
        lo = [max(0, m - stdv) for m, stdv in zip(mean_fix_dur, std_fix_dur)]
        hi = [m + stdv for m, stdv in zip(mean_fix_dur, std_fix_dur)]
        ax4.fill_between(x, lo, hi, alpha=0.15, color="#fd79a8")
        ax4.set_xticks(x); ax4.set_xticklabels(sent_labels, fontsize=8)
        ax4.set_title("Mean Fixation Duration \u00b1 Std (ms)", fontsize=9, fontweight="bold")
        ax4.grid(axis="y")

        ax5 = fig.add_subplot(gs[2, 1])
        reg_colors = ["#ff6348" if r > 0.1 else "#fdcb6e" if r > 0.02 else "#00b894" for r in regression]
        ax5.bar(x, regression, color=reg_colors, alpha=0.85)
        ax5.set_xticks(x); ax5.set_xticklabels(sent_labels, fontsize=8)
        ax5.set_title("Regression Rate per Sentence", fontsize=9, fontweight="bold")
        ax5.grid(axis="y")

        story.append(self._fig_to_image(fig, width=6.8 * inch))
        return story

    # ── Page 4: Typing Analysis (chart image + summary cards) ──────────────
    def _page_typing_analysis(self, metrics, typing_summary):
        s = self._styles
        story: List = [Paragraph("Typing Analysis", ParagraphStyle(
            "H1", parent=s["Heading1"], fontSize=15))]

        if not typing_summary:
            story.append(Paragraph(
                "No typing data recorded for this session. Typing is recorded when the "
                "participant types each word into the text box during the session.",
                s["Normal"]))
            return story

        n = len(metrics)
        sent_labels = [f"S{m.sentence_index + 1}" for m in metrics]
        x = np.arange(n)

        def _ts(i, key, default=0.0):
            return float(typing_summary.get(i, {}).get(key, default))

        def _ms(m, attr):
            v = getattr(m, attr, None)
            return float(v) if v is not None else 0.0

        words_typed = [_ts(i, "words_typed") for i in range(n)]
        total_words = [_ts(i, "total_words", 1) for i in range(n)]
        typed_pct = [wt / max(tot, 1) for wt, tot in zip(words_typed, total_words)]
        corrections = [_ts(i, "corrections") for i in range(n)]
        mean_stg = [_ms(m, "mean_stg_ms") for m in metrics]
        mean_evs_t = [_ms(m, "mean_evs_typing_ms") for m in metrics]
        mean_corr_r = [_ms(m, "mean_correction_rate") for m in metrics]

        fig = Figure(figsize=(9.5, 7.2), tight_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)
        bar_w = 0.35

        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(x - bar_w / 2, total_words, bar_w, label="Total Words", color="#636e72", alpha=0.75)
        ax1.bar(x + bar_w / 2, words_typed, bar_w, label="Words Typed", color="#a29bfe", alpha=0.9)
        ax1.set_xticks(x); ax1.set_xticklabels(sent_labels, fontsize=8)
        ax1.set_title("Words Typed vs Total per Sentence", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=7); ax1.grid(axis="y")

        ax2 = fig.add_subplot(gs[1, 0])
        colors2 = ["#00b894" if p >= 0.8 else "#fdcb6e" if p >= 0.5 else "#ff6348" for p in typed_pct]
        ax2.bar(x, typed_pct, color=colors2, alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(sent_labels, fontsize=8)
        ax2.set_ylim(0, 1.15)
        ax2.set_title("Typing Coverage per Sentence", fontsize=9, fontweight="bold")
        ax2.grid(axis="y")

        ax3 = fig.add_subplot(gs[1, 1])
        stg_colors = ["#00b894" if v > 0 else "#e17055" for v in mean_stg]
        ax3.bar(x, mean_stg, color=stg_colors, alpha=0.85)
        ax3.axhline(0, color="#888", linewidth=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(sent_labels, fontsize=8)
        ax3.set_title("Speech\u2013Typing Gap per Sentence (ms)", fontsize=9, fontweight="bold")
        ax3.grid(axis="y")

        story.append(self._fig_to_image(fig, width=6.8 * inch))
        story.append(Spacer(1, 0.15 * inch))

        total_typed = sum(int(w) for w in words_typed)
        total_all = sum(int(w) for w in total_words)
        total_corr = sum(int(c) for c in corrections)
        overall_cov = total_typed / max(total_all, 1)
        mean_stg_all = float(np.mean([v for v in mean_stg if v != 0])) if any(v != 0 for v in mean_stg) else 0.0
        mean_evst_all = float(np.mean([v for v in mean_evs_t if v != 0])) if any(v != 0 for v in mean_evs_t) else 0.0

        cards = Table([
            ["Words Typed", "Typing Coverage", "Total Corrections", "Mean STG", "Mean EVS Typing"],
            [f"{total_typed} / {total_all}", f"{overall_cov:.0%}", str(total_corr),
             f"{mean_stg_all:.0f} ms" if mean_stg_all else "N/A",
             f"{mean_evst_all:.0f} ms" if mean_evst_all else "N/A"],
        ], colWidths=[1.36 * inch] * 5)
        cards.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("FONTSIZE", (0, 1), (-1, 1), 11),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ]))
        story.append(cards)
        return story

    # ── Page 5: Full feature table ──────────────────────────────────────────
    def _page_feature_table(self, fv):
        s = self._styles
        story: List = [Paragraph("Feature Vector", ParagraphStyle(
            "H1", parent=s["Heading1"], fontSize=15))]
        fv_dict = self._fv_to_dict(fv)
        if not fv_dict:
            story.append(Paragraph("Feature vector not available or empty.", s["Normal"]))
            return story

        rows = [["Feature", "Value", "Description"]]
        for k in sorted(fv_dict.keys()):
            v = fv_dict[k]
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            rows.append([k, val_str, FEATURE_DESCRIPTIONS.get(k, "")])

        table = Table(rows, colWidths=[2.1 * inch, 0.9 * inch, 3.8 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, CARD_BG]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        return story

    # ── Page 6: Feature heatmap ──────────────────────────────────────────────
    def _page_feature_heatmap(self, fv):
        s = self._styles
        story: List = [Paragraph("Feature Heatmap", ParagraphStyle(
            "H1", parent=s["Heading1"], fontSize=15))]
        fv_dict = self._fv_to_dict(fv)
        if not fv_dict:
            story.append(Paragraph("Feature vector not available or empty.", s["Normal"]))
            return story

        keys_all = sorted(fv_dict.keys())
        vals_all = [fv_dict[k] for k in keys_all]

        def _minmax(arr):
            mn, mx = min(arr), max(arr)
            span = mx - mn
            if span < 1e-12:
                return [0.5] * len(arr)
            return [(v - mn) / span for v in arr]

        norm_vals = _minmax(vals_all)
        fig = Figure(figsize=(9.5, 3.2))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.array(norm_vals).reshape(1, -1), aspect="auto",
                        cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(keys_all)))
        ax.set_xticklabels(keys_all, rotation=55, ha="right", fontsize=7)
        ax.set_yticks([])
        ax.set_title("Feature Heatmap \u2014 per-feature min\u2192max normalised",
                     fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.45, fraction=0.08,
                     label="low \u2192 high (per feature)")
        fig.subplots_adjust(bottom=0.42, left=0.04, right=0.98, top=0.88)

        story.append(self._fig_to_image(fig, width=6.8 * inch))
        return story

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _fv_to_dict(fv) -> Dict[str, float]:
        if hasattr(fv, "__dict__"):
            return {k: v for k, v in fv.__dict__.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")}
        if isinstance(fv, dict):
            return {k: v for k, v in fv.items() if isinstance(v, (int, float))}
        return {}

    @staticmethod
    def _fig_to_image(fig: Figure, width) -> Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        aspect = fig.get_size_inches()[1] / fig.get_size_inches()[0]
        return Image(buf, width=width, height=width * aspect)
