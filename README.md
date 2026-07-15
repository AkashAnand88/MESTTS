# Multimodal Eye Speech Typing Temporal Synchronization Architecture
## Dyslexia Risk Screening System: Codebase Reference

---

### Overview

Research-grade implementation of a multimodal behavioral screening system
that synchronizes gaze tracking, speech recognition, and typing to compute
Eye-Voice Span (EVS) and related temporal metrics for dyslexia risk
indicator scoring.

**This system is not a clinical diagnostic tool.**

---

### Directory Structure

```
dyslexia_screen/
├── config/
│   ├── experiment_config.yaml      # All thresholds, weights, camera params
│   ├── sentence_protocol.json      # Stimulus sentences
│   └── calibration_params.json     # Per-session calibration output
├── core/
│   ├── clock.py                    # Monotonic session clock (singleton)
│   ├── config.py                   # Configuration loader
│   ├── event_bus.py                # Thread-safe pub/sub message broker
│   ├── state_machine.py            # Session/sentence FSM
│   └── word_slot.py                # WordSlot join-table unit
├── modules/
│   ├── vision/
│   │   ├── calibration.py          # 3-point gaze calibration (LinearRegression)
│   │   ├── capture.py              # OpenCV + MediaPipe iris pipeline thread
│   │   └── fixation_detector.py    # Rolling-buffer fixation + regression detection
│   ├── speech/
│   │   ├── capture.py              # Vosk streaming ASR thread
│   │   └── fuzzy_aligner.py        # Lookahead fuzzy word alignment
│   └── typing/
│       └── hook.py                 # pynput keyboard hook + word buffer
├── alignment/
│   ├── synchronizer.py             # Word-index join table + synchronizer thread
│   └── temporal_metrics.py         # EVS_speech, EVS_typing, STG aggregation
├── analytics/
│   ├── feature_extractor.py        # F01–F17 session feature vector
│   └── risk_scorer.py              # Weighted rule-based risk scoring (RS)
├── persistence/
│   ├── db_manager.py               # SQLite schema + CRUD
│   ├── session_logger.py           # Async write queue + JSON export
│   └── export.py                   # CSV research export
├── display/
│   └── sentence_renderer.py        # Tkinter stimulus display + trigger
├── tests/
│   ├── unit/
│   │   ├── test_temporal_metrics.py
│   │   ├── test_risk_scorer.py
│   │   └── test_fuzzy_aligner.py
│   └── integration/
│       └── test_alignment_pipeline.py
└── scripts/
    ├── run_session.py              # Main entry point
    └── run_calibration.py          # Standalone calibration tool
```

---

### Prerequisites

```
Python >= 3.9
IPCam running DroidCam / IP Webcam on local WiFi
Vosk model downloaded to models/
```

```bash
pip install -r requirements.txt
```

---

### Running a Session

```bash
python scripts/run_session.py --participant P001
```

Session data is written to `data/sessions.db`.
JSON export is written to `data/exports/<session_id>.json`.
CSV exports are written to `data/exports/`.

---

### Key Temporal Formulae

```
EVS_speech_i  = (P_i − F_i) / 1e6    [ms]
EVS_typing_i  = (Y_i − F_i) / 1e6    [ms]
STG_i         = (Y_i − P_i) / 1e6    [ms]

Where:
  F_i = fixation_onset_ns (camera-latency-corrected)
  P_i = Vosk word start timestamp (session epoch ns)
  Y_i = delimiter keystroke timestamp (session epoch ns)
```

---

### Risk Score

```
RS = Σ_k (w_k × I_k)     where I_k ∈ {0,1}

Risk Bands:
  [0.00, 0.20) → LOW
  [0.20, 0.45) → MODERATE
  [0.45, 0.70) → HIGH
  [0.70, 1.00] → SEVERE_FLAG
  data_completeness < 0.60 → INCONCLUSIVE (override, gated on worst
                                            scored sentence — see Changelog)
```

---

### Ethical Statement

This system produces a **behavioral risk indicator only**.
It does not constitute a clinical diagnosis of dyslexia or any other
condition. All results must be interpreted by a qualified specialist
within a comprehensive assessment context.

---

### Citation

If used in academic work, cite the architectural specification document
and this implementation as: *Multimodal Eye–Speech–Typing Temporal
Synchronization Architecture for Dyslexia Risk Screening*.

---

### Changelog

**Data-quality / race-condition fixes (2026-07-15)**

Root cause: fast typers/readers could advance to the next sentence
before Vosk finished transcribing the tail end of it. The synchronizer
flushed immediately on advance, so any word still waiting on a
`speech_record` was marked `TIMED_OUT` → `EXCLUDED`. Gaze data for
that word was salvaged as `DEGRADED`, but its speech-derived features
were lost. This showed up as low "Speech Coverage" for a sentence in
the PDF report, and could silently inflate variability-based features
(F01, F14) computed from a small, non-representative sample of
early-arriving words — while the *session-average* completeness still
looked fine, since it was mean-blended with fully-captured sentences.

1. **`alignment/synchronizer.py`** — added `pending_speech_count()`,
   a cheap lock-protected snapshot of slots still waiting on ASR.

2. **`core/config.py`** — new `flush_grace_s` property
   (`sentence_flow.flush_grace_s` in `experiment_config.yaml`,
   default `0.8`s). Bounds how long advance can wait for trailing ASR
   results before giving up and flushing anyway.

3. **`scripts/run_session.py`** (`_advance_sentence`) — grace-waits up
   to `flush_grace_s`, polling `pending_speech_count()`, before calling
   `flush_sentence()`. Pumps `QApplication.processEvents()` during the
   wait so the UI doesn't freeze (this runs on the Qt main thread).

4. **`analytics/feature_extractor.py`** — `FeatureExtractor` gained
   `min_sentence_completeness` (default `0.60`, sourced from
   `risk_bands.min_data_completeness`). Any scored sentence below this
   is dropped from feature aggregation entirely and logged, instead of
   being blended into the session mean where its outlier values go
   unnoticed.

5. **`scripts/run_session.py`** (`_close_session`) — the
   `INCONCLUSIVE` risk-band override is now gated on
   `min(session_average_completeness, worst_sentence_completeness)`,
   not the average alone. A session of sentence completeness
   `[1.0, 1.0, 0.5]` previously averaged to `0.83` (comfortably above
   the `0.60` threshold) and scored normally; it now correctly reports
   `INCONCLUSIVE` and prompts a re-run of the affected sentence,
   instead of issuing a confident risk band built on half-missing data.

No DB schema changes, no new dependencies. Existing `experiment_config.yaml`
files work unchanged (new `flush_grace_s` knob has a default).

