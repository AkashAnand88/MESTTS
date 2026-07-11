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
  data_completeness < 0.60 → INCONCLUSIVE (override)
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
