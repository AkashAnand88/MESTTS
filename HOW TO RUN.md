# MESTTS — How to Run
## Multimodal Eye–Speech–Typing Temporal Synchronization System

---

## Prerequisites Overview

| Component | What you need |
|-----------|--------------|
| Python | 3.9 or higher |
| Camera | Android phone running **DroidCam** or **IP Webcam** app, on the same WiFi as your PC |
| Microphone | USB or built-in, any standard mic |
| Vosk model | Downloaded manually (offline ASR — ~50MB) |
| OS | Windows 10+, macOS 12+, or Ubuntu 20.04+ |

---

## Step 1 — Clone / Place the Project

```
mestts/
├── config/
├── core/
├── modules/
├── alignment/
├── analytics/
├── persistence/
├── display/
├── scripts/
├── tests/
└── requirements.txt
```

All commands below are run from inside the `mestts/` root directory.

---

## Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv

# Activate it:
# Linux / macOS:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

---

## Step 3 — Install System-Level Dependencies

PyAudio requires PortAudio to be installed at the OS level **before** `pip install`.

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-dev
```

### macOS (Homebrew)
```bash
brew install portaudio
```

### Windows
PortAudio is bundled inside the PyAudio wheel — no extra step needed.
If `pip install pyaudio` fails on Windows, install the pre-built wheel:
```bash
pip install pipwin
pipwin install pyaudio
```

---

## Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

If `python-Levenshtein` fails to build (it needs a C compiler):
```bash
# Linux: sudo apt install python3-dev build-essential
# macOS: xcode-select --install
# Windows: install "Build Tools for Visual Studio" from microsoft.com
# --- OR just skip it (rapidfuzz covers the same functionality):
pip install -r requirements.txt --ignore-requires-python
```

Verify the install:
```bash
python3 -c "import cv2, mediapipe, vosk, pyaudio, pynput, rapidfuzz; print('All OK')"
```

---

## Step 5 — Download the Vosk Speech Model

MESTTS uses **offline** ASR — no internet during sessions.

1. Go to: https://alphacephei.com/vosk/models
2. Download **vosk-model-en-us-0.22** (recommended, ~1.8GB) or the smaller **vosk-model-small-en-us-0.15** (~40MB, lower accuracy)
3. Extract the folder and place it at:

```
mestts/
└── models/
    └── vosk-model-en-us-0.22/     ← the folder from the zip
        ├── am/
        ├── conf/
        ├── graph/
        └── ...
```

4. Update `config/experiment_config.yaml` if your folder name differs:

```yaml
speech:
  vosk_model_path: "models/vosk-model-en-us-0.22"
```

---

## Step 6 — Set Up the IPCam (Phone as Webcam)

### Option A — DroidCam (recommended, Android + Windows/Linux)
1. Install **DroidCam** on your Android phone (Play Store)
2. Install **DroidCam Client** on your PC (www.dev47apps.com)
3. Connect phone and PC to the same WiFi
4. Note the IP shown in the app (e.g. `192.168.1.105`)
5. Update your config:
```yaml
vision:
  ipcam_url: "http://192.168.1.105:4747/video"
```

### Option B — IP Webcam (Android)
1. Install **IP Webcam** app (Play Store)
2. Tap "Start server" — note the URL shown (e.g. `http://192.168.1.105:8080`)
3. Update your config:
```yaml
vision:
  ipcam_url: "http://192.168.1.105:8080/video"
```

### Option C — Built-in / USB Webcam
Replace the URL with a device index:
```yaml
vision:
  ipcam_url: "0"    # 0 = first webcam, 1 = second, etc.
```
> Note: built-in webcams typically have higher latency than IPCam.
> Increase `camera_latency_ms` in config accordingly (try 80–120ms).

### Test your camera URL
```bash
python3 -c "
import cv2
cap = cv2.VideoCapture('http://192.168.1.105:4747/video')
ret, frame = cap.read()
print('Camera OK, frame shape:', frame.shape if ret else 'FAILED')
cap.release()
"
```

---

## Step 7 — Configure the Experiment

Edit `config/experiment_config.yaml`. The most important settings:

```yaml
vision:
  ipcam_url: "http://192.168.1.105:4747/video"   # ← your camera URL
  camera_latency_ms: 50                            # increase if camera is slow

speech:
  vosk_model_path: "models/vosk-model-en-us-0.22" # ← your model folder

sentence_flow:
  practice_sentences: 2    # first 2 sentences are warm-up, excluded from scoring
```

All other defaults are fine to start with.

---

## Step 8 — Run Calibration (First Time and Each Session)

Calibration maps the camera's iris pixel coordinates to screen coordinates.
**Run this once before each screening session**, or whenever the participant
repositions.

```bash
python3 scripts/run_calibration.py --participant P001
```

With the optional live preview (10 seconds of gaze feedback after calibration):
```bash
python3 scripts/run_calibration.py --participant P001 --show-preview
```

**What happens:**
1. A fullscreen window opens showing instructions
2. A red dot appears at the LEFT of the screen — fixate on it
3. Dot moves to CENTER — fixate
4. Dot moves to RIGHT — fixate
5. The system fits a regression model and reports RMSE
   - `ACCEPTED` (RMSE < 50px) → calibration profile saved to `config/calibration_params.json`
   - `REJECTED` (RMSE ≥ 50px) → reposition camera or participant and retry

**Exit codes:**
- `0` = Accepted ✓
- `1` = Rejected (retry)
- `2` = Hardware error (camera not found, no face detected)

---

## Step 9 — Run a Screening Session

```bash
python3 scripts/run_session.py --participant P001
```

With a custom config file:
```bash
python3 scripts/run_session.py --participant P001 --config config/experiment_config.yaml
```

**What happens:**
1. Calibration runs automatically at session start
2. A stimulus sentence appears on screen
3. The participant reads the sentence aloud (speech channel) and types it (typing channel)
4. The eye tracker records fixations continuously
5. Press **ENTER** or the USB pedal to advance to the next sentence
6. After all sentences, a risk score is computed and printed
7. Session data is saved to `data/sessions.db` and exported to `data/exports/`

**Session output files:**
```
data/
├── sessions.db                        # SQLite — all structured data
├── exports/
│   ├── <session_id>.jsonl             # Raw event log (lossless)
│   ├── <session_id>_export.json       # Full session JSON
│   ├── session_summary.csv            # One row per session
│   ├── sentence_records.csv           # One row per sentence
│   └── word_records.csv               # One row per word slot
└── logs/
    └── session_YYYYMMDD_HHMMSS.log    # Full run log
```

---

## Step 10 — Run the Tests

The test suite does not require any hardware (no camera, mic, or display needed).

```bash
# Install pytest first:
pip install pytest

# Run all tests:
python3 -m pytest tests/ -v

# Run only unit tests:
python3 -m pytest tests/unit/ -v

# Run only integration tests:
python3 -m pytest tests/integration/ -v

# Run with coverage:
pip install pytest-cov
python3 -m pytest tests/ --cov=. --cov-report=term-missing
```

Expected output: all tests pass. Tests cover the core temporal metrics,
risk scorer, fuzzy aligner, and the full alignment pipeline.

---

## Physical Setup for Research-Grade Data

For best results, follow these constraints from the architecture spec:

| Parameter | Recommendation |
|-----------|---------------|
| Room lighting | Uniform ambient light; avoid strong side-lighting |
| Participant distance | 60–70 cm from screen (chin rest recommended) |
| Camera position | Top-center of monitor or below screen at eye level |
| Camera angle | < 15° lateral, < 10° vertical from face |
| Microphone | Directional USB mic, 20–40 cm from participant, away from keyboard |
| Advance trigger | USB foot pedal (preferred) or separate keyboard; avoid contaminating typing channel |

---

## Troubleshooting

### "Cannot open camera"
- Confirm the phone and PC are on the same WiFi network (not guest networks)
- Try opening the camera URL directly in a browser — you should see a video feed
- Check firewall rules; the camera port (4747 or 8080) must be reachable

### "No face/iris detected during calibration"
- Improve room lighting (avoid backlit positions)
- Ensure participant is within 60–80 cm of camera
- Try reducing camera resolution in the DroidCam/IP Webcam app settings

### PyAudio / PortAudio install fails
- Linux: `sudo apt install portaudio19-dev python3-dev`
- macOS: `brew install portaudio`
- Windows: use `pipwin install pyaudio`

### Vosk produces no speech results
- Confirm the model path in config matches the actual folder name exactly
- Verify the microphone is recognized: `python3 -c "import pyaudio; p=pyaudio.PyAudio(); print(p.get_device_count(), 'audio devices')"`
- Check that the mic is not muted in system settings

### Risk score shows INCONCLUSIVE
- `data_completeness < 0.60` — more than 40% of word slots failed to complete
- Usually caused by: camera dropout (fixation data missing), ASR not recognizing words, or participant completing sentences too quickly
- Check `data/logs/` for the session log to identify which module was dropping data

### Tkinter display not found (Linux)
```bash
sudo apt install python3-tk
```

---

## Modes of Operation

The system supports three modality combinations, controlled automatically
based on which modules successfully acquire data:

| Mode | Modalities active | Use case |
|------|------------------|----------|
| `dual` | Gaze + Speech + Typing | Full protocol (recommended) |
| `gaze_speech` | Gaze + Speech only | No typing task |
| `gaze_typing` | Gaze + Typing only | Silent reading + typing |

To run gaze+typing only (disable speech), stop the microphone from being
opened by pointing `vosk_model_path` to a non-existent path — the system
degrades gracefully and continues with the remaining two modalities.

---

## Important Disclaimer

> This system produces a **behavioral risk indicator only**.
> It does not constitute a clinical diagnosis of dyslexia or any other condition.
> All results must be interpreted by a qualified specialist within a
> comprehensive assessment context.
