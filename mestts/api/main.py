"""
FastAPI Backend for Dyslexia Detection System
Exposes ML models and real-time processing as REST API
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import joblib
import numpy as np
import pandas as pd
import asyncio
import json
import base64
import cv2
import tempfile
import subprocess
import sys
import time
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from multimodel_pipeline.realtime_prediction import RealTimePredictor
    from multimodel_pipeline.dyslexia_system import DyslexiaScreeningSystem
except ImportError as e:
    print(f"Warning: Could not import prediction modules: {e}")
    RealTimePredictor = None
    DyslexiaScreeningSystem = None

try:
    from multimodel_pipeline.audio_module.extract_features import (
        extract_acoustic_features,
        extract_speech_rate_features,
    )
except ImportError as e:
    print(f"Warning: Could not import audio feature extraction: {e}")
    extract_acoustic_features = None
    extract_speech_rate_features = None

# Feature order DyslexiaScreeningSystem.predict_live() expects — confirmed
# directly against each trained model's feature_names_in_, not guessed.
EYE_FEATURE_ORDER = [
    "sampling_rate_hz", "duration_sec", "fixation_count", "avg_fixation_duration_ms",
    "first_pass_fixation_duration_ms", "avg_saccade_length_px", "mean_saccade_velocity_deg_s",
    "regressions_per_sentence", "word_skips_per_sentence", "microsaccade_rate_per_sec",
    "blink_rate_per_min", "pupil_diameter_mm", "gaze_path_entropy",
]
TYPING_FEATURE_ORDER = [
    "Simulation_Error_Rate", "Words_Modified_Count", "Transposition_Count",
    "Mirror_Error_Count", "Phonetic_Error_Count", "Modification_Proportion",
]
AUDIO_FEATURE_ORDER = [
    "mfcc_mean", "jitter", "shimmer", "Pauses_Per_Minute",
    "Phoneme_Confusion_Rate", "Word_Articulation_Time",
]

# ==========================================
# FastAPI App Configuration
# ==========================================

# ==========================================
# Pydantic Models (Request/Response Schemas)
# ==========================================

class EyeTrackingData(BaseModel):
    """Eye tracking features (13 total)"""
    sampling_rate_hz: float = Field(default=60, description="Camera sample rate")
    duration_sec: float = Field(default=1.5, description="Analysis window")
    fixation_count: int = Field(..., ge=0, description="Number of fixations")
    avg_fixation_duration_ms: float = Field(..., ge=0, description="Average fixation duration")
    first_pass_fixation_duration_ms: float = Field(default=0, ge=0)
    avg_saccade_length_px: float = Field(..., ge=0, description="Average saccade distance")
    mean_saccade_velocity_deg_s: float = Field(..., ge=0, description="Saccade velocity")
    regressions_per_sentence: float = Field(default=0, ge=0)
    word_skips_per_sentence: float = Field(default=0, ge=0)
    microsaccade_rate_per_sec: float = Field(default=0, ge=0)
    blink_rate_per_min: float = Field(..., ge=0, description="Blink frequency")
    pupil_diameter_mm: float = Field(..., ge=0, description="Pupil size")
    gaze_path_entropy: float = Field(default=2.0, ge=0)

class TypingData(BaseModel):
    """Typing dynamics features (6 total)"""
    Simulation_Error_Rate: float = Field(..., ge=0, le=1, description="Overall error rate")
    Words_Modified_Count: int = Field(default=0, ge=0)
    Transposition_Count: int = Field(..., ge=0, description="Letter reversals")
    Mirror_Error_Count: int = Field(..., ge=0, description="b/d, p/q confusion")
    Phonetic_Error_Count: int = Field(..., ge=0, description="Sound-based errors")
    Modification_Proportion: float = Field(default=0, ge=0, le=1)

class AudioData(BaseModel):
    """Audio analysis features (6 total)"""
    mfcc_mean: float = Field(..., description="MFCC average")
    jitter: float = Field(..., ge=0, le=1, description="Frequency variation")
    shimmer: float = Field(..., ge=0, le=1, description="Amplitude variation")
    Pauses_Per_Minute: float = Field(..., ge=0, description="Speech disfluencies")
    Phoneme_Confusion_Rate: float = Field(..., ge=0, le=1)
    Word_Articulation_Time: float = Field(..., ge=0, description="Time per word")

class AssessmentRequest(BaseModel):
    """Complete multimodal assessment request"""
    eye_tracking: Optional[EyeTrackingData] = None
    typing: Optional[TypingData] = None
    audio: Optional[AudioData] = None
    session_id: Optional[str] = None

class RiskAssessment(BaseModel):
    """Risk assessment response"""
    risk_score: float = Field(..., ge=0, le=1, description="Overall risk probability")
    risk_level: str = Field(..., description="LOW/MEDIUM/HIGH")
    risk_icon: str = Field(..., description="Visual indicator emoji")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    modalities_active: int = Field(..., ge=0, le=3)
    details: Dict[str, float] = Field(..., description="Individual modality scores")
    timestamp: str

class SessionInfo(BaseModel):
    """Session metadata"""
    session_id: str
    start_time: str
    assessments_count: int
    average_risk: Optional[float] = None

# ==========================================
# Global State
# ==========================================

class ApplicationState:
    def __init__(self):
        self.predictor = None
        self.fusion_system = None
        self.sessions = {}
        self.models_loaded = False
        
    def load_models(self, models_dir: str = "../multimodel_pipeline/"):
        """Load ML models on startup"""
        try:
            if RealTimePredictor:
                from pathlib import Path
                models_path = Path(models_dir)
                # Ensure the path is absolute or correctly relative
                self.predictor = RealTimePredictor(models_dir=str(models_path.resolve()))
            
            if DyslexiaScreeningSystem:
                # BUG FIX: dyslexia_system.py builds paths via raw string
                # concatenation (model_dir + 'eye_model.pkl'), not os.path.join.
                # The caller (lifespan startup) passes str(Path(...)), which has
                # NO trailing slash — so without this fix, model_dir becomes
                # e.g. ".../multimodel_pipelineeye_model.pkl" and silently fails
                # to load, making fusion_system None regardless of anything else.
                fusion_model_dir = str(models_dir)
                if not fusion_model_dir.endswith(("/", "\\")):
                    fusion_model_dir += "/"
                self.fusion_system = DyslexiaScreeningSystem(model_dir=fusion_model_dir)
                print("✅ DyslexiaScreeningSystem loaded")
            
            self.models_loaded = True
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
            return False

state = ApplicationState()

# ==========================================
# Startup/Shutdown Events
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("🚀 Starting Dyslexia Detection API...")
    
    # Define paths relative to this script
    current_dir = Path(__file__).parent
    paths_to_try = [
        current_dir.parent / "multimodel_pipeline", # ../multimodel_pipeline
        current_dir / "multimodel_pipeline",        # ./multimodel_pipeline
        current_dir                                 # current directory
    ]
    
    success = False
    for path in paths_to_try:
        if path.exists() and path.is_dir():
            print(f"📁 Checking directory: {path}")
            if state.load_models(str(path)):
                print(f"✅ All models verified and loaded from: {path}")
                success = True
                break
    
    if not success:
        print("⚠️ Warning: Models not loaded. API will use fallback logic.")
    
    yield  
    
    # --- Shutdown Logic ---
    print("🛑 Shutting down Dyslexia Detection API...")
# Update your FastAPI app initialization:
app = FastAPI(
    title="Dyslexia Detection API",
    description="Real-time multimodal dyslexia screening API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration - Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8001",
        "http://127.0.0.1:8001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# REST API Endpoints
# ==========================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Dyslexia Detection API",
        "version": "1.0.0",
        "models_loaded": state.models_loaded
    }

@app.get("/health")
async def health_check():
    """Detailed health status"""
    return {
        "status": "healthy" if state.models_loaded else "degraded",
        "models": {
            "predictor": state.predictor is not None,
            "fusion_system": state.fusion_system is not None
        },
        "active_sessions": len(state.sessions)
    }

@app.post("/assess", response_model=RiskAssessment)
async def assess_dyslexia_risk(request: AssessmentRequest):
    """
    Perform dyslexia risk assessment using multimodal data
    
    - **eye_tracking**: Eye movement metrics (optional)
    - **typing**: Keyboard dynamics (optional)
    - **audio**: Speech analysis (optional)
    
    At least one modality is required.
    """
    from datetime import datetime
    
    # Validate at least one modality is provided
    if not any([request.eye_tracking, request.typing, request.audio]):
        raise HTTPException(
            status_code=400, 
            detail="At least one modality (eye_tracking, typing, or audio) must be provided"
        )
    
    try:
        # Convert Pydantic models to dicts
        eye_data = request.eye_tracking.dict() if request.eye_tracking else {}
        typing_data = request.typing.dict() if request.typing else {}
        audio_data = request.audio.dict() if request.audio else {}

        # PHASE 5 FIX: DyslexiaScreeningSystem (state.fusion_system) was being
        # loaded at startup but never called anywhere. It's the correct fusion
        # path — it actually feeds [p_eye, p_type, p_audio] through
        # fusion_layer_v1.pkl (a trained stacking meta-learner), unlike
        # RealTimePredictor which just averages the three probabilities.
        # DyslexiaScreeningSystem.predict_live() requires all three modalities
        # (no partial-modality support like RealTimePredictor attempted), so we
        # only use it when eye+typing+audio are all present; otherwise fall
        # back to the heuristic rather than crash.
        if state.fusion_system and eye_data and typing_data and audio_data:
            live_data = {
                "eye": [eye_data[f] for f in EYE_FEATURE_ORDER],
                "typing": [typing_data[f] for f in TYPING_FEATURE_ORDER],
                "audio": [audio_data[f] for f in AUDIO_FEATURE_ORDER],
            }
            result = state.fusion_system.predict_live(live_data)
            risk_score = result["final_score"]
            risk_level = result["risk_label"].replace(" RISK", "")  # "LOW RISK" -> "LOW"
            risk_icon = result["risk_icon"]

            # predict_proba doesn't give a calibrated "confidence" — this is a
            # standard proxy (distance from the 0.5 decision boundary, scaled
            # to 0-1), not a statistically calibrated confidence interval.
            # Flagging so it's not mistaken for something more rigorous than it is.
            confidence = round(2 * abs(risk_score - 0.5), 3)

            response = RiskAssessment(
                risk_score=float(risk_score),
                risk_level=risk_level,
                risk_icon=risk_icon,
                confidence=confidence,
                modalities_active=3,
                details=result["details"],  # real eye_conf/type_conf/audio_conf, not hardcoded 0.5s
                timestamp=datetime.utcnow().isoformat()
            )

        # Use RealTimePredictor for partial-modality requests (it at least
        # tries to handle 1-2 modalities, even though its fusion is a simple
        # average rather than the trained meta-learner)
        elif state.predictor:
            result = state.predictor.predict(eye_data, typing_data, audio_data)
            
            # Map to standard response format
            risk_score = result['risk_score']
            
            # Determine risk level
            if risk_score < 0.35:
                risk_level = "LOW"
                risk_icon = "🟢"
            elif risk_score < 0.75:
                risk_level = "MEDIUM"
                risk_icon = "🟡"
            else:
                risk_level = "HIGH"
                risk_icon = "🔴"
            
            response = RiskAssessment(
                risk_score=float(risk_score),
                risk_level=risk_level,
                risk_icon=risk_icon,
                confidence=0.7,  # lower than the fusion path's — partial modalities, weaker signal
                modalities_active=result.get('active_sensors', 0),
                details={
                    "eye_tracking": 0.5,
                    "typing": 0.5,
                    "audio": 0.5
                },
                timestamp=datetime.utcnow().isoformat()
            )
        
        else:
            # Fallback heuristic if models not loaded
            risk_score = _heuristic_assessment(eye_data, typing_data, audio_data)
            
            response = RiskAssessment(
                risk_score=risk_score,
                risk_level="MEDIUM" if risk_score > 0.5 else "LOW",
                risk_icon="🟡" if risk_score > 0.5 else "🟢",
                confidence=0.6,
                modalities_active=sum([bool(eye_data), bool(typing_data), bool(audio_data)]),
                details={
                    "eye_tracking": risk_score * 0.9 if eye_data else 0,
                    "typing": risk_score * 1.1 if typing_data else 0,
                    "audio": risk_score * 0.95 if audio_data else 0
                },
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Store in session if session_id provided
        if request.session_id:
            if request.session_id not in state.sessions:
                state.sessions[request.session_id] = {
                    "assessments": [],
                    "start_time": datetime.utcnow().isoformat()
                }
            state.sessions[request.session_id]["assessments"].append(response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

def _heuristic_assessment(eye_data: dict, typing_data: dict, audio_data: dict) -> float:
    """Fallback heuristic when models aren't loaded"""
    score = 0.0
    count = 0
    
    if eye_data:
        # High fixation duration suggests difficulty
        if eye_data.get('avg_fixation_duration_ms', 0) > 300:
            score += 0.7
        else:
            score += 0.3
        count += 1
    
    if typing_data:
        # High error rate suggests risk
        if typing_data.get('Simulation_Error_Rate', 0) > 0.15:
            score += 0.8
        else:
            score += 0.2
        count += 1
    
    if audio_data:
        # High pause rate suggests disfluency
        if audio_data.get('Pauses_Per_Minute', 0) > 15:
            score += 0.75
        else:
            score += 0.25
        count += 1
    
    return score / count if count > 0 else 0.5

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Retrieve session information and statistics"""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = state.sessions[session_id]
    assessments = session["assessments"]
    
    avg_risk = None
    if assessments:
        avg_risk = sum(a["risk_score"] for a in assessments) / len(assessments)
    
    return SessionInfo(
        session_id=session_id,
        start_time=session["start_time"],
        assessments_count=len(assessments),
        average_risk=avg_risk
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data"""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del state.sessions[session_id]
    return {"status": "deleted", "session_id": session_id}

@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    return {
        "loaded": state.models_loaded,
        "predictor_available": state.predictor is not None,
        "fusion_system_available": state.fusion_system is not None,
        "supported_modalities": ["eye_tracking", "typing", "audio"]
    }

# ==========================================
# Desktop App Launcher (replaces browser capture)
# ==========================================
# Per project decision: for the final-year presentation, the actual
# calibration/capture/report flow runs via the real main_gui.py (PyQt) —
# the one that already works — instead of the browser-based MediaPipe/
# MediaRecorder reimplementation from Phases 1-3. The web frontend now only
# launches it and later reads back whatever it exports.
#
# IMPORTANT CONSTRAINT: this only works when the backend and the person
# taking the test are on the SAME machine — subprocess.Popen opens a real
# GUI window on this process's display. It is not a remote/deployable flow.

MAIN_GUI_PATH = (Path(__file__).parent.parent / "multimodel_pipeline" / "main_gui.py").resolve()
EXPORT_DIR = (Path(__file__).parent.parent / "multimodel_pipeline" / "data" / "exports").resolve()


class DesktopSessionState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.launched_at: Optional[float] = None


desktop_session = DesktopSessionState()


@app.post("/launch-assessment")
async def launch_assessment(participant_id: str = "WEB_USER"):
    """
    Spawns main_gui.py as a real, separate GUI process. The user completes
    calibration + sentences + clicks "Export" in that window themselves —
    this endpoint does not control it beyond starting it.
    """
    if not MAIN_GUI_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"main_gui.py not found at {MAIN_GUI_PATH}. Check it's actually in multimodel_pipeline/.",
        )
    if desktop_session.process is not None and desktop_session.process.poll() is None:
        raise HTTPException(status_code=409, detail="An assessment window is already running.")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    desktop_session.launched_at = time.time()
    try:
        desktop_session.process = subprocess.Popen(
            [
                sys.executable, str(MAIN_GUI_PATH),
                "--participant", participant_id,
                "--export-dir", str(EXPORT_DIR),
            ],
            cwd=str(MAIN_GUI_PATH.parent),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch main_gui.py: {e}")

    return {"status": "launched", "pid": desktop_session.process.pid}


@app.get("/assessment-status")
async def assessment_status():
    """Poll target — 'running' while the GUI window is open, 'closed' once the user closes it."""
    if desktop_session.process is None:
        return {"status": "not_started"}
    exit_code = desktop_session.process.poll()
    if exit_code is None:
        return {"status": "running"}
    return {"status": "closed", "exit_code": exit_code}


@app.get("/latest-result")
async def latest_result():
    """
    Reads whatever _export_session() (the GUI's own Export button) most
    recently wrote to data/exports/, filtered to files created after this
    launch — so it doesn't accidentally return a stale result from a
    previous, unrelated run of main_gui.py.
    """
    if desktop_session.launched_at is None:
        raise HTTPException(status_code=404, detail="No assessment has been launched yet.")

    candidates = sorted(
        (f for f in EXPORT_DIR.glob("*_export.json") if f.stat().st_mtime >= desktop_session.launched_at),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No export found yet. In the desktop app, click the 'Export' button on the results screen once the session ends.",
        )

    with open(candidates[0], "r") as f:
        return json.load(f)

# ==========================================
# Audio Feature Extraction (Phase 3 addition)
# ==========================================
# NEW endpoint. Does not modify /assess, /health, or any existing route.
# Frontend records audio in-browser, uploads it here, gets back an
# AudioData-shaped JSON, and merges it into the existing /assess payload.
# Runs through the SAME librosa/parselmouth/whisper pipeline already in
# multimodel_pipeline/audio_module/extract_features.py — not reimplemented
# in JS, since Praat's jitter/shimmer algorithm can't be replicated client-side.

def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Browsers record webm/opus or ogg; Praat/librosa want a clean wav."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")


@app.post("/extract-audio-features", response_model=AudioData)
async def extract_audio_features(
    file: UploadFile = File(...),
    reference_text: str = Form(...),
):
    """
    Accepts a recorded audio clip + the sentence the user read aloud.
    Returns the exact AudioData shape /assess expects, so the frontend can
    drop it straight into the existing payload without any schema change.
    """
    if extract_acoustic_features is None or extract_speech_rate_features is None:
        raise HTTPException(
            status_code=503,
            detail="Audio feature extraction unavailable (librosa/parselmouth/whisper not installed on server)",
        )
    if not reference_text.strip():
        raise HTTPException(status_code=400, detail="reference_text is required")

    suffix = Path(file.filename or "").suffix or ".webm"

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = str(Path(tmpdir) / f"input{suffix}")
        wav_path = str(Path(tmpdir) / "converted.wav")

        with open(raw_path, "wb") as f:
            f.write(await file.read())

        try:
            _convert_to_wav(raw_path, wav_path)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=f"Could not process audio: {e}")

        try:
            acoustic = extract_acoustic_features(wav_path)
            speech = extract_speech_rate_features(wav_path, reference_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio feature extraction failed: {e}")

    # extract_acoustic_features() returns mfcc_mean as a 13-element list (one
    # per MFCC coefficient, already time-averaged). The trained audio_model.pkl
    # was fit on a SINGLE scalar mfcc_mean column (confirmed against
    # datasets/corrected_audio_features.csv) — so we take the grand mean
    # across the 13 coefficients here. This isn't a shortcut we invented for
    # the browser: the schema and the trained model only ever had room for
    # one number, so this collapse has to happen somewhere regardless of
    # whether the audio comes from a browser or the original desktop script.
    mfcc_scalar = float(np.mean(acoustic["mfcc_mean"]))

    return AudioData(
        mfcc_mean=mfcc_scalar,
        jitter=float(acoustic["jitter_local"]),
        shimmer=float(acoustic["shimmer_local"]),
        Pauses_Per_Minute=float(speech["Pauses_Per_Minute"]),
        Phoneme_Confusion_Rate=float(speech["Phoneme_Confusion_Rate"]),
        Word_Articulation_Time=float(speech["Word_Articulation_Time"]),
    )

# ==========================================
# WebSocket for Real-Time Streaming
# ==========================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming assessments
    
    Client sends: {"eye": {...}, "typing": {...}, "audio": {...}}
    Server responds: {"risk_score": 0.65, "risk_level": "MEDIUM", ...}
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Extract modalities
            eye_data = data.get("eye", {})
            typing_data = data.get("typing", {})
            audio_data = data.get("audio", {})
            
            # Perform assessment
            if state.predictor:
                result = state.predictor.predict(eye_data, typing_data, audio_data)
                risk_score = result['risk_score']
            else:
                risk_score = _heuristic_assessment(eye_data, typing_data, audio_data)
            
            # Determine level
            if risk_score < 0.35:
                level = "LOW"
                icon = "🟢"
            elif risk_score < 0.75:
                level = "MEDIUM"
                icon = "🟡"
            else:
                level = "HIGH"
                icon = "🔴"
            
            # Send response
            response = {
                "risk_score": float(risk_score),
                "risk_level": level,
                "risk_icon": icon,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from WebSocket")

# ==========================================
# Main Entry Point
# ==========================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   Dyslexia Detection API - FastAPI Backend Server    ║
    ╚═══════════════════════════════════════════════════════╝
    
    🌐 API Documentation: http://localhost:8000/docs
    🔗 Health Check: http://localhost:8000/health
    🔌 WebSocket: ws://localhost:8000/ws/stream
    
    📡 Endpoints:
       POST /assess          - Perform risk assessment
       GET  /session/{id}    - Get session info
       GET  /models/info     - Model status
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
