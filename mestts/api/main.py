from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import sys
import sqlite3
import json
from typing import Optional
from pathlib import Path

app = FastAPI(title="MESTTS Desktop Launcher API")

# Setup CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to keep track of the running process
current_process: Optional[subprocess.Popen] = None

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "sessions.db"
EXPORTS_DIR = ROOT_DIR / "data" / "exports"


@app.get("/health")
async def health():
    """Lightweight liveness check used by the frontend's isApiOnline badge."""
    return {"status": "ok"}


@app.post("/launch-assessment")
async def launch_assessment(participant_id: str = Query(...)):
    """Launches the PyQT5 session GUI as a background process."""
    global current_process
    
    # Check if process is already running
    if current_process is not None:
        if current_process.poll() is None:
            raise HTTPException(status_code=400, detail="A session is already running.")
    
    script_path = ROOT_DIR / "scripts" / "run_session.py"
    
    if not script_path.exists():
        raise HTTPException(status_code=500, detail="run_session.py not found.")
        
    try:
        # Launch the GUI script as a detached subprocess
        current_process = subprocess.Popen(
            [sys.executable, str(script_path), "--participant", participant_id],
            cwd=str(ROOT_DIR)
        )
        return {"status": "launched", "participant_id": participant_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch: {str(e)}")


@app.get("/assessment-status")
async def assessment_status():
    """Polls whether the GUI process has closed."""
    global current_process
    if current_process is None:
        return {"status": "closed"}
        
    retcode = current_process.poll()
    if retcode is None:
        return {"status": "running"}
    else:
        return {"status": "closed", "exit_code": retcode}


@app.get("/latest-result")
async def latest_result():
    """Fetches the exported JSON of the latest session."""
    global current_process
    if current_process is not None and current_process.poll() is None:
        raise HTTPException(status_code=400, detail="Session still running.")
        
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Database not found. No sessions run yet.")
        
    try:
        # 1. Find the latest session_id from the database
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute("SELECT session_id FROM sessions ORDER BY start_epoch_ts DESC LIMIT 1")
        session = cur.fetchone()
        conn.close()
        
        if not session:
            raise HTTPException(status_code=404, detail="No sessions found in database.")
            
        session_id = session["session_id"]
        
        # 2. Look for the corresponding export JSON file
        export_file = EXPORTS_DIR / f"{session_id}_export.json"
        
        if not export_file.exists():
            # Returns 404, which the frontend interprets as "waiting_for_export"
            raise HTTPException(status_code=404, detail="Result not exported yet.")
            
        # 3. Read and return the JSON payload
        with open(export_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {str(e)}")

# This block allows you to run the server directly via `python main.py` 
# instead of having to use the `uvicorn main:app` command!
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
