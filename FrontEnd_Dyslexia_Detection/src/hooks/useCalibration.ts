// src/hooks/useCalibration.ts
//
// Runs the real 3-point (left/center/right) gaze calibration before capture
// starts — mirrors main_gui.py's SessionWindow calibration flow (press 0,
// look at each dot in turn, quality computed from span+balance). This is a
// pre-capture quality gate, not a change to how eye features are computed in
// Phase 2 (that still uses the same PX_PER_DEGREE heuristic as video_module.py).

import { useCallback, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { WASM_BASE, FACE_LANDMARKER_MODEL_URL } from "@/lib/mediapipeConfig";
import {
  estimateNormalizedGaze,
  finalizeCalibration,
  CALIB_POINT_DURATION_SEC,
  type CalibrationResult,
} from "@/lib/calibrationMetrics";

export type CalibrationStage = "left" | "center" | "right";
const STAGES: CalibrationStage[] = ["left", "center", "right"];

export type CalibrationStatus =
  | "idle"
  | "requesting_permission"
  | "loading_model"
  | "calibrating"
  | "done"
  | "error";

export type CalibrationErrorReason =
  | "unsupported_browser"
  | "permission_denied"
  | "no_camera"
  | "model_load_failed";

export function useCalibration() {
  const [status, setStatus] = useState<CalibrationStatus>("idle");
  const [errorReason, setErrorReason] = useState<CalibrationErrorReason | null>(null);
  const [currentStage, setCurrentStage] = useState<CalibrationStage>("left");
  const [stageProgress, setStageProgress] = useState(0); // 0-100
  const [result, setResult] = useState<CalibrationResult | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const stageTimerRef = useRef<number | null>(null);

  const samples = useRef<{ left: number[]; center: number[]; right: number[] }>({
    left: [],
    center: [],
    right: [],
  });
  const stageIndexRef = useRef(0);
  const stageStartRef = useRef(0);

  const cleanup = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    if (stageTimerRef.current !== null) window.clearInterval(stageTimerRef.current);
    rafRef.current = null;
    stageTimerRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const detectionLoop = useCallback(() => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(detectionLoop);
      return;
    }
    const detection = landmarker.detectForVideo(video, performance.now());
    const landmarks = detection.faceLandmarks?.[0];
    if (landmarks) {
      const gaze = estimateNormalizedGaze(landmarks);
      if (gaze !== null) {
        const stage = STAGES[stageIndexRef.current];
        samples.current[stage].push(gaze);
      }
    }
    rafRef.current = requestAnimationFrame(detectionLoop);
  }, []);

  const advanceStage = useCallback(() => {
    stageIndexRef.current += 1;
    if (stageIndexRef.current >= STAGES.length) {
      // All 3 stages done — compute the real result
      if (stageTimerRef.current !== null) window.clearInterval(stageTimerRef.current);
      cleanup();
      const finalResult = finalizeCalibration(samples.current);
      setResult(finalResult);
      setStatus("done");
      return;
    }
    setCurrentStage(STAGES[stageIndexRef.current]);
    setStageProgress(0);
    stageStartRef.current = performance.now();
  }, [cleanup]);

  const start = useCallback(async () => {
    setErrorReason(null);
    setResult(null);
    samples.current = { left: [], center: [], right: [] };
    stageIndexRef.current = 0;
    setCurrentStage("left");
    setStageProgress(0);

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("error");
      setErrorReason("unsupported_browser");
      return;
    }

    setStatus("requesting_permission");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
    } catch (err) {
      setStatus("error");
      const name = (err as DOMException)?.name;
      setErrorReason(name === "NotFoundError" ? "no_camera" : "permission_denied");
      return;
    }
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }

    setStatus("loading_model");
    try {
      if (!landmarkerRef.current) {
        const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE);
        landmarkerRef.current = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
        });
      }
    } catch (err) {
      console.error("FaceLandmarker load failed:", err);
      setStatus("error");
      setErrorReason("model_load_failed");
      cleanup();
      return;
    }

    setStatus("calibrating");
    stageStartRef.current = performance.now();
    rafRef.current = requestAnimationFrame(detectionLoop);
    stageTimerRef.current = window.setInterval(() => {
      const elapsed = (performance.now() - stageStartRef.current) / 1000;
      const progress = Math.min(100, (elapsed / CALIB_POINT_DURATION_SEC) * 100);
      setStageProgress(progress);
      if (elapsed >= CALIB_POINT_DURATION_SEC) advanceStage();
    }, 100);
  }, [detectionLoop, advanceStage, cleanup]);

  const retry = useCallback(() => {
    start();
  }, [start]);

  return {
    videoRef,
    status,
    errorReason,
    currentStage,
    stageProgress,
    result,
    start,
    retry,
  };
}
