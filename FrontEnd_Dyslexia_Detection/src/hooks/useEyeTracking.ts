// src/hooks/useEyeTracking.ts
//
// Owns: camera permission request, MediaPipe FaceLandmarker init, the
// requestAnimationFrame capture loop, and buffer accumulation. Produces an
// EyeTrackingFeatures object matching the exact /assess schema shape.

import { useCallback, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import {
  processFrame,
  computeAdvancedMetrics,
  BLINK_EAR_THRESHOLD,
  type EyeTrackingFeatures,
  type GazeSample,
} from "@/lib/eyeTrackingMetrics";
import { WASM_BASE, FACE_LANDMARKER_MODEL_URL as MODEL_URL } from "@/lib/mediapipeConfig";

export type EyeTrackingStatus =
  | "idle"
  | "requesting_permission"
  | "loading_model"
  | "tracking"
  | "stopped"
  | "error";

export type EyeTrackingErrorReason =
  | "unsupported_browser"
  | "permission_denied"
  | "no_camera"
  | "model_load_failed"
  | "no_face_detected";


export function useEyeTracking() {
  const [status, setStatus] = useState<EyeTrackingStatus>("idle");
  const [errorReason, setErrorReason] = useState<EyeTrackingErrorReason | null>(null);
  const [liveStats, setLiveStats] = useState({ blinkCount: 0, faceDetected: false, elapsedSec: 0 });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);

  const gazePoints = useRef<GazeSample[]>([]);
  const pupilDiametersPx = useRef<number[]>([]);
  const blinkCount = useRef(0);
  const lastBlinkStatus = useRef<"Open" | "Closed">("Open");
  const frameCount = useRef(0);
  const anyFaceEverDetected = useRef(false);
  const startTimeRef = useRef<number | null>(null);

  const reset = useCallback(() => {
    gazePoints.current = [];
    pupilDiametersPx.current = [];
    blinkCount.current = 0;
    lastBlinkStatus.current = "Open";
    frameCount.current = 0;
    anyFaceEverDetected.current = false;
    startTimeRef.current = null;
    setLiveStats({ blinkCount: 0, faceDetected: false, elapsedSec: 0 });
  }, []);

  const stopStream = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const loop = useCallback(() => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }

    const now = performance.now();
    const result = landmarker.detectForVideo(video, now);
    const landmarks = result.faceLandmarks?.[0] ?? null;

    const t = (now - (startTimeRef.current ?? now)) / 1000;
    const frame = processFrame(landmarks, video.videoWidth, video.videoHeight, t);
    frameCount.current += 1;

    if (frame.faceDetected) {
      anyFaceEverDetected.current = true;
      if (frame.gaze) gazePoints.current.push(frame.gaze);
      if (frame.irisDiameterPx !== null) pupilDiametersPx.current.push(frame.irisDiameterPx);

      if (frame.ear !== null) {
        if (frame.ear < BLINK_EAR_THRESHOLD) {
          if (lastBlinkStatus.current === "Open") blinkCount.current += 1;
          lastBlinkStatus.current = "Closed";
        } else {
          lastBlinkStatus.current = "Open";
        }
      }
    }

    // Throttle React state updates to ~4/sec so the UI doesn't re-render every frame
    if (frameCount.current % 15 === 0) {
      setLiveStats({
        blinkCount: blinkCount.current,
        faceDetected: frame.faceDetected,
        elapsedSec: t,
      });
    }

    rafRef.current = requestAnimationFrame(loop);
  }, []);

  const start = useCallback(async () => {
    setErrorReason(null);
    reset();

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
      // NotAllowedError = user denied; NotFoundError = no camera device
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
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
        });
      }
    } catch (err) {
      console.error("FaceLandmarker load failed:", err);
      setStatus("error");
      setErrorReason("model_load_failed");
      stopStream();
      return;
    }

    startTimeRef.current = performance.now();
    setStatus("tracking");
    rafRef.current = requestAnimationFrame(loop);
  }, [loop, reset, stopStream]);

  const stop = useCallback((): EyeTrackingFeatures | null => {
    const durationSec = startTimeRef.current !== null ? (performance.now() - startTimeRef.current) / 1000 : 0;
    stopStream();
    setStatus("stopped");

    if (!anyFaceEverDetected.current) {
      setErrorReason("no_face_detected");
      return null;
    }

    const samplingRateHz = durationSec > 0 ? frameCount.current / durationSec : 0;
    return computeAdvancedMetrics(
      gazePoints.current,
      blinkCount.current,
      pupilDiametersPx.current,
      durationSec,
      samplingRateHz
    );
  }, [stopStream]);

  return { videoRef, status, errorReason, liveStats, start, stop };
}
