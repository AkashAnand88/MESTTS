// src/hooks/useAudioCapture.ts
//
// Replaces the "Speech Pauses / Minute" slider. Records the user reading
// referenceText aloud, uploads the clip to the new /extract-audio-features
// endpoint (backend runs it through librosa/parselmouth/whisper — see
// audio_module/extract_features.py), and returns the real AudioData.
// MFCC/jitter/shimmer are NOT computed in JS — see Phase 3 notes on why.

import { useCallback, useRef, useState } from "react";
import { API_BASE } from "@/hooks/useDyslexiaAssessment";

export interface AudioData {
  mfcc_mean: number;
  jitter: number;
  shimmer: number;
  Pauses_Per_Minute: number;
  Phoneme_Confusion_Rate: number;
  Word_Articulation_Time: number;
}

export type AudioCaptureStatus =
  | "idle"
  | "requesting_permission"
  | "recording"
  | "uploading"
  | "done"
  | "error";

export type AudioCaptureErrorReason =
  | "unsupported_browser"
  | "permission_denied"
  | "no_microphone"
  | "empty_recording"
  | "upload_failed"
  | "extraction_failed";

export function useAudioCapture(referenceText: string) {
  const [status, setStatus] = useState<AudioCaptureStatus>("idle");
  const [errorReason, setErrorReason] = useState<AudioCaptureErrorReason | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    setErrorReason(null);
    chunksRef.current = [];
    setElapsedSec(0);

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setStatus("error");
      setErrorReason("unsupported_browser");
      return;
    }

    setStatus("requesting_permission");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setStatus("error");
      const name = (err as DOMException)?.name;
      setErrorReason(name === "NotFoundError" ? "no_microphone" : "permission_denied");
      return;
    }
    streamRef.current = stream;

    const recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorderRef.current = recorder;
    recorder.start();

    startTimeRef.current = performance.now();
    setStatus("recording");
    timerRef.current = window.setInterval(() => {
      setElapsedSec((performance.now() - startTimeRef.current) / 1000);
    }, 250);
  }, []);

  /** Stops recording, uploads to backend, returns real AudioData (or null on failure). */
  const stopAndUpload = useCallback(async (): Promise<AudioData | null> => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return null;

    const blob: Blob = await new Promise((resolve) => {
      recorder.onstop = () => resolve(new Blob(chunksRef.current, { type: recorder.mimeType }));
      recorder.stop();
    });
    cleanupStream();

    if (blob.size < 1000) {
      setStatus("error");
      setErrorReason("empty_recording");
      return null;
    }

    setStatus("uploading");
    try {
      const formData = new FormData();
      const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("wav") ? "wav" : "webm";
      formData.append("file", blob, `recording.${ext}`);
      formData.append("reference_text", referenceText);

      const response = await fetch(`${API_BASE}/extract-audio-features`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const isServerSide = response.status >= 500 || response.status === 422;
        setStatus("error");
        setErrorReason(isServerSide ? "extraction_failed" : "upload_failed");
        return null;
      }

      const data: AudioData = await response.json();
      setStatus("done");
      return data;
    } catch (err) {
      console.error("Audio upload error:", err);
      setStatus("error");
      setErrorReason("upload_failed");
      return null;
    }
  }, [referenceText, cleanupStream]);

  const reset = useCallback(() => {
    cleanupStream();
    chunksRef.current = [];
    setElapsedSec(0);
    setStatus("idle");
    setErrorReason(null);
  }, [cleanupStream]);

  return { status, errorReason, elapsedSec, start, stopAndUpload, reset };
}
