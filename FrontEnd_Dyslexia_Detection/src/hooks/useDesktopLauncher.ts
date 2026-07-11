// src/hooks/useDesktopLauncher.ts
//
// Replaces useTypingCapture / useEyeTracking / useAudioCapture / useCalibration
// as the primary assessment flow. The web app's job is now just: launch
// main_gui.py, poll until the window closes, then fetch whatever it exported.
// The actual calibration/sentence/capture/report UI happens in that native
// window — not in the browser.

import { useCallback, useRef, useState } from "react";
import { API_BASE } from "@/hooks/useDyslexiaAssessment";

export type LaunchStatus =
  | "idle"
  | "launching"
  | "running"
  | "waiting_for_export"
  | "fetching_result"
  | "done"
  | "error";

export interface SentenceDetail {
  sentence_index: number;
  final_score: number;
  risk_label: string;
  eye_conf: number;
  type_conf: number;
  audio_conf: number;
}

export interface DesktopAssessmentResult {
  session_id: string;
  participant_id: string;
  timestamp: string;
  risk_band: string; // "LOW RISK" | "MEDIUM RISK" | "HIGH RISK"
  avg_fusion_score: number;
  avg_eye_conf: number;
  avg_type_conf: number;
  avg_audio_conf: number;
  per_sentence: SentenceDetail[];
}

const POLL_INTERVAL_MS = 2000;

export function useDesktopLauncher() {
  const [status, setStatus] = useState<LaunchStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const fetchResult = useCallback(async (): Promise<DesktopAssessmentResult | null> => {
    setStatus("fetching_result");
    try {
      const res = await fetch(`${API_BASE}/latest-result`);
      if (res.status === 404) {
        // The window closed but the user never clicked "Export" inside it
        setStatus("waiting_for_export");
        return null;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setStatus("error");
        setErrorMessage(body.detail || `Server error: ${res.status}`);
        return null;
      }
      const data: DesktopAssessmentResult = await res.json();
      setStatus("done");
      return data;
    } catch (err) {
      console.error("Fetch result error:", err);
      setStatus("error");
      setErrorMessage("Couldn't reach the backend to fetch the result.");
      return null;
    }
  }, []);

  const launch = useCallback(
    (participantId: string, onClosed: (result: DesktopAssessmentResult | null) => void) => {
      setErrorMessage(null);
      setStatus("launching");

      (async () => {
        try {
          const res = await fetch(
            `${API_BASE}/launch-assessment?participant_id=${encodeURIComponent(participantId)}`,
            { method: "POST" }
          );
          if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            setStatus("error");
            setErrorMessage(body.detail || `Failed to launch: ${res.status}`);
            return;
          }
        } catch (err) {
          console.error("Launch error:", err);
          setStatus("error");
          setErrorMessage("Couldn't reach the backend to launch the assessment.");
          return;
        }

        setStatus("running");
        pollRef.current = window.setInterval(async () => {
          try {
            const res = await fetch(`${API_BASE}/assessment-status`);
            const data = await res.json();
            if (data.status === "closed") {
              stopPolling();
              const result = await fetchResult();
              onClosed(result);
            }
          } catch (err) {
            console.error("Poll error:", err);
            // transient network hiccup — keep polling rather than erroring out
          }
        }, POLL_INTERVAL_MS);
      })();
    },
    [fetchResult, stopPolling]
  );

  const retryFetchResult = useCallback(
    async (onDone: (result: DesktopAssessmentResult | null) => void) => {
      const result = await fetchResult();
      onDone(result);
    },
    [fetchResult]
  );

  const reset = useCallback(() => {
    stopPolling();
    setStatus("idle");
    setErrorMessage(null);
  }, [stopPolling]);

  return { status, errorMessage, launch, retryFetchResult, reset };
}
