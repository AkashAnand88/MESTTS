// src/hooks/useTypingCapture.ts
//
// Replaces the "Typing Error Rate" slider with real keystroke capture.
// Mirrors Backend_MESTTS_Detection/multimodel_pipeline/typing_module.py
// (TypingFeatureExtractor.on_press / on_release / analyze_errors) so the
// computed fields match what the fusion model was trained against —
// same mirror-letter map, same transposition test (equal-length, same
// multiset of chars, different order), same Levenshtein-ratio error rate.

import { useCallback, useRef, useState } from "react";
import { SequenceMatcher } from "@/lib/sequenceMatcher";

export interface TypingData {
  Simulation_Error_Rate: number;
  Words_Modified_Count: number;
  Transposition_Count: number;
  Mirror_Error_Count: number;
  Phonetic_Error_Count: number;
  Modification_Proportion: number;
}

// Debug/telemetry only — not part of the /assess schema, but useful to show
// the user real numbers are being captured (not a hardcoded value).
export interface TypingDynamics {
  avgHoldTimeMs: number;
  avgFlightTimeMs: number;
  typingSpeedCpm: number;
  backspaceCount: number;
  totalKeystrokes: number;
}

const MIRRORS: Record<string, string> = {
  b: "d",
  d: "b",
  p: "q",
  q: "p",
  m: "w",
  w: "m",
};

function analyzeErrors(reference: string, typed: string) {
  const matcher = new SequenceMatcher(reference, typed);

  let transpositions = 0;
  let mirrorErrors = 0;

  for (const op of matcher.getOpcodes()) {
    if (op.tag !== "replace") continue;
    const s1 = reference.slice(op.i1, op.i2);
    const s2 = typed.slice(op.j1, op.j2);

    // Transposition: same letters, same length, different order (e.g. "teh" vs "the")
    if (s1.length === s2.length && s1.length > 0) {
      const sorted1 = s1.split("").sort().join("");
      const sorted2 = s2.split("").sort().join("");
      if (sorted1 === sorted2 && s1 !== s2) transpositions++;
    }

    // Mirror-letter confusion: single-char replace matching b/d, p/q, m/w
    if (s1.length === 1 && s2.length === 1 && MIRRORS[s1] === s2) {
      mirrorErrors++;
    }
  }

  const simError = 1.0 - matcher.ratio();
  const refWords = reference.trim().split(/\s+/).filter(Boolean);
  const typedWords = typed.trim().split(/\s+/).filter(Boolean);
  const wordsModified = Math.abs(refWords.length - typedWords.length);

  return {
    Simulation_Error_Rate: simError,
    Words_Modified_Count: wordsModified,
    Transposition_Count: transpositions,
    Mirror_Error_Count: mirrorErrors,
    // typing_module.py never populates a phonetic error count on this path
    // (only the separate audio/whisper path does) — kept at 0, not fabricated.
    Phonetic_Error_Count: 0,
    Modification_Proportion: simError,
  };
}

export function useTypingCapture(referenceText: string) {
  const [typedText, setTypedText] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [dynamics, setDynamics] = useState<TypingDynamics>({
    avgHoldTimeMs: 0,
    avgFlightTimeMs: 0,
    typingSpeedCpm: 0,
    backspaceCount: 0,
    totalKeystrokes: 0,
  });

  const keyDownTimes = useRef<Map<string, number>>(new Map());
  const holdTimes = useRef<number[]>([]);
  const flightTimes = useRef<number[]>([]);
  const lastReleaseTime = useRef<number | null>(null);
  const backspaceCount = useRef(0);
  const totalKeystrokes = useRef(0);
  const startTime = useRef<number | null>(null);

  const reset = useCallback(() => {
    setTypedText("");
    setIsComplete(false);
    keyDownTimes.current.clear();
    holdTimes.current = [];
    flightTimes.current = [];
    lastReleaseTime.current = null;
    backspaceCount.current = 0;
    totalKeystrokes.current = 0;
    startTime.current = null;
    setDynamics({
      avgHoldTimeMs: 0,
      avgFlightTimeMs: 0,
      typingSpeedCpm: 0,
      backspaceCount: 0,
      totalKeystrokes: 0,
    });
  }, []);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      const now = performance.now();
      if (startTime.current === null) startTime.current = now;

      const key = e.key;
      keyDownTimes.current.set(key, now);
      totalKeystrokes.current += 1;
      if (key === "Backspace" || key === "Delete") backspaceCount.current += 1;

      if (lastReleaseTime.current !== null) {
        const flight = now - lastReleaseTime.current;
        if (flight < 2000) flightTimes.current.push(flight); // same 2s cap as typing_module.py
      }
    },
    [],
  );

  const onKeyUp = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const now = performance.now();
    lastReleaseTime.current = now;
    const key = e.key;
    const down = keyDownTimes.current.get(key);
    if (down !== undefined) {
      holdTimes.current.push(now - down);
    }
  }, []);

  const onChange = useCallback((value: string) => {
    setTypedText(value);
  }, []);

  const finish = useCallback((): TypingData => {
    const avgHold =
      holdTimes.current.length > 0
        ? holdTimes.current.reduce((a, b) => a + b, 0) /
          holdTimes.current.length
        : 0;
    const avgFlight =
      flightTimes.current.length > 0
        ? flightTimes.current.reduce((a, b) => a + b, 0) /
          flightTimes.current.length
        : 0;
    const totalFlightSec =
      flightTimes.current.reduce((a, b) => a + b, 0) / 1000;
    const cpm =
      totalFlightSec > 0 ? (typedText.length / (totalFlightSec + 1)) * 60 : 0;

    setDynamics({
      avgHoldTimeMs: avgHold,
      avgFlightTimeMs: avgFlight,
      typingSpeedCpm: cpm,
      backspaceCount: backspaceCount.current,
      totalKeystrokes: totalKeystrokes.current,
    });
    setIsComplete(true);

    return analyzeErrors(referenceText, typedText);
  }, [referenceText, typedText]);

  return {
    typedText,
    onChange,
    onKeyDown,
    onKeyUp,
    finish,
    reset,
    isComplete,
    dynamics,
  };
}
