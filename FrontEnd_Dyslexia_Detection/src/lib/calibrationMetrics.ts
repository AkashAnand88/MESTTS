// src/lib/calibrationMetrics.ts
//
// Direct port of GazeTracker (main_gui.py) — the real 3-point left/center/right
// calibration math, not just its visual dots. Separate from eyeTrackingMetrics.ts
// (which handles the actual fixation/saccade/blink feature extraction) since
// calibration here is a pre-capture quality gate: confirms the camera can see
// a stable, well-lit, correctly-distanced face before the real capture starts.
// This does NOT feed into the MESTTS per-word gaze-to-screen mapping — that
// logic (word_boundaries, get_word_index) belongs to the separate MESTTS
// synchronized scorer, intentionally not ported here.

import type { Landmark } from "./eyeTrackingMetrics";

const LEFT_IRIS = 468;
const RIGHT_IRIS = 473;
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_OUTER = 362;
const NOSE_TIP = 1;

export const MIN_SAMPLES_PER_POINT = 20;
export const MIN_CALIBRATION_SPAN = 0.05;
export const CALIB_POINT_DURATION_SEC = 3.5;

/** Same formula as GazeTracker.estimate_normalized_gaze() — head-pose-invariant
 * horizontal gaze estimate: iris position relative to eye center, scaled by
 * inter-eye distance so it stays roughly constant regardless of distance to camera. */
export function estimateNormalizedGaze(landmarks: Landmark[]): number | null {
  try {
    const li = landmarks[LEFT_IRIS];
    const ri = landmarks[RIGHT_IRIS];
    const irisX = (li.x + ri.x) / 2.0;

    const lo = landmarks[LEFT_EYE_OUTER];
    const ro = landmarks[RIGHT_EYE_OUTER];
    const ecX = (lo.x + ro.x) / 2.0;
    const ied = Math.abs(ro.x - lo.x);

    if (ied < 0.05 || ied > 0.4) return null; // too far / too close

    const ng = (irisX - ecX) / ied;
    if (Math.abs(ng) > 2.0) return null; // implausible, likely a bad detection

    return ng;
  } catch {
    return null;
  }
}

/** Interquartile-range-filtered median, same as GazeTracker's rmed() helper. */
function robustMedian(samples: number[]): number {
  const arr = [...samples].sort((a, b) => a - b);
  const q1 = percentile(arr, 25);
  const q3 = percentile(arr, 75);
  const iqr = q3 - q1;
  const filtered = arr.filter((v) => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
  return median(filtered.length > 0 ? filtered : arr);
}

function percentile(sortedArr: number[], p: number): number {
  const idx = (p / 100) * (sortedArr.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedArr[lo];
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo);
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Least-squares linear fit — equivalent to np.polyfit(x, y, deg=1) for 3 points. */
function linearFit(x: number[], y: number[]): { slope: number; intercept: number } {
  const n = x.length;
  const xMean = x.reduce((a, b) => a + b, 0) / n;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const num = x.reduce((acc, xi, i) => acc + (xi - xMean) * (y[i] - yMean), 0);
  const den = x.reduce((acc, xi) => acc + (xi - xMean) ** 2, 0);
  const slope = den !== 0 ? num / den : 0;
  const intercept = yMean - slope * xMean;
  return { slope, intercept };
}

export interface CalibrationResult {
  isCalibrated: boolean;
  calibrationSpan: number;
  calibrationQuality: number; // 0-1, same weighting as GazeTracker (0.6 span + 0.4 balance)
  gazeToScreenSlope: number | null;
  gazeToScreenIntercept: number | null;
  reason?: "insufficient_samples" | "low_span";
}

/** Direct port of GazeTracker.finalize_calibration(). */
export function finalizeCalibration(samples: {
  left: number[];
  center: number[];
  right: number[];
}): CalibrationResult {
  const counts = {
    left: samples.left.length,
    center: samples.center.length,
    right: samples.right.length,
  };
  if (counts.left < MIN_SAMPLES_PER_POINT || counts.center < MIN_SAMPLES_PER_POINT || counts.right < MIN_SAMPLES_PER_POINT) {
    return {
      isCalibrated: false,
      calibrationSpan: 0,
      calibrationQuality: 0,
      gazeToScreenSlope: null,
      gazeToScreenIntercept: null,
      reason: "insufficient_samples",
    };
  }

  const lg = robustMedian(samples.left);
  const cg = robustMedian(samples.center);
  const rg = robustMedian(samples.right);

  const calibrationSpan = Math.abs(rg - lg);
  const spanScore = Math.min(1.0, calibrationSpan / 0.4);
  const midpoint = (lg + rg) / 2;
  const offset = Math.abs(cg - midpoint);
  const balanceScore = calibrationSpan > 0 ? Math.max(0.0, 1.0 - offset / (calibrationSpan * 0.5)) : 0.0;
  const calibrationQuality = spanScore * 0.6 + balanceScore * 0.4;

  const { slope, intercept } = linearFit([lg, cg, rg], [-1.0, 0.0, 1.0]);

  return {
    isCalibrated: true,
    calibrationSpan,
    calibrationQuality,
    gazeToScreenSlope: slope,
    gazeToScreenIntercept: intercept,
    reason: calibrationSpan < MIN_CALIBRATION_SPAN ? "low_span" : undefined,
  };
}
