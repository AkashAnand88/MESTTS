// src/lib/eyeTrackingMetrics.ts
//
// Direct port of Backend_Dyslexia_Detection/multimodel_pipeline/video_module.py.
// IMPORTANT: this is NOT calibrated eye-tracking. The original Python module
// itself uses a fixed heuristic constant (PX_PER_DEGREE = 18.0) to convert
// iris pixel movement into a fake "deg/s" and "mm" — it does not use real
// camera calibration or a physical distance measurement. `gaze_path_entropy`
// is hardcoded to 4.1 in the original too. We replicate ALL of this exactly,
// including the fakeness, because the fusion model was trained on features
// produced by this exact heuristic. Doing "better" physics here would move
// the feature distribution away from what the model has ever seen.

export const LEFT_EYE = [33, 160, 158, 133, 153, 144];
export const RIGHT_EYE = [362, 385, 387, 263, 373, 374];
export const LEFT_IRIS_CENTER = 468;
export const RIGHT_IRIS_CENTER = 473;
export const IRIS_DIAMETER_POINTS: [number, number] = [474, 476]; // same pair video_module.py uses
export const PX_PER_DEGREE = 18.0;
export const SACCADE_THRESH_DEG_S = 6.0;
export const BLINK_EAR_THRESHOLD = 0.2;
export const GAZE_PATH_ENTROPY_PLACEHOLDER = 4.1; // hardcoded in the original backend too

export interface Landmark {
  x: number;
  y: number;
  z?: number;
}

function dist(a: [number, number], b: [number, number]) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** Same formula as video_module.py's calculate_ear() */
export function calculateEAR(landmarks: Landmark[], indices: number[]): number {
  const coords = indices.map((i) => [landmarks[i].x, landmarks[i].y] as [number, number]);
  const v1 = dist(coords[1], coords[5]);
  const v2 = dist(coords[2], coords[4]);
  const h = dist(coords[0], coords[3]);
  return h > 0 ? (v1 + v2) / (2.0 * h) : 0;
}

/** Same formula as video_module.py's calculate_iris_diameter() */
export function calculateIrisDiameter(landmarks: Landmark[], imgW: number, imgH: number): number {
  const [i1, i2] = IRIS_DIAMETER_POINTS;
  const p1: [number, number] = [landmarks[i1].x * imgW, landmarks[i1].y * imgH];
  const p2: [number, number] = [landmarks[i2].x * imgW, landmarks[i2].y * imgH];
  return dist(p1, p2);
}

export interface GazeSample {
  x: number;
  y: number;
  t: number; // seconds
}

export interface FrameResult {
  gaze: GazeSample | null;
  ear: number | null;
  irisDiameterPx: number | null;
  faceDetected: boolean;
}

/** Called once per video frame, mirrors video_module.py's process_frame() */
export function processFrame(landmarks: Landmark[] | null, imgW: number, imgH: number, timestamp: number): FrameResult {
  if (!landmarks) {
    return { gaze: null, ear: null, irisDiameterPx: null, faceDetected: false };
  }
  const rx = landmarks[RIGHT_IRIS_CENTER].x * imgW;
  const ry = landmarks[RIGHT_IRIS_CENTER].y * imgH;
  const lx = landmarks[LEFT_IRIS_CENTER].x * imgW;
  const ly = landmarks[LEFT_IRIS_CENTER].y * imgH;
  const avgX = (rx + lx) / 2;
  const avgY = (ry + ly) / 2;

  const lEar = calculateEAR(landmarks, LEFT_EYE);
  const rEar = calculateEAR(landmarks, RIGHT_EYE);
  const avgEar = (lEar + rEar) / 2.0;

  const irisDiameterPx = calculateIrisDiameter(landmarks, imgW, imgH);

  return {
    gaze: { x: avgX, y: avgY, t: timestamp },
    ear: avgEar,
    irisDiameterPx,
    faceDetected: true,
  };
}

export interface EyeTrackingFeatures {
  sampling_rate_hz: number;
  duration_sec: number;
  fixation_count: number;
  avg_fixation_duration_ms: number;
  first_pass_fixation_duration_ms: number;
  avg_saccade_length_px: number;
  mean_saccade_velocity_deg_s: number;
  regressions_per_sentence: number;
  word_skips_per_sentence: number;
  microsaccade_rate_per_sec: number;
  blink_rate_per_min: number;
  pupil_diameter_mm: number;
  gaze_path_entropy: number;
}

/**
 * Direct port of video_module.py's get_advanced_metrics().
 * gazePoints must be in chronological order, t in seconds.
 */
export function computeAdvancedMetrics(
  gazePoints: GazeSample[],
  blinkCount: number,
  pupilDiametersPx: number[],
  durationSec: number,
  samplingRateHz: number
): EyeTrackingFeatures {
  if (gazePoints.length < 2) {
    return {
      sampling_rate_hz: samplingRateHz,
      duration_sec: durationSec,
      fixation_count: 0,
      avg_fixation_duration_ms: 0,
      first_pass_fixation_duration_ms: 0,
      avg_saccade_length_px: 0,
      mean_saccade_velocity_deg_s: 0,
      regressions_per_sentence: 0,
      word_skips_per_sentence: 0,
      microsaccade_rate_per_sec: 0,
      blink_rate_per_min: durationSec > 0 ? (blinkCount / durationSec) * 60 : 0,
      pupil_diameter_mm: 0,
      gaze_path_entropy: GAZE_PATH_ENTROPY_PLACEHOLDER,
    };
  }

  const diffDist: number[] = [];
  const diffTime: number[] = [];
  for (let i = 1; i < gazePoints.length; i++) {
    const dt = gazePoints[i].t - gazePoints[i - 1].t;
    if (dt <= 0) continue; // same "valid = diff_time > 0" filter as Python
    diffDist.push(dist([gazePoints[i].x, gazePoints[i].y], [gazePoints[i - 1].x, gazePoints[i - 1].y]));
    diffTime.push(dt);
  }

  if (diffTime.length === 0) {
    return {
      sampling_rate_hz: samplingRateHz,
      duration_sec: durationSec,
      fixation_count: 0,
      avg_fixation_duration_ms: 0,
      first_pass_fixation_duration_ms: 0,
      avg_saccade_length_px: 0,
      mean_saccade_velocity_deg_s: 0,
      regressions_per_sentence: 0,
      word_skips_per_sentence: 0,
      microsaccade_rate_per_sec: 0,
      blink_rate_per_min: durationSec > 0 ? (blinkCount / durationSec) * 60 : 0,
      pupil_diameter_mm: 0,
      gaze_path_entropy: GAZE_PATH_ENTROPY_PLACEHOLDER,
    };
  }

  const velocitiesDegS = diffDist.map((d, i) => d / diffTime[i] / PX_PER_DEGREE);

  const saccadeIdx = velocitiesDegS.reduce<number[]>((acc, v, i) => {
    if (v > SACCADE_THRESH_DEG_S) acc.push(i);
    return acc;
  }, []);

  const avgSaccadeLen = saccadeIdx.length > 0 ? mean(saccadeIdx.map((i) => diffDist[i])) : 0;
  const meanSaccadeVel = saccadeIdx.length > 0 ? mean(saccadeIdx.map((i) => velocitiesDegS[i])) : 0;

  // Fixation segmentation: consecutive sub-threshold samples accumulate duration
  const fixationEvents: number[] = [];
  let currentDur = 0;
  for (let i = 0; i < velocitiesDegS.length; i++) {
    if (velocitiesDegS[i] <= SACCADE_THRESH_DEG_S) {
      currentDur += diffTime[i];
    } else {
      if (currentDur > 0) fixationEvents.push(currentDur);
      currentDur = 0;
    }
  }
  if (currentDur > 0) fixationEvents.push(currentDur);

  const avgFixDurMs = fixationEvents.length > 0 ? mean(fixationEvents) * 1000 : 0;
  const firstPassDurMs = fixationEvents.length > 0 ? fixationEvents[0] * 1000 : 0;

  const pupilDiameterMm =
    pupilDiametersPx.length > 0 ? (mean(pupilDiametersPx) / PX_PER_DEGREE) * 4 : 0; // same odd conversion as Python

  return {
    sampling_rate_hz: samplingRateHz,
    duration_sec: durationSec,
    fixation_count: fixationEvents.length,
    avg_fixation_duration_ms: avgFixDurMs,
    first_pass_fixation_duration_ms: firstPassDurMs,
    avg_saccade_length_px: avgSaccadeLen,
    mean_saccade_velocity_deg_s: meanSaccadeVel,
    regressions_per_sentence: 0, // backend hardcodes this too — not worth building
    word_skips_per_sentence: 0,
    microsaccade_rate_per_sec: 0,
    blink_rate_per_min: durationSec > 0 ? (blinkCount / durationSec) * 60 : 0,
    pupil_diameter_mm: pupilDiameterMm,
    gaze_path_entropy: GAZE_PATH_ENTROPY_PLACEHOLDER,
  };
}

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
