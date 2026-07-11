// src/components/CalibrationScreen.tsx
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertCircle, CheckCircle2, RotateCcw } from "lucide-react";
import type {
  CalibrationStage,
  CalibrationStatus,
  CalibrationErrorReason,
} from "@/hooks/useCalibration";
import type { CalibrationResult } from "@/lib/calibrationMetrics";

interface CalibrationDotProps {
  active: boolean;
  progress: number; // 0-100
}

/** Visual port of CalibrationDot (QLabel/paintEvent) from main_gui.py:
 * inactive = faint grey circle, active = white ring + red center + green
 * progress arc that fills as samples are collected for that point. */
function CalibrationDot({ active, progress }: CalibrationDotProps) {
  const circumference = 2 * Math.PI * 30;
  const dashOffset = circumference * (1 - progress / 100);
  return (
    <div className="relative w-[70px] h-[70px] flex items-center justify-center">
      {active ? (
        <>
          <svg className="absolute inset-0 -rotate-90" viewBox="0 0 70 70">
            <circle cx="35" cy="35" r="30" fill="none" stroke="#2ecc71" strokeWidth="5"
              strokeDasharray={circumference} strokeDashoffset={dashOffset}
              strokeLinecap="round" style={{ transition: "stroke-dashoffset 0.1s linear" }} />
          </svg>
          <div className="w-9 h-9 rounded-full bg-white flex items-center justify-center border-[3px] border-white">
            <div className="w-5 h-5 rounded-full bg-[#e74c3c] flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-white" />
            </div>
          </div>
        </>
      ) : (
        <div className="w-6 h-6 rounded-full bg-[#bdc3c7]/40 border-2 border-[#95a5a6]" />
      )}
    </div>
  );
}

interface CalibrationScreenProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  status: CalibrationStatus;
  errorReason: CalibrationErrorReason | null;
  currentStage: CalibrationStage;
  stageProgress: number;
  result: CalibrationResult | null;
  onStart: () => void;
  onRetry: () => void;
  onContinue: () => void;
}

const errorMessages: Record<CalibrationErrorReason, string> = {
  unsupported_browser: "Your browser doesn't support camera access. Try Chrome, Edge, or Firefox.",
  permission_denied: "Camera permission denied. Enable it in your browser's site settings to continue.",
  no_camera: "No camera device found. Connect a webcam and try again.",
  model_load_failed: "Couldn't load the face-tracking model — check your connection and try again.",
};

export default function CalibrationScreen({
  videoRef,
  status,
  errorReason,
  currentStage,
  stageProgress,
  result,
  onStart,
  onRetry,
  onContinue,
}: CalibrationScreenProps) {
  const stagePositions: Record<CalibrationStage, string> = {
    left: "justify-start",
    center: "justify-center",
    right: "justify-end",
  };

  return (
    <Card className="max-w-xl mx-auto">
      <CardHeader>
        <CardTitle>Gaze Calibration</CardTitle>
        <CardDescription>
          {status === "calibrating"
            ? `Look at the ${currentStage} dot until it fills in`
            : "A quick 3-point check (left, center, right) before eye tracking starts — same as pressing 0 in the original desktop app."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative w-full aspect-video bg-black/90 rounded-md overflow-hidden">
          <video ref={videoRef} className="w-full h-full object-cover -scale-x-100" muted playsInline />
          {status === "calibrating" && (
            <div className={`absolute inset-x-6 top-1/2 -translate-y-1/2 flex ${stagePositions[currentStage]}`}>
              <CalibrationDot active progress={stageProgress} />
            </div>
          )}
        </div>

        {status === "error" && errorReason && (
          <div className="p-2 bg-destructive/10 text-destructive text-xs rounded-md flex items-start gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <p>{errorMessages[errorReason]}</p>
          </div>
        )}

        {status === "done" && result && (
          <div className="space-y-2">
            <div className={`p-3 rounded-md text-sm flex items-start gap-2 ${
              result.isCalibrated ? "bg-green-500/10 text-green-700" : "bg-destructive/10 text-destructive"
            }`}>
              {result.isCalibrated ? <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" /> : <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}
              <div>
                {result.isCalibrated ? (
                  <>
                    Calibration quality: <strong>{(result.calibrationQuality * 100).toFixed(0)}%</strong>
                    {result.reason === "low_span" && (
                      <p className="text-xs mt-1">Low movement range detected — try moving your eyes further left/right next time for better accuracy.</p>
                    )}
                  </>
                ) : (
                  <>Not enough samples collected — make sure your face is well-lit and centered, then retry.</>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          {status === "idle" && (
            <Button className="w-full" onClick={onStart}>Start calibration</Button>
          )}
          {(status === "requesting_permission" || status === "loading_model") && (
            <Button className="w-full" disabled>
              {status === "requesting_permission" ? "Requesting camera permission..." : "Loading face-tracking model..."}
            </Button>
          )}
          {status === "error" && (
            <Button className="w-full" variant="outline" onClick={onRetry}>
              <RotateCcw className="w-4 h-4 mr-2" /> Try again
            </Button>
          )}
          {status === "done" && (
            <>
              <Button variant="outline" className="flex-1" onClick={onRetry}>
                <RotateCcw className="w-4 h-4 mr-2" /> Re-calibrate
              </Button>
              <Button className="flex-1" onClick={onContinue} disabled={!result?.isCalibrated}>
                Continue to assessment
              </Button>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
