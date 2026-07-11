// src/components/AssessmentDashboard.tsx

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, CheckCircle2, Activity, Keyboard, Mic, MicOff, Play, Eye, Video, VideoOff } from "lucide-react";
import { Label } from "@/components/ui/label";
import type { TypingDynamics } from "@/hooks/useTypingCapture";
import type { EyeTrackingStatus, EyeTrackingErrorReason } from "@/hooks/useEyeTracking";
import type { AudioCaptureStatus, AudioCaptureErrorReason } from "@/hooks/useAudioCapture";

import type { AssessmentResult } from "@/hooks/useDyslexiaAssessment";

interface AssessmentDashboardProps {
  isApiOnline: boolean;
  isLoading: boolean;
  error: string | null;
  result: AssessmentResult | null;
  // --- Real eye tracking (replaces the old fixationCount slider) ---
  videoRef: React.RefObject<HTMLVideoElement>;
  eyeStatus: EyeTrackingStatus;
  eyeErrorReason: EyeTrackingErrorReason | null;
  eyeLiveStats: { blinkCount: number; faceDetected: boolean; elapsedSec: number };
  eyeCaptureDone: boolean;
  onStartEyeTracking: () => void;
  onStopEyeTracking: () => void;
  // --- Real typing capture (replaces the old typingErrorRate slider) ---
  referenceText: string;
  typedText: string;
  onTypedChange: (v: string) => void;
  onTypingKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onTypingKeyUp: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  typingDynamics: TypingDynamics;
  // --- Real audio capture (replaces the old pausesPerMinute slider) ---
  audioStatus: AudioCaptureStatus;
  audioErrorReason: AudioCaptureErrorReason | null;
  audioElapsedSec: number;
  audioCaptureDone: boolean;
  onStartAudio: () => void;
  onStopAudio: () => void;
  onRunAssessment: () => void;
}

export default function AssessmentDashboard({
  isApiOnline,
  isLoading,
  error,
  result,
  videoRef,
  eyeStatus,
  eyeErrorReason,
  eyeLiveStats,
  eyeCaptureDone,
  onStartEyeTracking,
  onStopEyeTracking,
  referenceText,
  typedText,
  onTypedChange,
  onTypingKeyDown,
  onTypingKeyUp,
  typingDynamics,
  audioStatus,
  audioErrorReason,
  audioElapsedSec,
  audioCaptureDone,
  onStartAudio,
  onStopAudio,
  onRunAssessment,
}: AssessmentDashboardProps) {
  const eyeErrorMessages: Record<string, string> = {
    unsupported_browser: "Your browser doesn't support camera access. Try Chrome, Edge, or Firefox.",
    permission_denied: "Camera permission denied. Enable it in your browser's site settings to continue.",
    no_camera: "No camera device found. Connect a webcam and try again.",
    model_load_failed: "Couldn't load the face-tracking model (check your connection) and try again.",
    no_face_detected: "No face was detected during the session. Make sure you're facing the camera in good lighting.",
  };
  const audioErrorMessages: Record<string, string> = {
    unsupported_browser: "Your browser doesn't support microphone recording.",
    permission_denied: "Microphone permission denied. Enable it in your browser's site settings to continue.",
    no_microphone: "No microphone found. Connect one and try again.",
    empty_recording: "That recording was too short or silent — try again and read the sentence aloud.",
    upload_failed: "Couldn't reach the server to process the recording. Check the API is running.",
    extraction_failed: "The server couldn't process that recording (audio format or backend issue).",
  };
  return (
    <section className="py-20 md:py-32 container mx-auto px-4">
      <div className="flex flex-col items-center mb-12 text-center">
        <Badge variant={isApiOnline ? "default" : "destructive"} className="mb-4">
          {isApiOnline ? (
            <span className="flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> API Online</span>
          ) : (
            <span className="flex items-center gap-1"><AlertCircle className="w-3 h-3" /> API Offline</span>
          )}
        </Badge>
        <h2 className="text-3xl md:text-5xl font-bold mb-4 font-serif">Assessment Dashboard</h2>
        <p className="text-muted-foreground text-lg max-w-2xl">
          Real multimodal screening system utilizing our advanced AI backend. Let's see it in action.
        </p>
      </div>

      <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">

        {/* Left column: Controls */}
        <div className="w-full">
          <Card className="border-primary/10 shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" /> Assessment Inputs
              </CardTitle>
              <CardDescription>Typing, eye tracking, and audio are all captured live — no more simulated inputs.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                     <Eye className="w-4 h-4 text-muted-foreground"/> Eye Tracking (webcam, real capture)
                  </Label>
                  {eyeStatus === "tracking" && (
                    <span className="text-xs text-muted-foreground">{eyeLiveStats.elapsedSec.toFixed(0)}s</span>
                  )}
                </div>

                <div className="relative w-full aspect-video bg-black/90 rounded-md overflow-hidden">
                  <video ref={videoRef} className="w-full h-full object-cover -scale-x-100" muted playsInline />
                  {eyeStatus === "tracking" && (
                    <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-black/60 rounded-full px-2 py-0.5 text-xs text-white">
                      <span className={`w-2 h-2 rounded-full ${eyeLiveStats.faceDetected ? "bg-green-400" : "bg-red-400"}`} />
                      {eyeLiveStats.faceDetected ? "Face detected" : "No face"}
                    </div>
                  )}
                  {eyeStatus === "idle" && (
                    <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
                      <VideoOff className="w-8 h-8 opacity-30" />
                    </div>
                  )}
                </div>

                {eyeStatus === "tracking" && (
                  <p className="text-xs text-muted-foreground">Blinks so far: {eyeLiveStats.blinkCount}</p>
                )}

                {eyeStatus === "error" && eyeErrorReason && (
                  <div className="p-2 bg-destructive/10 text-destructive text-xs rounded-md flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <p>{eyeErrorMessages[eyeErrorReason]}</p>
                  </div>
                )}

                {eyeCaptureDone && eyeStatus !== "tracking" && (
                  <p className="text-xs text-green-600 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Eye tracking sample captured
                  </p>
                )}

                <Button
                  type="button"
                  variant={eyeStatus === "tracking" ? "destructive" : "outline"}
                  size="sm"
                  className="w-full gap-2"
                  onClick={eyeStatus === "tracking" ? onStopEyeTracking : onStartEyeTracking}
                  disabled={eyeStatus === "requesting_permission" || eyeStatus === "loading_model"}
                >
                  {eyeStatus === "requesting_permission" && "Requesting camera permission..."}
                  {eyeStatus === "loading_model" && "Loading face-tracking model..."}
                  {eyeStatus === "tracking" && <><VideoOff className="w-4 h-4" /> Stop capture</>}
                  {(eyeStatus === "idle" || eyeStatus === "stopped" || eyeStatus === "error") && (
                    <><Video className="w-4 h-4" /> {eyeCaptureDone ? "Re-capture" : "Start webcam capture"}</>
                  )}
                </Button>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="typing-capture" className="flex items-center gap-2">
                     <Keyboard className="w-4 h-4 text-muted-foreground"/> Typing Sample (real keystroke capture)
                  </Label>
                  <span className="text-xs text-muted-foreground">{typingDynamics.totalKeystrokes} keys</span>
                </div>
                <p className="text-xs text-muted-foreground italic border-l-2 border-primary/30 pl-2">
                  "{referenceText}"
                </p>
                <textarea
                  id="typing-capture"
                  value={typedText}
                  onChange={(e) => onTypedChange(e.target.value)}
                  onKeyDown={onTypingKeyDown}
                  onKeyUp={onTypingKeyUp}
                  placeholder="Type the sentence above exactly as shown..."
                  rows={2}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary/40"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Backspaces: {typingDynamics.backspaceCount}</span>
                  <span>Avg hold: {typingDynamics.avgHoldTimeMs.toFixed(0)}ms</span>
                  <span>Speed: {typingDynamics.typingSpeedCpm.toFixed(0)} cpm</span>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                     <Mic className="w-4 h-4 text-muted-foreground"/> Read-Aloud Sample (real audio capture)
                  </Label>
                  {audioStatus === "recording" && (
                    <span className="text-xs text-muted-foreground">{audioElapsedSec.toFixed(0)}s</span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground italic border-l-2 border-primary/30 pl-2">
                  Read aloud: "{referenceText}"
                </p>

                {audioStatus === "error" && audioErrorReason && (
                  <div className="p-2 bg-destructive/10 text-destructive text-xs rounded-md flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <p>{audioErrorMessages[audioErrorReason]}</p>
                  </div>
                )}

                {audioCaptureDone && audioStatus === "done" && (
                  <p className="text-xs text-green-600 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Audio sample processed by backend
                  </p>
                )}

                <Button
                  type="button"
                  variant={audioStatus === "recording" ? "destructive" : "outline"}
                  size="sm"
                  className="w-full gap-2"
                  onClick={audioStatus === "recording" ? onStopAudio : onStartAudio}
                  disabled={audioStatus === "requesting_permission" || audioStatus === "uploading"}
                >
                  {audioStatus === "requesting_permission" && "Requesting microphone permission..."}
                  {audioStatus === "uploading" && "Extracting features on backend (MFCC/jitter/shimmer/Whisper)..."}
                  {audioStatus === "recording" && <><MicOff className="w-4 h-4" /> Stop recording</>}
                  {(audioStatus === "idle" || audioStatus === "done" || audioStatus === "error") && (
                    <><Mic className="w-4 h-4" /> {audioCaptureDone ? "Re-record" : "Start recording"}</>
                  )}
                </Button>
              </div>

            </CardContent>
            <CardFooter className="flex flex-col gap-2">
              <Button
                onClick={onRunAssessment}
                disabled={
                  isLoading ||
                  !isApiOnline ||
                  typedText.trim().length === 0 ||
                  !eyeCaptureDone ||
                  !audioCaptureDone
                }
                className="w-full gap-2"
              >
                {isLoading ? "Processing..." : <><Play className="w-4 h-4" /> Run Assessment</>}
              </Button>
              {(typedText.trim().length === 0 || !eyeCaptureDone || !audioCaptureDone) && (
                <p className="text-xs text-muted-foreground text-center">
                  Complete all three captures to enable assessment:
                  {typedText.trim().length === 0 && " typing,"}
                  {!eyeCaptureDone && " eye tracking,"}
                  {!audioCaptureDone && " audio"}
                </p>
              )}
            </CardFooter>
          </Card>
        </div>

        {/* Right column: Results */}
        <div className="w-full">
           <Card className="h-full border-primary/20 shadow-card-hover bg-card-gradient overflow-hidden relative">
              {/* Decorative background element */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />

              <CardHeader>
                <CardTitle>Analysis Results</CardTitle>
                <CardDescription>AI-generated risk assessment feedback</CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">

                {error && (
                  <div className="p-4 bg-destructive/10 text-destructive text-sm rounded-lg flex items-start gap-3 border border-destructive/20 animate-fade-in">
                    <AlertCircle className="w-5 h-5 shrink-0" />
                    <p>{error}</p>
                  </div>
                )}

                {!result && !isLoading && !error && (
                  <div className="h-64 flex flex-col items-center justify-center text-muted-foreground text-center px-4">
                     <Activity className="w-10 h-10 mb-4 opacity-20" />
                     <p>Run the assessment to see AI results here.</p>
                  </div>
                )}

                {isLoading && (
                   <div className="h-64 flex flex-col items-center justify-center space-y-4">
                      <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                      <p className="text-muted-foreground animate-pulse">Running AI models on backend...</p>
                   </div>
                )}

                {result && !isLoading && (
                  <div className="space-y-6 animate-fade-in">
                    <div className="flex items-center justify-between bg-background p-4 rounded-xl border">
                       <div>
                          <p className="text-sm text-muted-foreground mb-1">Risk Level</p>
                          <h3 className="text-2xl font-bold flex items-center gap-2">
                            {result.risk_level === 'LOW' && "🟢 Low"}
                            {result.risk_level === 'MEDIUM' && "🟡 Medium"}
                            {result.risk_level === 'HIGH' && "🔴 High"}
                          </h3>
                       </div>
                       <div className="text-right">
                          <p className="text-sm text-muted-foreground mb-1">Confidence</p>
                          <p className="text-xl font-bold text-primary">{(result.confidence * 100).toFixed(1)}%</p>
                       </div>
                    </div>

                    <div className="space-y-2">
                       <div className="flex justify-between items-center text-sm">
                         <span className="font-medium">Overall Risk Score</span>
                         <span>{(result.risk_score * 100).toFixed(1)}%</span>
                       </div>
                       <Progress value={result.risk_score * 100} className="h-3" />
                       <p className="text-xs text-muted-foreground mt-2">
                         Score determined via backend fusion of visual, acoustic, and kinetic models.
                       </p>
                    </div>
                  </div>
                )}
              </CardContent>
           </Card>
        </div>

      </div>
    </section>
  );
}
