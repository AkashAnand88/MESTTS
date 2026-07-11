// src/components/DesktopLauncher.tsx
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AlertCircle, Monitor, RotateCcw } from "lucide-react";
import type { LaunchStatus } from "@/hooks/useDesktopLauncher";

interface DesktopLauncherProps {
  status: LaunchStatus;
  errorMessage: string | null;
  isApiOnline: boolean;
  onLaunch: (participantId: string) => void;
  onRetryFetch: () => void;
  onReset: () => void;
}

export default function DesktopLauncher({
  status,
  errorMessage,
  isApiOnline,
  onLaunch,
  onRetryFetch,
  onReset,
}: DesktopLauncherProps) {
  const [participantId, setParticipantId] = useState("WEB_USER");

  return (
    <Card className="max-w-xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Monitor className="w-5 h-5 text-primary" /> Run Assessment
        </CardTitle>
        <CardDescription>
          This opens the full desktop assessment app on this computer — calibration, reading sentences,
          and the results dashboard all happen in that window. Come back here once you close it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {status === "idle" && (
          <>
            <div className="space-y-1.5">
              <Label htmlFor="participant">Participant ID</Label>
              <Input
                id="participant"
                value={participantId}
                onChange={(e) => setParticipantId(e.target.value)}
                placeholder="e.g. STUDENT_01"
              />
            </div>
            <Button
              className="w-full"
              disabled={!isApiOnline || participantId.trim().length === 0}
              onClick={() => onLaunch(participantId.trim())}
            >
              Launch Assessment
            </Button>
            {!isApiOnline && (
              <p className="text-xs text-destructive text-center">Backend is offline — start the FastAPI server first.</p>
            )}
          </>
        )}

        {status === "launching" && (
          <p className="text-sm text-muted-foreground text-center py-4">Launching the desktop app...</p>
        )}

        {status === "running" && (
          <div className="text-center py-6 space-y-2">
            <p className="text-sm font-medium">Assessment window is open</p>
            <p className="text-xs text-muted-foreground">
              Complete calibration, the reading sentences, and click <strong>Export</strong> on the results
              screen before closing the window. This page updates automatically once it closes.
            </p>
          </div>
        )}

        {status === "fetching_result" && (
          <p className="text-sm text-muted-foreground text-center py-4">Reading the exported results...</p>
        )}

        {status === "waiting_for_export" && (
          <div className="space-y-3">
            <div className="p-3 bg-yellow-500/10 text-yellow-700 text-sm rounded-md flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <p>The window closed, but no exported result was found. Reopen it and click <strong>Export</strong> on the results screen before closing next time.</p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" className="flex-1" onClick={onRetryFetch}>Check again</Button>
              <Button className="flex-1" onClick={onReset}>Run again</Button>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="space-y-3">
            <div className="p-3 bg-destructive/10 text-destructive text-sm rounded-md flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <p>{errorMessage}</p>
            </div>
            <Button variant="outline" className="w-full" onClick={onReset}>
              <RotateCcw className="w-4 h-4 mr-2" /> Try again
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
