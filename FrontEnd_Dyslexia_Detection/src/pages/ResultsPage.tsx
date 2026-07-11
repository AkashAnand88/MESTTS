// src/pages/ResultsPage.tsx
import { useLocation, useNavigate } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, LineChart, Line, CartesianGrid, Tooltip } from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, ArrowLeft } from "lucide-react";
import type { DesktopAssessmentResult } from "@/hooks/useDesktopLauncher";

const DISCLAIMER =
  "This system produces a behavioral risk indicator only. It does not constitute a clinical diagnosis of dyslexia or any other condition. Results must be interpreted by a qualified specialist.";

interface ResultsLocationState {
  result: DesktopAssessmentResult;
}

const RISK_COLORS: Record<string, string> = {
  "LOW RISK": "#00b894",
  "MEDIUM RISK": "#fdcb6e",
  "HIGH RISK": "#ff6348",
};

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as ResultsLocationState | null;

  if (!state?.result) {
    return (
      <div className="max-w-xl mx-auto mt-16 text-center space-y-4">
        <p className="text-muted-foreground">No assessment result to show yet.</p>
        <Button onClick={() => navigate("/")}>Run an assessment</Button>
      </div>
    );
  }

  const { result } = state;
  const riskColor = RISK_COLORS[result.risk_band] ?? RISK_COLORS["MEDIUM RISK"];

  const confidenceChartData = [
    { name: "Eye Tracking", value: result.avg_eye_conf * 100 },
    { name: "Typing", value: result.avg_type_conf * 100 },
    { name: "Audio", value: result.avg_audio_conf * 100 },
  ];

  const sentenceChartData = result.per_sentence.map((s) => ({
    name: `S${s.sentence_index + 1}`,
    score: s.final_score * 100,
  }));

  return (
    <div className="max-w-3xl mx-auto py-8 px-4 space-y-6">
      <Button variant="ghost" size="sm" onClick={() => navigate("/")} className="gap-2">
        <ArrowLeft className="w-4 h-4" /> Back
      </Button>

      <Card style={{ borderColor: riskColor, borderWidth: 2 }}>
        <CardContent className="pt-6 text-center space-y-2">
          <p className="text-sm text-muted-foreground">
            {result.participant_id} · {result.timestamp}
          </p>
          <p className="text-4xl font-bold" style={{ color: riskColor }}>
            {result.risk_band}
          </p>
          <p className="text-sm text-muted-foreground">
            Fusion score {(result.avg_fusion_score * 100).toFixed(1)}%
            {" · "}{result.per_sentence.length} sentence{result.per_sentence.length !== 1 ? "s" : ""} scored
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Per-Modality Confidence</CardTitle>
          <CardDescription>Average across all sentences in this session</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={confidenceChartData} layout="vertical" margin={{ left: 20 }}>
                <XAxis type="number" domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="name" width={90} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {confidenceChartData.map((entry, i) => (
                    <Cell key={i} fill={entry.value > 60 ? RISK_COLORS["HIGH RISK"] : entry.value > 30 ? RISK_COLORS["MEDIUM RISK"] : RISK_COLORS["LOW RISK"]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {result.per_sentence.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Score by Sentence</CardTitle>
            <CardDescription>Fusion score trend across the reading session</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sentenceChartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} unit="%" />
                  <Tooltip />
                  <Line type="monotone" dataKey="score" stroke={riskColor} strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 rounded-md border divide-y">
              {result.per_sentence.map((s) => (
                <div key={s.sentence_index} className="flex justify-between px-3 py-1.5 text-sm">
                  <span>Sentence {s.sentence_index + 1}</span>
                  <span className="text-muted-foreground">{s.risk_label}</span>
                  <span className="font-mono">{(s.final_score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="flex items-start gap-2 p-3 rounded-md bg-muted text-xs text-muted-foreground">
        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
        <p>{DISCLAIMER}</p>
      </div>

      <Button className="w-full" onClick={() => navigate("/")}>Run Another Assessment</Button>
    </div>
  );
}
