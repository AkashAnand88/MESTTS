import { useState, useEffect, useCallback } from "react";

export interface AssessmentResult {
  risk_level: string;
  risk_score: number;
  risk_icon: string;
  confidence: number;
  modalities_active: number;
  details: Record<string, number>;
  timestamp: string;
}

interface UseDyslexiaAssessmentReturn {
  isApiOnline: boolean;
  isLoading: boolean;
  error: string | null;
  result: AssessmentResult | null;
  checkHealth: () => Promise<void>;
  performAssessment: (payload: any) => Promise<void>;
}

export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export function useDyslexiaAssessment(): UseDyslexiaAssessmentReturn {
  const [isApiOnline, setIsApiOnline] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AssessmentResult | null>(null);

  // =========================
  // Health Check
  // =========================
  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) throw new Error("Health check failed");

      setIsApiOnline(true);
      setError(null);
    } catch (err) {
      setIsApiOnline(false);
      setError("API Offline");
    }
  }, []);

  // =========================
  // Perform Assessment
  // =========================
  const performAssessment = useCallback(async (payload: any) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/assess`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error("Assessment error:", err);
      setError("Failed to fetch assessment result");
    } finally {
      setIsLoading(false);
    }
  }, []);

  // =========================
  // Auto health check on mount
  // =========================
  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return {
    isApiOnline,
    isLoading,
    error,
    result,
    checkHealth,
    performAssessment,
  };
}