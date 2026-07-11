import { useNavigate } from "react-router-dom";
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import DetectionMethods from "@/components/DetectionMethods";
import HowItWorks from "@/components/HowItWorks";
import AboutSection from "@/components/AboutSection";
import Footer from "@/components/Footer";
import DesktopLauncher from "@/components/DesktopLauncher";
import { useDyslexiaAssessment } from "@/hooks/useDyslexiaAssessment";
import { useDesktopLauncher, type DesktopAssessmentResult } from "@/hooks/useDesktopLauncher";

// NOTE ON ARCHITECTURE (per project decision): the browser-based capture
// built in Phases 1-3 (useTypingCapture / useEyeTracking / useAudioCapture /
// useCalibration + CalibrationScreen) is no longer used in the main flow.
// Those files are still in the repo but unreferenced here. The actual
// calibration/capture/report experience now runs in the real main_gui.py
// (PyQt) desktop app, launched as a subprocess by the backend. This page's
// job is just: launch it, wait, then show whatever it exported.
//
// CONSTRAINT: this only works when the browser and the backend are running
// on the SAME machine as the person taking the test (subprocess.Popen opens
// a real GUI window on the backend process's display). Fine for an in-person
// demo on your own laptop — not a deployable remote flow.

const Index = () => {
  const navigate = useNavigate();
  const { isApiOnline } = useDyslexiaAssessment();
  const launcher = useDesktopLauncher();

  const handleLaunch = (participantId: string) => {
    launcher.launch(participantId, (result) => {
      if (result) navigate("/results", { state: { result } });
    });
  };

  const handleRetryFetch = () => {
    launcher.retryFetchResult((result: DesktopAssessmentResult | null) => {
      if (result) navigate("/results", { state: { result } });
    });
  };

  const handleBeginAssessment = () => {
    document.getElementById("assessment")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-background selection:bg-primary/20">
      <Navbar />
      <HeroSection onBeginAssessment={handleBeginAssessment} isLoading={launcher.status === "launching"} />
      <DetectionMethods />
      <HowItWorks />
      <div id="assessment" className="bg-gradient-to-b from-background to-secondary/20 py-20">
        <DesktopLauncher
          status={launcher.status}
          errorMessage={launcher.errorMessage}
          isApiOnline={isApiOnline}
          onLaunch={handleLaunch}
          onRetryFetch={handleRetryFetch}
          onReset={launcher.reset}
        />
      </div>
      <AboutSection />
      <Footer />
    </div>
  );
};

export default Index;
