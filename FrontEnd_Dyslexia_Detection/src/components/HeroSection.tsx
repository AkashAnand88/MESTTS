import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Sparkles, Activity, Loader2 } from "lucide-react";

interface HeroSectionProps {
  onBeginAssessment: () => void;
  isLoading: boolean;
}

const HeroSection = ({ onBeginAssessment, isLoading }: HeroSectionProps) => {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section
      id="home"
      className="relative min-h-[90vh] flex items-center justify-center overflow-hidden pt-24 pb-12"
    >
      {/* Background Elements */}
      <div className="absolute inset-0 bg-background overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-4 md:left-10 w-48 md:w-72 h-48 md:h-72 bg-teal-light rounded-full blur-3xl opacity-60 animate-pulse-slow" />
        <div className="absolute bottom-20 right-4 md:right-10 w-64 md:w-96 h-64 md:h-96 bg-coral-light rounded-full blur-3xl opacity-40 animate-pulse-slow" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] md:w-[600px] h-[300px] md:h-[600px] bg-teal-medium rounded-full blur-3xl opacity-20" />
      </div>

      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-4xl mx-auto text-center flex flex-col items-center">
          {/* Badge */}
          <Badge variant="outline" className="mb-8 animate-fade-in py-1.5 px-4 bg-background/50 backdrop-blur-md border-primary/20 text-primary gap-2">
            <Sparkles className="w-4 h-4" />
            AI-Powered Dyslexia Screening
          </Badge>

          {/* Heading */}
          <h1 className="font-serif text-5xl md:text-6xl lg:text-7xl font-bold text-foreground leading-tight mb-6 animate-fade-in [animation-delay:100ms] drop-shadow-sm">
            Early Detection for{" "}
            <span className="text-gradient">Better Outcomes</span>
          </h1>

          {/* Subheading */}
          <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in [animation-delay:200ms]">
            Advanced machine learning models analyze typing patterns, eye
            movements, and audio to provide comprehensive dyslexia screening in
            minutes.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in [animation-delay:300ms] w-full sm:w-auto">
            <Button
              onClick={onBeginAssessment}
              disabled={isLoading}
              size="lg"
              className="w-full sm:w-auto group rounded-full text-base h-14 px-8 shadow-glow"
            >
              {isLoading ? (
                <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Running Assessment...</>
              ) : (
                <>Begin Assessment
                  <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                </>
              )}
            </Button>
            <Button
              onClick={() => scrollTo("methods")}
              variant="outline"
              size="lg"
              className="w-full sm:w-auto rounded-full text-base h-14 px-8 border-primary/20 hover:bg-primary/5 bg-background/50 backdrop-blur-sm"
            >
              Learn More
            </Button>
          </div>

          {/* Stats Section */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 sm:gap-12 mt-16 max-w-3xl mx-auto animate-fade-in [animation-delay:400ms] bg-background/40 backdrop-blur-md p-8 rounded-3xl border border-white/10 shadow-xl">
            {[
              { value: "3", label: "Detection Methods", icon: Activity },
              { value: "95%", label: "Accuracy Rate", icon: Sparkles },
              { value: "5min", label: "Assessment Time", icon: ArrowRight },
            ].map((stat, index) => (
              <div key={index} className="flex flex-col items-center justify-center text-center space-y-2">
                <div className="text-3xl sm:text-4xl font-extrabold text-foreground tracking-tight">
                  {stat.value}
                </div>
                <div className="text-sm font-medium text-muted-foreground uppercase tracking-wider">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
