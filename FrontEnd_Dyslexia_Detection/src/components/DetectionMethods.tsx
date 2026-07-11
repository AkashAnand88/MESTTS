import { Keyboard, Eye, Mic, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const methods = [
  {
    icon: Keyboard,
    title: "Keystroke Analysis",
    description:
      "Our ML model analyzes typing patterns including speed, rhythm, and hesitations to identify potential dyslexia indicators.",
    features: [
      "Typing speed measurement",
      "Pattern recognition",
      "Error frequency analysis",
      "Rhythm consistency check",
    ],
    color: "primary",
    file: "typing_model.py",
  },
  {
    icon: Eye,
    title: "Eye Tracking",
    description:
      "Advanced eye movement analysis tracks reading patterns, fixation points, and saccadic movements during text processing.",
    features: [
      "Gaze pattern analysis",
      "Fixation duration",
      "Reading path tracking",
      "Regression detection",
    ],
    color: "accent",
    file: "eye_model.py",
  },
  {
    icon: Mic,
    title: "Audio Analysis",
    description:
      "Speech pattern recognition evaluates pronunciation, reading fluency, and phonological awareness through voice analysis.",
    features: [
      "Pronunciation accuracy",
      "Reading fluency score",
      "Phoneme recognition",
      "Speech rhythm analysis",
    ],
    color: "primary",
    file: "Audio_model.py",
  },
];

const DetectionMethods = () => {
  const scrollToAssessment = () => {
    document.getElementById("assessment")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section id="methods" className="py-24 bg-gradient-to-b from-background to-secondary/30">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <Badge variant="secondary" className="mb-4">Core Technology</Badge>
          <h2 className="font-serif text-4xl md:text-5xl font-bold text-foreground mb-4">
            Three Powerful Detection Methods
          </h2>
          <p className="text-lg text-muted-foreground">
            Our comprehensive approach combines multiple ML models trained on
            extensive datasets to provide accurate and reliable dyslexia
            screening.
          </p>
        </div>

        {/* Method Cards */}
        <div className="grid md:grid-cols-3 gap-8">
          {methods.map((method, index) => (
            <Card
              key={index}
              className="group border-primary/10 overflow-hidden shadow-card hover:shadow-card-hover transition-all duration-300 hover:-translate-y-2 bg-card/50 backdrop-blur-sm"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardHeader className="relative pb-0">
                 {/* Decorative gradient blob */}
                 <div className={`absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-20 pointer-events-none ${method.color === "accent" ? "bg-accent" : "bg-primary"}`} />
                
                {/* Icon */}
                <div
                  className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-6 transition-transform duration-500 group-hover:scale-110 group-hover:rotate-3 shadow-sm ${
                    method.color === "accent"
                      ? "bg-coral-light/50 border border-accent/20"
                      : "bg-teal-light/50 border border-primary/20"
                  }`}
                >
                  <method.icon
                    className={`w-8 h-8 ${
                      method.color === "accent" ? "text-accent" : "text-primary"
                    }`}
                  />
                </div>
                <CardTitle className="font-serif text-2xl font-bold">{method.title}</CardTitle>
              </CardHeader>

              <CardContent className="pt-4">
                <p className="text-muted-foreground mb-6 leading-relaxed">
                  {method.description}
                </p>

                {/* Features */}
                <ul className="space-y-3 mb-2">
                  {method.features.map((feature, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm">
                      <div
                        className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${
                          method.color === "accent" ? "bg-accent" : "bg-primary"
                        }`}
                      />
                      <span className="text-muted-foreground font-medium">{feature}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>

              <CardFooter className="pt-6 border-t border-border/50 flex items-center justify-between bg-muted/20">
                <Badge variant="outline" className="font-mono text-xs text-muted-foreground border-primary/20 bg-background/50">
                  {method.file}
                </Badge>
                <Button
                  onClick={scrollToAssessment}
                  variant="ghost"
                  size="sm"
                  className="text-primary hover:text-primary hover:bg-primary/10 gap-1 rounded-full group/btn"
                >
                  Try Now <ArrowRight className="w-3 h-3 group-hover/btn:translate-x-0.5 transition-transform" />
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default DetectionMethods;
