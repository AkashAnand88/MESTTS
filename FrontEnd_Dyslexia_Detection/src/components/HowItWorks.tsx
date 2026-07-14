import { Upload, Brain, BarChart3, FileText, ArrowRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const steps = [
  {
    icon: Upload,
    step: "01",
    title: "Provide Input",
    description:
      "Complete simple tasks like typing a passage, reading text while we track eye movements, or reading aloud for audio analysis.",
  },
  {
    icon: Brain,
    step: "02",
    title: "ML Processing",
    description:
      "Our trained machine learning models analyze your input data using patterns learned from our comprehensive training dataset.",
  },
  {
    icon: BarChart3,
    step: "03",
    title: "Analysis",
    description:
      "The system evaluates multiple indicators across all three detection methods to form a comprehensive assessment.",
  },
  {
    icon: FileText,
    step: "04",
    title: "Results Report",
    description:
      "Receive a detailed report with insights, risk indicators, and recommendations for next steps if needed.",
  },
];

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 bg-background">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <Badge variant="outline" className="mb-4">
            Process
          </Badge>
          <h2 className="font-serif text-4xl md:text-5xl font-bold text-foreground mb-4">
            How It Works
          </h2>
          <p className="text-lg text-muted-foreground">
            A simple four-step process to get comprehensive MESTTS screening
            results in minutes.
          </p>
        </div>

        {/* Steps */}
        <div className="relative max-w-5xl mx-auto mb-20">
          {/* Connection Line */}
          <div className="hidden lg:block absolute top-10 left-12 right-12 h-0.5 bg-border z-0" />

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 relative z-10">
            {steps.map((item, index) => (
              <div
                key={index}
                className="relative text-center group flex flex-col items-center"
              >
                {/* Step Number Circle */}
                <div className="relative inline-flex mb-6">
                  <div className="w-20 h-20 rounded-2xl bg-card shadow-sm flex items-center justify-center border border-border group-hover:border-primary group-hover:bg-primary/5 transition-all duration-300 transform group-hover:-translate-y-1">
                    <item.icon className="w-8 h-8 text-primary group-hover:scale-110 transition-transform duration-300" />
                  </div>
                  <span className="absolute -top-3 -right-3 w-8 h-8 bg-primary text-primary-foreground text-sm font-bold rounded-full flex items-center justify-center shadow-md ring-4 ring-background">
                    {index + 1}
                  </span>
                </div>

                {/* Content */}
                <h3 className="font-serif text-xl font-semibold text-foreground mb-3 px-2">
                  {item.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed px-4">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Training Data Info */}
        <div className="max-w-4xl mx-auto">
          <Card className="border-primary/20 bg-primary/5 overflow-hidden relative">
            <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-gradient-to-l from-primary/10 to-transparent pointer-events-none" />
            <CardContent className="p-8 sm:p-10">
              <div className="flex flex-col md:flex-row items-center gap-8 relative z-10">
                <div className="w-20 h-20 rounded-2xl bg-primary text-primary-foreground flex items-center justify-center shrink-0 shadow-lg shadow-primary/20">
                  <FileText className="w-10 h-10" />
                </div>
                <div className="text-center md:text-left flex-1">
                  <h4 className="font-serif text-2xl font-bold text-foreground mb-2">
                    Trained on Comprehensive Data
                  </h4>
                  <p className="text-muted-foreground text-lg leading-relaxed mb-4">
                    All three ML models are trained using our diverse dataset
                    containing thousands of samples with verified MESTTS
                    indicators, ensuring high accuracy and reliability in
                    detection.
                  </p>
                  <div className="inline-flex items-center gap-2 text-sm font-mono bg-background/80 backdrop-blur border rounded-md px-3 py-1.5 text-muted-foreground">
                    <FileText className="w-4 h-4 text-primary" />
                    training_data.csv
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
