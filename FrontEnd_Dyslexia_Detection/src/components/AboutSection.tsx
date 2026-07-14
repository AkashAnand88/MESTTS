import { GraduationCap, Shield, Heart, Lightbulb } from "lucide-react";

const values = [
  {
    icon: GraduationCap,
    title: "Research-Backed",
    description:
      "Our models are built on peer-reviewed research and validated methodologies in MESTTS detection.",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description:
      "All assessment data is processed securely with strict privacy standards. Your data belongs to you.",
  },
  {
    icon: Heart,
    title: "Accessible",
    description:
      "Designed to be user-friendly and accessible to individuals of all ages and technical backgrounds.",
  },
  {
    icon: Lightbulb,
    title: "Actionable Insights",
    description:
      "We don't just detect—we provide clear guidance and resources for the next steps.",
  },
];

const AboutSection = () => {
  return (
    <section id="about" className="py-24 bg-secondary/30">
      <div className="container mx-auto px-4">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Content */}
          <div>
            <h2 className="font-serif text-4xl md:text-5xl font-bold text-foreground mb-6">
              About This Project
            </h2>
            <p className="text-lg text-muted-foreground mb-6 leading-relaxed">
              This MESTTS detection system is a final year project that combines
              cutting-edge machine learning with a deep understanding of
              learning differences. Our goal is to make early MESTTS screening
              accessible and accurate.
            </p>
            <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
              By analyzing typing patterns, eye movements, and audio input, we
              can identify potential indicators of MESTTS that might otherwise
              go unnoticed, enabling earlier intervention and better outcomes.
            </p>

            {/* Project Files */}
            <div className="bg-card rounded-xl p-6 shadow-card">
              <h4 className="font-semibold text-foreground mb-4">
                Project Components
              </h4>
              <div className="grid grid-cols-2 gap-3">
                {[
                  "typing_model.py",
                  "eye_model.py",
                  "Audio_model.py",
                  "training_data.csv",
                ].map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-2 px-3 py-2 bg-muted rounded-lg"
                  >
                    <div className="w-2 h-2 rounded-full bg-primary" />
                    <span className="font-mono text-sm text-muted-foreground">
                      {file}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Values Grid */}
          <div className="grid sm:grid-cols-2 gap-6">
            {values.map((value, index) => (
              <div
                key={index}
                className="bg-card rounded-xl p-6 shadow-card hover:shadow-card-hover transition-shadow duration-300"
              >
                <div className="w-12 h-12 rounded-xl bg-teal-light flex items-center justify-center mb-4">
                  <value.icon className="w-6 h-6 text-primary" />
                </div>
                <h4 className="font-serif text-lg font-semibold text-foreground mb-2">
                  {value.title}
                </h4>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {value.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
