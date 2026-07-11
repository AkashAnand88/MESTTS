import { Brain, Github, Mail } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 bg-card border-t border-border">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-hero-gradient flex items-center justify-center shadow-md">
              <Brain className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-serif text-xl font-semibold text-foreground">
              DysDetect
            </span>
          </div>

          {/* Links */}
          <div className="flex items-center gap-6">
            <a
              href="#"
              className="text-muted-foreground hover:text-primary transition-colors flex items-center gap-2"
            >
              <Github className="w-4 h-4" />
              <span className="text-sm">Dyslexia awareness for inclusive learning</span>
            </a>
            <a
              href="#"
              className="text-muted-foreground hover:text-primary transition-colors flex items-center gap-2"
            >
              
              
            </a>
          </div>

          {/* Copyright */}
          <p className="text-sm text-muted-foreground">
            © Final Year Project.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
