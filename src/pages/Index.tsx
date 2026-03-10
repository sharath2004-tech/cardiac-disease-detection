import { Heart, Activity, FileText, Shield, ChevronRight, Users, Brain, Clock } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";

const features = [
  {
    icon: Brain,
    title: "AI-Powered Analysis",
    description: "Advanced machine learning algorithms analyze your health data with 95%+ accuracy.",
  },
  {
    icon: FileText,
    title: "Medical Records Management",
    description: "Securely upload and manage ECGs, lab reports, and medical documents in one place.",
  },
  {
    icon: Activity,
    title: "Risk Tracking",
    description: "Monitor your cardiac health over time with historical comparisons and trend analysis.",
  },
  {
    icon: Shield,
    title: "HIPAA Compliant",
    description: "Your health data is protected with enterprise-grade security and encryption.",
  },
];

const stats = [
  { value: "50K+", label: "Patients Analyzed" },
  { value: "95%", label: "Detection Accuracy" },
  { value: "500+", label: "Healthcare Partners" },
  { value: "24/7", label: "AI Monitoring" },
];

export default function Index() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
        <div className="container relative py-20 md:py-32">
          <div className="max-w-3xl mx-auto text-center animate-slide-up">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-sm font-medium mb-6">
              <Heart className="h-4 w-4" />
              <span>AI-Powered Cardiac Health Platform</span>
            </div>

            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
              <span className="text-gray-900 dark:text-white">AI Cardiac Disease</span>
              <span className="bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent"> Detection System</span>
            </h1>

            <p className="text-lg md:text-xl text-gray-700 dark:text-gray-300 mb-10 max-w-2xl mx-auto">
              Track, analyze, and predict heart disease using AI and your medical history. 
              Get personalized insights and early warning detection for better heart health.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/assessment">
                <Button variant="hero" size="xl" className="group">
                  Check Heart Risk
                  <ChevronRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
                </Button>
              </Link>
              <Link to="/profile">
                <Button variant="heroOutline" size="xl">
                  View Medical History
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="container py-16 -mt-12 relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((stat, index) => (
            <div
              key={stat.label}
              className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-lg text-center animate-slide-up"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <p className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">
                {stat.value}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="container py-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-foreground">
            Comprehensive Heart Health Platform
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Everything you need to monitor, analyze, and improve your cardiac health in one secure platform.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="group p-6 bg-card rounded-xl border border-border shadow-card hover:shadow-card-hover transition-all duration-300 animate-slide-up"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="h-14 w-14 rounded-xl healthcare-gradient flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <feature.icon className="h-7 w-7 text-primary-foreground" />
              </div>
              <h3 className="font-semibold text-lg mb-2 text-foreground">{feature.title}</h3>
              <p className="text-sm text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="container py-20">
        <div className="relative rounded-3xl overflow-hidden">
          <div className="absolute inset-0 healthcare-gradient opacity-90" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-white/20 via-transparent to-transparent" />
          
          <div className="relative p-8 md:p-16 text-center">
            <div className="flex justify-center mb-6">
              <div className="h-16 w-16 rounded-2xl bg-white/20 flex items-center justify-center backdrop-blur-sm">
                <Users className="h-8 w-8 text-primary-foreground" />
              </div>
            </div>
            <h2 className="text-2xl md:text-4xl font-bold text-primary-foreground mb-4">
              Ready to Take Control of Your Heart Health?
            </h2>
            <p className="text-primary-foreground/80 max-w-xl mx-auto mb-8">
              Join thousands of patients and healthcare providers using CardioAI for early detection and prevention.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/signup">
                <Button size="lg" className="bg-white text-primary hover:bg-white/90 shadow-lg">
                  Get Started Free
                </Button>
              </Link>
              <Link to="/doctor">
                <Button size="lg" variant="outline" className="border-white/30 text-primary-foreground hover:bg-white/10">
                  Doctor Portal
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Trust Indicators */}
      <section className="container py-16 border-t border-border">
        <div className="flex flex-wrap items-center justify-center gap-8 text-muted-foreground">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            <span className="text-sm font-medium">HIPAA Compliant</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            <span className="text-sm font-medium">Real-time Analysis</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            <span className="text-sm font-medium">FDA Registered</span>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
