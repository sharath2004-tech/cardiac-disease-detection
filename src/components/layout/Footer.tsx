import { Heart, Shield, Lock } from "lucide-react";
import { Link } from "react-router-dom";

export function Footer() {
  return (
    <footer className="border-t border-border bg-card">
      <div className="container py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl healthcare-gradient shadow-healthcare">
                <Heart className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-bold">CardioAI</span>
            </div>
            <p className="text-sm text-muted-foreground">
              AI-powered cardiac disease detection for better heart health outcomes.
            </p>
            <div className="flex items-center gap-4 text-muted-foreground">
              <div className="flex items-center gap-1 text-xs">
                <Shield className="h-3 w-3" />
                <span>HIPAA Compliant</span>
              </div>
              <div className="flex items-center gap-1 text-xs">
                <Lock className="h-3 w-3" />
                <span>256-bit Encryption</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Platform</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><Link to="/assessment" className="hover:text-foreground transition-colors">Risk Assessment</Link></li>
              <li><Link to="/profile" className="hover:text-foreground transition-colors">Medical Records</Link></li>
              <li><Link to="/dashboard" className="hover:text-foreground transition-colors">Results Dashboard</Link></li>
              <li><Link to="/doctor" className="hover:text-foreground transition-colors">Doctor Portal</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Support</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><a href="#" className="hover:text-foreground transition-colors">Help Center</a></li>
              <li><a href="#" className="hover:text-foreground transition-colors">Contact Us</a></li>
              <li><a href="#" className="hover:text-foreground transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-foreground transition-colors">Terms of Service</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Contact</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>support@cardioai.health</li>
              <li>1-800-CARDIO-AI</li>
              <li>Mon-Fri: 9AM - 6PM EST</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-border mt-8 pt-8 text-center text-sm text-muted-foreground">
          <p>© 2024 CardioAI Detection System. All rights reserved.</p>
          <p className="mt-2 text-xs">
            This tool is for informational purposes only and should not replace professional medical advice.
          </p>
        </div>
      </div>
    </footer>
  );
}
