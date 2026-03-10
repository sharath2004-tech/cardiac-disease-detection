import { cn } from "@/lib/utils";
import { AlertTriangle, CheckCircle, AlertCircle } from "lucide-react";

interface RiskIndicatorProps {
  level: "low" | "medium" | "high";
  score: number;
  className?: string;
}

const riskConfig = {
  low: {
    label: "Low Risk",
    color: "text-success",
    bgColor: "bg-success/10",
    borderColor: "border-success/30",
    icon: CheckCircle,
    gradient: "from-success/20 to-success/5",
  },
  medium: {
    label: "Medium Risk",
    color: "text-warning",
    bgColor: "bg-warning/10",
    borderColor: "border-warning/30",
    icon: AlertTriangle,
    gradient: "from-warning/20 to-warning/5",
  },
  high: {
    label: "High Risk",
    color: "text-destructive",
    bgColor: "bg-destructive/10",
    borderColor: "border-destructive/30",
    icon: AlertCircle,
    gradient: "from-destructive/20 to-destructive/5",
  },
};

export function RiskIndicator({ level, score, className }: RiskIndicatorProps) {
  const config = riskConfig[level];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "relative p-6 rounded-2xl border-2 overflow-hidden",
        config.borderColor,
        className
      )}
    >
      <div className={cn("absolute inset-0 bg-gradient-to-br", config.gradient)} />
      
      <div className="relative flex items-center gap-6">
        <div className={cn("h-20 w-20 rounded-full flex items-center justify-center", config.bgColor)}>
          <Icon className={cn("h-10 w-10", config.color)} />
        </div>
        
        <div className="flex-1">
          <div className="flex items-baseline gap-2">
            <span className={cn("text-4xl font-bold", config.color)}>{score}%</span>
            <span className="text-lg text-muted-foreground">risk score</span>
          </div>
          <p className={cn("text-xl font-semibold mt-1", config.color)}>
            {config.label}
          </p>
        </div>
      </div>

      <div className="relative mt-6">
        <div className="h-3 bg-secondary rounded-full overflow-hidden">
          <div
            className={cn("h-full rounded-full transition-all duration-1000", config.bgColor)}
            style={{ width: `${score}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs text-muted-foreground">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  );
}
