import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface TimelineItem {
  id: string;
  title: string;
  description: string;
  date: string;
  icon: LucideIcon;
  iconColor?: string;
}

interface TimelineProps {
  items: TimelineItem[];
  className?: string;
}

export function Timeline({ items, className }: TimelineProps) {
  return (
    <div className={cn("relative", className)}>
      {/* Vertical line */}
      <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-border" />

      <div className="space-y-6">
        {items.map((item, index) => {
          const Icon = item.icon;
          return (
            <div
              key={item.id}
              className="relative flex gap-4 animate-fade-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div
                className={cn(
                  "relative z-10 h-12 w-12 rounded-full flex items-center justify-center bg-card border-2 border-border shadow-sm",
                  item.iconColor
                )}
              >
                <Icon className="h-5 w-5" />
              </div>

              <div className="flex-1 pt-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">{item.title}</h4>
                  <span className="text-sm text-muted-foreground">{item.date}</span>
                </div>
                <p className="text-sm text-muted-foreground mt-1">{item.description}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
