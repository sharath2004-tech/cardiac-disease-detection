import { FileText, Image, FileSpreadsheet, Eye, Download, Trash2 } from "lucide-react";
import { Button } from "./button";
import { Badge } from "./badge";
import { cn } from "@/lib/utils";

interface RecordCardProps {
  id: string;
  fileName: string;
  recordType: "ECG" | "Lab" | "Report" | "Prescription" | "Scan";
  uploadDate: string;
  onView?: () => void;
  onDownload?: () => void;
  onDelete?: () => void;
  className?: string;
}

const recordTypeConfig = {
  ECG: { icon: Image, color: "bg-primary/10 text-primary" },
  Lab: { icon: FileSpreadsheet, color: "bg-accent/10 text-accent" },
  Report: { icon: FileText, color: "bg-warning/10 text-warning" },
  Prescription: { icon: FileText, color: "bg-success/10 text-success" },
  Scan: { icon: Image, color: "bg-primary/10 text-primary" },
};

export function RecordCard({
  fileName,
  recordType,
  uploadDate,
  onView,
  onDownload,
  onDelete,
  className,
}: RecordCardProps) {
  const config = recordTypeConfig[recordType];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "group flex items-center gap-4 p-4 bg-card rounded-xl border border-border shadow-card hover:shadow-card-hover transition-all duration-200",
        className
      )}
    >
      <div className={cn("h-12 w-12 rounded-lg flex items-center justify-center", config.color)}>
        <Icon className="h-6 w-6" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="font-medium truncate">{fileName}</p>
          <Badge variant="secondary" className="text-xs">
            {recordType}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground mt-0.5">{uploadDate}</p>
      </div>

      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onView}>
          <Eye className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onDownload}>
          <Download className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-destructive"
          onClick={onDelete}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
