import { useState } from "react";
import { Search, Download, User, Activity, FileText, ChevronRight, Calendar, Filter } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Timeline } from "@/components/ui/timeline";
import { RecordCard } from "@/components/ui/record-card";
import { RiskIndicator } from "@/components/ui/risk-indicator";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

// Mock patient data
const patients = [
  { id: "1", name: "John Anderson", age: 54, lastVisit: "Jan 15, 2024", riskLevel: "medium" as const, riskScore: 42 },
  { id: "2", name: "Sarah Mitchell", age: 62, lastVisit: "Jan 14, 2024", riskLevel: "high" as const, riskScore: 68 },
  { id: "3", name: "Michael Chen", age: 45, lastVisit: "Jan 12, 2024", riskLevel: "low" as const, riskScore: 18 },
  { id: "4", name: "Emily Brown", age: 58, lastVisit: "Jan 10, 2024", riskLevel: "medium" as const, riskScore: 35 },
  { id: "5", name: "Robert Davis", age: 67, lastVisit: "Jan 8, 2024", riskLevel: "high" as const, riskScore: 72 },
];

const patientRecords = [
  { id: "1", fileName: "ECG_Report_Jan2024.pdf", recordType: "ECG" as const, uploadDate: "Jan 15, 2024" },
  { id: "2", fileName: "Blood_Test_Results.csv", recordType: "Lab" as const, uploadDate: "Jan 10, 2024" },
  { id: "3", fileName: "Cardiac_MRI_Scan.jpg", recordType: "Scan" as const, uploadDate: "Dec 28, 2023" },
];

const patientTimeline = [
  { id: "1", title: "Latest Assessment", description: "Risk score: 42% - Moderate risk", date: "Jan 15, 2024", icon: Activity, iconColor: "text-warning" },
  { id: "2", title: "ECG Report", description: "Normal sinus rhythm detected", date: "Jan 10, 2024", icon: FileText, iconColor: "text-primary" },
  { id: "3", title: "Follow-up Visit", description: "Blood pressure monitoring", date: "Jan 5, 2024", icon: Calendar, iconColor: "text-muted-foreground" },
];

const riskBadgeVariant = {
  low: "bg-success/10 text-success border-success/20",
  medium: "bg-warning/10 text-warning border-warning/20",
  high: "bg-destructive/10 text-destructive border-destructive/20",
};

export default function DoctorDashboard() {
  const [selectedPatient, setSelectedPatient] = useState<typeof patients[0] | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterRisk, setFilterRisk] = useState("all");

  const filteredPatients = patients.filter((patient) => {
    const matchesSearch = patient.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterRisk === "all" || patient.riskLevel === filterRisk;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 container py-8">
        <div className="animate-fade-in">
          {/* Page Header */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
            <div>
              <h1 className="text-3xl font-bold mb-1">Doctor Dashboard</h1>
              <p className="text-muted-foreground">
                Monitor and manage your patients' cardiac health
              </p>
            </div>
            <Button className="gap-2">
              <Download className="h-4 w-4" />
              Export All Records
            </Button>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Patient List */}
            <div className="lg:col-span-1">
              <Card>
                <CardHeader className="pb-4">
                  <CardTitle>Patients</CardTitle>
                  <CardDescription>{patients.length} total patients</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Search and Filter */}
                  <div className="space-y-3">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search patients..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9"
                      />
                    </div>
                    <Select value={filterRisk} onValueChange={setFilterRisk}>
                      <SelectTrigger>
                        <Filter className="h-4 w-4 mr-2" />
                        <SelectValue placeholder="Filter by risk" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Risk Levels</SelectItem>
                        <SelectItem value="low">Low Risk</SelectItem>
                        <SelectItem value="medium">Medium Risk</SelectItem>
                        <SelectItem value="high">High Risk</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Patient List */}
                  <div className="space-y-2 max-h-[500px] overflow-y-auto">
                    {filteredPatients.map((patient) => (
                      <div
                        key={patient.id}
                        onClick={() => setSelectedPatient(patient)}
                        className={cn(
                          "p-4 rounded-xl border cursor-pointer transition-all",
                          selectedPatient?.id === patient.id
                            ? "border-primary bg-primary/5"
                            : "border-border hover:border-primary/50 hover:bg-secondary/50"
                        )}
                      >
                        <div className="flex items-center gap-3">
                          <div className="h-10 w-10 rounded-full bg-secondary flex items-center justify-center">
                            <User className="h-5 w-5 text-muted-foreground" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{patient.name}</p>
                            <p className="text-sm text-muted-foreground">
                              Age: {patient.age} • Last: {patient.lastVisit}
                            </p>
                          </div>
                          <Badge
                            variant="outline"
                            className={cn("text-xs", riskBadgeVariant[patient.riskLevel])}
                          >
                            {patient.riskScore}%
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Patient Details */}
            <div className="lg:col-span-2 space-y-6">
              {selectedPatient ? (
                <>
                  {/* Patient Header */}
                  <Card>
                    <CardContent className="pt-6">
                      <div className="flex flex-col md:flex-row gap-6">
                        <div className="flex items-center gap-4">
                          <div className="h-16 w-16 rounded-2xl healthcare-gradient flex items-center justify-center">
                            <User className="h-8 w-8 text-primary-foreground" />
                          </div>
                          <div>
                            <h2 className="text-xl font-bold">{selectedPatient.name}</h2>
                            <p className="text-muted-foreground">
                              Age: {selectedPatient.age} • Patient ID: PAT-2024-{selectedPatient.id.padStart(4, '0')}
                            </p>
                          </div>
                        </div>
                        <div className="flex gap-3 md:ml-auto">
                          <Button variant="outline" className="gap-2">
                            <Download className="h-4 w-4" />
                            Download All
                          </Button>
                          <Button className="gap-2">
                            View Full Profile
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Current Prediction */}
                  <RiskIndicator
                    level={selectedPatient.riskLevel}
                    score={selectedPatient.riskScore}
                  />

                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Medical Records */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-lg">
                          <FileText className="h-5 w-5 text-primary" />
                          Medical Records
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {patientRecords.map((record) => (
                          <RecordCard
                            key={record.id}
                            {...record}
                            onView={() => console.log("View", record.id)}
                            onDownload={() => console.log("Download", record.id)}
                          />
                        ))}
                      </CardContent>
                    </Card>

                    {/* Timeline */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-lg">
                          <Calendar className="h-5 w-5 text-primary" />
                          Report Timeline
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <Timeline items={patientTimeline} />
                      </CardContent>
                    </Card>
                  </div>
                </>
              ) : (
                <Card className="h-full min-h-[400px] flex items-center justify-center">
                  <div className="text-center">
                    <User className="h-16 w-16 text-muted-foreground/30 mx-auto mb-4" />
                    <h3 className="text-lg font-medium mb-2">Select a Patient</h3>
                    <p className="text-muted-foreground">
                      Choose a patient from the list to view their details
                    </p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
