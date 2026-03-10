import { useState } from "react";
import { User, Calendar, FileText, Plus, Search, Filter } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FileUpload } from "@/components/ui/file-upload";
import { RecordCard } from "@/components/ui/record-card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

// Mock data
const mockRecords = [
  { id: "1", fileName: "ECG_Report_Jan2024.pdf", recordType: "ECG" as const, uploadDate: "Jan 15, 2024" },
  { id: "2", fileName: "Blood_Test_Results.csv", recordType: "Lab" as const, uploadDate: "Jan 10, 2024" },
  { id: "3", fileName: "Cardiac_MRI_Scan.jpg", recordType: "Scan" as const, uploadDate: "Dec 28, 2023" },
  { id: "4", fileName: "Prescription_Dec2023.pdf", recordType: "Prescription" as const, uploadDate: "Dec 20, 2023" },
  { id: "5", fileName: "Annual_Checkup_Report.pdf", recordType: "Report" as const, uploadDate: "Dec 15, 2023" },
];

export default function PatientProfile() {
  const [records, setRecords] = useState(mockRecords);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [deleteRecordId, setDeleteRecordId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState("all");
  const [selectedRecordType, setSelectedRecordType] = useState<string>("");

  const filteredRecords = records.filter((record) => {
    const matchesSearch = record.fileName.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterType === "all" || record.recordType === filterType;
    return matchesSearch && matchesFilter;
  });

  const handleFilesSelected = (files: File[]) => {
    console.log("Files selected:", files);
  };

  const handleUpload = () => {
    // Here you would handle the actual upload
    setIsUploadOpen(false);
  };

  const handleDelete = () => {
    if (deleteRecordId) {
      setRecords(records.filter((r) => r.id !== deleteRecordId));
      setDeleteRecordId(null);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 container py-8">
        <div className="animate-fade-in">
          {/* Patient Info Card */}
          <div className="bg-card rounded-2xl border border-border shadow-card p-6 md:p-8 mb-8">
            <div className="flex flex-col md:flex-row gap-6">
              <div className="flex-shrink-0">
                <div className="h-24 w-24 rounded-2xl healthcare-gradient flex items-center justify-center shadow-healthcare">
                  <User className="h-12 w-12 text-primary-foreground" />
                </div>
              </div>
              
              <div className="flex-1 space-y-4">
                <div>
                  <h1 className="text-2xl font-bold">John Anderson</h1>
                  <p className="text-muted-foreground">Patient ID: PAT-2024-0892</p>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Age</p>
                    <p className="font-medium">54 years</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Gender</p>
                    <p className="font-medium">Male</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Blood Type</p>
                    <p className="font-medium">A+</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Last Visit</p>
                    <p className="font-medium">Jan 15, 2024</p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">Hypertension</Badge>
                  <Badge variant="secondary">Type 2 Diabetes</Badge>
                  <Badge variant="secondary">High Cholesterol</Badge>
                </div>
              </div>
            </div>
          </div>

          {/* Medical Records Section */}
          <div className="bg-card rounded-2xl border border-border shadow-card p-6 md:p-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
              <div>
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  Medical Records
                </h2>
                <p className="text-sm text-muted-foreground mt-1">
                  {records.length} records uploaded
                </p>
              </div>

              <Dialog open={isUploadOpen} onOpenChange={setIsUploadOpen}>
                <DialogTrigger asChild>
                  <Button className="gap-2">
                    <Plus className="h-4 w-4" />
                    Add Medical Record
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-lg">
                  <DialogHeader>
                    <DialogTitle>Upload Medical Record</DialogTitle>
                    <DialogDescription>
                      Upload your medical documents securely. Supported formats: PDF, Images, CSV.
                    </DialogDescription>
                  </DialogHeader>

                  <div className="py-4 space-y-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">Record Type</label>
                      <Select value={selectedRecordType} onValueChange={setSelectedRecordType}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select record type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ECG">ECG Report</SelectItem>
                          <SelectItem value="Lab">Lab Results</SelectItem>
                          <SelectItem value="Report">Medical Report</SelectItem>
                          <SelectItem value="Prescription">Prescription</SelectItem>
                          <SelectItem value="Scan">Scan/Imaging</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <FileUpload
                      onFilesSelected={handleFilesSelected}
                      accept=".pdf,.jpg,.jpeg,.png,.csv"
                      multiple
                    />
                  </div>

                  <DialogFooter>
                    <Button variant="outline" onClick={() => setIsUploadOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleUpload}>Upload Records</Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>

            {/* Search and Filter */}
            <div className="flex flex-col sm:flex-row gap-3 mb-6">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search records..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="w-full sm:w-40">
                  <Filter className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Filter" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="ECG">ECG</SelectItem>
                  <SelectItem value="Lab">Lab Results</SelectItem>
                  <SelectItem value="Report">Reports</SelectItem>
                  <SelectItem value="Prescription">Prescriptions</SelectItem>
                  <SelectItem value="Scan">Scans</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Records List */}
            <div className="space-y-3">
              {filteredRecords.length === 0 ? (
                <div className="text-center py-12">
                  <FileText className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                  <p className="text-muted-foreground">No records found</p>
                </div>
              ) : (
                filteredRecords.map((record) => (
                  <RecordCard
                    key={record.id}
                    {...record}
                    onView={() => console.log("View", record.id)}
                    onDownload={() => console.log("Download", record.id)}
                    onDelete={() => setDeleteRecordId(record.id)}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteRecordId} onOpenChange={() => setDeleteRecordId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Medical Record</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this medical record? This action cannot be undone.
              The record will be permanently removed from your profile.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete Record
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <Footer />
    </div>
  );
}
