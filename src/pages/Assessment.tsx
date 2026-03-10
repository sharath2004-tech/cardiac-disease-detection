import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { User, Activity, FileText, ChevronRight, ChevronLeft, Check } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { FileUpload } from "@/components/ui/file-upload";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

const steps = [
  { id: 1, title: "Basic Details", icon: User },
  { id: 2, title: "Health Parameters", icon: Activity },
  { id: 3, title: "Medical Records", icon: FileText },
];

const previousRecords = [
  { id: "1", name: "ECG Report - Jan 2024", type: "ECG" },
  { id: "2", name: "Blood Test - Jan 2024", type: "Lab" },
  { id: "3", name: "Cardiac MRI - Dec 2023", type: "Scan" },
];

export default function Assessment() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedRecords, setSelectedRecords] = useState<string[]>([]);
  const [usePreviousECG, setUsePreviousECG] = useState(false);

  const handleNext = () => {
    if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    } else {
      // Submit and navigate to results
      navigate("/dashboard");
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const toggleRecord = (id: string) => {
    setSelectedRecords((prev) =>
      prev.includes(id) ? prev.filter((r) => r !== id) : [...prev, id]
    );
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 container py-8">
        <div className="max-w-3xl mx-auto animate-fade-in">
          {/* Page Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-2">Heart Risk Assessment</h1>
            <p className="text-muted-foreground">
              Complete the form below for an AI-powered cardiac risk analysis
            </p>
          </div>

          {/* Progress Steps */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              {steps.map((step, index) => (
                <div key={step.id} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <div
                      className={cn(
                        "h-12 w-12 rounded-full flex items-center justify-center border-2 transition-all duration-300",
                        currentStep >= step.id
                          ? "healthcare-gradient border-transparent text-primary-foreground"
                          : "border-border bg-card text-muted-foreground"
                      )}
                    >
                      {currentStep > step.id ? (
                        <Check className="h-5 w-5" />
                      ) : (
                        <step.icon className="h-5 w-5" />
                      )}
                    </div>
                    <span
                      className={cn(
                        "text-sm mt-2 font-medium hidden sm:block",
                        currentStep >= step.id ? "text-foreground" : "text-muted-foreground"
                      )}
                    >
                      {step.title}
                    </span>
                  </div>
                  {index < steps.length - 1 && (
                    <div
                      className={cn(
                        "h-0.5 w-16 sm:w-24 md:w-32 mx-2 transition-colors duration-300",
                        currentStep > step.id ? "bg-primary" : "bg-border"
                      )}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Form Card */}
          <div className="bg-card rounded-2xl border border-border shadow-card p-6 md:p-8">
            {/* Step 1: Basic Details */}
            {currentStep === 1 && (
              <div className="space-y-6 animate-fade-in">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="firstName">First Name</Label>
                    <Input id="firstName" placeholder="Enter first name" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lastName">Last Name</Label>
                    <Input id="lastName" placeholder="Enter last name" />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="age">Age</Label>
                    <Input id="age" type="number" placeholder="Years" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="gender">Gender</Label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="male">Male</SelectItem>
                        <SelectItem value="female">Female</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="bloodType">Blood Type</Label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="A+">A+</SelectItem>
                        <SelectItem value="A-">A-</SelectItem>
                        <SelectItem value="B+">B+</SelectItem>
                        <SelectItem value="B-">B-</SelectItem>
                        <SelectItem value="O+">O+</SelectItem>
                        <SelectItem value="O-">O-</SelectItem>
                        <SelectItem value="AB+">AB+</SelectItem>
                        <SelectItem value="AB-">AB-</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input id="email" type="email" placeholder="your@email.com" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phone">Phone Number</Label>
                    <Input id="phone" type="tel" placeholder="+1 (555) 000-0000" />
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: Health Parameters */}
            {currentStep === 2 && (
              <div className="space-y-6 animate-fade-in">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="systolic">Systolic BP (mmHg)</Label>
                    <Input id="systolic" type="number" placeholder="e.g., 120" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="diastolic">Diastolic BP (mmHg)</Label>
                    <Input id="diastolic" type="number" placeholder="e.g., 80" />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="cholesterol">Cholesterol (mg/dL)</Label>
                    <Input id="cholesterol" type="number" placeholder="e.g., 200" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="glucose">Blood Glucose (mg/dL)</Label>
                    <Input id="glucose" type="number" placeholder="e.g., 100" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="heartRate">Resting Heart Rate (bpm)</Label>
                    <Input id="heartRate" type="number" placeholder="e.g., 72" />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="height">Height (cm)</Label>
                    <Input id="height" type="number" placeholder="e.g., 175" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="weight">Weight (kg)</Label>
                    <Input id="weight" type="number" placeholder="e.g., 70" />
                  </div>
                </div>

                <div className="space-y-4">
                  <Label>Lifestyle Factors</Label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-3 p-4 rounded-lg bg-secondary/50">
                      <Checkbox id="smoking" />
                      <Label htmlFor="smoking" className="cursor-pointer">Currently smoking</Label>
                    </div>
                    <div className="flex items-center space-x-3 p-4 rounded-lg bg-secondary/50">
                      <Checkbox id="alcohol" />
                      <Label htmlFor="alcohol" className="cursor-pointer">Regular alcohol consumption</Label>
                    </div>
                    <div className="flex items-center space-x-3 p-4 rounded-lg bg-secondary/50">
                      <Checkbox id="exercise" />
                      <Label htmlFor="exercise" className="cursor-pointer">Regular exercise (3+ times/week)</Label>
                    </div>
                    <div className="flex items-center space-x-3 p-4 rounded-lg bg-secondary/50">
                      <Checkbox id="familyHistory" />
                      <Label htmlFor="familyHistory" className="cursor-pointer">Family history of heart disease</Label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Medical Records */}
            {currentStep === 3 && (
              <div className="space-y-6 animate-fade-in">
                <div className="space-y-4">
                  <Label>Select Previous Records for Analysis</Label>
                  <div className="space-y-3">
                    {previousRecords.map((record) => (
                      <div
                        key={record.id}
                        className={cn(
                          "flex items-center space-x-3 p-4 rounded-lg border transition-all cursor-pointer",
                          selectedRecords.includes(record.id)
                            ? "border-primary bg-primary/5"
                            : "border-border bg-secondary/30 hover:bg-secondary/50"
                        )}
                        onClick={() => toggleRecord(record.id)}
                      >
                        <Checkbox
                          checked={selectedRecords.includes(record.id)}
                          onCheckedChange={() => toggleRecord(record.id)}
                        />
                        <div className="flex-1">
                          <p className="font-medium">{record.name}</p>
                          <p className="text-sm text-muted-foreground">{record.type}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-4 rounded-lg border border-primary/30 bg-primary/5">
                  <div className="flex items-center space-x-3">
                    <Checkbox
                      id="usePreviousECG"
                      checked={usePreviousECG}
                      onCheckedChange={(checked) => setUsePreviousECG(checked as boolean)}
                    />
                    <div>
                      <Label htmlFor="usePreviousECG" className="cursor-pointer font-medium">
                        Use previous ECG for analysis
                      </Label>
                      <p className="text-sm text-muted-foreground mt-0.5">
                        Compare with your most recent ECG report
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <Label>Upload New ECG or Medical Records</Label>
                  <FileUpload
                    onFilesSelected={(files) => console.log("Files:", files)}
                    accept=".pdf,.jpg,.jpeg,.png,.csv"
                    multiple
                  />
                </div>
              </div>
            )}

            {/* Navigation Buttons */}
            <div className="flex items-center justify-between mt-8 pt-6 border-t border-border">
              <Button
                variant="outline"
                onClick={handleBack}
                disabled={currentStep === 1}
                className="gap-2"
              >
                <ChevronLeft className="h-4 w-4" />
                Back
              </Button>

              <Button onClick={handleNext} className="gap-2">
                {currentStep === 3 ? (
                  <>
                    Submit Assessment
                    <Check className="h-4 w-4" />
                  </>
                ) : (
                  <>
                    Continue
                    <ChevronRight className="h-4 w-4" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
