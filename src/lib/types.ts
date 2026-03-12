// User types
export interface User {
  _id: string;
  email: string;
  name: string;
  role: 'patient' | 'doctor' | 'admin';
  createdAt: string;
  updatedAt: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
  // role is intentionally omitted — server always assigns 'patient' on self-registration
}

export interface LoginData {
  email: string;
  password: string;
}

export interface AuthResponse {
  _id: string;
  name: string;
  email: string;
  role: string;
  token: string;
}

// Patient types
export interface MedicalHistoryItem {
  condition: string;
  diagnosedDate: Date | string;
  notes?: string;
}

export interface Medication {
  name: string;
  dosage: string;
  frequency: string;
  startDate: Date | string;
}

export interface EmergencyContact {
  name: string;
  relationship: string;
  phone: string;
}

export interface Patient {
  _id: string;
  userId: string;
  dateOfBirth: Date | string;
  gender: 'male' | 'female' | 'other';
  bloodType?: 'A+' | 'A-' | 'B+' | 'B-' | 'AB+' | 'AB-' | 'O+' | 'O-';
  height?: number;
  weight?: number;
  medicalHistory?: MedicalHistoryItem[];
  allergies?: string[];
  medications?: Medication[];
  emergencyContact?: EmergencyContact;
  createdAt: string;
  updatedAt: string;
}

export interface CreatePatientData {
  userId: string;
  dateOfBirth: Date | string;
  gender: 'male' | 'female' | 'other';
  bloodType?: 'A+' | 'A-' | 'B+' | 'B-' | 'AB+' | 'AB-' | 'O+' | 'O-';
  height?: number;
  weight?: number;
  medicalHistory?: MedicalHistoryItem[];
  allergies?: string[];
  medications?: Medication[];
  emergencyContact?: EmergencyContact;
}

export type UpdatePatientData = Partial<CreatePatientData>;

// Assessment types
export interface Symptom {
  name: string;
  severity: number;
  duration: string;
}

export interface VitalSigns {
  bloodPressure: {
    systolic: number;
    diastolic: number;
  };
  heartRate: number;
  temperature: number;
  oxygenSaturation: number;
  respiratoryRate: number;
}

export interface Assessment {
  _id: string;
  patientId: string;
  assessmentType: 'initial' | 'follow-up' | 'routine' | 'emergency';
  symptoms?: Symptom[];
  vitalSigns?: VitalSigns;
  riskScore?: number;
  riskLevel?: 'low' | 'moderate' | 'high' | 'critical';
  notes?: string;
  recommendations?: string[];
  doctorId?: string;
  assessmentDate: Date | string;
  followUpRequired?: boolean;
  followUpDate?: Date | string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateAssessmentData {
  patientId: string;
  assessmentType?: 'initial' | 'follow-up' | 'routine' | 'emergency';
  symptoms?: Symptom[];
  vitalSigns?: VitalSigns;
  riskScore?: number;
  riskLevel?: 'low' | 'moderate' | 'high' | 'critical';
  notes?: string;
  recommendations?: string[];
  assessmentDate?: Date | string;
  followUpRequired?: boolean;
  followUpDate?: Date | string;
}

export type UpdateAssessmentData = Partial<CreateAssessmentData>;
