import axios from 'axios';
import type {
  AuthResponse,
  CreateAssessmentData,
  CreatePatientData,
  LoginData,
  RegisterData,
  UpdateAssessmentData,
  UpdatePatientData,
  User,
} from './types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  timeout: 15000, // 15 s — prevent requests hanging indefinitely
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if it exists
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle response errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear token and redirect to login
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/signin';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  register: (data: RegisterData) =>
    api.post<AuthResponse>('/auth/register', data),
  
  login: (data: LoginData) =>
    api.post<AuthResponse>('/auth/login', data),
  
  getCurrentUser: () =>
    api.get<User>('/auth/me'),
};

// Patients API
export const patientsAPI = {
  getAll: () =>
    api.get('/patients'),
  
  getById: (id: string) =>
    api.get(`/patients/${id}`),
  
  create: (data: CreatePatientData) =>
    api.post('/patients', data),
  
  update: (id: string, data: UpdatePatientData) =>
    api.put(`/patients/${id}`, data),
  
  delete: (id: string) =>
    api.delete(`/patients/${id}`),
};

// Assessments API
export const assessmentsAPI = {
  getAll: () =>
    api.get('/assessments'),
  
  getByPatientId: (patientId: string) =>
    api.get(`/assessments/patient/${patientId}`),
  
  getById: (id: string) =>
    api.get(`/assessments/${id}`),
  
  create: (data: CreateAssessmentData) =>
    api.post('/assessments', data),
  
  update: (id: string, data: UpdateAssessmentData) =>
    api.put(`/assessments/${id}`, data),
  
  delete: (id: string) =>
    api.delete(`/assessments/${id}`),
};

// Health check
export const healthCheck = () =>
  api.get('/health');

export default api;
