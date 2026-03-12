import express from 'express';
import mongoose from 'mongoose';
import { authenticate } from '../middleware/auth.js';
import Assessment from '../models/Assessment.js';
import Patient from '../models/Patient.js';

const router = express.Router();

const isValidObjectId = (id) => mongoose.Types.ObjectId.isValid(id);

// Whitelist of fields a client is allowed to set/update on a patient profile
const PATIENT_ALLOWED_FIELDS = [
  'dateOfBirth', 'gender', 'bloodType', 'height', 'weight',
  'medicalHistory', 'allergies', 'medications', 'emergencyContact',
];
const pickAllowed = (body, fields) =>
  Object.fromEntries(fields.filter((f) => f in body).map((f) => [f, body[f]]));

// @route   GET /api/patients
// @desc    Doctors/admins see all patients; patients see only their own profile
// @access  Private
router.get('/', authenticate, async (req, res) => {
  try {
    if (req.user.role === 'patient') {
      const patient = await Patient.findOne({ userId: req.user.id });
      return res.json(patient ? [patient] : []);
    }
    // doctor or admin
    const patients = await Patient.find().populate('userId', 'name email');
    res.json(patients);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   GET /api/patients/:id
// @desc    Get patient by ID — only own profile or doctor/admin
// @access  Private
router.get('/:id', authenticate, async (req, res) => {
  try {
    if (!isValidObjectId(req.params.id)) {
      return res.status(400).json({ message: 'Invalid patient ID' });
    }

    const patient = await Patient.findById(req.params.id).populate('userId', 'name email');

    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }

    // Patients may only view their own record
    if (req.user.role === 'patient' && patient.userId._id.toString() !== req.user.id) {
      return res.status(403).json({ message: 'Access denied' });
    }

    res.json(patient);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   POST /api/patients
// @desc    Create a new patient profile for the current user
// @access  Private
router.post('/', authenticate, async (req, res) => {
  try {
    const fields = pickAllowed(req.body, PATIENT_ALLOWED_FIELDS);
    const patient = await Patient.create({
      userId: req.user.id,
      ...fields,
    });

    res.status(201).json(patient);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   PUT /api/patients/:id
// @desc    Update own patient profile (or doctor/admin can update any)
// @access  Private
router.put('/:id', authenticate, async (req, res) => {
  try {
    if (!isValidObjectId(req.params.id)) {
      return res.status(400).json({ message: 'Invalid patient ID' });
    }

    const patient = await Patient.findById(req.params.id);
    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }

    // Patients may only update their own record
    if (req.user.role === 'patient' && patient.userId.toString() !== req.user.id) {
      return res.status(403).json({ message: 'Access denied' });
    }

    const fields = pickAllowed(req.body, PATIENT_ALLOWED_FIELDS);
    const updated = await Patient.findByIdAndUpdate(
      req.params.id,
      { ...fields, updatedAt: Date.now() },
      { new: true, runValidators: true }
    );

    res.json(updated);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   DELETE /api/patients/:id
// @desc    Delete own patient profile (or admin); cascades assessments
// @access  Private
router.delete('/:id', authenticate, async (req, res) => {
  try {
    if (!isValidObjectId(req.params.id)) {
      return res.status(400).json({ message: 'Invalid patient ID' });
    }

    const patient = await Patient.findById(req.params.id);
    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }

    // Only the patient themselves or an admin may delete
    if (req.user.role === 'patient' && patient.userId.toString() !== req.user.id) {
      return res.status(403).json({ message: 'Access denied' });
    }
    if (req.user.role === 'doctor') {
      return res.status(403).json({ message: 'Doctors cannot delete patient records' });
    }

    // Cascade-delete associated assessments
    await Assessment.deleteMany({ patientId: req.params.id });
    await Patient.findByIdAndDelete(req.params.id);

    res.json({ message: 'Patient and associated assessments removed' });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;
