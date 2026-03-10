import express from 'express';
import { authenticate } from '../middleware/auth.js';
import Patient from '../models/Patient.js';

const router = express.Router();

// @route   GET /api/patients
// @desc    Get all patients (for doctors/admin)
// @access  Private
router.get('/', authenticate, async (req, res) => {
  try {
    const patients = await Patient.find().populate('userId', 'name email');
    res.json(patients);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   GET /api/patients/:id
// @desc    Get patient by ID
// @access  Private
router.get('/:id', authenticate, async (req, res) => {
  try {
    const patient = await Patient.findById(req.params.id).populate('userId', 'name email');
    
    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }
    
    res.json(patient);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   POST /api/patients
// @desc    Create a new patient profile
// @access  Private
router.post('/', authenticate, async (req, res) => {
  try {
    const patient = await Patient.create({
      userId: req.user.id,
      ...req.body
    });
    
    res.status(201).json(patient);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   PUT /api/patients/:id
// @desc    Update patient profile
// @access  Private
router.put('/:id', authenticate, async (req, res) => {
  try {
    const patient = await Patient.findByIdAndUpdate(
      req.params.id,
      { ...req.body, updatedAt: Date.now() },
      { new: true, runValidators: true }
    );
    
    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }
    
    res.json(patient);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   DELETE /api/patients/:id
// @desc    Delete patient
// @access  Private
router.delete('/:id', authenticate, async (req, res) => {
  try {
    const patient = await Patient.findByIdAndDelete(req.params.id);
    
    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }
    
    res.json({ message: 'Patient removed' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
