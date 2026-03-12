import express from 'express';
import mongoose from 'mongoose';
import { authenticate } from '../middleware/auth.js';
import Assessment from '../models/Assessment.js';
import Patient from '../models/Patient.js';

const router = express.Router();

const isValidObjectId = (id) => mongoose.Types.ObjectId.isValid(id);

// Whitelist of fields clients may supply when creating/updating an assessment
const ASSESSMENT_ALLOWED_FIELDS = [
  'patientId', 'assessmentType', 'symptoms', 'vitalSigns',
  'riskScore', 'riskLevel', 'notes', 'recommendations',
  'assessmentDate', 'followUpRequired', 'followUpDate',
];
const pickAllowed = (body, fields) =>
  Object.fromEntries(fields.filter((f) => f in body).map((f) => [f, body[f]]));

// Helper: verify the requesting user has access to the given assessment
const canAccess = (user, assessment) => {
  if (user.role === 'doctor' || user.role === 'admin') return true;
  return assessment.doctorId?.toString() === user.id;
};

// @route   GET /api/assessments
// @desc    Doctors/admins see all; patients see only assessments for their own profile
// @access  Private
router.get('/', authenticate, async (req, res) => {
  try {
    if (req.user.role === 'patient') {
      const patient = await Patient.findOne({ userId: req.user.id });
      if (!patient) return res.json([]);
      const assessments = await Assessment.find({ patientId: patient._id })
        .populate('doctorId', 'name email')
        .sort({ assessmentDate: -1 });
      return res.json(assessments);
    }
    // doctor or admin
    const assessments = await Assessment.find()
      .populate('patientId')
      .populate('doctorId', 'name email')
      .sort({ assessmentDate: -1 });
    res.json(assessments);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   GET /api/assessments/patient/:patientId
// @desc    Get assessments by patient ID
// @access  Private
router.get('/patient/:patientId', authenticate, async (req, res) => {
  try {
    if (!isValidObjectId(req.params.patientId)) {
      return res.status(400).json({ message: 'Invalid patient ID' });
    }

    // Patients may only query their own assessments
    if (req.user.role === 'patient') {
      const patient = await Patient.findOne({ userId: req.user.id });
      if (!patient || patient._id.toString() !== req.params.patientId) {
        return res.status(403).json({ message: 'Access denied' });
      }
    }

    const assessments = await Assessment.find({ patientId: req.params.patientId })
      .populate('doctorId', 'name email')
      .sort({ assessmentDate: -1 });

    res.json(assessments);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   GET /api/assessments/:id
// @desc    Get assessment by ID
// @access  Private
router.get('/:id', authenticate, async (req, res) => {
  try {
    if (!isValidObjectId(req.params.id)) {
      return res.status(400).json({ message: 'Invalid assessment ID' });
    }

    const assessment = await Assessment.findById(req.params.id)
      .populate('patientId')
      .populate('doctorId', 'name email');

    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }

    if (!canAccess(req.user, assessment)) {
      return res.status(403).json({ message: 'Access denied' });
    }

    res.json(assessment);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   POST /api/assessments
// @desc    Create a new assessment (doctors/admins only)
// @access  Private
router.post('/', authenticate, async (req, res) => {
  try {
    if (req.user.role === 'patient') {
      return res.status(403).json({ message: 'Only doctors can create assessments' });
    }

    const fields = pickAllowed(req.body, ASSESSMENT_ALLOWED_FIELDS);

    if (!fields.patientId || !isValidObjectId(fields.patientId)) {
      return res.status(400).json({ message: 'A valid patientId is required' });
    }

    const patientExists = await Patient.exists({ _id: fields.patientId });
    if (!patientExists) {
      return res.status(404).json({ message: 'Patient not found' });
    }

    const assessment = await Assessment.create({
      ...fields,
      doctorId: req.user.id,
    });

    res.status(201).json(assessment);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// @route   PUT /api/assessments/:id
// @desc    Update assessment (the creating doctor or admin only)
// @access  Private
router.put('/:id', authenticate, async (req, res) => {
  try {
    if (!isValidObjectId(req.params.id)) {
      return res.status(400).json({ message: 'Invalid assessment ID' });
    }

    const assessment = await Assessment.findById(req.params.id);
    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }

    if (!canAccess(req.user, assessment)) {
      return res.status(403).json({ message: 'Access denied' });
    }

    // Prevent reassigning patientId or doctorId via update
    const { patientId, doctorId, ...rest } = req.body;
    const fields = pickAllowed(rest, ASSESSMENT_ALLOWED_FIELDS.filter(
      (f) => f !== 'patientId'
    ));

    const updated = await Assessment.findByIdAndUpdate(
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

// @route   DELETE /api/assessments/:id
// @desc    Delete assessment (the creating doctor or admin only)
// @access  Private
router.delete('/:id', authenticate, async (req, res) => {
  try {
    if (!isValidObjectId(req.params.id)) {
      return res.status(400).json({ message: 'Invalid assessment ID' });
    }

    const assessment = await Assessment.findById(req.params.id);
    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }

    if (!canAccess(req.user, assessment)) {
      return res.status(403).json({ message: 'Access denied' });
    }

    await Assessment.findByIdAndDelete(req.params.id);
    res.json({ message: 'Assessment removed' });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;
