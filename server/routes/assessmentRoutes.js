import express from 'express';
import { authenticate } from '../middleware/auth.js';
import Assessment from '../models/Assessment.js';

const router = express.Router();

// @route   GET /api/assessments
// @desc    Get all assessments
// @access  Private
router.get('/', authenticate, async (req, res) => {
  try {
    const assessments = await Assessment.find()
      .populate('patientId')
      .populate('doctorId', 'name email')
      .sort({ assessmentDate: -1 });
    
    res.json(assessments);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   GET /api/assessments/patient/:patientId
// @desc    Get assessments by patient ID
// @access  Private
router.get('/patient/:patientId', authenticate, async (req, res) => {
  try {
    const assessments = await Assessment.find({ patientId: req.params.patientId })
      .populate('doctorId', 'name email')
      .sort({ assessmentDate: -1 });
    
    res.json(assessments);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   GET /api/assessments/:id
// @desc    Get assessment by ID
// @access  Private
router.get('/:id', authenticate, async (req, res) => {
  try {
    const assessment = await Assessment.findById(req.params.id)
      .populate('patientId')
      .populate('doctorId', 'name email');
    
    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }
    
    res.json(assessment);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   POST /api/assessments
// @desc    Create a new assessment
// @access  Private
router.post('/', authenticate, async (req, res) => {
  try {
    const assessment = await Assessment.create({
      ...req.body,
      doctorId: req.user.id
    });
    
    res.status(201).json(assessment);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   PUT /api/assessments/:id
// @desc    Update assessment
// @access  Private
router.put('/:id', authenticate, async (req, res) => {
  try {
    const assessment = await Assessment.findByIdAndUpdate(
      req.params.id,
      { ...req.body, updatedAt: Date.now() },
      { new: true, runValidators: true }
    );
    
    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }
    
    res.json(assessment);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// @route   DELETE /api/assessments/:id
// @desc    Delete assessment
// @access  Private
router.delete('/:id', authenticate, async (req, res) => {
  try {
    const assessment = await Assessment.findByIdAndDelete(req.params.id);
    
    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }
    
    res.json({ message: 'Assessment removed' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
