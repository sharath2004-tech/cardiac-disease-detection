# ACRMF-Net: Imbalanced Classification Training

## Overview

Production-ready training script for ACRMF-Net that handles:
- **Severe class imbalance:** 16.95x ratio (Class 0: 55.8%, Class 4: 3.3%)
- **Overfitting:** 12.25% train-validation gap
- **Target:** ≥90% validation accuracy + improved minority class performance

## Quick Start

```bash
cd ACRMF_Net_Model
python train_90plus_optimized.py
```

**Runtime:** 6-8 hours on GPU for all 8 experiments

## Current Status

| Metric | Current | Target |
|--------|---------|--------|
| Val Accuracy | 88.80% | ≥90% |
| Train Accuracy | 99.78% | <95% |
| Train-Val Gap | 12.25% | <5% |
| Class 4 F1 | Poor | ≥0.60 |

## What the Script Does

Runs 8 systematic experiments:

1. **Exp0:** Baseline (no special handling)
2. **Exp1:** Smoothed inverse frequency weights
3. **Exp2:** Effective-number weighting (β=0.9999) ⭐
4. **Exp3:** Weighted random sampling only
5. **Exp4:** Best weights + higher weight decay
6. **Exp5:** Best + label smoothing
7. **Exp6:** Best + moderate sampling
8. **Exp7:** Best + MixUp augmentation

Each experiment:
- Trains for up to 100 epochs
- Uses early stopping (patience=20)
- Saves best model based on **Macro F1** (not accuracy)
- Generates confusion matrix + classification report
- Monitors validation macro F1 with ReduceLROnPlateau

## Output Structure

```
experiments/
├── Exp0_Baseline_NoWeights/
│   ├── best_model.pth              # Best checkpoint
│   ├── results.json                # All metrics
│   ├── training_curves.png         # Plots
│   ├── confusion_matrix.png        # Confusion matrix
│   └── classification_report.txt   # Per-class metrics
├── Exp1_SmoothedInvFreq/
├── Exp2_EffectiveNum_0.9999/      # Usually best for 16.95x imbalance
├── ...
└── comparison_results.json         # All experiments compared
```

## Key Features

### 1. Class Imbalance Handling

**Effective-Number Weighting:**
```python
weight_i = (1 - beta) / (1 - beta^n_i)
where beta = 0.9999 for severe imbalance
```

**Weighted Random Sampling:**
- Minority classes sampled more frequently
- Prevents overfitting to majority class

**Class-Aware Augmentation:**
- Stronger augmentation for Classes 3 & 4 (70% prob)
- Moderate augmentation for Classes 0-2 (50% prob)

### 2. Overfitting Reduction

- **AdamW optimizer** with weight decay (1e-4 to 5e-4)
- **Label smoothing** (0.1)
- **ReduceLROnPlateau** scheduler
- **Early stopping** based on macro F1

### 3. Comprehensive Evaluation

**Per Experiment:**
- Overall accuracy
- Macro F1, Weighted F1
- Per-class F1, precision, recall
- Confusion matrix
- Training/validation curves

## Configuration

**Model:**
- ACRMF-Net (256D embeddings)
- ~2-3M parameters

**Optimizer:**
- AdamW (lr=1e-4, weight_decay varies)

**Training:**
- Batch size: 32
- Max epochs: 100 per experiment
- Early stopping: 20 epochs patience
- Scheduler: ReduceLROnPlateau (monitor macro F1)

**Data Split (Stratified):**
- Train: 11,370 samples (70%)
- Val: 2,437 samples (15%)
- Test: 2,437 samples (15%)

## Class Distribution

```
Class 0: 9,069 samples (55.8%) ████████████████████████
Class 1: 2,532 samples (15.6%) ███████
Class 2: 2,400 samples (14.8%) ██████
Class 3: 1,708 samples (10.5%) ████
Class 4:   535 samples ( 3.3%) █   ← Critical minority class

Imbalance Ratio: 16.95x
```

## Expected Results

**Conservative:**
- Val accuracy: 90-91%
- Macro F1: 0.87-0.89
- Class 4 F1: 0.60-0.70
- Train-val gap: <6%

**Optimistic:**
- Val accuracy: 91-92%
- Macro F1: 0.89-0.91
- Class 4 F1: 0.70-0.80
- Train-val gap: <5%

## Loading Best Model

```python
import torch
from test_complete_model import ACRMFNet

# Load model
model = ACRMFNet(num_classes=5, embedding_dim=256)
checkpoint = torch.load('experiments/Exp7_BestConfig_MixUp/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Check performance
print(f"Macro F1: {checkpoint['val_metrics']['macro_f1']:.4f}")
print(f"Val Acc: {checkpoint['val_metrics']['accuracy']:.2f}%")
print(f"Class 4 F1: {checkpoint['val_metrics']['per_class_f1'][4]:.4f}")
```

## Troubleshooting

**If validation accuracy < 90%:**
1. Check which experiment had best macro F1
2. Review Class 4 performance in classification reports
3. Examine confusion matrices - which classes confused?
4. Check if still overfitting (train-val gap > 8%)

**Common Issues:**
- GPU memory error → Reduce batch size to 16
- Dataset not found → Run `prepare_dataset_once.py` first
- Slow training → Check GPU usage with `nvidia-smi`

## Files

**Essential:**
- `train_90plus_optimized.py` - Main training (run this)
- `test_complete_model.py` - Model architecture
- `preprocessed_dataset_full.npz` - Preprocessed data

**Generated:**
- `experiments/` - All experiment results
- `experiments/comparison_results.json` - Summary

## Requirements

```
torch >= 1.10
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

## Model Architecture

ACRMF-Net combines three modalities:
- **Clinical:** 13 features → 256D embedding
- **ECG:** 12 leads × 1000 samples → 256D embedding
- **PCG:** 128×128 spectrogram → 256D embedding

Uses adaptive reliability-weighted fusion with:
- Reliability Estimation Network (REN)
- Confidence Estimation Network (CEN)
- Adaptive Weight Generator (AWG)
- ACRMF Fusion module

## Citation

[Add paper citation when published]

---

**Status:** Production Ready
**Last Updated:** 2024
