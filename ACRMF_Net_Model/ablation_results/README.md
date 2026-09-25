# Ablation Study: REN, CEN, AWG Module Analysis

## Overview

This ablation study analyzes the contribution of each module in the ACRMF-Net architecture:
- **Module 7 (REN)**: Reliability Estimation Network
- **Module 8 (CEN)**: Confidence Estimation Network
- **Module 9 (AWG)**: Adaptive Weight Generator

## Experiments Conducted

The following 7 experiments test different module combinations:

| Experiment | REN (Module 7) | CEN (Module 8) | AWG (Module 9) | Description |
|------------|----------------|----------------|----------------|-------------|
| 1. REN_Only | ✅ | ❌ | ❌ | Only reliability estimation active |
| 2. CEN_Only | ❌ | ✅ | ❌ | Only confidence estimation active |
| 3. AWG_Only | ❌ | ❌ | ✅ | Only adaptive weight generation active |
| 4. REN_CEN | ✅ | ✅ | ❌ | Reliability + Confidence (no adaptive fusion) |
| 5. CEN_AWG | ❌ | ✅ | ✅ | Confidence + Adaptive fusion (no reliability) |
| 6. REN_AWG | ✅ | ❌ | ✅ | Reliability + Adaptive fusion (no confidence) |
| 7. ALL_Modules | ✅ | ✅ | ✅ | **Full model with all modules** |

## How to Run

### Prerequisites
```bash
pip install torch numpy matplotlib seaborn tqdm
```

### Execution
```bash
cd d:\heartsense-ai-main\cardiac-disease-detection\ACRMF_Net_Model
python ablation_study_modules.py
```

### Using Real Data (Optional)
To use your actual trained model and dataset, modify the script:

```python
# Replace generate_synthetic_data() with:
from data.loaders import load_preprocessed_data

clinical, ecg, pcg, labels = load_preprocessed_data('preprocessed_dataset_full.npz')
```

## Expected Outputs

All results will be saved in the `ablation_results/` directory:

### 1. Comprehensive Visualization
**File**: `ablation_study_comprehensive.png`

A single figure containing:
- Overall accuracy comparison across all experiments
- Loss comparison
- Per-class accuracy heatmap
- Weight distribution (Wc, We, Wp)
- Reliability scores (Rc, Re, Rp)
- Confidence scores (Cc, Ce, Cp, Cf)

### 2. Individual Comparison Plots
- `accuracy_comparison.png` - Horizontal bar chart of accuracies
- `module_impact.png` - Impact analysis showing contribution of each module

### 3. Results Data
- `ablation_results_YYYYMMDD_HHMMSS.json` - Complete results in JSON format
- `ablation_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary report

## Interpretation Guide

### Expected Patterns

1. **Single Module Performance**
   - Each module alone provides baseline functionality
   - REN provides reliability-based weighting
   - CEN estimates prediction confidence
   - AWG generates adaptive fusion weights

2. **Two Module Combinations**
   - REN + CEN: Quality-aware predictions but fixed fusion weights
   - CEN + AWG: Adaptive fusion based on confidence only
   - REN + AWG: Adaptive fusion based on reliability only

3. **All Modules (Full Model)**
   - Should show **best performance**
   - Combines reliability, confidence, and adaptive weighting
   - Provides most robust and accurate predictions

### Key Metrics to Analyze

#### 1. Accuracy Improvement
```
Baseline (Single Module) → Dual Modules → All Modules
Expected trend: Increasing accuracy
```

#### 2. Weight Adaptation
- **Without AWG**: Weights are equal (Wc = We = Wp = 0.33)
- **With AWG**: Weights adapt based on modality quality
- **Best scenario**: Weights reflect actual modality reliability

#### 3. Confidence Calibration
- **Without CEN**: Fixed confidence values
- **With CEN**: Dynamic confidence based on prediction quality
- **Importance**: Enables uncertainty quantification

## Module Interactions

### REN (Module 7) Impact
- Estimates data quality for each modality
- Influences weight generation in AWG
- **Key output**: Rc, Re, Rp ∈ [0, 1]

### CEN (Module 8) Impact
- Estimates prediction confidence
- Considers both embeddings and predictions
- **Key output**: Cc, Ce, Cp, Cf ∈ [0, 1]

### AWG (Module 9) Impact
- Generates normalized fusion weights
- Uses both reliability and confidence
- **Constraint**: Wc + We + Wp = 1

### Information Flow
```
Input Data
    ↓
Encoders → Embeddings
    ↓
REN: Estimates reliability (Rc, Re, Rp)
    ↓
Prediction Heads → Logits
    ↓
CEN: Estimates confidence (Cc, Ce, Cp, Cf)
    ↓
AWG: Generates weights (Wc, We, Wp)
    ↓
Weighted Fusion
    ↓
Final Prediction
```

## Example Results Interpretation

### Scenario 1: REN is Critical
```
REN_Only: 0.85 accuracy
CEN_Only: 0.82 accuracy
AWG_Only: 0.80 accuracy
ALL_Modules: 0.92 accuracy
```
**Interpretation**: Reliability estimation is most important factor

### Scenario 2: Synergistic Effect
```
REN_CEN: 0.87 accuracy
CEN_AWG: 0.86 accuracy
REN_AWG: 0.88 accuracy
ALL_Modules: 0.93 accuracy
```
**Interpretation**: All modules work together synergistically

### Scenario 3: Module Redundancy
```
REN_AWG: 0.89 accuracy
ALL_Modules: 0.90 accuracy (small improvement)
```
**Interpretation**: CEN provides limited additional benefit

## Customization Options

### Modify Module Configurations
Edit `ablation_study_modules.py`:

```python
# Test only specific combinations
experiments = [
    {'name': 'Baseline', 'ren': False, 'cen': False, 'awg': False},
    {'name': 'Full_Model', 'ren': True, 'cen': True, 'awg': True}
]
```

### Change Model Hyperparameters
```python
model = AblationModel(
    embedding_dim=256,  # Increase embedding size
    num_classes=5,
    use_ren=True,
    use_cen=True,
    use_awg=True
)
```

### Use Different Datasets
```python
# Use validation set
clinical_val, ecg_val, pcg_val, labels_val = load_validation_data()
results = run_ablation_study(clinical_val, ecg_val, pcg_val, labels_val)
```

## Troubleshooting

### Issue: Python not found
**Solution**: Ensure Python 3.8+ is installed and in PATH

### Issue: Missing dependencies
```bash
pip install -r ../config/requirements.txt
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use CPU
```python
device = 'cpu'  # In ablation_study_modules.py
```

### Issue: Import errors
**Solution**: Ensure you're in the correct directory
```bash
cd d:\heartsense-ai-main\cardiac-disease-detection\ACRMF_Net_Model
```

## Citation

If you use this ablation study in your research, please cite:

```bibtex
@article{acrmfnet2026,
  title={ACRMF-Net: Adaptive Confidence-Reliability Multimodal Fusion Network for Cardiac Disease Detection},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your.email@example.com

---

**Generated**: 2026-08-29  
**Version**: 1.0.0  
**Status**: Ready for execution
