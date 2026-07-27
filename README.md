# ACRMF-Net: Adaptive Clinically-aware Reliability-aware Multimodal Fusion Network

> **Cardiac Disease Detection** using Clinical Features, ECG, and PCG (Phonocardiogram) data with Adaptive Multimodal Fusion

---

## Overview

ACRMF-Net is a novel multimodal deep learning architecture for cardiac disease detection that intelligently fuses:
- **Clinical Features** (tabular patient data)
- **ECG Signals** (electrocardiography)
- **PCG Signals** (phonocardiography - heart sounds)

The model uses adaptive fusion with reliability and confidence estimation to handle modality quality variations in real-world clinical scenarios.

---

## Architecture Components

### Core Modules (30 Total)

<cite index="1-1,1-2,1-3,1-4,1-5">**Stage 1 — Project Initialization**
- Module 1-4: Project Folder Structure, Configuration, Requirements, Logging</cite>

<cite index="1-5,1-6,1-7,1-8,1-9">**Stage 2 — Dataset Preparation**
- Module 5-8: Clinical Dataset Loader, ECG Dataset Loader, PCG Dataset Loader, Dataset Split Module</cite>

<cite index="1-9,1-10,1-11,1-12">**Stage 3 — Data Preprocessing**
- Module 9-12: Clinical Preprocessor, ECG Preprocessor, PCG Preprocessor, Data Quality Assessment Module</cite>

<cite index="1-13">**Stage 4 — Clinical Feature Learning**
- Module 13: Clinical Encoder → (B,128)</cite>

<cite index="1-14">**Stage 5 — ECG Feature Learning**
- Module 14: ECG Encoder → (B,128)</cite>

<cite index="1-15">**Stage 6 — PCG Feature Learning**
- Module 15: PCG Encoder → (B,128)</cite>

<cite index="1-16">**Stage 7 — Reliability Learning**
- Module 16: Reliability Estimation Network (REN) → Rc, Re, Rp</cite>

<cite index="1-17">**Stage 8 — Confidence Learning**
- Module 17: Confidence Estimation Network (CEN) → Cc, Ce, Cp, Cf</cite>

<cite index="1-18">**Stage 9 — Adaptive Decision Making**
- Module 18: Adaptive Weight Generator (AWG) → Wc, We, Wp (sum=1)</cite>

<cite index="1-19">**Stage 10 — Proposed Fusion (Main Contribution)**
- Module 19: ACRMF Fusion Module → Fusion Feature (B,128)</cite>

<cite index="1-20,1-21">**Stage 11 — Disease Prediction**
- Module 20-21: Decision Head, Probability Estimator</cite>

<cite index="1-21,1-22,1-23,1-24">**Stage 12 — Model Optimization**
- Module 22-25: Composite Loss, Training Engine, Validation Engine, Testing Engine
- Includes: AdamW, Scheduler, Early Stopping, Checkpoint Saving</cite>

<cite index="1-24,1-25,1-26,1-27,1-28,1-29">**Stage 13 — Experimental Evaluation**
- Module 26-30: Performance Evaluation, Explainability, Ablation Study, Statistical Analysis, Result Visualization
- Produces: ROC, PR Curve, SHAP, Confusion Matrix, Reliability plots, Calibration plots</cite>

---

## Project Structure

```
cardiac-disease-detection/
├── .env                          # Environment configuration
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── LICENSE                       # Project license
├── README.md                     # This file
│
├── config/                       # Stage 1: Configuration
│   ├── __init__.py
│   ├── config.py                 # Module 2: Configuration settings
│   ├── requirements.txt          # Module 3: Dependencies
│   └── logging_config.py         # Module 4: Logging setup
│
├── data/                         # Stage 2-3: Data handling
│   ├── __init__.py
│   ├── loaders/                  # Stage 2: Dataset loaders
│   │   ├── __init__.py
│   │   ├── clinical_loader.py    # Module 5
│   │   ├── ecg_loader.py         # Module 6
│   │   ├── pcg_loader.py         # Module 7
│   │   └── dataset_split.py      # Module 8
│   │
│   └── preprocessing/            # Stage 3: Preprocessing
│       ├── __init__.py
│       ├── clinical_preprocessor.py  # Module 9
│       ├── ecg_preprocessor.py       # Module 10
│       ├── pcg_preprocessor.py       # Module 11
│       └── quality_assessment.py     # Module 12
│
├── models/                       # Stage 4-11: Neural architectures
│   ├── __init__.py
│   ├── encoders/                 # Stage 4-6: Feature extractors
│   │   ├── __init__.py
│   │   ├── clinical_encoder.py   # Module 13
│   │   ├── ecg_encoder.py        # Module 14
│   │   └── pcg_encoder.py        # Module 15
│   │
│   ├── reliability/              # Stage 7-9: Adaptive components
│   │   ├── __init__.py
│   │   ├── ren.py                # Module 16: Reliability Estimation Network
│   │   ├── cen.py                # Module 17: Confidence Estimation Network
│   │   └── awg.py                # Module 18: Adaptive Weight Generator
│   │
│   ├── fusion/                   # Stage 10: Core contribution
│   │   ├── __init__.py
│   │   └── acrmf_fusion.py       # Module 19: ACRMF Fusion (MAIN)
│   │
│   ├── heads/                    # Stage 11: Output layers
│   │   ├── __init__.py
│   │   ├── decision_head.py      # Module 20
│   │   └── probability_estimator.py  # Module 21
│   │
│   └── acrmf_net.py              # Complete assembled model
│
├── training/                     # Stage 12: Training pipeline
│   ├── __init__.py
│   ├── losses.py                 # Module 22: Composite loss
│   ├── trainer.py                # Module 23: Training engine
│   ├── validator.py              # Module 24: Validation engine
│   └── tester.py                 # Module 25: Testing engine
│
├── evaluation/                   # Stage 13: Experiments
│   ├── __init__.py
│   ├── metrics.py                # Module 26: Performance evaluation
│   ├── explainability.py         # Module 27: SHAP, saliency
│   ├── ablation.py               # Module 28: Ablation studies
│   ├── statistical_tests.py      # Module 29: Statistical analysis
│   └── visualizations.py         # Module 30: Result plots
│
├── train_acrmf.py                # Main training script
├── test_acrmf.py                 # Testing script
├── infer_acrmf.py                # Inference script
│
├── notebooks/                    # Jupyter notebooks
│   └── ACRMF_Net_Analysis.ipynb
│
├── results/                      # Experiment outputs
│   ├── checkpoints/              # Saved models
│   ├── logs/                     # Training logs
│   ├── figures/                  # Thesis figures
│   └── metrics/                  # Performance metrics
│
├── datasets/                     # Data storage (not in repo)
│   ├── ptb-xl/                   # ECG dataset
│   ├── cinc2016/                 # PCG dataset (archive/)
│   └── clinical/                 # Clinical features
│
└── docs/                         # Documentation
    ├── Final_Roadmap_Implementation_of_ACRMF-Net (1).pdf
    └── Final_Model_Document_II.docx
```

---

## Datasets

### Required Datasets (Download Separately)

1. **PTB-XL** (ECG Data)
   - 21,837 ECG recordings, 12-lead, 100/500 Hz
   - Download: https://physionet.org/content/ptb-xl/1.0.3/
   - Place in: `datasets/ptb-xl/`

2. **CinC Challenge 2016** (PCG Data)
   - Heart sound recordings from training-a through training-f
   - Download: https://physionet.org/content/challenge-2016/1.0.0/
   - Already in: `archive/` directory

3. **Clinical Features**
   - UCI Heart Disease Dataset: `heart_disease_uci.csv`
   - Already in project root

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd cardiac-disease-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r config/requirements.txt
```

---

## Usage

### 1. Training

```bash
# Train complete ACRMF-Net model
python train_acrmf.py --epochs 100 --batch_size 32 --lr 0.001

# Train with specific configuration
python train_acrmf.py --config config/config.py
```

### 2. Testing

```bash
# Test trained model
python test_acrmf.py --checkpoint results/checkpoints/best_model.pth
```

### 3. Inference

```bash
# Run inference on new data
python infer_acrmf.py --input <path-to-data> --checkpoint results/checkpoints/best_model.pth
```

---

## Key Features

✅ <cite index="1-16">**Reliability Estimation Network (REN)** - Assesses quality of each modality</cite>

✅ <cite index="1-17">**Confidence Estimation Network (CEN)** - Estimates prediction confidence per modality</cite>

✅ <cite index="1-18">**Adaptive Weight Generator (AWG)** - Dynamically balances modality contributions (Wc+We+Wp=1)</cite>

✅ <cite index="1-19">**ACRMF Fusion Module** - Novel adaptive fusion mechanism (Main Research Contribution)</cite>

✅ <cite index="1-29">**Comprehensive Evaluation** - ROC, PR Curve, SHAP, Confusion Matrix, Reliability plots, Calibration plots</cite>

---

## Implementation Progress

- [x] Stage 1: Project Initialization
- [ ] Stage 2: Dataset Preparation
- [ ] Stage 3: Data Preprocessing
- [ ] Stage 4: Clinical Feature Learning
- [ ] Stage 5: ECG Feature Learning
- [ ] Stage 6: PCG Feature Learning
- [ ] Stage 7: Reliability Learning
- [ ] Stage 8: Confidence Learning
- [ ] Stage 9: Adaptive Decision Making
- [ ] Stage 10: Proposed Fusion (Main Contribution)
- [ ] Stage 11: Disease Prediction
- [ ] Stage 12: Model Optimization
- [ ] Stage 13: Experimental Evaluation

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{acrmf-net-2024,
  title={ACRMF-Net: Adaptive Clinically-aware Reliability-aware Multimodal Fusion Network for Cardiac Disease Detection},
  author={Your Name},
  year={2024}
}
```

---

## License

See LICENSE file for details.

---

## Contact

For questions or collaboration, please open an issue or contact: [your-email]

---

**Note**: This is a research implementation. Follow the 13-stage roadmap systematically for complete implementation.
