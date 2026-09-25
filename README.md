# Cardiac Disease Detection Project

> **Multi-Model Cardiac Disease Detection System**  
> Comparing different deep learning architectures for cardiac disease diagnosis using multimodal data.

---

## 📁 Project Structure

```
cardiac-disease-detection/
│
├── 📊 SHARED RESOURCES
│   ├── archive/                          # CinC 2016 PCG Dataset (Heart Sounds)
│   ├── ptb-xl-.../                       # PTB-XL ECG Dataset
│   ├── heart_disease_uci.csv             # UCI Clinical Features Dataset
│   ├── results/                          # Shared results directory
│   ├── local_test_results/               # Test outputs
│   ├── papers/                           # Research papers
│   └── diagrams/                         # Architecture diagrams
│
├── 🏗️ MODEL IMPLEMENTATIONS
│   ├── CardioM3Net_Model/                # Original CardioM3Net implementation
│   │   ├── cardiom3net/                  # Model code
│   │   ├── train_cardiom3net.py          # Training script
│   │   ├── CardioM3Net_Kaggle.ipynb      # Kaggle notebook
│   │   └── *.pdf / *.docx                # Documentation
│   │
│   └── ACRMF_Net_Model/                  # ⭐ NEW: ACRMF-Net implementation
│       ├── config/                       # Configuration system
│       ├── data/                         # Data loaders & preprocessing
│       ├── models/                       # Neural network architectures
│       ├── training/                     # Training pipeline
│       ├── evaluation/                   # Evaluation & metrics
│       ├── notebooks/                    # Jupyter notebooks
│       ├── results/                      # Model-specific results
│       ├── setup_project.py              # Project setup & verification
│       └── STAGE1_COMPLETE.md            # Implementation progress
│
└── 📄 DOCUMENTATION
    ├── .env / .env.example               # Environment configuration
    ├── .gitignore                        # Git ignore rules
    ├── LICENSE                           # Project license
    ├── Final_Roadmap_Implementation...   # ACRMF-Net roadmap PDF
    ├── Final_Model_Document_II.docx      # Model documentation
    ├── FAEDL_CVD_Implementation.ipynb    # Implementation notebook
    └── README.md                         # This file
```

---

## 🔬 Models Overview

### 1. CardioM3Net (Original)
**Location:** `CardioM3Net_Model/`

**Architecture:**
- Multimodal Meta-Learning Framework
- Components: SimCLR + MAML + Federated Learning
- Features: ECG (ResNet1D) + PCG (2D CNN) + Clinical (MLP)
- Fusion: CrossAttention + ModalityGate

**Key Features:**
- Self-supervised pretraining (SimCLR)
- Meta-learning adaptation (MAML)
- Federated learning across hospitals
- Domain adaptation with GRL

---

### 2. ACRMF-Net (New Implementation) ⭐
**Location:** `ACRMF_Net_Model/`

**Architecture:**
- Adaptive Clinically-aware Reliability-aware Multimodal Fusion Network
- Components: REN + CEN + AWG + ACRMF Fusion
- Features: Clinical (13→128) + ECG (12×1000→128) + PCG (1×2000→128)

**Key Innovation:**
<cite index="1-16,1-17,1-18,1-19">
- **REN** - Reliability Estimation Network (assesses modality quality)
- **CEN** - Confidence Estimation Network (prediction confidence)
- **AWG** - Adaptive Weight Generator (dynamic modality weights: Wc+We+Wp=1)
- **ACRMF Fusion** - Main research contribution
</cite>

**Implementation Status:**
- ✅ Stage 1 Complete: Project Initialization (4/4 modules)
- ⏳ Stage 2-13 Pending: 26 modules remaining

---

## 📊 Shared Datasets

All models use the same datasets (located in root directory):

### 1. PTB-XL (ECG Data)
- **Location:** `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`
- **Size:** 21,837 ECG recordings
- **Format:** 12-lead, 100/500 Hz
- **Source:** https://physionet.org/content/ptb-xl/1.0.3/

### 2. CinC 2016 (PCG Data)
- **Location:** `archive/`
- **Content:** Heart sound recordings (training-a through training-f)
- **Format:** Audio files with annotations
- **Source:** https://physionet.org/content/challenge-2016/1.0.0/

### 3. UCI Heart Disease (Clinical Data)
- **Location:** `heart_disease_uci.csv`
- **Features:** 13 clinical features
- **Format:** CSV

---

## 🚀 Getting Started

### CardioM3Net (Original Model)

```bash
cd CardioM3Net_Model

# Install dependencies (if needed)
pip install torch numpy pandas scikit-learn wfdb scipy matplotlib shap tqdm

# Train model
python train_cardiom3net.py --epochs 30

# Or use Kaggle notebook
jupyter notebook CardioM3Net_Kaggle.ipynb
```

### ACRMF-Net (New Implementation)

```bash
cd ACRMF_Net_Model

# Verify setup
python setup_project.py

# Install dependencies
pip install -r config/requirements.txt

# Training will be available after Stage 2-12 implementation
# python train_acrmf.py --epochs 100
```

---

## 📈 Results

Results from both models are stored in their respective directories:
- **CardioM3Net:** Uses root `results/` folder
- **ACRMF-Net:** Uses `ACRMF_Net_Model/results/` folder

---

## 🔄 Implementation Roadmap (ACRMF-Net)

<cite index="1-1,1-2,1-3,1-4,1-5,1-6,1-7,1-8,1-9,1-10,1-11,1-12">
**13 Stages | 30 Modules**

- [x] **Stage 1:** Project Initialization (Modules 1-4) ✅
- [ ] **Stage 2:** Dataset Preparation (Modules 5-8)
- [ ] **Stage 3:** Data Preprocessing (Modules 9-12)
- [ ] **Stage 4:** Clinical Feature Learning (Module 13)
- [ ] **Stage 5:** ECG Feature Learning (Module 14)
- [ ] **Stage 6:** PCG Feature Learning (Module 15)
- [ ] **Stage 7:** Reliability Learning (Module 16)
- [ ] **Stage 8:** Confidence Learning (Module 17)
- [ ] **Stage 9:** Adaptive Decision Making (Module 18)
- [ ] **Stage 10:** Proposed Fusion - Main Contribution (Module 19)
- [ ] **Stage 11:** Disease Prediction (Modules 20-21)
- [ ] **Stage 12:** Model Optimization (Modules 22-25)
- [ ] **Stage 13:** Experimental Evaluation (Modules 26-30)
</cite>

**Progress:** 4/30 modules (13.3%)

---

## 📚 Documentation

- **ACRMF-Net Roadmap:** `Final_Roadmap_Implementation_of_ACRMF-Net (1).pdf`
- **Model Documentation:** `Final_Model_Document_II.docx`
- **Stage 1 Complete:** `ACRMF_Net_Model/STAGE1_COMPLETE.md`

---

## 🎯 Current Focus

**Next Implementation:** Stage 2 - Dataset Preparation
- Module 5: Clinical Dataset Loader
- Module 6: ECG Dataset Loader  
- Module 7: PCG Dataset Loader
- Module 8: Dataset Split Module

---

## 🤝 Contributing

Each model implementation is independent:
- CardioM3Net: Maintains original architecture
- ACRMF-Net: New implementation following roadmap

---

## 📝 License

See LICENSE file for details.

---

## 📧 Contact

For questions about specific implementations:
- **CardioM3Net:** Check original documentation in `CardioM3Net_Model/`
- **ACRMF-Net:** Follow roadmap in `ACRMF_Net_Model/`

---

**Last Updated:** July 27, 2026  
**Project Status:** Active Development - ACRMF-Net Implementation in Progress
