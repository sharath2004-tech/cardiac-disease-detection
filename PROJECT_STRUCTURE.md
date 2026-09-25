# Project Structure - Cardiac Disease Detection

**Last Updated:** July 27, 2026

---

## 📂 Directory Organization

```
cardiac-disease-detection/                   
                                 # ROOT PROJECT
│
├── 🗂️ SHARED RESOURCES (Both Models)
│   ├── archive/                              # PCG Dataset (CinC 2016)  
│   ├── ptb-xl-...../                         # ECG Dataset (PTB-XL)
│   ├── heart_disease_uci.csv                 # Clinical Features
│   ├── results/                              # Shared results
│   ├── local_test_results/                   # Test outputs
│   ├── papers/                               # Research papers
│   └── diagrams/                             # Architecture diagrams
│
├── 🏗️ MODEL 1: CardioM3Net (Original)
│   └── CardioM3Net_Model/
│       ├── cardiom3net/                      # Model implementation
│       ├── train_cardiom3net.py              # Training script
│       ├── CardioM3Net_Kaggle.ipynb          # Kaggle notebook
│       └── *.pdf/*.docx                      # Documentation
│
├── 🏗️ MODEL 2: ACRMF-Net (New) ⭐
│   └── ACRMF_Net_Model/
│       ├── config/                           # Configuration & logging
│       ├── data/                             # Loaders & preprocessing
│       ├── models/                           # Neural architectures
│       ├── training/                         # Training pipeline
│       ├── evaluation/                       # Metrics & evaluation
│       ├── notebooks/                        # Jupyter notebooks
│       ├── results/                          # Model-specific results
│       ├── setup_project.py                  # Setup verification
│       ├── STAGE1_COMPLETE.md                # Progress tracker
│       └── README.md                         # Model documentation
│
└── 📄 ROOT FILES
    ├── .env / .env.example                   # Environment config
    ├── .gitignore                            # Git ignore
    ├── LICENSE                               # License
    ├── README.md                             # Main documentation
    ├── PROJECT_STRUCTURE.md                  # This file
    ├── Final_Roadmap_Implementation...pdf    # ACRMF-Net roadmap
    ├── Final_Model_Document_II.docx          # Documentation
    └── FAEDL_CVD_Implementation.ipynb        # Implementation notebook
```

---

## 🎯 Key Design Decisions

### 1. **Separated Model Implementations**
- **CardioM3Net_Model/**: Original implementation preserved
- **ACRMF_Net_Model/**: New implementation from scratch
- Both models are independent and self-contained

### 2. **Shared Resources**
- **Datasets**: Stored in root, accessed by both models
- **Papers**: Research references in root `/papers`
- **Diagrams**: Architecture diagrams in root `/diagrams`

### 3. **No Web Files**
- Removed all web-related files (React, Node.js, TypeScript configs)
- Focus: Pure ML/DL implementations
- Removed: `src/`, `public/`, `server/`, `package.json`, etc.

### 4. **Independent Results**
- **CardioM3Net**: Uses root `/results` folder
- **ACRMF-Net**: Uses `/ACRMF_Net_Model/results` folder

---

## 🔄 Path Configuration

### CardioM3Net Paths
```python
# Relative to: cardiac-disease-detection/CardioM3Net_Model/
datasets → ../archive, ../ptb-xl-...
results → ../results
```

### ACRMF-Net Paths
```python
# Relative to: cardiac-disease-detection/ACRMF_Net_Model/
datasets → ../archive, ../ptb-xl-..., ../heart_disease_uci.csv
results → ./results
config → ./config
```

**Configuration automatically handles parent directory references.**

---

## 📊 Dataset Organization

All datasets remain in root for easy access:

| Dataset | Path | Size | Used By |
|---------|------|------|---------|
| **PTB-XL** | `ptb-xl-.../` | ~1GB | Both Models |
| **CinC 2016** | `archive/` | ~2GB | Both Models |
| **Clinical** | `heart_disease_uci.csv` | ~50KB | Both Models |

---

## 🚀 Running Models

### CardioM3Net
```bash
cd CardioM3Net_Model
python train_cardiom3net.py
```

### ACRMF-Net
```bash
cd ACRMF_Net_Model
python setup_project.py  # Verify setup
python train_acrmf.py    # After implementation complete
```

---

## ✅ Clean-Up Summary

**Removed:**
- ❌ Web application files (`src/`, `public/`, `server/`)
- ❌ Node.js configs (`package.json`, `tsconfig.json`, etc.)
- ❌ Build tools (`vite.config.ts`, `tailwind.config.ts`, etc.)
- ❌ Duplicate results folders
- ❌ Utility scripts not needed

**Kept:**
- ✅ All datasets (PTB-XL, CinC 2016, Clinical CSV)
- ✅ All model implementations
- ✅ Research papers and documentation
- ✅ Results and test outputs
- ✅ Git configuration

---

## 📈 Implementation Progress

### CardioM3Net
- ✅ Complete and functional
- Can be run independently

### ACRMF-Net
- ✅ Stage 1: Complete (4/4 modules)
- 🔄 Stage 2: In Progress
- ⏳ Stages 3-13: Pending (22 modules)

**Total:** 4/30 modules (13.3%)

---

## 🎯 Next Steps

1. **Complete Stage 2** (Dataset Preparation)
   - Module 5: Clinical Dataset Loader
   - Module 6: ECG Dataset Loader
   - Module 7: PCG Dataset Loader
   - Module 8: Dataset Split Module

2. **Continue through Stage 13**
   - Follow roadmap systematically
   - Test each module independently
   - Maintain documentation

---

## 📝 Notes

- Both models can coexist and run independently
- Datasets are shared to save disk space
- Each model has its own Python environment (can use same or separate)
- Results are separated by model for comparison

---

**Maintained by:** ACRMF-Net Development Team  
**Project Status:** Active Development
