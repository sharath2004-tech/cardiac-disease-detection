# CardioM3Net — Multimodal Meta-Learning Framework for Cardiovascular Disease Diagnosis

> **HeartSense AI** · End-to-end cardiac risk assessment combining ECG, PCG (heart sounds) and clinical tabular data with self-supervised pretraining, MAML meta-learning, federated learning and explainable AI.

---

## Architecture Overview

### Diagram 1 — Full Model Architecture

![CardioM3Net Architecture](cardiom3net_results/diagram1_architecture.png)

### Diagram 2 — Training Pipeline, Deployment & Key Novelty

![CardioM3Net Pipeline](cardiom3net_results/diagram2_pipeline.png)

---

## Key Contributions

| Component | Details |
|---|---|
| **Trimodal Fusion** | ECG (256-d) + PCG (128-d) + Clinical (64-d) fused via CrossAttention + learned ModalityGate (floor=0.20, diversity λ=0.1) |
| **ECG Encoder** | ResNet1D + Self-Attention `(B, 12, T) → 256-dim` |
| **PCG Encoder** | 4-layer 2D CNN on log-mel spectrogram `(B, 1, 64, 99) → 128-dim` |
| **Clinical Encoder** | MLP with BatchNorm + Dropout `→ 64-dim` |
| **SimCLR Pretraining** | Contrastive self-supervised ECG encoder with NT-Xent loss (Phase 1) |
| **MAML** | 5-way 5-shot fast cross-dataset adaptation · inner SGD + outer Adam (Phase 3) |
| **Federated Learning** | Weighted FedAvg across 3 simulated hospital clients · 5 rounds (Phase 4) |
| **Domain Adaptation** | GRL-based adversarial discriminator on ECG features (Phase 2) |
| **Explainable AI** | SHAP clinical importance + ECG saliency + modality gate weight visualization (Phase 5) |

---

## 5-Phase Training Pipeline

```
Phase 1 → SimCLR ECG Pretraining      (self-supervised, NT-Xent loss)
Phase 2 → Supervised Multi-Task       (Binary + Disease + Severity + GRL Domain Adaptation)
Phase 3 → MAML Meta-Learning          (fast cross-dataset adaptation)
Phase 4 → Federated Learning          (Weighted FedAvg, 3 hospital clients)
Phase 5 → Explainability              (SHAP + saliency + modality weights)
```

---

## Datasets

> **Datasets are NOT included in this repository** due to size. Download them separately:

| Dataset | Use | Download |
|---|---|---|
| **PTB-XL** (21,837 ECGs, 12-lead, 100/500 Hz) | ECG + clinical features | [physionet.org/content/ptb-xl](https://physionet.org/content/ptb-xl/1.0.3/) |
| **CinC 2016** (heart sound recordings) | PCG branch | [physionet.org/content/challenge-2016](https://physionet.org/content/challenge-2016/1.0.0/) |

After downloading, place them at:
```
cardiac-disease-detection/
  ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
  archive/          ← CinC 2016 training-a .. training-f folders
```

---

## Quick Start

### Python (Model Training)

```sh
# Install dependencies
pip install torch numpy pandas scikit-learn wfdb scipy matplotlib shap tqdm

# Run full 5-phase training
cd cardiac-disease-detection
py train_cardiom3net.py --epochs 30 --pcg_archive_dir archive

# Skip PCG (bimodal ECG + Clinical only)
py train_cardiom3net.py --epochs 30 --skip_pcg

# Regenerate architecture diagrams
py generate_diagrams.py
```

### Web Application (Frontend + Backend)

```sh
# Install frontend dependencies
npm install

# Install backend dependencies
npm run server:install

# Configure MongoDB Atlas — update server/.env:
#   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/heartsense-ai
#   JWT_SECRET=your-secure-random-string

# Start backend (terminal 1)
npm run server

# Start frontend (terminal 2)
npm run dev
```

---

## Project Structure

```
cardiac-disease-detection/
  cardiom3net/
    models/
      cardiom3net.py          ← Full assembled model
      ecg_encoder.py          ← ResNet1D + Self-Attention
      pcg_encoder.py          ← 2D CNN on log-mel spectrogram
      clinical_encoder.py     ← MLP encoder
      fusion.py               ← CrossAttentionFusion + ModalityGate
      multitask_head.py       ← Binary / Disease / Severity heads
      domain_adaptation.py    ← GRL adversarial discriminator
    training/
      self_supervised.py      ← SimCLR pretraining
      supervised.py           ← Multi-task training + diversity loss
      maml_trainer.py         ← MAML meta-learning
      federated.py            ← Weighted FedAvg
    data/
      ecg_loader.py           ← PTB-XL loader
      pcg_loader.py           ← CinC 2016 loader + log-mel
      clinical_loader.py      ← Clinical feature builder
      multimodal_dataset.py   ← Unified dataset class
    explainability/
      shap_analysis.py        ← SHAP + modality weight plot
      gradcam_1d.py           ← ECG saliency
  train_cardiom3net.py        ← Master training script
  generate_diagrams.py        ← Architecture diagram generator
  CardioM3Net_Kaggle.ipynb    ← Kaggle-ready notebook
  src/                        ← React frontend
  server/                     ← Node.js + Express backend
```

---

## Web Application Stack

### Frontend
- React 18 · TypeScript · Vite · shadcn/ui · Tailwind CSS · React Router

### Backend
- Node.js · Express.js · MongoDB Atlas · Mongoose · JWT · bcryptjs

### Environment Variables

**Frontend** (`.env`):
```
VITE_API_URL=http://localhost:5000/api
```

**Backend** (`server/.env`):
```
PORT=5000
MONGODB_URI=your-mongodb-connection-string
JWT_SECRET=your-jwt-secret
FRONTEND_URL=http://localhost:5173
```

---

## Available Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start frontend dev server |
| `npm run build` | Build frontend for production |
| `npm run test` | Run frontend tests |
| `npm run server` | Start backend dev server |
| `npm run server:install` | Install backend dependencies |

---

## API Documentation

See [server/README.md](server/README.md) for full API reference.
