"""
FEMT-Net Training Script — Kaggle GPU Ready
============================================
Upload this file to Kaggle, then run in a notebook cell:

    !python /kaggle/working/train_kaggle.py

Or with custom args:

    !python /kaggle/working/train_kaggle.py --epochs 20 --batch_size 64 --lr 0.001

Prerequisites — Add these datasets via Kaggle "Add Data":
  1. m0hamedyousry/ptb-xl-a-large-publicly-available-ecg-dataset
  2. redwankarimsony/heart-disease-data
"""

import argparse
import ast
import copy
import os
import pickle
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  CLI ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="FEMT-Net Kaggle Training")
    p.add_argument("--epochs",     type=int,   default=10,    help="Training epochs")
    p.add_argument("--batch_size", type=int,   default=32,    help="Batch size")
    p.add_argument("--lr",         type=float, default=1e-3,  help="Learning rate")
    p.add_argument("--max_records",type=int,   default=21837, help="Max ECG records to load")
    p.add_argument("--fed_rounds", type=int,   default=5,     help="Federated learning rounds")
    p.add_argument("--fed_clients",type=int,   default=3,     help="Number of federated clients")
    p.add_argument("--local_epochs",type=int,  default=2,     help="Local epochs per client per round")
    p.add_argument("--output_dir", type=str,   default="models", help="Where to save models")
    p.add_argument("--seed",       type=int,   default=42,    help="Random seed")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════════════
def setup_device_and_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    return device


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
ECG_FIXED_LEN = 5000
UCI_FEATURE_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


def find_ptbxl_root():
    candidates = [
        Path('/kaggle/input/ptb-xl-a-large-publicly-available-ecg-dataset/'
             'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'),
        Path('/kaggle/input/ptb-xl-a-large-publicly-available-ecg-dataset'),
        Path('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'),
        Path.cwd() / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
    ]
    for c in candidates:
        if (c / 'ptbxl_database.csv').exists():
            return c
    # fallback: recursive search
    for search_root in [Path('/kaggle/input'), Path.cwd()]:
        if search_root.exists():
            for csv in search_root.rglob('ptbxl_database.csv'):
                return csv.parent
    return None


def find_uci_csv():
    candidates = [
        Path('/kaggle/input/heart-disease-data/heart_disease_uci.csv'),
        Path('heart_disease_uci.csv'),
        Path.cwd() / 'heart_disease_uci.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    for search_root in [Path('/kaggle/input'), Path.cwd()]:
        if search_root.exists():
            for csv in search_root.rglob('heart_disease_uci.csv'):
                return csv
    return None


def normalize_ecg(signal):
    signal = signal.astype(np.float32)
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-6
    return (signal - mean) / std


def load_data(max_records):
    """Load PTB-XL ECG + UCI clinical data. Returns dataset bundle dict."""

    # ── PTB-XL ──────────────────────────────────────────────────────────
    dataset_root = find_ptbxl_root()
    if dataset_root is None:
        raise FileNotFoundError(
            "PTB-XL dataset not found. Add 'm0hamedyousry/ptb-xl-a-large-publicly-"
            "available-ecg-dataset' via Kaggle Add Data."
        )
    print(f"PTB-XL root: {dataset_root}")

    df_raw = pd.read_csv(dataset_root / 'ptbxl_database.csv', index_col='ecg_id')
    df_raw['scp_codes'] = df_raw['scp_codes'].apply(ast.literal_eval)

    scp_df = pd.read_csv(dataset_root / 'scp_statements.csv', index_col=0)
    scp_df = scp_df[scp_df['diagnostic'] == 1]

    def aggregate_diagnostic(scp_codes):
        classes = []
        for code in scp_codes.keys():
            if code in scp_df.index:
                classes.append(scp_df.loc[code, 'diagnostic_class'])
        return sorted(set(classes))

    df_raw['diagnostic_superclass'] = df_raw['scp_codes'].apply(aggregate_diagnostic)
    df_raw['target'] = df_raw['diagnostic_superclass'].apply(
        lambda c: 0 if 'NORM' in c else 1 if len(c) > 0 else -1
    )
    df_raw = df_raw[df_raw['target'] != -1].copy()

    # Balanced sampling
    normal_df = df_raw[df_raw['target'] == 0]
    disease_df = df_raw[df_raw['target'] == 1]
    per_class = max_records // 2
    sampled = pd.concat([
        normal_df.sample(n=min(len(normal_df), per_class), random_state=42),
        disease_df.sample(n=min(len(disease_df), per_class), random_state=42),
    ]).sample(frac=1.0, random_state=42)

    # Pick resolution
    use_hr = False
    if 'filename_hr' in sampled.columns:
        use_hr = (dataset_root / f"{sampled.iloc[0]['filename_hr']}.hea").exists()
    record_field = 'filename_hr' if use_hr else 'filename_lr'
    target_length = ECG_FIXED_LEN if use_hr else 1000

    ptbxl_meta_cols = [c for c in ['age', 'sex', 'height', 'weight'] if c in df_raw.columns]
    ptbxl_meta_df = sampled[ptbxl_meta_cols].apply(pd.to_numeric, errors='coerce')

    ecg_signals, meta_rows, label_list, loaded_ids = [], [], [], []
    skipped = 0
    for ecg_id, row in sampled.iterrows():
        try:
            signal, _ = wfdb.rdsamp(str(dataset_root / row[record_field]))
            signal = normalize_ecg(signal)
            if signal.shape[0] >= target_length:
                signal = signal[:target_length]
            else:
                pad = np.zeros((target_length - signal.shape[0], signal.shape[1]), np.float32)
                signal = np.vstack([signal, pad])
            ecg_signals.append(signal.T.astype(np.float32))
            meta_rows.append(ptbxl_meta_df.loc[ecg_id].values.astype(np.float32))
            label_list.append(int(row['target']))
            loaded_ids.append(ecg_id)
        except Exception:
            skipped += 1

    ecg_array = np.stack(ecg_signals)
    labels = np.array(label_list, dtype=np.int64)
    hz = "500 Hz" if use_hr else "100 Hz"
    print(f"Loaded {len(labels)} ECG records ({hz}) | leads={ecg_array.shape[1]} "
          f"| timesteps={ecg_array.shape[2]} | skipped={skipped}")

    # ── UCI Heart Disease ───────────────────────────────────────────────
    uci_path = find_uci_csv()
    if uci_path is not None:
        uci_df = pd.read_csv(uci_path)
        lbl_col = 'num' if 'num' in uci_df.columns else 'target'
        uci_df[lbl_col] = (uci_df[lbl_col] > 0).astype(int)
        uci_feature_cols = [c for c in UCI_FEATURE_COLS if c in uci_df.columns]
        uci_feat_df = uci_df[uci_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        idx = np.arange(len(ecg_array)) % len(uci_feat_df)
        clinical_array = uci_feat_df.values[idx].astype(np.float32)
        clinical_df = pd.DataFrame(clinical_array, columns=uci_feature_cols)
        clinical_feature_names = uci_feature_cols
        print(f"UCI clinical: {len(uci_feat_df)} rows → cyclic-mapped to {len(ecg_array)} ECGs")
    else:
        clinical_df = pd.DataFrame(meta_rows, columns=ptbxl_meta_cols, index=loaded_ids)
        clinical_feature_names = ptbxl_meta_cols
        print("UCI not found — using PTB-XL metadata as clinical features")

    print(f"ECG shape: {ecg_array.shape} | Clinical shape: {clinical_df.shape} "
          f"| Labels: {np.bincount(labels)}")

    return {
        'ecg_array': ecg_array,
        'clinical_df': clinical_df,
        'clinical_feature_names': clinical_feature_names,
        'labels': labels,
        'signal_length': int(ecg_array.shape[2]),
        'num_leads': int(ecg_array.shape[1]),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════
class MultimodalCardiacDataset(Dataset):
    def __init__(self, ecg, clinical, labels):
        self.ecg = torch.tensor(ecg, dtype=torch.float32)
        self.clinical = torch.tensor(clinical, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ecg[idx], self.clinical[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════
class ECGCNNEncoder(nn.Module):
    def __init__(self, num_leads):
        super().__init__()
        self.conv1 = nn.Conv1d(num_leads, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        return x


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class FEMTNet(nn.Module):
    def __init__(self, num_leads, clinical_dim, d_model=128, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.ecg_encoder = ECGCNNEncoder(num_leads)
        self.clinical_encoder = ClinicalEncoder(clinical_dim)
        self.ecg_projection = nn.Linear(128, d_model)
        self.clinical_projection = nn.Linear(32, d_model)
        self.modality_gate = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, ecg, clinical):
        ecg_emb = self.ecg_encoder(ecg)
        clin_emb = self.clinical_encoder(clinical)
        gate_input = torch.cat([ecg_emb, clin_emb], dim=1)
        modality_w = torch.softmax(self.modality_gate(gate_input), dim=1)
        ecg_tok = self.ecg_projection(ecg_emb) * modality_w[:, 0:1]
        clin_tok = self.clinical_projection(clin_emb) * modality_w[:, 1:2]
        tokens = torch.stack([ecg_tok, clin_tok], dim=1)
        transformed = self.transformer(tokens)
        pooled = transformed.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, modality_w, transformed


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS & TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    m = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }
    m['roc_auc'] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    return m


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for ecg_b, clin_b, lbl_b in loader:
        ecg_b, clin_b, lbl_b = ecg_b.to(device), clin_b.to(device), lbl_b.to(device)
        with torch.set_grad_enabled(training):
            logits, _, _ = model(ecg_b, clin_b)
            loss = criterion(logits, lbl_b)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(lbl_b)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.extend(lbl_b.cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    return total_loss / len(loader.dataset), compute_metrics(all_labels, all_probs)


# ═══════════════════════════════════════════════════════════════════════════
#  FEDERATED LEARNING
# ═══════════════════════════════════════════════════════════════════════════
def weighted_fedavg(state_dicts, weights):
    total = float(sum(weights))
    averaged = {}
    for key in state_dicts[0]:
        acc = None
        for sd, w in zip(state_dicts, weights):
            val = sd[key].detach().clone().float() * (w / total)
            acc = val if acc is None else acc + val
        averaged[key] = acc.type_as(state_dicts[0][key])
    return averaged


def run_federated(model, train_dataset, test_loader, args, device):
    criterion = nn.CrossEntropyLoss()
    indices = np.random.permutation(len(train_dataset))
    splits = np.array_split(indices, args.fed_clients)
    client_subsets = [Subset(train_dataset, s.tolist()) for s in splits if len(s) > 0]

    fed_history = []
    for rnd in range(1, args.fed_rounds + 1):
        local_states, local_weights = [], []
        for cid, subset in enumerate(client_subsets, 1):
            local_model = copy.deepcopy(model)
            local_opt = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-5)
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
            for _ in range(args.local_epochs):
                run_epoch(local_model, loader, criterion, device, local_opt)
            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(len(subset))
            print(f"  Round {rnd} | Client {cid}: {len(subset)} samples")

        global_state = weighted_fedavg(local_states, local_weights)
        model.load_state_dict(global_state)
        _, metrics = run_epoch(model, test_loader, criterion, device)
        fed_history.append(metrics)
        print(f"  Round {rnd} global → AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

    return model, fed_history


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
def save_training_plots(history, output_dir, title="FEMT-Net"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_title(f'{title} Loss'); axes[0].legend()
    axes[1].plot(epochs, history['train_auc'], label='Train')
    axes[1].plot(epochs, history['val_auc'], label='Val')
    axes[1].set_title(f'{title} AUC'); axes[1].legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = setup_device_and_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    data = load_data(args.max_records)

    ecg_array = data['ecg_array']
    clinical_df = data['clinical_df']
    labels = data['labels']
    num_leads = data['num_leads']
    clinical_feature_names = data['clinical_feature_names']

    # ── Preprocessing ─────────────────────────────────────────────────
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=labels)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    train_clin = scaler.fit_transform(imputer.fit_transform(clinical_df.iloc[train_idx])).astype(np.float32)
    test_clin = scaler.transform(imputer.transform(clinical_df.iloc[test_idx])).astype(np.float32)

    train_ds = MultimodalCardiacDataset(ecg_array[train_idx], train_clin, labels[train_idx])
    test_ds  = MultimodalCardiacDataset(ecg_array[test_idx],  test_clin,  labels[test_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    clinical_dim = train_clin.shape[1]
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Clinical dim: {clinical_dim}")

    # ── Centralized Training ──────────────────────────────────────────
    print("\n" + "="*60)
    print("CENTRALIZED TRAINING")
    print("="*60)

    model = FEMTNet(num_leads, clinical_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    best_state = copy.deepcopy(model.state_dict())
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        t_loss, t_m = run_epoch(model, train_loader, criterion, device, optimizer)
        v_loss, v_m = run_epoch(model, test_loader,  criterion, device)
        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_auc'].append(t_m['roc_auc'])
        history['val_auc'].append(v_m['roc_auc'])

        vauc = np.nan_to_num(v_m['roc_auc'], nan=-1.0)
        if vauc > best_auc:
            best_auc = vauc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"T-Loss: {t_loss:.4f} V-Loss: {v_loss:.4f} | "
              f"T-AUC: {t_m['roc_auc']:.4f} V-AUC: {v_m['roc_auc']:.4f} | "
              f"V-Acc: {v_m['accuracy']:.4f} V-F1: {v_m['f1']:.4f}")

    # Load best & evaluate
    model.load_state_dict(best_state)
    _, central_metrics = run_epoch(model, test_loader, criterion, device)
    print("\nCentralized results:")
    for k, v in central_metrics.items():
        print(f"  {k}: {v:.4f}")

    save_training_plots(history, args.output_dir, title="Central FEMT-Net")

    # Save centralized model
    central_path = os.path.join(args.output_dir, 'femtnet_central.pth')
    torch.save({
        'state_dict': model.state_dict(),
        'num_leads': num_leads,
        'clinical_dim': clinical_dim,
        'clinical_feature_names': clinical_feature_names,
        'results': central_metrics,
    }, central_path)
    print(f"Saved: {central_path}")

    # ── Federated Training ────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"FEDERATED TRAINING ({args.fed_clients} clients, {args.fed_rounds} rounds)")
    print("="*60)

    fed_model = FEMTNet(num_leads, clinical_dim).to(device)
    fed_model, fed_history = run_federated(fed_model, train_ds, test_loader, args, device)

    _, fed_metrics = run_epoch(fed_model, test_loader, criterion, device)
    print("\nFederated results:")
    for k, v in fed_metrics.items():
        print(f"  {k}: {v:.4f}")

    fed_path = os.path.join(args.output_dir, 'femtnet_federated.pth')
    torch.save({
        'state_dict': fed_model.state_dict(),
        'num_leads': num_leads,
        'clinical_dim': clinical_dim,
        'clinical_feature_names': clinical_feature_names,
        'results': fed_metrics,
    }, fed_path)
    print(f"Saved: {fed_path}")

    # Save preprocessors
    with open(os.path.join(args.output_dir, 'clinical_imputer.pkl'), 'wb') as f:
        pickle.dump(imputer, f)
    with open(os.path.join(args.output_dir, 'clinical_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.output_dir, 'clinical_feature_names.pkl'), 'wb') as f:
        pickle.dump(clinical_feature_names, f)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Metric':<12} {'Central':>10} {'Federated':>10}")
    print("-" * 34)
    for k in central_metrics:
        print(f"{k:<12} {central_metrics[k]:>10.4f} {fed_metrics[k]:>10.4f}")
    print("="*60)
    print("Training complete! All models saved to:", args.output_dir)


if __name__ == "__main__":
    main()
