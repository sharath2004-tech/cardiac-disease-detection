"""
CardioM3Net — Master Training Script
=====================================
Complete end-to-end pipeline:
  Phase 1: Self-Supervised ECG Pretraining (SimCLR)
  Phase 2: Multi-Task Supervised Training
  Phase 3: MAML Meta-Learning
  Phase 4: Domain Adaptation (built into Phase 2)
  Phase 5: Federated Learning
  Phase 6: Explainability Analysis

For Kaggle:
    !pip install -q wfdb shap
    !python train_cardiom3net.py

Customize:
    !python train_cardiom3net.py --epochs 50 --ssl_epochs 30 --batch_size 64
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

# ── Path resolution: works locally AND on Kaggle input datasets ────────────
def _setup_path():
    """
    On Kaggle, input dataset files are read-only under /kaggle/input/.
    We copy the cardiom3net/ package to /kaggle/working/ so Python can
    import from it, then add /kaggle/working/ to sys.path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_src = os.path.join(script_dir, 'cardiom3net')

    # Already importable (local run) — just add script dir
    if os.path.isdir(pkg_src):
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        return script_dir

    # Kaggle: script in /kaggle/input/... — copy package to /kaggle/working/
    kaggle_working = '/kaggle/working'
    if os.path.isdir(kaggle_working):
        dst_pkg = os.path.join(kaggle_working, 'cardiom3net')
        # Search for cardiom3net/ anywhere under /kaggle/input/
        for root, dirs, _ in os.walk('/kaggle/input'):
            if 'cardiom3net' in dirs:
                src = os.path.join(root, 'cardiom3net')
                if not os.path.isdir(dst_pkg):
                    shutil.copytree(src, dst_pkg)
                    print(f"[setup] Copied cardiom3net/ from {src} → {dst_pkg}")
                if kaggle_working not in sys.path:
                    sys.path.insert(0, kaggle_working)
                return kaggle_working
        print("[setup] WARNING: cardiom3net/ not found under /kaggle/input/")
        print("[setup] Upload cardiom3net/ folder alongside train_cardiom3net.py in your dataset.")
        sys.exit(1)

    # Fallback: add cwd
    sys.path.insert(0, os.getcwd())
    return os.getcwd()

_setup_path()

from cardiom3net.config import Config
from cardiom3net.utils import (seed_everything, get_device, plot_training_curves,
                                plot_confusion_matrices, plot_roc_curves)
from cardiom3net.data.ecg_loader import load_ptbxl
from cardiom3net.data.clinical_loader import load_uci_clinical
from cardiom3net.data.multimodal_dataset import (MultimodalCardiacDataset, MAMLTaskDataset)
from cardiom3net.models.cardiom3net import CardioM3Net
from cardiom3net.training.self_supervised import pretrain_ecg_encoder
from cardiom3net.training.supervised import train_supervised, run_epoch, MultiTaskLoss
from cardiom3net.training.maml_trainer import train_maml
from cardiom3net.training.federated import run_federated
from cardiom3net.explainability.gradcam_1d import plot_ecg_saliency
from cardiom3net.explainability.shap_analysis import run_shap_analysis, plot_modality_weights

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="CardioM3Net Training")
    # Data
    p.add_argument("--max_records", type=int, default=21837)
    p.add_argument("--use_100hz", action="store_true", default=True)
    p.add_argument("--test_size", type=float, default=0.2)
    # Self-supervised
    p.add_argument("--ssl_epochs", type=int, default=20)
    p.add_argument("--ssl_batch_size", type=int, default=64)
    p.add_argument("--skip_ssl", action="store_true", default=False)
    # Supervised
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    # MAML
    p.add_argument("--maml_episodes", type=int, default=200)
    p.add_argument("--skip_maml", action="store_true", default=False)
    # Federated
    p.add_argument("--fed_rounds", type=int, default=5)
    p.add_argument("--fed_clients", type=int, default=3)
    p.add_argument("--skip_fed", action="store_true", default=False)
    # Output
    p.add_argument("--output_dir", type=str, default="cardiom3net_results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        max_ecg_records=args.max_records,
        ssl_epochs=args.ssl_epochs,
        ssl_batch_size=args.ssl_batch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        maml_episodes=args.maml_episodes,
        fed_rounds=args.fed_rounds,
        fed_clients=args.fed_clients,
        output_dir=args.output_dir,
    )

    seed_everything(cfg.seed)
    device = get_device()
    os.makedirs(cfg.output_dir, exist_ok=True)
    start_time = time.time()

    # ══════════════════════════════════════════════════════════════════
    #  DATA LOADING
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    ptbxl = load_ptbxl(max_records=cfg.max_ecg_records, use_100hz=args.use_100hz)
    if ptbxl is None:
        print("FATAL: PTB-XL dataset not found.")
        print("  On Kaggle: Add 'm0hamedyousry/ptb-xl-a-large-publicly-available-ecg-dataset'")
        print("  Local: Place the ptb-xl folder in the project directory.")
        sys.exit(1)

    ecg_array = ptbxl['ecg_array']
    binary_labels = ptbxl['binary_labels']
    disease_labels = ptbxl['disease_labels']
    severity_labels = ptbxl['severity_labels']
    domain_labels = ptbxl['domain_labels']
    num_leads = ptbxl['num_leads']
    signal_length = ptbxl['signal_length']
    cfg.ecg_length = signal_length
    cfg.num_leads = num_leads

    # Clinical features
    clinical_array, clinical_names = load_uci_clinical(cfg.uci_feature_cols, len(ecg_array))
    if clinical_array is None:
        # Fallback to PTB-XL metadata
        meta_df = ptbxl['meta_df'].apply(pd.to_numeric, errors='coerce').fillna(0)
        clinical_array = meta_df.values.astype(np.float32)
        clinical_names = ptbxl['meta_cols']
        print("  Using PTB-XL metadata as clinical features (fallback)")

    clinical_dim = clinical_array.shape[1]

    # ══════════════════════════════════════════════════════════════════
    #  PREPROCESSING & SPLITS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    indices = np.arange(len(ecg_array))
    train_idx, test_idx = train_test_split(
        indices, test_size=cfg.test_size, random_state=cfg.seed, stratify=binary_labels
    )

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    train_clinical = scaler.fit_transform(imputer.fit_transform(clinical_array[train_idx])).astype(np.float32)
    test_clinical = scaler.transform(imputer.transform(clinical_array[test_idx])).astype(np.float32)

    train_ecg = ecg_array[train_idx]
    test_ecg = ecg_array[test_idx]

    train_ds = MultimodalCardiacDataset(
        train_ecg, train_clinical,
        binary_labels[train_idx], disease_labels[train_idx],
        severity_labels[train_idx], domain_labels[train_idx],
        augment=True,
    )
    test_ds = MultimodalCardiacDataset(
        test_ecg, test_clinical,
        binary_labels[test_idx], disease_labels[test_idx],
        severity_labels[test_idx], domain_labels[test_idx],
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")
    print(f"  ECG: ({num_leads}, {signal_length}) | Clinical dim: {clinical_dim}")
    print(f"  Disease classes: {cfg.disease_classes}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: SELF-SUPERVISED PRETRAINING
    # ══════════════════════════════════════════════════════════════════
    pretrained_encoder = None
    if not args.skip_ssl:
        pretrained_encoder = pretrain_ecg_encoder(train_ecg, cfg, device)
    else:
        print("\n  Skipping SSL pretraining (--skip_ssl)")

    # ══════════════════════════════════════════════════════════════════
    #  BUILD MODEL
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("BUILDING CardioM3Net")
    print("=" * 60)

    model = CardioM3Net(
        num_leads=num_leads,
        clinical_dim=clinical_dim,
        ecg_embed_dim=cfg.ecg_embed_dim,
        clinical_embed_dim=cfg.clinical_embed_dim,
        fusion_dim=cfg.fusion_dim,
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
        transformer_ff_dim=cfg.transformer_ff_dim,
        num_disease_classes=cfg.num_disease_classes,
        num_severity_classes=cfg.severity_bins,
        num_domains=3,
        da_lambda=cfg.da_lambda,
        dropout=cfg.dropout,
    ).to(device)

    # Load pretrained ECG encoder weights
    if pretrained_encoder is not None:
        model.ecg_encoder.load_state_dict(pretrained_encoder.state_dict())
        print("  Loaded self-supervised ECG encoder weights")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: MULTI-TASK SUPERVISED TRAINING (with Domain Adaptation)
    # ══════════════════════════════════════════════════════════════════
    model, history, supervised_results = train_supervised(
        model, train_loader, test_loader, cfg, device, use_domain=True
    )

    # Save training plots
    plot_training_curves(history, cfg.output_dir, title="CardioM3Net")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3: MAML META-LEARNING
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_maml:
        maml_task_ds = MAMLTaskDataset(
            train_ecg, train_clinical, disease_labels[train_idx],
            n_way=cfg.maml_n_way, k_shot=cfg.maml_k_shot, q_query=cfg.maml_q_query,
        )
        if len(maml_task_ds.available_classes) >= 2:
            model, maml_losses = train_maml(model, maml_task_ds, cfg, device)
        else:
            print("  Insufficient classes for MAML — skipping")
    else:
        print("\n  Skipping MAML (--skip_maml)")

    # Re-evaluate after MAML
    criterion = MultiTaskLoss(cfg.lambda_binary, cfg.lambda_disease, cfg.lambda_severity)
    post_maml = run_epoch(model, test_loader, criterion, device, use_domain=True)
    print(f"\n  Post-MAML eval:")
    print(f"    Binary Acc: {post_maml['binary']['accuracy']:.4f} | "
          f"AUC: {post_maml['binary']['roc_auc']:.4f}")
    print(f"    Disease Acc: {post_maml['disease']['accuracy']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 5: FEDERATED LEARNING
    # ══════════════════════════════════════════════════════════════════
    fed_model = None
    fed_results = None
    if not args.skip_fed:
        fed_model_init = CardioM3Net(
            num_leads=num_leads, clinical_dim=clinical_dim,
            ecg_embed_dim=cfg.ecg_embed_dim, clinical_embed_dim=cfg.clinical_embed_dim,
            fusion_dim=cfg.fusion_dim, transformer_heads=cfg.transformer_heads,
            transformer_layers=cfg.transformer_layers, transformer_ff_dim=cfg.transformer_ff_dim,
            num_disease_classes=cfg.num_disease_classes, num_severity_classes=cfg.severity_bins,
            num_domains=3, da_lambda=cfg.da_lambda, dropout=cfg.dropout,
        ).to(device)

        # Start federated from pretrained encoder too
        if pretrained_encoder is not None:
            fed_model_init.ecg_encoder.load_state_dict(pretrained_encoder.state_dict())

        fed_model, fed_history, fed_results = run_federated(
            fed_model_init, train_ds, test_loader, cfg, device
        )
    else:
        print("\n  Skipping Federated Learning (--skip_fed)")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 6: EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 6: Explainability Analysis")
    print("=" * 60)

    # Grad-CAM ECG saliency
    plot_ecg_saliency(model, test_ecg[0], test_clinical[0], device, cfg.output_dir)

    # SHAP clinical feature importance
    run_shap_analysis(
        train_clinical, binary_labels[train_idx], test_clinical,
        clinical_names, cfg.output_dir
    )

    # Modality weights
    model.eval()
    all_weights = []
    with torch.no_grad():
        for batch in test_loader:
            _, mw, _ = model(batch['ecg'].to(device), batch['clinical'].to(device))
            all_weights.append(mw.cpu().numpy())
    all_weights = np.vstack(all_weights)
    plot_modality_weights(all_weights, cfg.output_dir)

    # Confusion matrices
    plot_confusion_matrices(
        supervised_results['binary_labels'],
        (np.array(supervised_results['binary_probs']) >= 0.5).astype(int),
        supervised_results['disease_labels'],
        supervised_results['disease_preds'],
        cfg.disease_classes, cfg.output_dir,
    )

    # ROC curves
    roc_data = {
        'CardioM3Net (Central)': (
            supervised_results['binary_labels'],
            supervised_results['binary_probs'],
        ),
    }
    if fed_results is not None:
        roc_data['CardioM3Net (Federated)'] = (
            fed_results['binary_labels'],
            fed_results['binary_probs'],
        )
    plot_roc_curves(roc_data, cfg.output_dir)

    # ══════════════════════════════════════════════════════════════════
    #  SAVE MODELS & RESULTS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Central model
    torch.save({
        'state_dict': model.state_dict(),
        'config': cfg.__dict__,
        'clinical_feature_names': clinical_names,
        'results': {
            'binary': supervised_results['binary'],
            'disease': supervised_results['disease'],
            'severity': supervised_results['severity'],
        },
    }, os.path.join(cfg.output_dir, 'cardiom3net_central.pth'))
    print(f"  Saved: {cfg.output_dir}/cardiom3net_central.pth")

    # Federated model
    if fed_model is not None:
        torch.save({
            'state_dict': fed_model.state_dict(),
            'config': cfg.__dict__,
            'clinical_feature_names': clinical_names,
            'results': {
                'binary': fed_results['binary'],
                'disease': fed_results['disease'],
                'severity': fed_results['severity'],
            },
        }, os.path.join(cfg.output_dir, 'cardiom3net_federated.pth'))
        print(f"  Saved: {cfg.output_dir}/cardiom3net_federated.pth")

    # Preprocessors
    with open(os.path.join(cfg.output_dir, 'clinical_imputer.pkl'), 'wb') as f:
        pickle.dump(imputer, f)
    with open(os.path.join(cfg.output_dir, 'clinical_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # ══════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Task':<22} {'Metric':<12} {'Central':>10}", end="")
    if fed_results:
        print(f" {'Federated':>10}", end="")
    print()
    print("-" * 56)

    rows = [
        ('Binary (N vs D)', 'Accuracy', supervised_results['binary']['accuracy'],
         fed_results['binary']['accuracy'] if fed_results else None),
        ('Binary (N vs D)', 'AUC', supervised_results['binary']['roc_auc'],
         fed_results['binary']['roc_auc'] if fed_results else None),
        ('Binary (N vs D)', 'F1', supervised_results['binary']['f1'],
         fed_results['binary']['f1'] if fed_results else None),
        ('Disease (5-class)', 'Accuracy', supervised_results['disease']['accuracy'],
         fed_results['disease']['accuracy'] if fed_results else None),
        ('Disease (5-class)', 'F1(w)', supervised_results['disease']['f1_weighted'],
         fed_results['disease']['f1_weighted'] if fed_results else None),
        ('Severity (3-class)', 'Accuracy', supervised_results['severity']['accuracy'],
         fed_results['severity']['accuracy'] if fed_results else None),
    ]

    for task, metric, central_v, fed_v in rows:
        line = f"{task:<22} {metric:<12} {central_v:>10.4f}"
        if fed_v is not None:
            line += f" {fed_v:>10.4f}"
        print(line)

    print(f"\n  Total training time: {elapsed / 60:.1f} minutes")
    print(f"  All outputs saved to: {cfg.output_dir}/")
    print(f"\n  Files:")
    for f in sorted(os.listdir(cfg.output_dir)):
        size = os.path.getsize(os.path.join(cfg.output_dir, f))
        print(f"    {f} ({size / 1024:.0f} KB)")

    print("\n" + "=" * 60)
    print("CardioM3Net training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
