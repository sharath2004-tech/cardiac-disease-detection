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

# Force UTF-8 output on Windows consoles (avoids UnicodeEncodeError with special chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

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
                    print(f"[setup] Copied cardiom3net/ from {src} -> {dst_pkg}")
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
                                plot_confusion_matrices, plot_roc_curves,
                                plot_model_comparison)
from cardiom3net.data.ecg_loader import load_ptbxl
from cardiom3net.data.pcg_loader import (load_pcg_cinc2016, build_pcg_pool,
                                          assign_pcg_to_ecg)
from cardiom3net.data.clinical_loader import build_augmented_clinical
from cardiom3net.data.multimodal_dataset import (MultimodalCardiacDataset, MAMLTaskDataset)
from cardiom3net.models.cardiom3net    import CardioM3Net
from cardiom3net.models.baseline_models import (ECGOnlyModel, ClinicalOnlyModel,
                                                 ConcatFusionModel)
from cardiom3net.training.self_supervised import pretrain_ecg_encoder
from cardiom3net.training.supervised import train_supervised, run_epoch, MultiTaskLoss
from cardiom3net.training.maml_trainer  import train_maml
from cardiom3net.training.federated     import run_federated
from cardiom3net.explainability.gradcam_1d    import plot_ecg_saliency
from cardiom3net.explainability.shap_analysis import run_shap_analysis, plot_modality_weights

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="CardioM3Net Training")
    # Data
    p.add_argument("--max_records", type=int, default=21837)
    p.add_argument("--use_100hz", action="store_true", default=True)
    p.add_argument("--test_size", type=float, default=0.2)
    # PCG
    p.add_argument("--pcg_archive_dir", type=str, default="archive",
                   help="Path to the CinC 2016 archive/ folder")
    p.add_argument("--skip_pcg", action="store_true", default=False,
                   help="Skip PCG loading; train bimodal ECG+Clinical model")
    p.add_argument("--skip_comparison", action="store_true", default=False,
                   help="Skip baseline model comparison phase")
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

    # ── Clinical features: augmented PTB-XL demographics + UCI diagnostics ──
    # PTB-XL metadata alone (nurse, site, device_code) are too weak to
    # contribute meaningfully to the modality gate.  We replace them with a
    # 14-dim vector: real PTB-XL age/sex/height + UCI diagnostic features
    # (cholesterol, BP, chest-pain type, oldpeak, etc.) assigned via
    # label-stratified sampling (normal ECG → UCI-normal row, disease ECG →
    # UCI-disease row) so the clinically relevant signal is preserved.
    ptbxl_meta = ptbxl['clinical_array']   # (N, 7) — age,sex,height,weight,...
    meta_cols   = ptbxl['meta_cols']

    _age_idx    = meta_cols.index('age')    if 'age'    in meta_cols else None
    _sex_idx    = meta_cols.index('sex')    if 'sex'    in meta_cols else None
    _ht_idx     = meta_cols.index('height') if 'height' in meta_cols else None

    ptbxl_age    = ptbxl_meta[:, _age_idx] if _age_idx is not None else np.full(len(ptbxl_meta), np.nan)
    ptbxl_sex    = ptbxl_meta[:, _sex_idx] if _sex_idx is not None else np.full(len(ptbxl_meta), np.nan)
    ptbxl_height = ptbxl_meta[:, _ht_idx]  if _ht_idx  is not None else np.full(len(ptbxl_meta), np.nan)

    clinical_array, clinical_names = build_augmented_clinical(
        ptbxl_age, ptbxl_sex, ptbxl_height, binary_labels,
        rng=np.random.default_rng(cfg.seed),
    )
    print(f"  Augmented clinical features ({len(clinical_names)}): {clinical_names}")
    print(f"  Clinical shape: {clinical_array.shape} | NaN count: {np.isnan(clinical_array).sum()}")

    clinical_dim = clinical_array.shape[1]

    # ══════════════════════════════════════════════════════════════════
    #  PCG DATA LOADING  (PhysioNet CinC 2016)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PCG DATA LOADING (CinC 2016 Heart Sounds)")
    print("=" * 60)

    pcg_data     = None
    use_pcg_flag = False
    train_pcg    = None
    test_pcg     = None

    if not args.skip_pcg:
        pcg_cache = os.path.join(cfg.output_dir, 'pcg_cache.npz')
        pcg_data  = load_pcg_cinc2016(
            archive_dir=args.pcg_archive_dir,
            cache_path=pcg_cache,
        )

    if pcg_data is not None:
        use_pcg_flag          = True
        normal_pool, abn_pool = build_pcg_pool(pcg_data)
        print(f"  PCG pools — Normal: {len(normal_pool)}, Abnormal: {len(abn_pool)}")
    else:
        print("  PCG not available — running bimodal (ECG + Clinical) mode.")
        print("  Use --pcg_archive_dir to point to the CinC 2016 archive/ folder.")

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

    # ── PCG cross-dataset assignment (needs train_idx / test_idx) ────
    if pcg_data is not None:
        rng     = np.random.default_rng(cfg.seed)
        all_pcg = assign_pcg_to_ecg(binary_labels, normal_pool, abn_pool, rng=rng)
        train_pcg = all_pcg[train_idx]
        test_pcg  = all_pcg[test_idx]
        print(f"  PCG assigned: train={len(train_pcg)}, test={len(test_pcg)}")
        print(f"  PCG spec shape: {train_pcg.shape[1:]}")

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
        pcg=train_pcg,
        augment=True,
    )
    test_ds = MultimodalCardiacDataset(
        test_ecg, test_clinical,
        binary_labels[test_idx], disease_labels[test_idx],
        severity_labels[test_idx], domain_labels[test_idx],
        pcg=test_pcg,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")
    print(f"  ECG: ({num_leads}, {signal_length}) | Clinical dim: {clinical_dim}")

    # Compute class weights to handle disease imbalance
    train_disease = disease_labels[train_idx]
    train_severity = severity_labels[train_idx]
    disease_cw = compute_class_weight('balanced', classes=np.unique(train_disease), y=train_disease)
    severity_cw = compute_class_weight('balanced', classes=np.unique(train_severity), y=train_severity)
    # Pad to full num_classes in case some classes missing
    disease_weights_full = np.ones(cfg.num_disease_classes, dtype=np.float32)
    for i, c in enumerate(np.unique(train_disease)):
        disease_weights_full[c] = disease_cw[i]
    severity_weights_full = np.ones(cfg.severity_bins, dtype=np.float32)
    for i, c in enumerate(np.unique(train_severity)):
        severity_weights_full[c] = severity_cw[i]
    disease_weights_t = torch.tensor(disease_weights_full).to(device)
    severity_weights_t = torch.tensor(severity_weights_full).to(device)
    print(f"  Disease class weights: {disease_weights_full.round(2)}")
    print(f"  Severity class weights: {severity_weights_full.round(2)}")
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
        pcg_embed_dim=cfg.pcg_embed_dim,
        fusion_dim=cfg.fusion_dim,
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
        transformer_ff_dim=cfg.transformer_ff_dim,
        num_disease_classes=cfg.num_disease_classes,
        num_severity_classes=cfg.severity_bins,
        num_domains=3,
        da_lambda=cfg.da_lambda,
        dropout=cfg.dropout,
        use_pcg=use_pcg_flag,
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
        model, train_loader, test_loader, cfg, device, use_domain=True,
        disease_weights=disease_weights_t, severity_weights=severity_weights_t,
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
            print("  Insufficient classes for MAML -- skipping")
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
            pcg_embed_dim=cfg.pcg_embed_dim,
            fusion_dim=cfg.fusion_dim, transformer_heads=cfg.transformer_heads,
            transformer_layers=cfg.transformer_layers, transformer_ff_dim=cfg.transformer_ff_dim,
            num_disease_classes=cfg.num_disease_classes, num_severity_classes=cfg.severity_bins,
            num_domains=3, da_lambda=cfg.da_lambda, dropout=cfg.dropout,
            use_pcg=use_pcg_flag,
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
    #  PHASE 7: BASELINE MODEL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_comparison:
        print("\n" + "=" * 60)
        print("PHASE 7: BASELINE MODEL COMPARISON")
        print("=" * 60)
        print("  Training baselines with supervised-only protocol for fair comparison.")

        import copy as _copy

        def _train_baseline(bmodel, bname):
            """Quick supervised training for a single baseline model."""
            bmodel = bmodel.to(device)
            bopt   = torch.optim.Adam(bmodel.parameters(), lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
            bcrit  = MultiTaskLoss(cfg.lambda_binary, cfg.lambda_disease,
                                   cfg.lambda_severity,
                                   disease_weights=disease_weights_t,
                                   severity_weights=severity_weights_t)
            best_auc   = -1.0
            best_state = _copy.deepcopy(bmodel.state_dict())

            for ep in range(cfg.epochs):
                run_epoch(bmodel, train_loader, bcrit, device,
                          optimizer=bopt, use_domain=False)
                val = run_epoch(bmodel, test_loader, bcrit, device, use_domain=False)
                if val['binary']['roc_auc'] > best_auc:
                    best_auc   = val['binary']['roc_auc']
                    best_state = _copy.deepcopy(bmodel.state_dict())
                if (ep + 1) % 10 == 0:
                    print(f"    [{bname}] Epoch {ep+1}/{cfg.epochs} "
                          f"| Binary AUC: {val['binary']['roc_auc']:.4f}")

            bmodel.load_state_dict(best_state)
            final = run_epoch(bmodel, test_loader, bcrit, device, use_domain=False)
            print(f"  [{bname}] Final — "
                  f"Binary Acc: {final['binary']['accuracy']:.4f} "
                  f"AUC: {final['binary']['roc_auc']:.4f} "
                  f"Disease Acc: {final['disease']['accuracy']:.4f}")
            return final

        comparison_results = {}

        # ── Sklearn baselines (traditional ML on clinical + ECG stats) ──────
        def _ecg_stat_features(ecg_np):
            """Extract 60-dim statistical features per lead (mean, std, rms, max, min)."""
            # ecg_np: (N, leads, time)
            mean = ecg_np.mean(axis=2)                              # (N, leads)
            std  = ecg_np.std(axis=2)
            rms  = np.sqrt((ecg_np ** 2).mean(axis=2))
            mx   = ecg_np.max(axis=2)
            mn   = ecg_np.min(axis=2)
            return np.concatenate([mean, std, rms, mx, mn], axis=1)  # (N, leads*5)

        train_ecg_stats = _ecg_stat_features(train_ecg)
        test_ecg_stats  = _ecg_stat_features(test_ecg)
        # Combine with scaled clinical features
        X_train_ml = np.concatenate([train_clinical, train_ecg_stats], axis=1)
        X_test_ml  = np.concatenate([test_clinical,  test_ecg_stats],  axis=1)
        y_train_bin = binary_labels[train_idx]
        y_test_bin  = binary_labels[test_idx]
        y_train_dis = disease_labels[train_idx]
        y_test_dis  = disease_labels[test_idx]
        y_train_sev = severity_labels[train_idx]
        y_test_sev  = severity_labels[test_idx]

        from cardiom3net.utils import compute_binary_metrics, compute_multiclass_metrics

        def _train_sklearn_baseline(clf_bin, clf_dis, clf_sev, bname):
            """Fit 3 separate sklearn classifiers for each task, return comparison dict."""
            clf_bin.fit(X_train_ml, y_train_bin)
            clf_dis.fit(X_train_ml, y_train_dis)
            clf_sev.fit(X_train_ml, y_train_sev)

            # Binary: get probability of positive class
            if hasattr(clf_bin, 'predict_proba'):
                bin_probs = clf_bin.predict_proba(X_test_ml)[:, 1].tolist()
            else:
                bin_probs = clf_bin.decision_function(X_test_ml).tolist()
            bin_labels = y_test_bin.tolist()

            dis_preds = clf_dis.predict(X_test_ml).tolist()
            sev_preds = clf_sev.predict(X_test_ml).tolist()

            bin_m = compute_binary_metrics(bin_labels, bin_probs)
            dis_m = compute_multiclass_metrics(y_test_dis, dis_preds, cfg.num_disease_classes)
            sev_m = compute_multiclass_metrics(y_test_sev, sev_preds, cfg.severity_bins)

            print(f"  [{bname}] Final — "
                  f"Binary Acc: {bin_m['accuracy']:.4f} "
                  f"AUC: {bin_m['roc_auc']:.4f} "
                  f"Disease Acc: {dis_m['accuracy']:.4f}")
            return {
                'binary':   bin_m,
                'disease':  dis_m,
                'severity': sev_m,
                'binary_labels': bin_labels,
                'binary_probs':  bin_probs,
            }

        # B0a: Logistic Regression
        comparison_results['B0a: Logistic Regression'] = _train_sklearn_baseline(
            LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                               multi_class='ovr', random_state=cfg.seed),
            LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                               multi_class='ovr', random_state=cfg.seed),
            LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                               multi_class='ovr', random_state=cfg.seed),
            'Logistic Regression',
        )

        # B0b: Random Forest
        comparison_results['B0b: Random Forest'] = _train_sklearn_baseline(
            RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=cfg.seed, n_jobs=-1),
            RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=cfg.seed, n_jobs=-1),
            RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=cfg.seed, n_jobs=-1),
            'Random Forest',
        )

        # B1: ECG only
        comparison_results['B1: ECG-Only'] = _train_baseline(
            ECGOnlyModel(
                num_leads=num_leads,
                ecg_embed_dim=cfg.ecg_embed_dim,
                num_disease_classes=cfg.num_disease_classes,
                num_severity_classes=cfg.severity_bins,
                dropout=cfg.dropout,
            ),
            'ECG-Only',
        )

        # B2: Clinical only
        comparison_results['B2: Clinical-Only'] = _train_baseline(
            ClinicalOnlyModel(
                clinical_dim=clinical_dim,
                clinical_embed_dim=cfg.clinical_embed_dim,
                num_disease_classes=cfg.num_disease_classes,
                num_severity_classes=cfg.severity_bins,
                dropout=cfg.dropout,
            ),
            'Clinical-Only',
        )

        # B3: Concat fusion (bimodal, no attention)
        comparison_results['B3: Concat Fusion (no attn)'] = _train_baseline(
            ConcatFusionModel(
                num_leads=num_leads,
                clinical_dim=clinical_dim,
                ecg_embed_dim=cfg.ecg_embed_dim,
                clinical_embed_dim=cfg.clinical_embed_dim,
                hidden_dim=cfg.fusion_dim,
                num_disease_classes=cfg.num_disease_classes,
                num_severity_classes=cfg.severity_bins,
                dropout=cfg.dropout,
            ),
            'Concat Fusion',
        )

        # B4: Bimodal attention (CardioM3Net without PCG)
        comparison_results['B4: Bimodal Attention (no PCG)'] = _train_baseline(
            CardioM3Net(
                num_leads=num_leads,
                clinical_dim=clinical_dim,
                ecg_embed_dim=cfg.ecg_embed_dim,
                clinical_embed_dim=cfg.clinical_embed_dim,
                pcg_embed_dim=cfg.pcg_embed_dim,
                fusion_dim=cfg.fusion_dim,
                transformer_heads=cfg.transformer_heads,
                transformer_layers=cfg.transformer_layers,
                transformer_ff_dim=cfg.transformer_ff_dim,
                num_disease_classes=cfg.num_disease_classes,
                num_severity_classes=cfg.severity_bins,
                num_domains=3, da_lambda=cfg.da_lambda,
                dropout=cfg.dropout,
                use_pcg=False,
            ),
            'Bimodal Attention',
        )

        # Proposed model results (Phase 2 supervised output = fair comparison baseline)
        proposed_label = ('B5: CardioM3Net Trimodal (Proposed)'
                          if use_pcg_flag else 'B5: CardioM3Net Bimodal (Full Pipeline)')
        comparison_results[proposed_label] = supervised_results

        # B6: Federated CardioM3Net (if Phase 5 ran)
        if fed_results is not None:
            comparison_results['B6: CardioM3Net Federated'] = fed_results

        # Plot and print comparison table
        plot_model_comparison(comparison_results, cfg.output_dir)

        # Add all baselines to ROC plot
        for bname, bres in comparison_results.items():
            if bname == proposed_label:
                continue
            roc_data[bname] = (bres['binary_labels'], bres['binary_probs'])
        plot_roc_curves(roc_data, cfg.output_dir)
    else:
        print("\n  Skipping baseline comparison (--skip_comparison)")

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
        'use_pcg': use_pcg_flag,
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
            'use_pcg': use_pcg_flag,
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
