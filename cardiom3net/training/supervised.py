"""
Multi-task supervised training for CardioM3Net.
Joint loss: L = λ1·L_binary + λ2·L_disease + λ3·L_severity + λ_da·L_domain
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import compute_binary_metrics, compute_multiclass_metrics


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_binary=1.0, lambda_disease=1.0, lambda_severity=0.5,
                 disease_weights=None, severity_weights=None):
        super().__init__()
        self.lambda_binary = lambda_binary
        self.lambda_disease = lambda_disease
        self.lambda_severity = lambda_severity

        self.binary_loss = nn.CrossEntropyLoss()
        self.disease_loss = nn.CrossEntropyLoss(
            weight=disease_weights if disease_weights is not None else None
        )
        self.severity_loss = nn.CrossEntropyLoss(
            weight=severity_weights if severity_weights is not None else None
        )
        self.domain_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, domain_logits=None, domain_targets=None):
        loss_b = self.binary_loss(predictions['binary'], targets['binary'])
        loss_d = self.disease_loss(predictions['disease'], targets['disease'])
        loss_s = self.severity_loss(predictions['severity'], targets['severity'])

        total = (self.lambda_binary * loss_b +
                 self.lambda_disease * loss_d +
                 self.lambda_severity * loss_s)

        losses = {'binary': loss_b.item(), 'disease': loss_d.item(),
                  'severity': loss_s.item(), 'total': total.item()}

        if domain_logits is not None and domain_targets is not None:
            loss_da = self.domain_loss(domain_logits, domain_targets)
            total = total + loss_da
            losses['domain'] = loss_da.item()
            losses['total'] = total.item()

        return total, losses


def run_epoch(model, loader, criterion, device, optimizer=None, use_domain=False):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_binary_labels, all_binary_probs = [], []
    all_disease_labels, all_disease_preds = [], []
    all_severity_labels, all_severity_preds = [], []
    n_samples = 0

    for batch in loader:
        ecg = batch['ecg'].to(device)
        clinical = batch['clinical'].to(device)
        targets = {
            'binary': batch['binary'].to(device),
            'disease': batch['disease'].to(device),
            'severity': batch['severity'].to(device),
        }
        domain_targets = batch.get('domain')
        if domain_targets is not None:
            domain_targets = domain_targets.to(device)

        with torch.set_grad_enabled(training):
            predictions, modality_w, domain_logits = model(
                ecg, clinical, return_domain=use_domain
            )
            loss, _ = criterion(predictions, targets,
                                domain_logits if use_domain else None,
                                domain_targets if use_domain else None)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = ecg.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        # Collect predictions
        binary_prob = torch.softmax(predictions['binary'], dim=1)[:, 1]
        all_binary_labels.extend(targets['binary'].cpu().numpy().tolist())
        all_binary_probs.extend(binary_prob.detach().cpu().numpy().tolist())

        disease_pred = predictions['disease'].argmax(dim=1)
        all_disease_labels.extend(targets['disease'].cpu().numpy().tolist())
        all_disease_preds.extend(disease_pred.detach().cpu().numpy().tolist())

        severity_pred = predictions['severity'].argmax(dim=1)
        all_severity_labels.extend(targets['severity'].cpu().numpy().tolist())
        all_severity_preds.extend(severity_pred.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(n_samples, 1)
    binary_metrics = compute_binary_metrics(all_binary_labels, all_binary_probs)
    disease_metrics = compute_multiclass_metrics(all_disease_labels, all_disease_preds, 5)
    severity_metrics = compute_multiclass_metrics(all_severity_labels, all_severity_preds, 3)

    return {
        'loss': avg_loss,
        'binary': binary_metrics,
        'disease': disease_metrics,
        'severity': severity_metrics,
        'binary_labels': all_binary_labels,
        'binary_probs': all_binary_probs,
        'disease_labels': all_disease_labels,
        'disease_preds': all_disease_preds,
    }


def train_supervised(model, train_loader, val_loader, config, device, use_domain=False):
    """Full supervised training loop with early stopping."""
    print("\n" + "=" * 60)
    print("PHASE 2: Multi-Task Supervised Training")
    print("=" * 60)

    criterion = MultiTaskLoss(
        lambda_binary=config.lambda_binary,
        lambda_disease=config.lambda_disease,
        lambda_severity=config.lambda_severity,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config.scheduler_patience,
        factor=config.scheduler_factor
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'val_binary_acc': [], 'val_binary_auc': [],
        'val_disease_acc': [], 'val_severity_acc': [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -1.0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_res = run_epoch(model, train_loader, criterion, device, optimizer, use_domain)
        val_res = run_epoch(model, val_loader, criterion, device, use_domain=use_domain)
        scheduler.step(val_res['loss'])

        history['train_loss'].append(train_res['loss'])
        history['val_loss'].append(val_res['loss'])
        history['val_binary_acc'].append(val_res['binary']['accuracy'])
        history['val_binary_auc'].append(val_res['binary']['roc_auc'])
        history['val_disease_acc'].append(val_res['disease']['accuracy'])
        history['val_severity_acc'].append(val_res['severity']['accuracy'])

        val_auc = np.nan_to_num(val_res['binary']['roc_auc'], nan=-1.0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == config.epochs:
            print(
                f"  Epoch {epoch:3d}/{config.epochs} | "
                f"T-Loss: {train_res['loss']:.4f} V-Loss: {val_res['loss']:.4f} | "
                f"Bin-Acc: {val_res['binary']['accuracy']:.3f} "
                f"Bin-AUC: {val_res['binary']['roc_auc']:.3f} | "
                f"Dis-Acc: {val_res['disease']['accuracy']:.3f} "
                f"Sev-Acc: {val_res['severity']['accuracy']:.3f}"
            )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch} (patience={config.patience})")
            break

    model.load_state_dict(best_state)
    final = run_epoch(model, val_loader, criterion, device, use_domain=use_domain)

    print(f"\n  Best model results:")
    print(f"    Binary  — Acc: {final['binary']['accuracy']:.4f} | "
          f"AUC: {final['binary']['roc_auc']:.4f} | F1: {final['binary']['f1']:.4f}")
    print(f"    Disease — Acc: {final['disease']['accuracy']:.4f} | "
          f"F1(w): {final['disease']['f1_weighted']:.4f}")
    print(f"    Severity— Acc: {final['severity']['accuracy']:.4f} | "
          f"F1(w): {final['severity']['f1_weighted']:.4f}")

    return model, history, final
