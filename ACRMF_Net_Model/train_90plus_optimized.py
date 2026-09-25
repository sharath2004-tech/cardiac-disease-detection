"""
SYSTEMATIC IMBALANCED CLASSIFICATION TRAINING FOR 90%+ ACCURACY
Addresses severe 16.95x class imbalance (Class 0: 55.8%, Class 4: 3.3%)
Reduces 12.25% train-validation gap through proper regularization

Key Improvements:
1. Effective-number class weighting (beta=0.9999)
2. WeightedRandomSampler for minority classes
3. AdamW optimizer with proper weight decay
4. ReduceLROnPlateau scheduler monitoring macro F1
5. Best checkpoint based on validation macro F1
6. Early stopping (patience=20)
7. Label smoothing for better calibration
8. Enhanced data augmentation for minority classes
9. Comprehensive per-class metrics
10. Controlled experiments to identify best configuration
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score, 
                             accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).parent))
from models.acrmf import ACRMFNet

print("\n" + "="*80)
print("IMBALANCED CLASSIFICATION TRAINING: CLASS IMBALANCE 16.95x")
print("Target: 90%+ Val Accuracy + Improved Minority Class Performance")
print("WITH AUTOMATIC VISUALIZATION GENERATION")
print("="*80 + "\n")


# ============================================================================
# VISUALIZATION FUNCTIONS (Integrated into Training)
# ============================================================================
def plot_training_progress(history, save_path='training_progress.png'):
    """Plot training curves in real-time"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].axhline(y=0.90, color='g', linestyle='--', label='90% Target')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    axes[0, 2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[0, 2].axhline(y=0.80, color='g', linestyle='--', label='80% Target')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('Macro F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    if 'lr' in history and history['lr']:
        axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Train-Val Gap
    if len(history['train_acc']) > 0 and len(history['val_acc']) > 0:
        gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gap (Train - Val)')
        axes[1, 1].set_title('Overfitting Monitor (Lower is Better)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Best Metrics Summary
    axes[1, 2].axis('off')
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    
    summary_text = f"""
    TRAINING SUMMARY
    
    Current Epoch: {len(epochs)}
    
    Best Val Accuracy: {best_val_acc*100:.2f}%
    Best Val F1: {best_val_f1:.4f}
    
    Latest Train Acc: {final_train_acc*100:.2f}%
    Latest Val Acc: {history['val_acc'][-1]*100:.2f}%
    
    Overfitting Gap: {(final_train_acc - history['val_acc'][-1])*100:.2f}%
    
    Status: {"[OK] On Track!" if best_val_acc >= 0.90 else "[WARN] Below 90%"}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_live(labels, predictions, num_classes, epoch, save_path='confusion_matrix.png'):
    """Generate confusion matrix visualization"""
    cm = confusion_matrix(labels, predictions)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Confusion Matrix - Epoch {epoch} (Counts)')
    
    # Plot 2: Normalized (recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'Confusion Matrix - Epoch {epoch} (Recall %)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_performance(per_class_metrics, epoch, save_path='per_class_performance.png'):
    """Plot per-class accuracy, precision, recall, F1"""
    num_classes = len(per_class_metrics['accuracy'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    classes = list(range(num_classes))
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [per_class_metrics[metric].get(cls, 0) * 100 for cls in classes]
        
        bars = ax.bar(classes, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=80, color='red', linestyle='--', label='80% Target', linewidth=2)
        ax.axhline(y=90, color='green', linestyle='--', label='90% Target', linewidth=2)
        ax.set_xlabel('Class')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(f'Per-Class {title} - Epoch {epoch}')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_final_report(history, per_class_metrics, labels, predictions, 
                         num_classes, save_dir='results/final_report'):
    """Generate comprehensive final report with all visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating final visualizations...")
    
    # 1. Training curves
    plot_training_progress(history, save_path=save_dir / 'training_curves.png')
    print("  [OK] Training curves saved")
    
    # 2. Final confusion matrix
    plot_confusion_matrix_live(labels, predictions, num_classes, 
                               len(history['train_loss']), 
                               save_path=save_dir / 'final_confusion_matrix.png')
    print("  [OK] Confusion matrix saved")
    
    # 3. Per-class performance
    plot_per_class_performance(per_class_metrics, len(history['train_loss']),
                              save_path=save_dir / 'per_class_performance.png')
    print("  [OK] Per-class performance saved")
    
    # 4. Summary dashboard
    create_summary_dashboard(history, per_class_metrics, num_classes,
                            save_path=save_dir / 'summary_dashboard.png')
    print("  [OK] Summary dashboard saved")
    
    print(f"\n[OK] All visualizations saved to: {save_dir}/")


def create_summary_dashboard(history, per_class_metrics, num_classes, save_path):
    """Create comprehensive summary dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
    
    fig.suptitle(f'Training Summary Dashboard - Best Val Accuracy: {best_val_acc*100:.2f}%',
                fontsize=18, fontweight='bold')
    
    # 1. Accuracy over time
    ax = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_acc']) + 1)
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    ax.axhline(y=0.90, color='g', linestyle='--', label='Target')
    ax.set_title('Accuracy Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Per-class accuracy bars
    ax = fig.add_subplot(gs[0, 1])
    classes = list(range(num_classes))
    accs = [per_class_metrics['accuracy'].get(cls, 0) * 100 for cls in classes]
    colors_bar = ['green' if a >= 80 else 'orange' if a >= 70 else 'red' for a in accs]
    ax.bar(classes, accs, color=colors_bar, alpha=0.7)
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2)
    ax.set_title('Final Per-Class Accuracy')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. F1 Score over time
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    ax.set_title('F1 Score Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Loss over time
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_title('Loss Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Overfitting monitor
    ax = fig.add_subplot(gs[1, 1])
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    ax.plot(epochs, gap, 'purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.set_title('Train-Val Gap (Overfitting)')
    ax.grid(True, alpha=0.3)
    
    # 6. Class distribution
    ax = fig.add_subplot(gs[1, 2])
    # Placeholder - would need actual class counts
    ax.text(0.5, 0.5, 'Class Distribution\n(See training logs)', 
           ha='center', va='center', fontsize=12)
    ax.axis('off')
    
    # 7. Metrics table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    metrics_text = f"""
    FINAL TRAINING METRICS
    
    Epochs Trained: {len(epochs)}
    Best Val Accuracy: {best_val_acc*100:.2f}%
    Best Val F1 Score: {best_val_f1:.4f}
    
    Final Train Accuracy: {history['train_acc'][-1]*100:.2f}%
    Final Val Accuracy: {history['val_acc'][-1]*100:.2f}%
    
    Overfitting Gap: {(history['train_acc'][-1] - history['val_acc'][-1])*100:.2f}%
    
    Per-Class Accuracy:
    """
    
    for cls in range(num_classes):
        acc = per_class_metrics['accuracy'].get(cls, 0) * 100
        status = "[OK]" if acc >= 80 else "[WARN]"
        metrics_text += f"\n      {status} Class {cls}: {acc:.2f}%"
    
    ax.text(0.1, 0.5, metrics_text, fontsize=13, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# EFFECTIVE NUMBER CLASS WEIGHTING (for severe imbalance)
# ============================================================================
def get_effective_number_weights(labels, num_classes, beta=0.9999):
    """
    Calculate effective number of samples for each class
    Paper: "Class-Balanced Loss Based on Effective Number of Samples"
    
    For severe imbalance (16.95x), effective-number weighting works better
    than simple inverse frequency weighting.
    
    Args:
        labels: training labels
        num_classes: number of classes
        beta: balancing parameter (0.9999 for severe imbalance)
    """
    samples_per_class = np.bincount(labels, minlength=num_classes)
    
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return torch.FloatTensor(weights)


def get_inverse_freq_weights(labels, num_classes, smooth=True):
    """Simple inverse frequency weighting with optional smoothing"""
    samples_per_class = np.bincount(labels, minlength=num_classes)
    
    if smooth:
        # Softened inverse frequency (less aggressive)
        weights = 1.0 / np.sqrt(samples_per_class)
    else:
        weights = 1.0 / samples_per_class
    
    weights = weights / weights.mean()  # Normalize to mean=1
    return torch.FloatTensor(weights)


# ============================================================================
# ENHANCED DATASET WITH CLASS-AWARE AUGMENTATION
# ============================================================================
class ImbalancedDataset(Dataset):
    """Dataset with stronger augmentation for minority classes"""
    
    def __init__(self, clinical, ecg, pcg, labels, mode='train', 
                 minority_class_ids=[3, 4], aug_strength=1.0):
        self.clinical = torch.FloatTensor(clinical)
        self.ecg = torch.FloatTensor(ecg)
        self.pcg = torch.FloatTensor(pcg)
        self.labels = torch.LongTensor(labels)
        self.mode = mode
        self.minority_class_ids = minority_class_ids
        self.aug_strength = aug_strength
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        clinical = self.clinical[idx].clone()
        ecg = self.ecg[idx].clone()
        pcg = self.pcg[idx].clone()
        label = self.labels[idx].item()
        
        if self.mode == 'train':
            # Base augmentation probability
            aug_prob = 0.5
            
            # Increase augmentation for minority classes
            if label in self.minority_class_ids:
                aug_prob = 0.7  # Stronger augmentation
            
            if torch.rand(1) < aug_prob:
                strength = self.aug_strength
                
                # ECG augmentation (time series)
                if torch.rand(1) > 0.5:
                    # Amplitude scaling (±10%)
                    ecg = ecg * (0.9 + torch.rand(1) * 0.2 * strength)
                if torch.rand(1) > 0.5:
                    # Gaussian noise
                    noise = torch.randn_like(ecg) * 0.01 * strength
                    ecg = ecg + noise
                if torch.rand(1) > 0.6:
                    # Time shifting
                    shift = int(torch.randint(-25, 26, (1,)).item() * strength)
                    ecg = torch.roll(ecg, shifts=shift, dims=0)
                
                # PCG augmentation (spectrogram - SpecAugment style)
                if torch.rand(1) > 0.5:
                    # Frequency masking
                    f_mask = int(torch.randint(2, 6, (1,)).item() * strength)
                    f0 = torch.randint(0, max(1, 128 - f_mask), (1,)).item()
                    pcg[f0:f0+f_mask, :] = 0
                if torch.rand(1) > 0.5:
                    # Time masking
                    t_mask = int(torch.randint(2, 6, (1,)).item() * strength)
                    t0 = torch.randint(0, max(1, 128 - t_mask), (1,)).item()
                    pcg[:, t0:t0+t_mask] = 0
                if torch.rand(1) > 0.6:
                    # Add noise
                    pcg = pcg + torch.randn_like(pcg) * 0.008 * strength
                
                # Clinical augmentation (very gentle)
                if torch.rand(1) > 0.6:
                    clinical = clinical + torch.randn_like(clinical) * 0.025 * strength
        
        return clinical, ecg, pcg, label


# ============================================================================
# MIXUP AUGMENTATION FOR BETTER GENERALIZATION
# ============================================================================
def mixup_data(clinical, ecg, pcg, labels, alpha=0.2):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = clinical.size(0)
    index = torch.randperm(batch_size).to(clinical.device)
    
    mixed_clinical = lam * clinical + (1 - lam) * clinical[index]
    mixed_ecg = lam * ecg + (1 - lam) * ecg[index]
    mixed_pcg = lam * pcg + (1 - lam) * pcg[index]
    
    labels_a, labels_b = labels, labels[index]
    return mixed_clinical, mixed_ecg, mixed_pcg, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# SIMPLIFIED LOSS FUNCTION FOR IMBALANCED CLASSIFICATION
# ============================================================================
class ImbalancedLoss(nn.Module):
    """Focused loss for imbalanced classification"""
    
    def __init__(self, class_weights, num_classes=5, label_smoothing=0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing
        )
        self.num_classes = num_classes
    
    def forward(self, outputs, labels, labels_b=None, lam=None):
        # Main classification loss
        if labels_b is not None and lam is not None:
            # MixUp case
            loss = mixup_criterion(self.ce_loss, outputs['final_logits'], 
                                  labels, labels_b, lam)
        else:
            loss = self.ce_loss(outputs['final_logits'], labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            loss = F.cross_entropy(outputs['final_logits'], labels, reduction='mean')
        
        return {'total': loss, 'classification': loss.item()}


# ============================================================================
# TRAINING AND VALIDATION WITH COMPREHENSIVE METRICS
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device, epoch, use_mixup=False):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch:3d}", leave=False)
    for clinical, ecg, pcg, labels in pbar:
        clinical = clinical.to(device)
        ecg = ecg.to(device)
        pcg = pcg.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Apply MixUp if enabled
        labels_b, lam = None, None
        if use_mixup and torch.rand(1) < 0.5:
            clinical, ecg, pcg, labels_a, labels_b, lam = mixup_data(
                clinical, ecg, pcg, labels, alpha=0.2
            )
            labels = labels_a
        
        outputs = model(clinical, ecg, pcg)
        
        if torch.isnan(outputs['final_logits']).any():
            continue
        
        losses = criterion(outputs, labels, labels_b, lam)
        loss = losses['total']
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if labels_b is None:
            preds = outputs['predictions']
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    
    if len(all_preds) > 0:
        train_acc = accuracy_score(all_labels, all_preds)
    else:
        train_acc = 0.0
    
    return avg_loss, train_acc * 100


def validate(model, loader, criterion, device, num_classes=5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for clinical, ecg, pcg, labels in tqdm(loader, desc="Validating", leave=False):
            clinical = clinical.to(device)
            ecg = ecg.to(device)
            pcg = pcg.to(device)
            labels = labels.to(device)
            
            outputs = model(clinical, ecg, pcg)
            
            if not torch.isnan(outputs['final_logits']).any():
                losses = criterion(outputs, labels)
                loss = losses['total']
                if not torch.isnan(loss):
                    total_loss += loss.item()
            
            preds = outputs['predictions']
            probs = outputs['probabilities']
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate comprehensive metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    avg_loss = total_loss / len(loader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy * 100,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_accuracy': [x * 100 for x in per_class_acc],
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return metrics


# ============================================================================
# PLOTTING AND REPORTING FUNCTIONS
# ============================================================================
def plot_training_curves(history, save_path):
    """Plot training/validation loss and accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].axhline(y=90, color='r', linestyle='--', label='90% Target', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=range(5), yticklabels=range(5))
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_class_distribution(labels, split_name="Dataset"):
    """Print and analyze class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\n{split_name} Class Distribution:")
    print("-" * 60)
    for cls, count in zip(unique, counts):
        pct = 100 * count / total
        print(f"  Class {cls}: {count:5d} samples ({pct:5.1f}%)")
    
    if len(counts) > 1:
        imbalance_ratio = counts.max() / counts.min()
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}x")
        if imbalance_ratio > 2:
            print(f"  WARNING: Significant class imbalance detected!")
    print("-" * 60)
    
    return counts


# ============================================================================
# COMPREHENSIVE AUTOMATIC VISUALIZATION
# ============================================================================
def generate_comprehensive_plots(experiments, best_exp):
    """
    Generate comprehensive comparison plots automatically after training
    
    Creates 10 different visualizations:
    1. Validation Accuracy Comparison
    2. Macro F1 Score Comparison
    3. Per-Class F1 Scores Heatmap
    4. Train-Val Gap Analysis
    5. Training History - Best Experiment
    6. Class 4 Performance Across Experiments
    7. Accuracy vs Epoch - All Experiments
    8. Best Experiment Confusion Matrix
    9. Performance Radar Chart
    10. Experiment Rankings
    """
    
    # Create output directory
    plot_dir = Path('experiments/comparison_plots')
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating 10 comprehensive visualizations...")
    
    # ========================================================================
    # PLOT 1: VALIDATION ACCURACY COMPARISON BAR CHART
    # ========================================================================
    plt.figure(figsize=(14, 6))
    exp_names = [exp['experiment_name'].replace('_', '\n') for exp in experiments]
    val_accs = [exp['best_val_acc'] for exp in experiments]
    colors = ['green' if exp['target_achieved'] else 'orange' if exp['best_val_acc'] > 88 else 'red' 
              for exp in experiments]
    
    bars = plt.bar(range(len(experiments)), val_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% Target', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, val_accs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Experiment', fontsize=13, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title('Validation Accuracy Comparison Across All Experiments', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(experiments)), exp_names, fontsize=9)
    plt.ylim([min(val_accs) - 5, 95])
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / '01_validation_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 1: Validation Accuracy Comparison")
    
    # ========================================================================
    # PLOT 2: MACRO F1 SCORE COMPARISON
    # ========================================================================
    plt.figure(figsize=(14, 6))
    macro_f1s = [exp['macro_f1'] for exp in experiments]
    bars = plt.bar(range(len(experiments)), macro_f1s, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Highlight best
    best_idx = macro_f1s.index(max(macro_f1s))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2.5)
    
    for i, (bar, f1) in enumerate(zip(bars, macro_f1s)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Experiment', fontsize=13, fontweight='bold')
    plt.ylabel('Macro F1 Score', fontsize=13, fontweight='bold')
    plt.title('Macro F1 Score Comparison (Higher is Better)', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(experiments)), exp_names, fontsize=9)
    plt.ylim([min(macro_f1s) - 0.05, max(macro_f1s) + 0.1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / '02_macro_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 2: Macro F1 Score Comparison")
    
    # ========================================================================
    # PLOT 3: PER-CLASS F1 SCORES HEATMAP
    # ========================================================================
    plt.figure(figsize=(12, 8))
    per_class_f1_matrix = np.array([exp['per_class_f1'] for exp in experiments])
    
    sns.heatmap(per_class_f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'F1 Score'}, vmin=0, vmax=1,
                xticklabels=[f'Class {i}' for i in range(5)],
                yticklabels=[exp['experiment_name'] for exp in experiments],
                linewidths=0.5, linecolor='gray')
    
    plt.title('Per-Class F1 Scores Across All Experiments\n(Green = Good, Red = Poor)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Experiment', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir / '03_per_class_f1_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 3: Per-Class F1 Scores Heatmap")
    
    # ========================================================================
    # PLOT 4: TRAIN-VAL GAP ANALYSIS
    # ========================================================================
    plt.figure(figsize=(14, 6))
    gaps = [exp['train_val_gap'] for exp in experiments]
    colors_gap = ['green' if gap < 8 else 'orange' if gap < 12 else 'red' for gap in gaps]
    
    bars = plt.bar(range(len(experiments)), gaps, color=colors_gap, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='10% (Caution)', alpha=0.6)
    plt.axhline(y=8, color='green', linestyle='--', linewidth=2, label='8% (Good)', alpha=0.6)
    
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{gap:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Experiment', fontsize=13, fontweight='bold')
    plt.ylabel('Train-Validation Gap (%)', fontsize=13, fontweight='bold')
    plt.title('Overfitting Analysis: Train-Validation Gap\n(Lower is Better)', 
              fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(experiments)), exp_names, fontsize=9)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / '04_train_val_gap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 4: Train-Val Gap Analysis")
    
    # ========================================================================
    # PLOT 5: TRAINING HISTORY - BEST EXPERIMENT
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    history = best_exp['history']
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, color='blue', alpha=0.7)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'Training/Validation Loss\n{best_exp["experiment_name"]}', 
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2, color='blue', alpha=0.7)
    axes[0, 1].plot(epochs, history['val_acc'], label='Val Acc', linewidth=2, color='red', alpha=0.7)
    axes[0, 1].axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% Target', alpha=0.6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training/Validation Accuracy', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3)
    
    # Macro F1 evolution
    axes[1, 0].plot(epochs, history['val_macro_f1'], label='Val Macro F1', linewidth=2, color='purple', alpha=0.7)
    axes[1, 0].axhline(y=best_exp['macro_f1'], color='red', linestyle='--', 
                       linewidth=2, label=f'Best: {best_exp["macro_f1"]:.4f}', alpha=0.6)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Validation Macro F1 Evolution', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # Weighted F1 evolution
    axes[1, 1].plot(epochs, history['val_weighted_f1'], label='Val Weighted F1', linewidth=2, color='orange', alpha=0.7)
    axes[1, 1].axhline(y=best_exp['weighted_f1'], color='red', linestyle='--', 
                       linewidth=2, label=f'Best: {best_exp["weighted_f1"]:.4f}', alpha=0.6)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Weighted F1', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Validation Weighted F1 Evolution', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'Detailed Training History - BEST EXPERIMENT\n{best_exp["experiment_name"]}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plot_dir / '05_best_experiment_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 5: Best Experiment Training History")
    
    # ========================================================================
    # PLOT 6: CLASS 4 (MINORITY CLASS) PERFORMANCE
    # ========================================================================
    plt.figure(figsize=(14, 6))
    class4_f1s = [exp['per_class_f1'][4] for exp in experiments]
    colors_c4 = ['green' if f1 > 0.7 else 'orange' if f1 > 0.5 else 'red' for f1 in class4_f1s]
    
    bars = plt.bar(range(len(experiments)), class4_f1s, color=colors_c4, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Target: 0.70', alpha=0.6)
    
    for i, (bar, f1) in enumerate(zip(bars, class4_f1s)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Experiment', fontsize=13, fontweight='bold')
    plt.ylabel('Class 4 F1 Score', fontsize=13, fontweight='bold')
    plt.title('Class 4 (Minority Class) Performance Across Experiments\n(Most Important for Imbalanced Dataset)', 
              fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(experiments)), exp_names, fontsize=9)
    plt.ylim([0, max(class4_f1s) + 0.15])
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / '06_class4_minority_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 6: Class 4 Minority Performance")
    
    # ========================================================================
    # PLOT 7: VALIDATION ACCURACY EVOLUTION - ALL EXPERIMENTS
    # ========================================================================
    plt.figure(figsize=(16, 8))
    
    for i, exp in enumerate(experiments):
        history = exp['history']
        epochs = range(1, len(history['val_acc']) + 1)
        plt.plot(epochs, history['val_acc'], label=exp['experiment_name'], 
                linewidth=2, alpha=0.7, marker='o', markersize=2, markevery=10)
    
    plt.axhline(y=90, color='red', linestyle='--', linewidth=3, label='90% Target', alpha=0.8)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title('Validation Accuracy Evolution - All Experiments', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / '07_all_experiments_accuracy_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 7: All Experiments Accuracy Evolution")
    
    # ========================================================================
    # PLOT 8: CONFUSION MATRIX - BEST EXPERIMENT
    # ========================================================================
    # Load the confusion matrix from the best experiment
    best_exp_dir = Path('experiments') / best_exp['experiment_name'].replace(' ', '_')
    cm_path = best_exp_dir / 'confusion_matrix.png'
    
    if cm_path.exists():
        # Copy to comparison plots
        import shutil
        shutil.copy(cm_path, plot_dir / '08_best_experiment_confusion_matrix.png')
        print("  [OK] Plot 8: Best Experiment Confusion Matrix")
    else:
        print("   Plot 8: Confusion matrix not found")
    
    # ========================================================================
    # PLOT 9: PERFORMANCE RADAR CHART
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Select top 5 experiments by macro F1
    top_exps = sorted(experiments, key=lambda x: x['macro_f1'], reverse=True)[:5]
    
    categories = ['Val Acc\n(norm)', 'Macro F1', 'Weighted F1', 'Class 4 F1', 'Low Gap\n(inv)']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    colors_radar = plt.cm.Set2(np.linspace(0, 1, len(top_exps)))
    
    for i, exp in enumerate(top_exps):
        values = [
            exp['best_val_acc'] / 100,  # Normalize to 0-1
            exp['macro_f1'],
            exp['weighted_f1'],
            exp['per_class_f1'][4],
            1 - (min(exp['train_val_gap'], 20) / 20)  # Invert and normalize gap
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=exp['experiment_name'], 
                color=colors_radar[i], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors_radar[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Multi-Metric Performance Comparison\nTop 5 Experiments', 
              fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig(plot_dir / '09_performance_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 9: Performance Radar Chart")
    
    # ========================================================================
    # PLOT 10: EXPERIMENT RANKINGS
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Rank by Val Accuracy
    sorted_by_acc = sorted(enumerate(experiments), key=lambda x: x[1]['best_val_acc'], reverse=True)
    ranks_acc = [i+1 for i, _ in sorted_by_acc]
    names_acc = [experiments[i]['experiment_name'] for i, _ in sorted_by_acc]
    vals_acc = [experiments[i]['best_val_acc'] for i, _ in sorted_by_acc]
    
    axes[0, 0].barh(range(len(experiments)), vals_acc, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_yticks(range(len(experiments)))
    axes[0, 0].set_yticklabels([f"#{r}: {n}" for r, n in zip(ranks_acc, names_acc)], fontsize=9)
    axes[0, 0].set_xlabel('Val Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Ranked by Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].invert_yaxis()
    
    # Rank by Macro F1
    sorted_by_f1 = sorted(enumerate(experiments), key=lambda x: x[1]['macro_f1'], reverse=True)
    ranks_f1 = [i+1 for i, _ in sorted_by_f1]
    names_f1 = [experiments[i]['experiment_name'] for i, _ in sorted_by_f1]
    vals_f1 = [experiments[i]['macro_f1'] for i, _ in sorted_by_f1]
    
    axes[0, 1].barh(range(len(experiments)), vals_f1, color='purple', alpha=0.7, edgecolor='black')
    axes[0, 1].set_yticks(range(len(experiments)))
    axes[0, 1].set_yticklabels([f"#{r}: {n}" for r, n in zip(ranks_f1, names_f1)], fontsize=9)
    axes[0, 1].set_xlabel('Macro F1', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Ranked by Macro F1 Score', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].invert_yaxis()
    
    # Rank by Class 4 F1
    sorted_by_c4 = sorted(enumerate(experiments), key=lambda x: x[1]['per_class_f1'][4], reverse=True)
    ranks_c4 = [i+1 for i, _ in sorted_by_c4]
    names_c4 = [experiments[i]['experiment_name'] for i, _ in sorted_by_c4]
    vals_c4 = [experiments[i]['per_class_f1'][4] for i, _ in sorted_by_c4]
    
    axes[1, 0].barh(range(len(experiments)), vals_c4, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_yticks(range(len(experiments)))
    axes[1, 0].set_yticklabels([f"#{r}: {n}" for r, n in zip(ranks_c4, names_c4)], fontsize=9)
    axes[1, 0].set_xlabel('Class 4 F1', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Ranked by Class 4 (Minority) F1', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].invert_yaxis()
    
    # Rank by Low Gap (inverse - lower is better)
    sorted_by_gap = sorted(enumerate(experiments), key=lambda x: x[1]['train_val_gap'], reverse=False)
    ranks_gap = [i+1 for i, _ in sorted_by_gap]
    names_gap = [experiments[i]['experiment_name'] for i, _ in sorted_by_gap]
    vals_gap = [experiments[i]['train_val_gap'] for i, _ in sorted_by_gap]
    
    colors_gap_rank = ['green' if g < 8 else 'orange' if g < 12 else 'red' for g in vals_gap]
    axes[1, 1].barh(range(len(experiments)), vals_gap, color=colors_gap_rank, alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(experiments)))
    axes[1, 1].set_yticklabels([f"#{r}: {n}" for r, n in zip(ranks_gap, names_gap)], fontsize=9)
    axes[1, 1].set_xlabel('Train-Val Gap (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Ranked by Train-Val Gap (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].invert_yaxis()
    
    plt.suptitle('Experiment Rankings by Different Metrics', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plot_dir / '10_experiment_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 10: Experiment Rankings")
    
    print(f"\n{'='*80}")
    print(f"[OK] ALL 10 PLOTS GENERATED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Location: {plot_dir.absolute()}")
    print(f"\nGenerated plots:")
    for i in range(1, 11):
        plot_files = list(plot_dir.glob(f'{i:02d}_*.png'))
        if plot_files:
            print(f"  {i:2d}. {plot_files[0].name}")


# ============================================================================
# MAIN: CONTROLLED EXPERIMENTS FOR IMBALANCED CLASSIFICATION
# ============================================================================
def run_experiment(
    exp_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    max_epochs=100,
    patience=20,
    use_mixup=False,
    num_classes=5
):
    """Run single experiment with comprehensive tracking"""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}\n")
    
    best_macro_f1 = 0
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_macro_f1': [],
        'val_weighted_f1': []
    }
    
    for epoch in range(max_epochs):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch + 1, use_mixup=use_mixup
        )
        
        # Validation
        val_metrics = validate(model, val_loader, criterion, device, num_classes)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['macro_f1'])
            else:
                scheduler.step()
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        
        # Print epoch summary
        gap = train_acc - val_metrics['accuracy']
        print(f"Epoch {epoch+1:3d}/{max_epochs} | LR: {current_lr:.6f}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f} | Weighted F1: {val_metrics['weighted_f1']:.4f}")
        print(f"  Gap: {gap:+.2f}% | Per-Class F1: {[f'{x:.3f}' for x in val_metrics['per_class_f1']]}")
        
        # Check for best model
        score = val_metrics['macro_f1']  # Primary metric
        if score > best_macro_f1:
            best_macro_f1 = score
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            save_dir = Path('experiments') / exp_name.replace(' ', '_')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint (convert numpy to lists for safe serialization)
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                        [x.tolist() if isinstance(x, np.ndarray) else x for x in v] if isinstance(v, list) else v)
                    for k, v in val_metrics.items() if k not in ['predictions', 'labels']
                },
                'train_acc': float(train_acc),
                'history': {k: [float(x) for x in v] for k, v in history.items()}
            }
            torch.save(checkpoint_data, save_dir / 'best_model.pth')
            
            print(f"  >>> NEW BEST! Macro F1: {score:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            
            if val_metrics['accuracy'] >= 90.0:
                print(f"\n{'='*80}")
                print(f"*** TARGET ACHIEVED! Validation Accuracy: {val_metrics['accuracy']:.2f}% >= 90%! ***")
                print(f"{'='*80}\n")
                break
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    save_dir = Path('experiments') / exp_name.replace(' ', '_')
    checkpoint = torch.load(save_dir / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    final_metrics = validate(model, val_loader, criterion, device, num_classes)
    
    # Plot results
    plot_training_curves(history, save_dir / 'training_curves.png')
    plot_confusion_matrix(final_metrics['labels'], final_metrics['predictions'], 
                         save_dir / 'confusion_matrix.png')
    
    # Save classification report
    report = classification_report(
        final_metrics['labels'], 
        final_metrics['predictions'],
        digits=4,
        target_names=[f'Class {i}' for i in range(num_classes)]
    )
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save results
    results = {
        'experiment_name': exp_name,
        'best_epoch': best_epoch,
        'best_macro_f1': float(best_macro_f1),
        'best_val_acc': float(best_val_acc),
        'final_val_acc': float(final_metrics['accuracy']),
        'final_train_acc': float(checkpoint['train_acc']),
        'train_val_gap': float(checkpoint['train_acc'] - final_metrics['accuracy']),
        'macro_f1': float(final_metrics['macro_f1']),
        'weighted_f1': float(final_metrics['weighted_f1']),
        'macro_precision': float(final_metrics['macro_precision']),
        'macro_recall': float(final_metrics['macro_recall']),
        'per_class_f1': final_metrics['per_class_f1'],
        'per_class_accuracy': final_metrics['per_class_accuracy'],
        'target_achieved': final_metrics['accuracy'] >= 90.0,
        'history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"{'='*80}")
    print(f"Best Macro F1:     {best_macro_f1:.4f} (epoch {best_epoch})")
    print(f"Best Val Acc:      {best_val_acc:.2f}%")
    print(f"Train-Val Gap:     {checkpoint['train_acc'] - final_metrics['accuracy']:.2f}%")
    print(f"Per-Class F1:")
    for i, f1 in enumerate(final_metrics['per_class_f1']):
        print(f"  Class {i}: {f1:.4f}")
    target_msg = "ACHIEVED!" if results['target_achieved'] else f"{90.0 - final_metrics['accuracy']:.2f}% away"
    print(f"Target (90%+):     {target_msg}")
    print(f"{'='*80}\n")
    
    return results
# ============================================================================
# MAIN: CONTROLLED EXPERIMENTS FOR IMBALANCED CLASSIFICATION
# ============================================================================
def main():
    print("Loading data...")
    data_path = Path('preprocessed_dataset_full.npz')
    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
        return
    
    data = np.load(data_path)
    clinical = data['clinical']
    ecg = data['ecg']
    pcg = data['pcg']
    labels = data['labels']
    
    print(f"\nTotal dataset: {len(labels)} samples")
    print(f"  Clinical: {clinical.shape}")
    print(f"  ECG: {ecg.shape}")
    print(f"  PCG: {pcg.shape}")
    
    # Print overall class distribution
    class_counts = print_class_distribution(labels, "Overall Dataset")
    
    # STRATIFIED SPLIT (critical for imbalanced data)
    print("\nPerforming stratified train/val/test split...")
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Verify stratification
    print_class_distribution(labels[train_idx], "Training Set")
    print_class_distribution(labels[val_idx], "Validation Set")
    print_class_distribution(labels[test_idx], "Test Set")
    
    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_idx):5d} samples ({100*len(train_idx)/len(labels):.1f}%)")
    print(f"  Val:   {len(val_idx):5d} samples ({100*len(val_idx)/len(labels):.1f}%)")
    print(f"  Test:  {len(test_idx):5d} samples ({100*len(test_idx)/len(labels):.1f}%)")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Calculate class weights using training data only
    train_labels = labels[train_idx]
    num_classes = 5
    
    print("\n" + "="*80)
    print("CLASS WEIGHT STRATEGIES")
    print("="*80)
    
    # Strategy 1: Inverse frequency (standard)
    inv_freq_weights = get_inverse_freq_weights(train_labels, num_classes, smooth=False)
    print(f"\n1. Inverse Frequency Weights: {inv_freq_weights.numpy()}")
    
    # Strategy 2: Smoothed inverse frequency
    smooth_inv_freq_weights = get_inverse_freq_weights(train_labels, num_classes, smooth=True)
    print(f"2. Smoothed Inverse Freq Weights: {smooth_inv_freq_weights.numpy()}")
    
    # Strategy 3: Effective number (best for severe imbalance)
    eff_num_weights_0999 = get_effective_number_weights(train_labels, num_classes, beta=0.999)
    eff_num_weights_09999 = get_effective_number_weights(train_labels, num_classes, beta=0.9999)
    eff_num_weights_099999 = get_effective_number_weights(train_labels, num_classes, beta=0.99999)
    
    print(f"3. Effective Number (=0.999):   {eff_num_weights_0999.numpy()}")
    print(f"4. Effective Number (=0.9999):  {eff_num_weights_09999.numpy()}")
    print(f"5. Effective Number (=0.99999): {eff_num_weights_099999.numpy()}")
    
    # ========================================================================
    # EXPERIMENT 0: BASELINE (NO SPECIAL IMBALANCE HANDLING)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 0: BASELINE (NO CLASS WEIGHTS)")
    print("="*80)
    
    train_dataset = ImbalancedDataset(
        clinical[train_idx], ecg[train_idx], pcg[train_idx], train_labels,
        mode='train', minority_class_ids=[3, 4], aug_strength=0.5
    )
    val_dataset = ImbalancedDataset(
        clinical[val_idx], ecg[val_idx], pcg[val_idx], labels[val_idx],
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # No class weights
    criterion = ImbalancedLoss(
        class_weights=torch.ones(num_classes).to(device), 
        num_classes=num_classes,
        label_smoothing=0.0
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp0_results = run_experiment(
        "Exp0_Baseline_NoWeights",
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=False, num_classes=num_classes
    )
    
    # ========================================================================
    # EXPERIMENT 1: SMOOTHED INVERSE FREQUENCY WEIGHTS
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: SMOOTHED INVERSE FREQUENCY WEIGHTS")
    print("="*80)
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    criterion = ImbalancedLoss(
        class_weights=smooth_inv_freq_weights.to(device),
        num_classes=num_classes,
        label_smoothing=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp1_results = run_experiment(
        "Exp1_SmoothedInvFreq",
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=False, num_classes=num_classes
    )
    
    # ========================================================================
    # EXPERIMENT 2: EFFECTIVE NUMBER WEIGHTS (β=0.9999)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: EFFECTIVE NUMBER WEIGHTS (=0.9999)")
    print("="*80)
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    criterion = ImbalancedLoss(
        class_weights=eff_num_weights_09999.to(device),
        num_classes=num_classes,
        label_smoothing=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp2_results = run_experiment(
        "Exp2_EffectiveNum_0.9999",
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=False, num_classes=num_classes
    )
    
    # ========================================================================
    # EXPERIMENT 3: WEIGHTED RANDOM SAMPLER ONLY
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: WEIGHTED RANDOM SAMPLER ONLY")
    print("="*80)
    
    # Create weighted sampler
    sample_weights = smooth_inv_freq_weights[train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )
    
    train_loader_sampled = DataLoader(
        train_dataset, batch_size=32, sampler=sampler, num_workers=0
    )
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    criterion = ImbalancedLoss(
        class_weights=torch.ones(num_classes).to(device),  # No loss weights
        num_classes=num_classes,
        label_smoothing=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp3_results = run_experiment(
        "Exp3_WeightedSampler",
        model, train_loader_sampled, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=False, num_classes=num_classes
    )
    
    # ========================================================================
    # EXPERIMENT 4: BEST WEIGHTS + STRONGER REGULARIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 4: EFFECTIVE NUM WEIGHTS + HIGHER WEIGHT DECAY")
    print("="*80)
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    criterion = ImbalancedLoss(
        class_weights=eff_num_weights_09999.to(device),
        num_classes=num_classes,
        label_smoothing=0.0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=5e-4  # Increased
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp4_results = run_experiment(
        "Exp4_BestWeights_HigherWD",
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=False, num_classes=num_classes
    )
    
    # ========================================================================
    # EXPERIMENT 5: BEST CONFIG + LABEL SMOOTHING
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 5: BEST CONFIG + LABEL SMOOTHING")
    print("="*80)
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    criterion = ImbalancedLoss(
        class_weights=eff_num_weights_09999.to(device),
        num_classes=num_classes,
        label_smoothing=0.1  # Added
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp5_results = run_experiment(
        "Exp5_BestConfig_LabelSmooth",
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=False, num_classes=num_classes
    )
    
    # ========================================================================
    # EXPERIMENT 6: BEST CONFIG + WEIGHTED SAMPLING
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 6: EFFECTIVE NUM WEIGHTS + MODERATE SAMPLING")
    print("="*80)
    
    # Moderate sampling (less aggressive than exp3)
    moderate_weights = torch.sqrt(smooth_inv_freq_weights)  # Less aggressive
    sample_weights_moderate = moderate_weights[train_labels]
    sampler_moderate = WeightedRandomSampler(
        sample_weights_moderate, len(sample_weights_moderate), replacement=True
    )
    
    train_loader_moderate = DataLoader(
        train_dataset, batch_size=32, sampler=sampler_moderate, num_workers=0
    )
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    criterion = ImbalancedLoss(
        class_weights=eff_num_weights_09999.to(device),
        num_classes=num_classes,
        label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp6_results = run_experiment(
        "Exp6_BestConfig_ModerateSampling",
        model, train_loader_moderate, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=False, num_classes=num_classes
    )
    
    # ========================================================================
    # EXPERIMENT 7: BEST CONFIG + MIXUP
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 7: BEST CONFIGURATION + MIXUP")
    print("="*80)
    
    model = ACRMFNet(num_classes=5, embedding_dim=256).to(device)
    criterion = ImbalancedLoss(
        class_weights=eff_num_weights_09999.to(device),
        num_classes=num_classes,
        label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    exp7_results = run_experiment(
        "Exp7_BestConfig_MixUp",
        model, train_loader_moderate, val_loader, criterion, optimizer, scheduler, device,
        max_epochs=100, patience=20, use_mixup=True, num_classes=num_classes
    )
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON OF ALL EXPERIMENTS")
    print("="*80 + "\n")
    
    experiments = [
        exp0_results, exp1_results, exp2_results, exp3_results,
        exp4_results, exp5_results, exp6_results, exp7_results
    ]
    
    print(f"{'Experiment':<35} {'Val Acc':<10} {'Macro F1':<10} {'Class4 F1':<10} {'Gap':<8} {'Target'}")
    print("-" * 95)
    
    for exp in experiments:
        target = "PASS" if exp['target_achieved'] else f"FAIL -{90.0 - exp['best_val_acc']:.2f}%"
        class4_f1 = exp['per_class_f1'][4]
        print(f"{exp['experiment_name']:<35} {exp['best_val_acc']:>6.2f}%  "
              f"{exp['macro_f1']:>7.4f}  {class4_f1:>7.4f}  "
              f"{exp['train_val_gap']:>6.2f}%  {target}")
    
    # Find best experiment
    best_exp = max(experiments, key=lambda x: x['macro_f1'])
    print(f"\n{'='*80}")
    print(f"BEST EXPERIMENT: {best_exp['experiment_name']}")
    print(f"{'='*80}")
    print(f"  Best Val Accuracy:     {best_exp['best_val_acc']:.2f}%")
    print(f"  Macro F1:              {best_exp['macro_f1']:.4f}")
    print(f"  Train-Val Gap:         {best_exp['train_val_gap']:.2f}%")
    print(f"  Per-Class F1 Scores:")
    for i, f1 in enumerate(best_exp['per_class_f1']):
        print(f"    Class {i}: {f1:.4f}")
    target_msg2 = "ACHIEVED!" if best_exp['target_achieved'] else "Not achieved"
    print(f"  Target (90%+):         {target_msg2}")
    print(f"{'='*80}\n")
    
    # Save comparison
    comparison = {
        'experiments': experiments,
        'best_experiment': best_exp['experiment_name'],
        'best_macro_f1': float(best_exp['macro_f1']),
        'best_val_acc': float(best_exp['best_val_acc'])
    }
    
    with open('experiments/comparison_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("All results saved to experiments/ directory")
    
    # ========================================================================
    # AUTOMATIC COMPREHENSIVE VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE PLOTS")
    print("="*80 + "\n")
    
    generate_comprehensive_plots(experiments, best_exp)
    
    print("\n" + "="*80)
    print("TRAINING AND VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nBest Experiment: {best_exp['experiment_name']}")
    print(f"Best Validation Accuracy: {best_exp['best_val_acc']:.2f}%")
    print(f"Target (90%+): {target_msg2}")
    print(f"\nAll plots saved to: experiments/comparison_plots/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
