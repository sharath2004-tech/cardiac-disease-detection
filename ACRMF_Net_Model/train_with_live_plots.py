"""
Training with Automatic Live Visualization
===========================================
Trains model and automatically generates plots during and after training.

Features:
- Real-time training progress plots (updated every epoch)
- Confusion matrix visualization
- Per-class performance charts
- Final comprehensive report with all metrics
- No need for separate visualization scripts!
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score, 
                             accuracy_score)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from test_complete_model import ACRMFNet

print("\n" + "="*80)
print("TRAINING WITH AUTOMATIC LIVE VISUALIZATION")
print("="*80)


# ============================================================================
# DATASET
# ============================================================================
class SimpleDataset(Dataset):
    def __init__(self, clinical, ecg, pcg, labels, mode='train'):
        self.clinical = torch.FloatTensor(clinical)
        self.ecg = torch.FloatTensor(ecg)
        self.pcg = torch.FloatTensor(pcg)
        self.labels = torch.LongTensor(labels)
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.clinical[idx], self.ecg[idx], self.pcg[idx], self.labels[idx]


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_live_progress(history, epoch, save_dir='results/live_plots'):
    """Generate live training progress plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title(f'Loss - Epoch {epoch}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].axhline(y=0.90, color='g', linestyle='--', label='90% Target')
    axes[0, 1].set_title(f'Accuracy - Epoch {epoch}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title(f'F1 Score - Epoch {epoch}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary
    axes[1, 1].axis('off')
    best_val_acc = max(history['val_acc'])
    summary = f"""
    CURRENT STATUS (Epoch {epoch})
    
    Best Val Acc: {best_val_acc*100:.2f}%
    Current Val Acc: {history['val_acc'][-1]*100:.2f}%
    Current Train Acc: {history['train_acc'][-1]*100:.2f}%
    
    Gap: {(history['train_acc'][-1] - history['val_acc'][-1])*100:.2f}%
    
    {"[OK] Target Reached!" if best_val_acc >= 0.90 else "[...] In Progress..."}
    """
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'live_training_progress.png', dpi=150)
    plt.close()


def plot_confusion_matrix_final(labels, predictions, num_classes, save_dir='results/final_plots'):
    """Plot final confusion matrix"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(labels, predictions)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='RdYlGn', ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Recall %)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150)
    plt.close()


def plot_per_class_performance(per_class_acc, num_classes, save_dir='results/final_plots'):
    """Plot per-class accuracy"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(range(num_classes))
    accs = [per_class_acc.get(cls, 0) * 100 for cls in classes]
    colors = ['green' if a >= 80 else 'orange' if a >= 70 else 'red' for a in accs]
    
    bars = ax.bar(classes, accs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=80, color='red', linestyle='--', label='80% Threshold', linewidth=2)
    ax.axhline(y=90, color='green', linestyle='--', label='90% Target', linewidth=2)
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_accuracy.png', dpi=150)
    plt.close()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for clinical, ecg, pcg, labels in tqdm(loader, desc="Training"):
        clinical, ecg, pcg, labels = clinical.to(device), ecg.to(device), pcg.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(clinical, ecg, pcg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return total_loss / len(loader), acc, f1


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for clinical, ecg, pcg, labels in tqdm(loader, desc="Validation"):
            clinical, ecg, pcg, labels = clinical.to(device), ecg.to(device), pcg.to(device), labels.to(device)
            
            outputs = model(clinical, ecg, pcg)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class accuracy
    per_class_acc = {}
    for cls in range(num_classes):
        mask = np.array(all_labels) == cls
        if mask.sum() > 0:
            per_class_acc[cls] = accuracy_score(
                np.array(all_labels)[mask],
                np.array(all_preds)[mask]
            )
    
    return total_loss / len(loader), acc, f1, per_class_acc, all_labels, all_preds


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Load data
    print("\nLoading cleaned data...")
    clinical_data = np.load('cleaned_data/clinical_cleaned.npy')
    ecg_data = np.load('cleaned_data/ecg_cleaned.npy')
    pcg_data = np.load('cleaned_data/pcg_cleaned.npy')
    labels = np.load('cleaned_data/labels_cleaned.npy')
    
    with open('cleaned_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata['num_classes']
    print(f"[OK] Loaded {len(labels)} samples, {num_classes} classes")
    
    # Split
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])
    
    # Datasets
    train_dataset = SimpleDataset(clinical_data[train_idx], ecg_data[train_idx], 
                                 pcg_data[train_idx], labels[train_idx], 'train')
    val_dataset = SimpleDataset(clinical_data[val_idx], ecg_data[val_idx], 
                               pcg_data[val_idx], labels[val_idx], 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = ACRMFNet(num_classes=num_classes, embedding_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
    
    # Training
    print("\n" + "="*80)
    print("TRAINING START - Plots will be generated automatically!")
    print("="*80)
    
    num_epochs = 100
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, val_f1, per_class_acc, val_labels, val_preds = validate(
            model, val_loader, criterion, device, num_classes
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
        print(f"Per-class acc: {per_class_acc}")
        
        # Generate live plots every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            plot_live_progress(history, epoch)
            print(f"  [OK] Live plots updated: results/live_plots/")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'per_class_acc': per_class_acc
            }, 'checkpoints/best_model.pth')
            print(f"  [OK] Best model saved! Val Acc: {val_acc*100:.2f}%")
        
        scheduler.step(val_acc)
        
        # Early stopping
        if epoch > 20 and val_acc < 0.6:
            print("\n[WARN] Low accuracy, consider checking data/model")
    
    # Final visualizations
    print("\n" + "="*80)
    print("GENERATING FINAL VISUALIZATIONS")
    print("="*80)
    
    plot_live_progress(history, num_epochs, 'results/final_plots')
    plot_confusion_matrix_final(val_labels, val_preds, num_classes)
    plot_per_class_performance(per_class_acc, num_classes)
    
    print("\n[OK] All visualizations saved!")
    print("     Live plots: results/live_plots/")
    print("     Final plots: results/final_plots/")
    
    print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")
    print("\n" + "="*80)
    print("[OK] TRAINING COMPLETE WITH AUTO-VISUALIZATION!")
    print("="*80)


if __name__ == "__main__":
    main()
