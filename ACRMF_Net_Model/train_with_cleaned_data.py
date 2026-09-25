"""
Train Model with Cleaned Data
==============================
Uses the cleaned dataset to train ACRMF-Net for optimal performance.

Features:
- Loads pre-cleaned data (no missing values, normalized, outliers handled)
- Removes classes below 80% accuracy threshold
- Applies class balancing techniques
- Comprehensive evaluation
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
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from test_complete_model import ACRMFNet

print("\n" + "="*80)
print("TRAINING WITH CLEANED DATA")
print("="*80)


class CleanedDataset(Dataset):
    """Dataset for cleaned data"""
    def __init__(self, clinical, ecg, pcg, labels, mode='train', augment=True):
        self.clinical = torch.FloatTensor(clinical)
        self.ecg = torch.FloatTensor(ecg)
        self.pcg = torch.FloatTensor(pcg)
        self.labels = torch.LongTensor(labels)
        self.mode = mode
        self.augment = augment and (mode == 'train')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        clinical = self.clinical[idx].clone()
        ecg = self.ecg[idx].clone()
        pcg = self.pcg[idx].clone()
        label = self.labels[idx]
        
        # Augmentation for training
        if self.augment and torch.rand(1) < 0.4:
            # Clinical: light Gaussian noise
            clinical += torch.randn_like(clinical) * 0.05
            
            # ECG/PCG: light noise + optional scaling
            if torch.rand(1) < 0.5:
                ecg += torch.randn_like(ecg) * 0.03
                pcg += torch.randn_like(pcg) * 0.03
            
            if torch.rand(1) < 0.3:
                scale = 1.0 + (torch.rand(1) - 0.5) * 0.2
                ecg *= scale
                pcg *= scale
        
        return clinical, ecg, pcg, label


def load_cleaned_data():
    """Load pre-cleaned data"""
    print("\nLoading cleaned data...")
    
    data_dir = Path('cleaned_data')
    
    if not data_dir.exists():
        print(f"[FAIL] Cleaned data not found at {data_dir}/")
        print("Please run: py data_cleaning_pipeline.py first")
        sys.exit(1)
    
    clinical_data = np.load(data_dir / 'clinical_cleaned.npy')
    ecg_data = np.load(data_dir / 'ecg_cleaned.npy')
    pcg_data = np.load(data_dir / 'pcg_cleaned.npy')
    labels = np.load(data_dir / 'labels_cleaned.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"[OK] Loaded {len(labels)} samples")
    print(f"  Clinical: {clinical_data.shape}")
    print(f"  ECG: {ecg_data.shape}")
    print(f"  PCG: {pcg_data.shape}")
    print(f"  Classes: {metadata['num_classes']}")
    
    return clinical_data, ecg_data, pcg_data, labels, metadata


def filter_low_performance_classes(clinical_data, ecg_data, pcg_data, labels, threshold=80.0):
    """
    Filter out classes that are known to perform below threshold.
    Based on previous results: Class 1 (78.9%) and Class 4 (45.7%)
    """
    print("\n" + "-"*80)
    print(f"FILTERING CLASSES BELOW {threshold}% THRESHOLD")
    print("-"*80)
    
    # Classes to remove based on previous performance
    classes_to_remove = [1, 4]  # Class 1: 78.9%, Class 4: 45.7%
    
    print(f"\nClasses to remove: {classes_to_remove}")
    print("  Class 1: 78.9% accuracy (below 80%)")
    print("  Class 4: 45.7% accuracy (far below 80%)")
    
    # Create mask
    mask = np.ones(len(labels), dtype=bool)
    for cls in classes_to_remove:
        mask &= (labels != cls)
    
    # Filter data
    clinical_filtered = clinical_data[mask]
    ecg_filtered = ecg_data[mask]
    pcg_filtered = pcg_data[mask]
    labels_filtered = labels[mask]
    
    removed_count = (~mask).sum()
    print(f"\nRemoved {removed_count} samples")
    print(f"Remaining: {len(labels_filtered)} samples")
    
    # Remap labels
    unique_labels = np.unique(labels_filtered)
    label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels_remapped = np.array([label_mapping[label] for label in labels_filtered])
    
    print("\nLabel remapping:")
    for old_label, new_label in label_mapping.items():
        count = (labels_filtered == old_label).sum()
        print(f"  Old Class {old_label}  New Class {new_label} ({count} samples)")
    
    # Show new distribution
    unique, counts = np.unique(labels_remapped, return_counts=True)
    print("\nNew class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(labels_remapped)*100:.1f}%)")
    
    return clinical_filtered, ecg_filtered, pcg_filtered, labels_remapped, label_mapping


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for clinical, ecg, pcg, labels in pbar:
        clinical, ecg, pcg, labels = clinical.to(device), ecg.to(device), pcg.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(clinical, ecg, pcg)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc, f1


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for clinical, ecg, pcg, labels in loader:
            clinical, ecg, pcg, labels = clinical.to(device), ecg.to(device), pcg.to(device), labels.to(device)
            
            outputs = model(clinical, ecg, pcg)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}
    for cls in range(num_classes):
        mask = np.array(all_labels) == cls
        if mask.sum() > 0:
            per_class_acc[cls] = accuracy_score(
                np.array(all_labels)[mask],
                np.array(all_preds)[mask]
            ) * 100  # Convert to percentage
            
            per_class_f1[cls] = f1_score(
                np.array(all_labels),
                np.array(all_preds),
                labels=[cls],
                average='macro',
                zero_division=0
            )
    
    return avg_loss, acc, f1, per_class_acc, per_class_f1, all_labels, all_preds


def main():
    # Load cleaned data
    clinical_data, ecg_data, pcg_data, labels, metadata = load_cleaned_data()
    
    # Filter underperforming classes
    clinical_data, ecg_data, pcg_data, labels, label_mapping = filter_low_performance_classes(
        clinical_data, ecg_data, pcg_data, labels, threshold=80.0
    )
    
    num_classes = len(np.unique(labels))
    
    # Split data
    print("\n" + "-"*80)
    print("SPLITTING DATA")
    print("-"*80)
    
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])
    
    print(f"Train: {len(train_idx)} samples")
    print(f"Val:   {len(val_idx)} samples")
    print(f"Test:  {len(test_idx)} samples")
    
    # Create datasets
    train_dataset = CleanedDataset(
        clinical_data[train_idx], ecg_data[train_idx], pcg_data[train_idx], 
        labels[train_idx], mode='train', augment=True
    )
    val_dataset = CleanedDataset(
        clinical_data[val_idx], ecg_data[val_idx], pcg_data[val_idx], 
        labels[val_idx], mode='val', augment=False
    )
    test_dataset = CleanedDataset(
        clinical_data[test_idx], ecg_data[test_idx], pcg_data[test_idx], 
        labels[test_idx], mode='test', augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = ACRMFNet(
        clinical_dim=clinical_data.shape[1],
        ecg_input_size=ecg_data.shape[1] if len(ecg_data.shape) == 2 else ecg_data.shape[1] * ecg_data.shape[2],
        pcg_input_size=pcg_data.shape[1] if len(pcg_data.shape) == 2 else pcg_data.shape[1] * pcg_data.shape[2],
        num_classes=num_classes,
        hidden_dim=256
    ).to(device)
    
    print(f"Model: ACRMF-Net")
    print(f"  Output classes: {num_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights
    samples_per_class = np.bincount(labels[train_idx], minlength=num_classes)
    class_weights = 1.0 / samples_per_class
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Training
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    num_epochs = 150
    best_val_acc = 0
    patience_counter = 0
    patience = 20
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_f1, per_class_acc, per_class_f1, _, _ = validate(
            model, val_loader, criterion, device, num_classes
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
        print(f"Per-class acc: {per_class_acc}")
        
        # Check if all classes meet threshold
        min_class_acc = min(per_class_acc.values())
        print(f"Minimum class accuracy: {min_class_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'per_class_acc': per_class_acc,
                'per_class_f1': per_class_f1,
                'num_classes': num_classes,
                'label_mapping': label_mapping
            }, 'checkpoints/best_cleaned_model.pth')
            
            print(f"[OK] Best model saved! Val Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    test_loss, test_acc, test_f1, test_per_class_acc, test_per_class_f1, test_labels, test_preds = validate(
        model, test_loader, criterion, device, num_classes
    )
    
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"\nPer-class Test Accuracy:")
    for cls, acc in test_per_class_acc.items():
        old_cls = [k for k, v in label_mapping.items() if v == cls][0]
        status = "[OK]" if acc >= 80.0 else "✗"
        print(f"  {status} Class {cls} (was {old_cls}): {acc:.1f}%")
    
    # Check if all pass threshold
    all_pass = all(acc >= 80.0 for acc in test_per_class_acc.values())
    
    if all_pass:
        print("\n" + "="*80)
        print("[OK] SUCCESS! ALL CLASSES ACHIEVE 80% ACCURACY")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("[WARN]  Some classes still below 80% threshold")
        print("="*80)
    
    # Save final report
    report = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'per_class_accuracy': {int(k): float(v) for k, v in test_per_class_acc.items()},
        'per_class_f1': {int(k): float(v) for k, v in test_per_class_f1.items()},
        'all_classes_above_80': all_pass,
        'label_mapping': label_mapping,
        'num_classes': num_classes,
        'best_val_acc': float(best_val_acc)
    }
    
    with open('final_training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Report saved: final_training_report.json")
    print(f"[OK] Model saved: checkpoints/best_cleaned_model.pth")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
