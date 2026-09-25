"""
ACRMF-Net: Direct Training Script
Single unified training pipeline as per architecture diagram
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import model and loss
from models.acrmf import ACRMFNet
from losses.composite_loss import CompositeLoss
from handle_imbalance import compute_effective_number_weights

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

print("\n" + "="*80)
print("ACRMF-Net: Adaptive Confidence-Reliability Multimodal Fusion")
print("End-to-End Training Pipeline")
print("="*80 + "\n")


class CardiacDataset(Dataset):
    """Dataset class for multimodal cardiac data with augmentation support"""
    
    def __init__(self, clinical, ecg, pcg, labels, augment=False):
        self.clinical = torch.FloatTensor(clinical)
        self.ecg = torch.FloatTensor(ecg)
        self.pcg = torch.FloatTensor(pcg)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def augment_ecg(self, ecg):
        """Apply random augmentation to ECG signal"""
        ecg = ecg.clone()
        
        # Random jitter (add noise)
        if np.random.rand() < 0.5:
            noise = torch.randn_like(ecg) * 0.05
            ecg = ecg + noise
        
        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            ecg = ecg * scale
        
        # Random shift
        if np.random.rand() < 0.3:
            shift = np.random.randint(-50, 50)
            ecg = torch.roll(ecg, shift, dims=0)
        
        return ecg
    
    def augment_pcg(self, pcg):
        """Apply random augmentation to PCG spectrogram"""
        pcg = pcg.clone()
        
        # Frequency masking
        if np.random.rand() < 0.5:
            freq_mask_size = np.random.randint(5, 15)
            freq_start = np.random.randint(0, max(1, pcg.shape[0] - freq_mask_size))
            pcg[freq_start:freq_start+freq_mask_size, :] = 0.0
        
        # Time masking
        if np.random.rand() < 0.5:
            time_mask_size = np.random.randint(5, 15)
            time_start = np.random.randint(0, max(1, pcg.shape[1] - time_mask_size))
            pcg[:, time_start:time_start+time_mask_size] = 0.0
        
        # Random scaling
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            pcg = pcg * scale
        
        return pcg
    
    def __getitem__(self, idx):
        clinical = self.clinical[idx]
        ecg = self.ecg[idx]
        pcg = self.pcg[idx]
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment:
            ecg = self.augment_ecg(ecg)
            pcg = self.augment_pcg(pcg)
        
        return {
            'clinical': clinical,
            'ecg': ecg,
            'pcg': pcg,
            'label': label
        }


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        clinical = batch['clinical'].to(device)
        ecg = batch['ecg'].to(device)
        pcg = batch['pcg'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(clinical, ecg, pcg)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs['fused_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            clinical = batch['clinical'].to(device)
            ecg = batch['ecg'].to(device)
            pcg = batch['pcg'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(clinical, ecg, pcg)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs['fused_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("1. Loading preprocessed data...")
    data_path = Path('preprocessed_dataset_full.npz')
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
        return
    
    data = np.load(data_path)
    clinical = data['clinical']
    ecg = data['ecg']
    pcg = data['pcg']
    labels = data['labels']
    
    print(f"   Total samples: {len(labels)}")
    print(f"   Clinical: {clinical.shape}")
    print(f"   ECG: {ecg.shape}")
    print(f"   PCG: {pcg.shape}")
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n   Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"     Class {cls}: {count} ({100*count/len(labels):.1f}%)")
    
    # ========================================================================
    # 2. Split Data (70% train, 15% val, 15% test)
    # ========================================================================
    print("\n2. Splitting data (stratified)...")
    
    # Train/temp split
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    )
    
    # Val/test split
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    print(f"   Train: {len(train_idx)} samples ({100*len(train_idx)/len(labels):.1f}%)")
    print(f"   Val:   {len(val_idx)} samples ({100*len(val_idx)/len(labels):.1f}%)")
    print(f"   Test:  {len(test_idx)} samples ({100*len(test_idx)/len(labels):.1f}%)")
    
    # Create datasets (with augmentation for training)
    train_dataset = CardiacDataset(
        clinical[train_idx], ecg[train_idx], pcg[train_idx], labels[train_idx],
        augment=True  # Enable augmentation for training data
    )
    val_dataset = CardiacDataset(
        clinical[val_idx], ecg[val_idx], pcg[val_idx], labels[val_idx],
        augment=False  # No augmentation for validation
    )
    test_dataset = CardiacDataset(
        clinical[test_idx], ecg[test_idx], pcg[test_idx], labels[test_idx],
        augment=False  # No augmentation for test
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # ========================================================================
    # 3. Initialize Model
    # ========================================================================
    print("\n3. Initializing ACRMF-Net model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    model = ACRMFNet(
        clinical_input_dim=13,
        ecg_input_dim=1000,
        pcg_input_dim=1000,
        embedding_dim=128,
        num_classes=5,
        dropout=0.3,
        use_ren=True,  # Module 7: Reliability Estimation
        use_cen=True,  # Module 8: Confidence Estimation
        use_awg=True   # Module 9: Adaptive Weight Generator
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # ========================================================================
    # 4. Initialize Loss and Optimizer
    # ========================================================================
    print("\n4. Initializing training components...")
    
    # Compute class weights to handle imbalance
    print("   Computing class weights for imbalanced dataset...")
    train_labels = labels[train_idx]
    class_weights = compute_effective_number_weights(train_labels, beta=0.9999)
    class_weights = class_weights.to(device)
    
    # Composite loss (combines all loss components)
    criterion = CompositeLoss(
        num_classes=5,
        lambda_cls=1.0,      # Classification
        lambda_rel=0.1,      # Reliability
        lambda_conf=0.1,     # Confidence
        lambda_fusion=0.05,  # Fusion
        lambda_consist=0.1,  # Consistency
        class_weights=class_weights  # Add class weights
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    print("   Loss: CompositeLoss")
    print("   Optimizer: AdamW (lr=1e-4, wd=1e-4)")
    print("   Scheduler: ReduceLROnPlateau")
    
    # ========================================================================
    # 5. Training Loop
    # ========================================================================
    print("\n5. Starting training...")
    print("="*80)
    
    max_epochs = 100
    patience = 20
    best_val_f1 = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }, 'best_model.pth')
            
            print(f"  ✓ New best model saved! (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # ========================================================================
    # 6. Final Evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("6. Final Evaluation on Test Set")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  Macro F1: {test_f1:.4f}")
    
    # Classification report
    print("\nPer-Class Performance:")
    print(classification_report(test_labels, test_preds, 
                                target_names=[f'Class {i}' for i in range(5)]))
    
    # ========================================================================
    # 7. Save Results
    # ========================================================================
    print("\n7. Saving results...")
    
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_f1': float(best_val_f1),
        'best_val_acc': float(checkpoint['val_acc'] * 100),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc * 100),
        'test_f1': float(test_f1),
        'history': history
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("   ✓ Results saved to training_results.json")
    print("   ✓ Best model saved to best_model.pth")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nBest Validation:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Accuracy: {checkpoint['val_acc']*100:.2f}%")
    print(f"  Macro F1: {best_val_f1:.4f}")
    print(f"\nFinal Test:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  Macro F1: {test_f1:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
