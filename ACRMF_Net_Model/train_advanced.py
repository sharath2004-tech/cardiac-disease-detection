"""
Advanced Training Pipeline with All Improvements
================================================

Improvements Implemented:
1. ✓ Focal Loss (focuses on hard examples)
2. ✓ Label Smoothing (prevents overconfidence)
3. ✓ Mixup Augmentation (mixes samples during training)
4. ✓ SMOTE-Tomek balanced dataset
5. ✓ Class weights (Effective Number)
6. ✓ WeightedRandomSampler
7. ✓ Advanced data augmentation
8. ✓ CosineAnnealingWarmRestarts scheduler

Expected Performance: 90-92% accuracy
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import logging
from collections import Counter
from datetime import datetime

# Import model and losses
sys.path.append(str(Path(__file__).parent))
from models.acrmf import ACRMFNet
from losses.composite_loss import FocalLoss, LabelSmoothingLoss


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data
    use_balanced_data = True  # Use SMOTE-Tomek balanced dataset
    data_dir = "balanced_data" if use_balanced_data else "cleaned_data"
    
    # Model
    num_classes = 5
    ecg_channels = 12
    pcg_size = 128
    clinical_features = 13
    
    # Training
    batch_size = 32
    num_epochs = 150
    learning_rate = 0.0001
    weight_decay = 0.01
    
    # Loss configuration
    use_focal_loss = True
    focal_gamma = 2.0
    focal_alpha_weight = 0.7  # Weight for focal loss in combination
    
    use_label_smoothing = True
    label_smoothing = 0.1
    
    # Mixup
    use_mixup = True
    mixup_alpha = 0.2  # Beta distribution parameter
    mixup_prob = 0.5   # Probability of applying mixup
    
    # Augmentation
    augment_train = True
    
    # Scheduler
    scheduler_type = 'cosine_warm_restarts'  # 'cosine_warm_restarts' or 'reduce_on_plateau'
    T_0 = 10  # Cosine annealing: epochs until first restart
    T_mult = 2  # Cosine annealing: factor to increase T_0 after restart
    
    # Early stopping
    patience = 30
    min_delta = 0.001
    
    # Weights & Sampling
    use_class_weights = True
    use_weighted_sampler = True
    
    # Output
    output_dir = "experiments/advanced_training"
    save_best_only = True
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Random seed
    random_seed = 42


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset with Mixup Support
# ============================================================================

class CardiacDataset(Dataset):
    """Dataset with support for Mixup augmentation"""
    
    def __init__(self, ecg, pcg, clinical, labels, augment=False, mixup=False):
        # Transpose ECG from (N, 1000, 12) to (N, 12, 1000) for Conv1D
        self.ecg = torch.FloatTensor(ecg).transpose(1, 2)  # (N, 12, 1000)
        self.pcg = torch.FloatTensor(pcg)  # (N, 2000)
        self.clinical = torch.FloatTensor(clinical)  # (N, 8)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.mixup = mixup
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        ecg = self.ecg[idx]
        pcg = self.pcg[idx]
        clinical = self.clinical[idx]
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augment and torch.rand(1) < 0.5:
            ecg = self._augment_ecg(ecg)
            pcg = self._augment_pcg(pcg)
        
        return ecg, pcg, clinical, label
    
    def _augment_ecg(self, ecg):
        """ECG augmentation - expects (12, 1000) shape"""
        # Jitter
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(ecg) * 0.03
            ecg = ecg + noise
        
        # Scaling
        if torch.rand(1) < 0.5:
            scale = torch.FloatTensor(1).uniform_(0.95, 1.05)
            ecg = ecg * scale
        
        return ecg
    
    def _augment_pcg(self, pcg):
        """PCG augmentation - expects (2000,) shape"""
        # Add jitter
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(pcg) * 0.03
            pcg = pcg + noise
        
        # Scaling
        if torch.rand(1) < 0.5:
            scale = torch.FloatTensor(1).uniform_(0.95, 1.05)
            pcg = pcg * scale
        
        return pcg


def mixup_data(ecg, pcg, clinical, labels, alpha=0.2):
    """
    Apply Mixup augmentation
    
    Returns mixed inputs and pairs of targets with lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = ecg.size(0)
    index = torch.randperm(batch_size).to(ecg.device)
    
    mixed_ecg = lam * ecg + (1 - lam) * ecg[index]
    mixed_pcg = lam * pcg + (1 - lam) * pcg[index]
    mixed_clinical = lam * clinical + (1 - lam) * clinical[index]
    
    labels_a, labels_b = labels, labels[index]
    
    return mixed_ecg, mixed_pcg, mixed_clinical, labels_a, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Calculate loss for mixup"""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


# ============================================================================
# Loss Functions
# ============================================================================

def compute_effective_number_weights(labels, beta=0.9999):
    """Compute class weights using Effective Number of Samples"""
    counter = Counter(labels)
    num_classes = len(counter)
    
    effective_num = {}
    for cls in range(num_classes):
        n = counter.get(cls, 0)
        if n == 0:
            effective_num[cls] = 0
        else:
            effective_num[cls] = (1.0 - np.power(beta, n)) / (1.0 - beta)
    
    weights = []
    for cls in range(num_classes):
        if effective_num[cls] == 0:
            weights.append(0)
        else:
            weights.append(1.0 / effective_num[cls])
    
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes
    
    return torch.FloatTensor(weights)


class CombinedLoss(nn.Module):
    """
    Combines Focal Loss and Label Smoothing Loss
    """
    def __init__(self, num_classes, class_weights, focal_gamma=2.0, 
                 label_smoothing=0.1, focal_weight=0.7):
        super(CombinedLoss, self).__init__()
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.smooth_loss = LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=label_smoothing
        )
        self.focal_weight = focal_weight
        
    def forward(self, outputs, targets):
        focal = self.focal_loss(outputs, targets)
        smooth = self.smooth_loss(outputs, targets)
        
        return self.focal_weight * focal + (1 - self.focal_weight) * smooth


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, config, epoch):
    """Train for one epoch with Mixup support"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (ecg, pcg, clinical, labels) in enumerate(loader):
        ecg = ecg.to(config.device)
        pcg = pcg.to(config.device)
        clinical = clinical.to(config.device)
        labels = labels.to(config.device)
        
        # Apply Mixup
        if config.use_mixup and np.random.rand() < config.mixup_prob:
            ecg, pcg, clinical, labels_a, labels_b, lam = mixup_data(
                ecg, pcg, clinical, labels, config.mixup_alpha
            )
            
            optimizer.zero_grad()
            outputs_dict = model(clinical, ecg, pcg)
            outputs = outputs_dict['fused_logits']
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # For accuracy, use original labels
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).float() + 
                       (1 - lam) * predicted.eq(labels_b).float()).sum().item()
        else:
            # Normal training
            optimizer.zero_grad()
            outputs_dict = model(clinical, ecg, pcg)
            outputs = outputs_dict['fused_logits']
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.2f}%")
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, config):
    """Evaluate model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for ecg, pcg, clinical, labels in loader:
            ecg = ecg.to(config.device)
            pcg = pcg.to(config.device)
            clinical = clinical.to(config.device)
            labels = labels.to(config.device)
            
            outputs_dict = model(clinical, ecg, pcg)
            outputs = outputs_dict['fused_logits']
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            
            total_loss += loss.item()
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    # Calculate per-class metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, macro_f1, macro_precision, macro_recall


# ============================================================================
# Main Training
# ============================================================================

def main():
    config = Config()
    set_seed(config.random_seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("ADVANCED TRAINING WITH ALL IMPROVEMENTS")
    logging.info("="*80)
    logging.info(f"Device: {config.device}")
    logging.info(f"Using balanced data: {config.use_balanced_data}")
    logging.info(f"Focal Loss: {config.use_focal_loss} (gamma={config.focal_gamma})")
    logging.info(f"Label Smoothing: {config.use_label_smoothing} (epsilon={config.label_smoothing})")
    logging.info(f"Mixup: {config.use_mixup} (alpha={config.mixup_alpha})")
    logging.info(f"Scheduler: {config.scheduler_type}")
    logging.info("="*80)
    
    # Load data
    logging.info("\nLoading data...")
    data_path = Path(config.data_dir)
    
    ecg = np.load(data_path / f"ecg_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    pcg = np.load(data_path / f"pcg_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    clinical = np.load(data_path / f"clinical_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    labels = np.load(data_path / f"labels_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    
    logging.info(f"Loaded {len(labels)} samples")
    logging.info(f"  ECG: {ecg.shape}")
    logging.info(f"  PCG: {pcg.shape}")
    logging.info(f"  Clinical: {clinical.shape}")
    
    # Print distribution
    counter = Counter(labels)
    logging.info("\nClass distribution:")
    for cls in sorted(counter.keys()):
        count = counter[cls]
        pct = 100 * count / len(labels)
        logging.info(f"  Class {cls}: {count} ({pct:.2f}%)")
    
    # Split data (80/10/10)
    from sklearn.model_selection import train_test_split
    
    # First split: 80% train, 20% temp
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=config.random_seed
    )
    
    # Second split: 10% val, 10% test from temp
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=config.random_seed
    )
    
    logging.info(f"\nData split:")
    logging.info(f"  Train: {len(train_idx)} samples")
    logging.info(f"  Val: {len(val_idx)} samples")
    logging.info(f"  Test: {len(test_idx)} samples")
    
    # Create datasets
    train_dataset = CardiacDataset(
        ecg[train_idx], pcg[train_idx], clinical[train_idx], labels[train_idx],
        augment=config.augment_train
    )
    val_dataset = CardiacDataset(
        ecg[val_idx], pcg[val_idx], clinical[val_idx], labels[val_idx],
        augment=False
    )
    test_dataset = CardiacDataset(
        ecg[test_idx], pcg[test_idx], clinical[test_idx], labels[test_idx],
        augment=False
    )
    
    # Create weighted sampler for training
    if config.use_weighted_sampler:
        train_labels = labels[train_idx]
        class_weights = compute_effective_number_weights(train_labels)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
        logging.info("\n✓ Using WeightedRandomSampler")
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logging.info("\nCreating model...")
    model = ACRMFNet(
        clinical_input_dim=config.clinical_features,
        ecg_input_dim=1000,  # ECG sequence length
        pcg_input_dim=2000,  # PCG sequence length
        num_classes=config.num_classes,
        embedding_dim=128,
        dropout=0.3
    ).to(config.device)
    
    # Create loss function
    if config.use_class_weights:
        class_weights = compute_effective_number_weights(labels[train_idx])
        logging.info(f"\n✓ Class weights: {class_weights.numpy()}")
        class_weights = class_weights.to(config.device)
    else:
        class_weights = None
    
    if config.use_focal_loss and config.use_label_smoothing:
        criterion = CombinedLoss(
            num_classes=config.num_classes,
            class_weights=class_weights,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            focal_weight=config.focal_alpha_weight
        )
        logging.info("✓ Using Combined Loss (Focal + Label Smoothing)")
    elif config.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
        logging.info("✓ Using Focal Loss")
    elif config.use_label_smoothing:
        criterion = LabelSmoothingLoss(
            num_classes=config.num_classes,
            smoothing=config.label_smoothing
        )
        logging.info("✓ Using Label Smoothing Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info("✓ Using CrossEntropy Loss")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    if config.scheduler_type == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult
        )
        logging.info(f"✓ Using CosineAnnealingWarmRestarts (T_0={config.T_0}, T_mult={config.T_mult})")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        logging.info("✓ Using ReduceLROnPlateau")
    
    # Training loop
    logging.info("\n" + "="*80)
    logging.info("STARTING TRAINING")
    logging.info("="*80)
    
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': []
    }
    
    for epoch in range(config.num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        logging.info("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_precision, val_recall = evaluate(
            model, val_loader, criterion, config
        )
        
        # Update scheduler
        if config.scheduler_type == 'cosine_warm_restarts':
            scheduler.step()
        else:
            scheduler.step(val_f1)
        
        # Log
        logging.info(f"\nEpoch {epoch+1} Summary:")
        logging.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logging.info(f"  Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        logging.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        
        # Check for improvement
        if val_f1 > best_val_f1 + config.min_delta:
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': vars(config)
            }, output_dir / 'best_model.pth')
            
            logging.info(f"  ✓ New best model saved! (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            logging.info(f"  No improvement ({patience_counter}/{config.patience})")
        
        # Early stopping
        if patience_counter >= config.patience:
            logging.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    logging.info("\n" + "="*80)
    logging.info("FINAL EVALUATION")
    logging.info("="*80)
    
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(
        model, test_loader, criterion, config
    )
    
    logging.info(f"\nTest Results:")
    logging.info(f"  Accuracy: {test_acc:.2f}%")
    logging.info(f"  Macro F1: {test_f1:.4f}")
    logging.info(f"  Precision: {test_precision:.4f}")
    logging.info(f"  Recall: {test_recall:.4f}")
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"\n✓ Training completed!")
    logging.info(f"✓ Best model saved to: {output_dir / 'best_model.pth'}")
    logging.info(f"✓ History saved to: {output_dir / 'history.json'}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
