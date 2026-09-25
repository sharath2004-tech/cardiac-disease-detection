"""
Ensemble Training Script
========================

Trains multiple models with different seeds and combines predictions
for maximum accuracy improvement.

Expected gain: +2-4% accuracy over single model
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import logging
from collections import Counter
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.append(str(Path(__file__).parent))
from models.acrmf import ACRMFNet
from losses.composite_loss import FocalLoss, LabelSmoothingLoss


class Config:
    # Ensemble
    n_models = 5  # Number of models in ensemble
    ensemble_seeds = [42, 123, 456, 789, 1024]  # Different random seeds
    
    # Data
    use_balanced_data = True
    data_dir = "balanced_data" if use_balanced_data else "cleaned_data"
    
    # Model
    num_classes = 5
    ecg_channels = 12
    pcg_size = 128
    clinical_features = 13
    
    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0001
    weight_decay = 0.01
    
    # Loss
    use_focal_loss = True
    focal_gamma = 2.0
    label_smoothing = 0.1
    focal_weight = 0.7
    
    # Augmentation
    use_mixup = True
    mixup_alpha = 0.2
    mixup_prob = 0.5
    augment_train = True
    
    # Scheduler
    T_0 = 10
    T_mult = 2
    
    # Early stopping
    patience = 20
    min_delta = 0.001
    
    # Output
    output_dir = "experiments/ensemble"
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    """Set random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CardiacDataset(Dataset):
    def __init__(self, ecg, pcg, clinical, labels, augment=False):
        # Transpose ECG from (N, 1000, 12) to (N, 12, 1000)
        self.ecg = torch.FloatTensor(ecg).transpose(1, 2)
        self.pcg = torch.FloatTensor(pcg)
        self.clinical = torch.FloatTensor(clinical)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        ecg = self.ecg[idx]
        pcg = self.pcg[idx]
        clinical = self.clinical[idx]
        label = self.labels[idx]
        
        if self.augment and torch.rand(1) < 0.5:
            ecg = self._augment_ecg(ecg)
            pcg = self._augment_pcg(pcg)
        
        return ecg, pcg, clinical, label
    
    def _augment_ecg(self, ecg):
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(ecg) * 0.03
            ecg = ecg + noise
        if torch.rand(1) < 0.5:
            scale = torch.FloatTensor(1).uniform_(0.95, 1.05)
            ecg = ecg * scale
        return ecg
    
    def _augment_pcg(self, pcg):
        if torch.rand(1) < 0.5:
            freq_mask_size = torch.randint(8, 16, (1,)).item()
            freq_start = torch.randint(0, pcg.shape[0] - freq_mask_size, (1,)).item()
            pcg[freq_start:freq_start+freq_mask_size, :] *= torch.FloatTensor(1).uniform_(0.0, 0.3)
        if torch.rand(1) < 0.5:
            time_mask_size = torch.randint(8, 16, (1,)).item()
            time_start = torch.randint(0, pcg.shape[1] - time_mask_size, (1,)).item()
            pcg[:, time_start:time_start+time_mask_size] *= torch.FloatTensor(1).uniform_(0.0, 0.3)
        return pcg


def mixup_data(ecg, pcg, clinical, labels, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = ecg.size(0)
    index = torch.randperm(batch_size).to(ecg.device)
    
    mixed_ecg = lam * ecg + (1 - lam) * ecg[index]
    mixed_pcg = lam * pcg + (1 - lam) * pcg[index]
    mixed_clinical = lam * clinical + (1 - lam) * clinical[index]
    
    return mixed_ecg, mixed_pcg, mixed_clinical, labels, labels[index], lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def compute_effective_number_weights(labels, beta=0.9999):
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
    def __init__(self, num_classes, class_weights, focal_gamma=2.0, 
                 label_smoothing=0.1, focal_weight=0.7):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.smooth_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing)
        self.focal_weight = focal_weight
    
    def forward(self, outputs, targets):
        focal = self.focal_loss(outputs, targets)
        smooth = self.smooth_loss(outputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * smooth


def train_single_model(model_id, seed, train_loader, val_loader, config, class_weights):
    """Train a single model in the ensemble"""
    
    logging.info(f"\n{'='*80}")
    logging.info(f"TRAINING MODEL {model_id+1}/{config.n_models} (seed={seed})")
    logging.info(f"{'='*80}")
    
    set_seed(seed)
    
    # Create model
    model = ACRMFNet(
        clinical_input_dim=config.clinical_features,
        ecg_input_dim=1000,
        pcg_input_dim=2000,
        num_classes=config.num_classes,
        embedding_dim=128,
        dropout=0.3
    ).to(config.device)
    
    # Create loss
    criterion = CombinedLoss(
        num_classes=config.num_classes,
        class_weights=class_weights,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
        focal_weight=config.focal_weight
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult
    )
    
    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for ecg, pcg, clinical, labels in train_loader:
            ecg = ecg.to(config.device)
            pcg = pcg.to(config.device)
            clinical = clinical.to(config.device)
            labels = labels.to(config.device)
            
            # Mixup
            if config.use_mixup and np.random.rand() < config.mixup_prob:
                ecg, pcg, clinical, labels_a, labels_b, lam = mixup_data(
                    ecg, pcg, clinical, labels, config.mixup_alpha
                )
                optimizer.zero_grad()
                outputs_dict = model(clinical, ecg, pcg)
                outputs = outputs_dict['fused_logits']
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                _, predicted = outputs.max(1)
                train_correct += (lam * predicted.eq(labels_a).float() + 
                                (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            else:
                optimizer.zero_grad()
                outputs_dict = model(clinical, ecg, pcg)
                outputs = outputs_dict['fused_logits']
                loss = criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_total += labels.size(0)
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for ecg, pcg, clinical, labels in val_loader:
                ecg = ecg.to(config.device)
                pcg = pcg.to(config.device)
                clinical = clinical.to(config.device)
                
                outputs = model(ecg, pcg, clinical)
                _, predicted = outputs.max(1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.numpy())
        
        val_preds = np.array(val_preds)
        val_labels_array = np.array(val_labels_list)
        
        val_acc = accuracy_score(val_labels_array, val_preds) * 100
        val_f1 = f1_score(val_labels_array, val_preds, average='macro')
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                        f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        # Save best
        if val_f1 > best_val_f1 + config.min_delta:
            best_val_f1 = val_f1
            patience_counter = 0
            
            model_path = Path(config.output_dir) / f"model_{model_id}_seed_{seed}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'seed': seed,
                'epoch': epoch + 1
            }, model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logging.info(f"✓ Model {model_id+1} completed - Best Val F1: {best_val_f1:.4f}")
    
    return best_val_f1


def ensemble_predict(models, ecg, pcg, clinical, device):
    """Get ensemble predictions by averaging probabilities"""
    all_probs = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            ecg_batch = ecg.to(device)
            pcg_batch = pcg.to(device)
            clinical_batch = clinical.to(device)
            
            outputs = model(ecg_batch, pcg_batch, clinical_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)
    
    return predictions, avg_probs


def evaluate_ensemble(models, loader, config):
    """Evaluate ensemble on a dataset"""
    all_preds = []
    all_labels = []
    
    for ecg, pcg, clinical, labels in loader:
        preds, _ = ensemble_predict(models, ecg, pcg, clinical, config.device)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return accuracy, f1, precision, recall


def main():
    config = Config()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info(f"ENSEMBLE TRAINING - {config.n_models} Models")
    logging.info("="*80)
    logging.info(f"Seeds: {config.ensemble_seeds}")
    logging.info(f"Device: {config.device}")
    
    # Load data
    logging.info("\nLoading data...")
    data_path = Path(config.data_dir)
    
    ecg = np.load(data_path / f"ecg_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    pcg = np.load(data_path / f"pcg_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    clinical = np.load(data_path / f"clinical_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    labels = np.load(data_path / f"labels_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    
    logging.info(f"Loaded {len(labels)} samples")
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    logging.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
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
    
    # Weighted sampler
    train_labels = labels[train_idx]
    class_weights = compute_effective_number_weights(train_labels).to(config.device)
    sample_weights = compute_effective_number_weights(train_labels)[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Train ensemble models
    best_f1_scores = []
    
    for i, seed in enumerate(config.ensemble_seeds[:config.n_models]):
        f1 = train_single_model(i, seed, train_loader, val_loader, config, class_weights)
        best_f1_scores.append(f1)
    
    # Load all trained models
    logging.info("\n" + "="*80)
    logging.info("LOADING ENSEMBLE MODELS")
    logging.info("="*80)
    
    models = []
    for i, seed in enumerate(config.ensemble_seeds[:config.n_models]):
        model = ACRMFNet(
            num_classes=config.num_classes,
            ecg_channels=config.ecg_channels,
            pcg_size=config.pcg_size,
            clinical_features=config.clinical_features
        ).to(config.device)
        
        model_path = output_dir / f"model_{i}_seed_{seed}.pth"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)
        
        logging.info(f"✓ Loaded model {i+1} (Val F1: {checkpoint['val_f1']:.4f})")
    
    # Evaluate ensemble
    logging.info("\n" + "="*80)
    logging.info("ENSEMBLE EVALUATION")
    logging.info("="*80)
    
    val_acc, val_f1, val_prec, val_rec = evaluate_ensemble(models, val_loader, config)
    logging.info(f"\nValidation Ensemble:")
    logging.info(f"  Accuracy: {val_acc:.2f}%")
    logging.info(f"  Macro F1: {val_f1:.4f}")
    logging.info(f"  Precision: {val_prec:.4f}")
    logging.info(f"  Recall: {val_rec:.4f}")
    
    test_acc, test_f1, test_prec, test_rec = evaluate_ensemble(models, test_loader, config)
    logging.info(f"\nTest Ensemble:")
    logging.info(f"  Accuracy: {test_acc:.2f}%")
    logging.info(f"  Macro F1: {test_f1:.4f}")
    logging.info(f"  Precision: {test_prec:.4f}")
    logging.info(f"  Recall: {test_rec:.4f}")
    
    # Save summary
    summary = {
        'n_models': config.n_models,
        'seeds': config.ensemble_seeds[:config.n_models],
        'individual_val_f1': best_f1_scores,
        'ensemble_val_acc': float(val_acc),
        'ensemble_val_f1': float(val_f1),
        'ensemble_test_acc': float(test_acc),
        'ensemble_test_f1': float(test_f1),
    }
    
    with open(output_dir / 'ensemble_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\n✓ Ensemble training completed!")
    logging.info(f"✓ Models saved in: {output_dir}/")
    logging.info("="*80)


if __name__ == "__main__":
    main()
