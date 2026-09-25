"""
Ablation Study for ACRMF-Net Components
========================================

This script systematically evaluates the contribution of each component:
- REN (Reliability Estimation Network)
- AWG (Adaptive Weight Generator) 
- CEN (Confidence Estimation Network)

Experiments:
1. Full Model (REN + AWG + CEN) - Baseline
2. Without REN (Only AWG + CEN)
3. Without AWG (Only REN + CEN)
4. Without CEN (Only REN + AWG)
5. Without REN & AWG (Only CEN)
6. Without REN & CEN (Only AWG)
7. Without AWG & CEN (Only REN)
8. No Reliability/Confidence (Basic Fusion)
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
    # Data
    use_balanced_data = True
    data_dir = "balanced_data" if use_balanced_data else "cleaned_data"
    
    # Model
    num_classes = 2  # Binary classification
    clinical_features = 8
    
    # Training
    batch_size = 32
    num_epochs = 50  # Reduced for ablation study
    learning_rate = 0.0001
    weight_decay = 0.01
    
    # Loss
    use_focal_loss = True
    focal_gamma = 2.0
    label_smoothing = 0.1
    focal_weight = 0.7
    
    # Output
    output_dir = "experiments/ablation_study"
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Random seed
    random_seed = 42


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CardiacDataset(Dataset):
    def __init__(self, ecg, pcg, clinical, labels):
        # Transpose ECG from (N, 1000, 12) to (N, 12, 1000)
        self.ecg = torch.FloatTensor(ecg).transpose(1, 2)
        
        # Convert PCG from 1D signal to 2D spectrogram if needed
        if pcg.ndim == 2:  # (N, 2000) - 1D signal
            pcg = self._convert_pcg_to_spectrogram(pcg)
        
        self.pcg = torch.FloatTensor(pcg)
        self.clinical = torch.FloatTensor(clinical)
        self.labels = torch.LongTensor(labels)
    
    def _convert_pcg_to_spectrogram(self, pcg_signals):
        """Convert 1D PCG signals to 2D spectrograms"""
        from scipy import signal as scipy_signal
        from scipy.ndimage import zoom
        
        spectrograms = []
        for i in range(len(pcg_signals)):
            # Compute spectrogram
            f, t, Sxx = scipy_signal.spectrogram(
                pcg_signals[i], 
                fs=2000,  # Sampling rate
                nperseg=256,
                noverlap=128
            )
            
            # Resize to 128x128
            zoom_factors = (128 / Sxx.shape[0], 128 / Sxx.shape[1])
            Sxx_resized = zoom(Sxx, zoom_factors, order=1)
            
            # Normalize
            Sxx_resized = (Sxx_resized - Sxx_resized.mean()) / (Sxx_resized.std() + 1e-8)
            
            spectrograms.append(Sxx_resized)
        
        return np.array(spectrograms, dtype=np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.ecg[idx], self.pcg[idx], self.clinical[idx], self.labels[idx]


def compute_effective_number_weights(labels, beta=0.9999):
    """Compute class weights using Effective Number of Samples"""
    counter = Counter(labels)
    num_classes = len(counter)
    
    weights = []
    for cls in range(num_classes):
        n = counter.get(cls, 0)
        if n == 0:
            weights.append(0)
        else:
            effective_num = (1.0 - np.power(beta, n)) / (1.0 - beta)
            weights.append(1.0 / effective_num)
    
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes
    
    return torch.FloatTensor(weights)


class CombinedLoss(nn.Module):
    """Combines Focal Loss and Label Smoothing Loss"""
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


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for ecg, pcg, clinical, labels in loader:
        ecg = ecg.to(device)
        pcg = pcg.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (model uses internal flags)
        outputs_dict = model(clinical, ecg, pcg)
        
        outputs = outputs_dict['fused_logits']
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for ecg, pcg, clinical, labels in loader:
            ecg = ecg.to(device)
            pcg = pcg.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)
            
            # Forward pass (model uses internal flags)
            outputs_dict = model(clinical, ecg, pcg)
            
            outputs = outputs_dict['fused_logits']
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            
            total_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, f1_macro, precision, recall


def run_experiment(exp_name, use_ren, use_awg, use_cen, config, 
                   train_loader, val_loader, test_loader, class_weights):
    """Run a single ablation experiment"""
    
    logging.info("="*80)
    logging.info(f"EXPERIMENT: {exp_name}")
    logging.info(f"REN: {use_ren}, AWG: {use_awg}, CEN: {use_cen}")
    logging.info("="*80)
    
    set_seed(config.random_seed)
    
    # Create model with component flags
    model = ACRMFNet(
        clinical_input_dim=config.clinical_features,
        ecg_input_dim=1000,
        pcg_input_dim=2000,
        num_classes=config.num_classes,
        embedding_dim=128,
        dropout=0.3,
        use_ren=use_ren,      # Enable/disable REN
        use_awg=use_awg,      # Enable/disable AWG
        use_cen=use_cen       # Enable/disable CEN
    ).to(config.device)
    
    # Loss and optimizer
    criterion = CombinedLoss(
        num_classes=config.num_classes,
        class_weights=class_weights,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
        focal_weight=config.focal_weight
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(
            model, val_loader, criterion, config.device
        )
        
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        # Track best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            
            # Save best model
            exp_dir = Path(config.output_dir) / exp_name.replace(" ", "_").lower()
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, exp_dir / 'best_model.pth')
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(exp_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_prec, test_rec = evaluate(
        model, test_loader, criterion, config.device
    )
    
    logging.info(f"\nBest Model (Epoch {best_epoch}):")
    logging.info(f"  Val F1: {best_val_f1:.4f}")
    logging.info(f"  Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}")
    logging.info(f"  Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    
    # Save results
    results = {
        'experiment': exp_name,
        'components': {
            'REN': use_ren,
            'AWG': use_awg,
            'CEN': use_cen
        },
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'history': history
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    config = Config()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', errors='replace'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("ABLATION STUDY - ACRMF-Net Components")
    logging.info("="*80)
    logging.info(f"Device: {config.device}")
    logging.info(f"Epochs per experiment: {config.num_epochs}")
    
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
        indices, test_size=0.2, stratify=labels, random_state=config.random_seed
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=config.random_seed
    )
    
    logging.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create datasets
    train_dataset = CardiacDataset(
        ecg[train_idx], pcg[train_idx], clinical[train_idx], labels[train_idx]
    )
    val_dataset = CardiacDataset(
        ecg[val_idx], pcg[val_idx], clinical[val_idx], labels[val_idx]
    )
    test_dataset = CardiacDataset(
        ecg[test_idx], pcg[test_idx], clinical[test_idx], labels[test_idx]
    )
    
    # Create weighted sampler
    train_labels = labels[train_idx]
    class_weights = compute_effective_number_weights(train_labels).to(config.device)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             sampler=sampler, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=2, pin_memory=False)
    
    # Define ablation experiments
    experiments = [
        # (name, use_ren, use_awg, use_cen)
        ("Full Model (REN + AWG + CEN)", True, True, True),
        ("Without REN (AWG + CEN)", False, True, True),
        ("Without AWG (REN + CEN)", True, False, True),
        ("Without CEN (REN + AWG)", True, True, False),
        ("Without REN & AWG (CEN only)", False, False, True),
        ("Without REN & CEN (AWG only)", False, True, False),
        ("Without AWG & CEN (REN only)", True, False, False),
        ("Basic Fusion (No REN/AWG/CEN)", False, False, False),
    ]
    
    # Run all experiments
    all_results = []
    
    for exp_name, use_ren, use_awg, use_cen in experiments:
        try:
            results = run_experiment(
                exp_name, use_ren, use_awg, use_cen, config,
                train_loader, val_loader, test_loader, class_weights
            )
            all_results.append(results)
        except Exception as e:
            logging.error(f"Experiment '{exp_name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    logging.info("\n" + "="*80)
    logging.info("ABLATION STUDY SUMMARY")
    logging.info("="*80)
    
    logging.info(f"\n{'Experiment':<40} {'Test Acc':<12} {'Test F1':<12}")
    logging.info("-" * 64)
    
    for result in all_results:
        logging.info(f"{result['experiment']:<40} {result['test_accuracy']:>10.2f}%  {result['test_f1']:>10.4f}")
    
    # Save complete summary
    summary = {
        'experiments': all_results,
        'config': vars(config)
    }
    
    with open(output_dir / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\nComplete summary saved to: {output_dir / 'ablation_summary.json'}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
