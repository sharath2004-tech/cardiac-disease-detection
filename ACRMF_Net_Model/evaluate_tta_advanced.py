"""
Test-Time Augmentation (TTA) Evaluation
========================================

Applies multiple augmentations during testing and averages predictions
for improved accuracy.

Expected gain: +1-2% accuracy
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import json

sys.path.append(str(Path(__file__).parent))
from models.acrmf import ACRMFNet


class Config:
    # TTA settings
    n_tta_iterations = 10  # Number of augmented versions per sample
    
    # Data
    use_balanced_data = True
    data_dir = "balanced_data" if use_balanced_data else "cleaned_data"
    
    # Model
    num_classes = 5
    ecg_channels = 12
    pcg_size = 128
    clinical_features = 13
    
    # Model path
    model_path = "experiments/advanced_training/best_model.pth"
    
    # Batch size
    batch_size = 16  # Smaller for TTA
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def augment_ecg_tta(ecg, strength='medium'):
    """
    Apply TTA augmentation to ECG
    
    Args:
        ecg: (1000, 12) ECG signal
        strength: 'light', 'medium', 'strong'
    """
    ecg_aug = ecg.clone()
    
    if strength == 'light':
        noise_std = 0.02
        scale_range = (0.98, 1.02)
        shift_range = 10
    elif strength == 'medium':
        noise_std = 0.03
        scale_range = (0.95, 1.05)
        shift_range = 20
    else:  # strong
        noise_std = 0.05
        scale_range = (0.90, 1.10)
        shift_range = 30
    
    # Jitter
    if torch.rand(1) < 0.7:
        noise = torch.randn_like(ecg_aug) * noise_std
        ecg_aug = ecg_aug + noise
    
    # Scaling
    if torch.rand(1) < 0.7:
        scale = torch.FloatTensor(1).uniform_(*scale_range)
        ecg_aug = ecg_aug * scale
    
    # Time shift
    if torch.rand(1) < 0.5:
        shift = torch.randint(-shift_range, shift_range, (1,)).item()
        ecg_aug = torch.roll(ecg_aug, shifts=shift, dims=0)
    
    return ecg_aug


def augment_pcg_tta(pcg, strength='medium'):
    """
    Apply TTA augmentation to PCG spectrogram
    
    Args:
        pcg: (128, 128) PCG spectrogram
        strength: 'light', 'medium', 'strong'
    """
    pcg_aug = pcg.clone()
    
    if strength == 'light':
        mask_size_range = (5, 10)
        mask_value_range = (0.2, 0.5)
        scale_range = (0.98, 1.02)
    elif strength == 'medium':
        mask_size_range = (8, 15)
        mask_value_range = (0.0, 0.3)
        scale_range = (0.95, 1.05)
    else:  # strong
        mask_size_range = (10, 20)
        mask_value_range = (0.0, 0.2)
        scale_range = (0.90, 1.10)
    
    # Frequency masking
    if torch.rand(1) < 0.6:
        mask_size = torch.randint(*mask_size_range, (1,)).item()
        mask_start = torch.randint(0, pcg.shape[0] - mask_size, (1,)).item()
        mask_value = torch.FloatTensor(1).uniform_(*mask_value_range)
        pcg_aug[mask_start:mask_start+mask_size, :] *= mask_value
    
    # Time masking
    if torch.rand(1) < 0.6:
        mask_size = torch.randint(*mask_size_range, (1,)).item()
        mask_start = torch.randint(0, pcg.shape[1] - mask_size, (1,)).item()
        mask_value = torch.FloatTensor(1).uniform_(*mask_value_range)
        pcg_aug[:, mask_start:mask_start+mask_size] *= mask_value
    
    # Scaling
    if torch.rand(1) < 0.5:
        scale = torch.FloatTensor(1).uniform_(*scale_range)
        pcg_aug = pcg_aug * scale
    
    return pcg_aug


def predict_with_tta(model, ecg, pcg, clinical, n_iterations=10, device='cuda'):
    """
    Predict with Test-Time Augmentation
    
    Args:
        model: Trained model
        ecg: (B, 1000, 12) ECG batch
        pcg: (B, 128, 128) PCG batch
        clinical: (B, 13) Clinical batch
        n_iterations: Number of TTA iterations
        device: Device to run on
    
    Returns:
        predictions: (B,) Predicted classes
        probabilities: (B, num_classes) Average probabilities
    """
    model.eval()
    
    batch_size = ecg.shape[0]
    num_classes = 5
    
    all_probs = []
    
    with torch.no_grad():
        # Original (no augmentation)
        ecg_batch = ecg.to(device)
        pcg_batch = pcg.to(device)
        clinical_batch = clinical.to(device)
        
        outputs = model(ecg_batch, pcg_batch, clinical_batch)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        
        # Augmented versions
        for i in range(n_iterations - 1):
            # Augment each sample
            ecg_aug_batch = []
            pcg_aug_batch = []
            
            for j in range(batch_size):
                ecg_aug = augment_ecg_tta(ecg[j], strength='medium')
                pcg_aug = augment_pcg_tta(pcg[j], strength='medium')
                
                ecg_aug_batch.append(ecg_aug)
                pcg_aug_batch.append(pcg_aug)
            
            ecg_aug_batch = torch.stack(ecg_aug_batch).to(device)
            pcg_aug_batch = torch.stack(pcg_aug_batch).to(device)
            
            outputs = model(ecg_aug_batch, pcg_aug_batch, clinical_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)
    
    return predictions, avg_probs


def evaluate_with_tta(model, dataset, config):
    """Evaluate model with TTA on entire dataset"""
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    all_preds = []
    all_labels = []
    
    total_batches = len(loader)
    
    logging.info(f"Running TTA with {config.n_tta_iterations} iterations per sample...")
    
    for batch_idx, (ecg, pcg, clinical, labels) in enumerate(loader):
        preds, _ = predict_with_tta(
            model, ecg, pcg, clinical,
            n_iterations=config.n_tta_iterations,
            device=config.device
        )
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  Processed {batch_idx+1}/{total_batches} batches...")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'predictions': all_preds,
        'labels': all_labels
    }


def evaluate_without_tta(model, dataset, config):
    """Baseline evaluation without TTA"""
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for ecg, pcg, clinical, labels in loader:
            ecg = ecg.to(config.device)
            pcg = pcg.to(config.device)
            clinical = clinical.to(config.device)
            
            outputs = model(ecg, pcg, clinical)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'predictions': all_preds,
        'labels': all_labels
    }


class CardiacDataset(Dataset):
    def __init__(self, ecg, pcg, clinical, labels):
        self.ecg = torch.FloatTensor(ecg)
        self.pcg = torch.FloatTensor(pcg)
        self.clinical = torch.FloatTensor(clinical)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.ecg[idx], self.pcg[idx], self.clinical[idx], self.labels[idx]


def main():
    config = Config()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    logging.info("="*80)
    logging.info("TEST-TIME AUGMENTATION (TTA) EVALUATION")
    logging.info("="*80)
    logging.info(f"Model: {config.model_path}")
    logging.info(f"TTA iterations: {config.n_tta_iterations}")
    logging.info(f"Device: {config.device}")
    
    # Load data
    logging.info("\nLoading data...")
    data_path = Path(config.data_dir)
    
    ecg = np.load(data_path / f"ecg_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    pcg = np.load(data_path / f"pcg_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    clinical = np.load(data_path / f"clinical_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    labels = np.load(data_path / f"labels_{'balanced' if config.use_balanced_data else 'cleaned'}.npy")
    
    logging.info(f"Loaded {len(labels)} samples")
    
    # Split data (use same split as training)
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Create test dataset
    test_dataset = CardiacDataset(
        ecg[test_idx], pcg[test_idx], clinical[test_idx], labels[test_idx]
    )
    
    logging.info(f"Test set: {len(test_dataset)} samples")
    
    # Load model
    logging.info("\nLoading model...")
    model = ACRMFNet(
        clinical_input_dim=config.clinical_features,
        ecg_input_dim=1000,
        pcg_input_dim=2000,
        num_classes=config.num_classes,
        embedding_dim=128,
        dropout=0.3
    ).to(config.device)
    
    if Path(config.model_path).exists():
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"✓ Model loaded from {config.model_path}")
    else:
        logging.error(f"❌ Model not found at {config.model_path}")
        return
    
    # Evaluate without TTA (baseline)
    logging.info("\n" + "="*80)
    logging.info("BASELINE (No TTA)")
    logging.info("="*80)
    
    baseline_results = evaluate_without_tta(model, test_dataset, config)
    
    logging.info(f"\nBaseline Results:")
    logging.info(f"  Accuracy: {baseline_results['accuracy']:.2f}%")
    logging.info(f"  Macro F1: {baseline_results['f1_macro']:.4f}")
    logging.info(f"  Weighted F1: {baseline_results['f1_weighted']:.4f}")
    logging.info(f"  Precision: {baseline_results['precision']:.4f}")
    logging.info(f"  Recall: {baseline_results['recall']:.4f}")
    
    logging.info(f"\nPer-class F1:")
    for cls, f1 in enumerate(baseline_results['f1_per_class']):
        logging.info(f"  Class {cls}: {f1:.4f}")
    
    # Evaluate with TTA
    logging.info("\n" + "="*80)
    logging.info(f"WITH TTA ({config.n_tta_iterations} iterations)")
    logging.info("="*80)
    
    tta_results = evaluate_with_tta(model, test_dataset, config)
    
    logging.info(f"\nTTA Results:")
    logging.info(f"  Accuracy: {tta_results['accuracy']:.2f}%")
    logging.info(f"  Macro F1: {tta_results['f1_macro']:.4f}")
    logging.info(f"  Weighted F1: {tta_results['f1_weighted']:.4f}")
    logging.info(f"  Precision: {tta_results['precision']:.4f}")
    logging.info(f"  Recall: {tta_results['recall']:.4f}")
    
    logging.info(f"\nPer-class F1:")
    for cls, f1 in enumerate(tta_results['f1_per_class']):
        logging.info(f"  Class {cls}: {f1:.4f}")
    
    # Comparison
    logging.info("\n" + "="*80)
    logging.info("IMPROVEMENT WITH TTA")
    logging.info("="*80)
    
    acc_gain = tta_results['accuracy'] - baseline_results['accuracy']
    f1_gain = tta_results['f1_macro'] - baseline_results['f1_macro']
    
    logging.info(f"Accuracy gain: +{acc_gain:.2f}%")
    logging.info(f"Macro F1 gain: +{f1_gain:.4f}")
    
    logging.info(f"\nPer-class F1 gain:")
    for cls in range(len(tta_results['f1_per_class'])):
        gain = tta_results['f1_per_class'][cls] - baseline_results['f1_per_class'][cls]
        logging.info(f"  Class {cls}: {gain:+.4f}")
    
    # Save results
    output_dir = Path("experiments/tta_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'baseline': {
            'accuracy': float(baseline_results['accuracy']),
            'f1_macro': float(baseline_results['f1_macro']),
            'f1_weighted': float(baseline_results['f1_weighted']),
            'precision': float(baseline_results['precision']),
            'recall': float(baseline_results['recall']),
            'f1_per_class': baseline_results['f1_per_class'].tolist()
        },
        'tta': {
            'accuracy': float(tta_results['accuracy']),
            'f1_macro': float(tta_results['f1_macro']),
            'f1_weighted': float(tta_results['f1_weighted']),
            'precision': float(tta_results['precision']),
            'recall': float(tta_results['recall']),
            'f1_per_class': tta_results['f1_per_class'].tolist(),
            'n_iterations': config.n_tta_iterations
        },
        'improvement': {
            'accuracy_gain': float(acc_gain),
            'f1_macro_gain': float(f1_gain)
        }
    }
    
    with open(output_dir / 'tta_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\n✓ Results saved to {output_dir / 'tta_comparison.json'}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
