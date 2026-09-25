"""
Dataset Imbalance Handling Techniques for ACRMF-Net
Addresses class imbalance in cardiac disease detection

Current Distribution:
  Class 0: 9069 (55.8%)
  Class 1: 2532 (15.6%)
  Class 2: 2400 (14.8%)
  Class 3: 1708 (10.5%)
  Class 4: 535 (3.3%)
"""

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import Dict, Tuple
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.class_weight import compute_class_weight


# ============================================================================
# Method 1: Class Weights (Easiest, No Data Modification)
# ============================================================================

def compute_balanced_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies
    
    Usage:
        class_weights = compute_balanced_class_weights(labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    """
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    weights_tensor = torch.FloatTensor(class_weights)
    
    print("Class Weights:")
    for cls, weight in zip(unique_classes, class_weights):
        print(f"  Class {cls}: {weight:.4f}")
    
    return weights_tensor


def compute_effective_number_weights(labels: np.ndarray, beta: float = 0.9999) -> torch.Tensor:
    """
    Compute class weights using Effective Number of Samples
    Paper: "Class-Balanced Loss Based on Effective Number of Samples"
    
    Args:
        labels: Array of labels
        beta: Hyperparameter (0.9-0.9999), higher = more emphasis on rare classes
    
    Formula:
        E_n = (1 - β^n) / (1 - β)
        weight = 1 / E_n
    """
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
    
    # Normalize
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes
    
    weights_tensor = torch.FloatTensor(weights)
    
    print(f"Effective Number Weights (beta={beta}):")
    for cls, weight in enumerate(weights):
        print(f"  Class {cls}: {weight:.4f}")
    
    return weights_tensor


# ============================================================================
# Method 2: Focal Loss (Focuses on Hard Examples)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: Addresses class imbalance by down-weighting easy examples
    Paper: "Focal Loss for Dense Object Detection"
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for each class (like class weights)
        gamma: Focusing parameter (2.0 is typical, higher = more focus on hard examples)
    """
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, num_classes) - raw logits
            targets: Ground truth (B,) - class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


# ============================================================================
# Method 3: SMOTE - Synthetic Minority Over-sampling
# ============================================================================

def apply_smote_clinical(clinical: np.ndarray, labels: np.ndarray, 
                         strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to clinical data (works well for tabular data)
    
    Args:
        clinical: Clinical features (N, 13)
        labels: Labels (N,)
        strategy: 'auto', 'minority', 'not majority', or dict {class: n_samples}
    
    Returns:
        resampled_clinical, resampled_labels
    """
    smote = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=5)
    clinical_resampled, labels_resampled = smote.fit_resample(clinical, labels)
    
    print(f"\nSMOTE Applied (Clinical Data):")
    print(f"  Original: {len(labels)} samples")
    print(f"  Resampled: {len(labels_resampled)} samples")
    print(f"  New distribution:")
    for cls, count in Counter(labels_resampled).items():
        print(f"    Class {cls}: {count} ({100*count/len(labels_resampled):.1f}%)")
    
    return clinical_resampled, labels_resampled


def apply_borderline_smote_clinical(clinical: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Borderline-SMOTE: Only generates synthetic samples near decision boundaries
    Better for high-dimensional data
    """
    smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    clinical_resampled, labels_resampled = smote.fit_resample(clinical, labels)
    
    print(f"\nBorderline-SMOTE Applied:")
    print(f"  Original: {len(labels)} samples")
    print(f"  Resampled: {len(labels_resampled)} samples")
    
    return clinical_resampled, labels_resampled


def apply_adasyn(clinical: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ADASYN: Adaptive Synthetic Sampling
    Generates more synthetic samples for harder-to-learn minority examples
    """
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    clinical_resampled, labels_resampled = adasyn.fit_resample(clinical, labels)
    
    print(f"\nADASYN Applied:")
    print(f"  Original: {len(labels)} samples")
    print(f"  Resampled: {len(labels_resampled)} samples")
    
    return clinical_resampled, labels_resampled


# ============================================================================
# Method 4: Combine Over-sampling + Under-sampling
# ============================================================================

def apply_smote_tomek(clinical: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTETomek: SMOTE + Tomek links removal
    1. Over-sample minority classes with SMOTE
    2. Clean overlapping samples using Tomek links
    """
    smt = SMOTETomek(random_state=42)
    clinical_resampled, labels_resampled = smt.fit_resample(clinical, labels)
    
    print(f"\nSMOTE-Tomek Applied:")
    print(f"  Original: {len(labels)} samples")
    print(f"  Resampled: {len(labels_resampled)} samples")
    print(f"  New distribution:")
    for cls, count in Counter(labels_resampled).items():
        print(f"    Class {cls}: {count} ({100*count/len(labels_resampled):.1f}%)")
    
    return clinical_resampled, labels_resampled


def apply_smote_enn(clinical: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE-ENN: SMOTE + Edited Nearest Neighbors
    More aggressive cleaning than Tomek links
    """
    smote_enn = SMOTEENN(random_state=42)
    clinical_resampled, labels_resampled = smote_enn.fit_resample(clinical, labels)
    
    print(f"\nSMOTE-ENN Applied:")
    print(f"  Original: {len(labels)} samples")
    print(f"  Resampled: {len(labels_resampled)} samples")
    
    return clinical_resampled, labels_resampled


# ============================================================================
# Method 5: Weighted Random Sampler (PyTorch DataLoader)
# ============================================================================

def create_balanced_sampler(labels: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a weighted sampler for PyTorch DataLoader
    Each sample has weight inversely proportional to its class frequency
    
    Usage:
        sampler = create_balanced_sampler(labels)
        train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    sample_weights = np.array([class_weights[label] for label in labels])
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print("Weighted Sampler Created:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls}: weight={weight:.6f} (count={class_counts[cls]})")
    
    return sampler


# ============================================================================
# Method 6: Data Augmentation for Time-Series (ECG/PCG)
# ============================================================================

def augment_ecg_signal(ecg: np.ndarray, method: str = 'jitter') -> np.ndarray:
    """
    Augment ECG signals for minority classes
    
    Methods:
        - jitter: Add random noise
        - scaling: Random amplitude scaling
        - time_warp: Non-linear time warping
        - window_slice: Random window extraction
    """
    if method == 'jitter':
        noise = np.random.normal(0, 0.05, ecg.shape)
        return ecg + noise
    
    elif method == 'scaling':
        scale = np.random.uniform(0.9, 1.1)
        return ecg * scale
    
    elif method == 'time_warp':
        # Simple time warping by interpolation
        from scipy.interpolate import interp1d
        old_indices = np.linspace(0, ecg.shape[0]-1, ecg.shape[0])
        warp = np.random.uniform(0.9, 1.1, size=ecg.shape[0])
        new_indices = np.cumsum(warp)
        new_indices = new_indices / new_indices[-1] * (ecg.shape[0]-1)
        
        warped = np.zeros_like(ecg)
        for i in range(ecg.shape[1]):  # For each lead
            f = interp1d(old_indices, ecg[:, i], kind='linear', fill_value='extrapolate')
            warped[:, i] = f(new_indices)
        
        return warped
    
    elif method == 'window_slice':
        # Random window with slight shift
        shift = np.random.randint(-50, 50)
        return np.roll(ecg, shift, axis=0)
    
    return ecg


def augment_pcg_spectrogram(pcg: np.ndarray) -> np.ndarray:
    """
    Augment PCG spectrograms
    
    Methods:
        - freq_mask: Mask random frequency bands
        - time_mask: Mask random time segments
        - mixup: Mix with another sample
    """
    # Frequency masking
    if np.random.rand() < 0.5:
        freq_mask_size = np.random.randint(5, 15)
        freq_start = np.random.randint(0, pcg.shape[0] - freq_mask_size)
        pcg[freq_start:freq_start+freq_mask_size, :] *= 0.0
    
    # Time masking
    if np.random.rand() < 0.5:
        time_mask_size = np.random.randint(5, 15)
        time_start = np.random.randint(0, pcg.shape[1] - time_mask_size)
        pcg[:, time_start:time_start+time_mask_size] *= 0.0
    
    return pcg


# ============================================================================
# Integrated Solution: Combine Multiple Techniques
# ============================================================================

def balanced_dataset_pipeline(
    clinical: np.ndarray,
    ecg: np.ndarray,
    pcg: np.ndarray,
    labels: np.ndarray,
    method: str = 'smote_tomek',
    augment: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete pipeline to handle imbalance
    
    Args:
        clinical: Clinical data (N, 13)
        ecg: ECG signals (N, 1000, 12)
        pcg: PCG spectrograms (N, 128, 128)
        labels: Labels (N,)
        method: 'smote', 'borderline_smote', 'adasyn', 'smote_tomek', 'smote_enn'
        augment: Apply augmentation to ECG/PCG for minority classes
    
    Returns:
        balanced_clinical, balanced_ecg, balanced_pcg, balanced_labels
    """
    print(f"Original Dataset Distribution:")
    for cls, count in Counter(labels).items():
        print(f"  Class {cls}: {count} ({100*count/len(labels):.1f}%)")
    
    # Step 1: Apply SMOTE/resampling to clinical data
    if method == 'smote':
        clinical_balanced, labels_balanced = apply_smote_clinical(clinical, labels)
    elif method == 'borderline_smote':
        clinical_balanced, labels_balanced = apply_borderline_smote_clinical(clinical, labels)
    elif method == 'adasyn':
        clinical_balanced, labels_balanced = apply_adasyn(clinical, labels)
    elif method == 'smote_tomek':
        clinical_balanced, labels_balanced = apply_smote_tomek(clinical, labels)
    elif method == 'smote_enn':
        clinical_balanced, labels_balanced = apply_smote_enn(clinical, labels)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 2: Map resampled indices back to ECG/PCG
    # For synthetic samples, use augmentation; for real samples, copy directly
    n_original = len(labels)
    n_new = len(labels_balanced)
    
    ecg_balanced = []
    pcg_balanced = []
    
    for i in range(n_new):
        if i < n_original:
            # Real sample
            ecg_balanced.append(ecg[i])
            pcg_balanced.append(pcg[i])
        else:
            # Synthetic sample - find nearest real sample of same class
            label = labels_balanced[i]
            class_indices = np.where(labels == label)[0]
            nearest_idx = np.random.choice(class_indices)
            
            if augment:
                # Apply augmentation
                ecg_aug = augment_ecg_signal(ecg[nearest_idx], method='jitter')
                pcg_aug = augment_pcg_spectrogram(pcg[nearest_idx].copy())
                ecg_balanced.append(ecg_aug)
                pcg_balanced.append(pcg_aug)
            else:
                # Just duplicate
                ecg_balanced.append(ecg[nearest_idx])
                pcg_balanced.append(pcg[nearest_idx])
    
    ecg_balanced = np.array(ecg_balanced)
    pcg_balanced = np.array(pcg_balanced)
    
    print(f"\n✓ Balanced Dataset Created:")
    print(f"  Total samples: {len(labels_balanced)}")
    for cls, count in Counter(labels_balanced).items():
        print(f"    Class {cls}: {count} ({100*count/len(labels_balanced):.1f}%)")
    
    return clinical_balanced, ecg_balanced, pcg_balanced, labels_balanced


# ============================================================================
# Recommendation: Best Practices
# ============================================================================

def print_recommendations():
    """Print best practices for handling imbalance"""
    print("\n" + "="*80)
    print("RECOMMENDED STRATEGIES FOR YOUR DATASET")
    print("="*80)
    
    print("\n1. CLASS WEIGHTS (Easiest, No Data Change)")
    print("   ✓ Use compute_effective_number_weights() with beta=0.9999")
    print("   ✓ Pass weights to CompositeLoss")
    print("   ✓ Pros: Fast, no data modification")
    print("   ✓ Cons: May not fully address severe imbalance")
    
    print("\n2. FOCAL LOSS (Recommended for Medical Data)")
    print("   ✓ Use FocalLoss with gamma=2.0")
    print("   ✓ Combine with class weights (alpha parameter)")
    print("   ✓ Pros: Focuses on hard examples, works well with deep learning")
    print("   ✓ Cons: Requires hyperparameter tuning")
    
    print("\n3. SMOTE-TOMEK (Best for Clinical Features)")
    print("   ✓ Use balanced_dataset_pipeline with method='smote_tomek'")
    print("   ✓ Pros: Balanced dataset, removes noisy samples")
    print("   ✓ Cons: Increases dataset size, synthetic samples")
    
    print("\n4. WEIGHTED SAMPLER (PyTorch Native)")
    print("   ✓ Use create_balanced_sampler() in DataLoader")
    print("   ✓ Pros: Each epoch sees balanced batches")
    print("   ✓ Cons: Training takes longer (more batches per epoch)")
    
    print("\n5. COMBINATION APPROACH (BEST RESULTS)")
    print("   ✓ SMOTE-Tomek for data balancing")
    print("   ✓ + Focal Loss for training")
    print("   ✓ + Data augmentation for ECG/PCG")
    print("   ✓ Pros: Addresses imbalance at multiple levels")
    print("   ✓ Cons: More complex pipeline")
    
    print("\n" + "="*80)
    print("FOR YOUR DATASET (Class 4 only 3.3%):")
    print("  → Start with: Class weights + Focal Loss")
    print("  → If not enough: Add SMOTE-Tomek + Augmentation")
    print("  → Monitor: Per-class F1 scores (not just macro F1)")
    print("="*80 + "\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_recommendations()
    
    print("\nTo use these methods:")
    print("1. For quick fix: Add class weights to your loss")
    print("2. For best results: Run balanced_dataset_pipeline on your data")
    print("3. Save balanced data: Use np.savez to save resampled arrays")
