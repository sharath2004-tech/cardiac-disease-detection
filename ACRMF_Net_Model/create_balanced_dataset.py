"""
Create Balanced Dataset using SMOTE-Tomek
Addresses severe class imbalance for better training

This script creates a balanced dataset that will be used for training
with all the advanced techniques (Focal Loss, Mixup, Ensemble, etc.)
"""

import sys
import os

# Set UTF-8 encoding for stdout to handle special characters
if sys.platform == 'win32':
    # Windows-specific encoding fix
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pathlib import Path
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import json

def load_cleaned_data():
    """Load the cleaned dataset"""
    data_path = Path("cleaned_data")
    
    print("Loading cleaned data...")
    ecg = np.load(data_path / "ecg_cleaned.npy")
    pcg = np.load(data_path / "pcg_cleaned.npy")
    clinical = np.load(data_path / "clinical_cleaned.npy")
    labels = np.load(data_path / "labels_cleaned.npy")
    
    print(f"Loaded {len(labels)} samples")
    print(f"  ECG shape: {ecg.shape}")
    print(f"  PCG shape: {pcg.shape}")
    print(f"  Clinical shape: {clinical.shape}")
    
    return ecg, pcg, clinical, labels


def print_distribution(labels, title="Distribution"):
    """Print class distribution"""
    counter = Counter(labels)
    total = len(labels)
    
    print(f"\n{title}:")
    for cls in sorted(counter.keys()):
        count = counter[cls]
        pct = 100 * count / total
        print(f"  Class {cls}: {count:5d} ({pct:5.2f}%)")
    print(f"  Total: {total}")


def augment_ecg(ecg, method='combined'):
    """
    Augment ECG signal for synthetic samples
    
    Args:
        ecg: ECG signal (1000, 12)
        method: 'jitter', 'scaling', 'shift', 'combined'
    
    Returns:
        Augmented ECG
    """
    ecg_aug = ecg.copy()
    
    if method in ['jitter', 'combined']:
        # Add random noise
        noise = np.random.normal(0, 0.03, ecg.shape)
        ecg_aug = ecg_aug + noise
    
    if method in ['scaling', 'combined']:
        # Random amplitude scaling
        scale = np.random.uniform(0.95, 1.05)
        ecg_aug = ecg_aug * scale
    
    if method in ['shift', 'combined']:
        # Time shift
        shift = np.random.randint(-30, 30)
        ecg_aug = np.roll(ecg_aug, shift, axis=0)
    
    return ecg_aug


def augment_pcg(pcg):
    """
    Augment PCG spectrogram or signal for synthetic samples
    
    Args:
        pcg: PCG spectrogram (128, 128) or signal (2000,)
    
    Returns:
        Augmented PCG
    """
    pcg_aug = pcg.copy()
    
    # Check if 2D (spectrogram) or 1D (signal)
    if pcg_aug.ndim == 2:
        # Spectrogram augmentation
        # Frequency masking
        if np.random.rand() < 0.5:
            freq_mask_size = np.random.randint(8, 16)
            freq_start = np.random.randint(0, pcg.shape[0] - freq_mask_size)
            pcg_aug[freq_start:freq_start+freq_mask_size, :] *= np.random.uniform(0.0, 0.3)
        
        # Time masking
        if np.random.rand() < 0.5:
            time_mask_size = np.random.randint(8, 16)
            time_start = np.random.randint(0, pcg.shape[1] - time_mask_size)
            pcg_aug[:, time_start:time_start+time_mask_size] *= np.random.uniform(0.0, 0.3)
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        pcg_aug = pcg_aug * scale
    
    else:
        # 1D signal augmentation
        # Add jitter
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.03, pcg_aug.shape)
            pcg_aug = pcg_aug + noise
        
        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            pcg_aug = pcg_aug * scale
        
        # Time shift
        if np.random.rand() < 0.5:
            shift = np.random.randint(-50, 50)
            pcg_aug = np.roll(pcg_aug, shift)
    
    return pcg_aug


def create_balanced_dataset():
    """
    Create balanced dataset using SMOTE-Tomek
    
    Steps:
    1. Load cleaned data
    2. Apply SMOTE-Tomek to clinical features
    3. Map synthetic samples to ECG/PCG with augmentation
    4. Save balanced dataset
    """
    
    print("="*80)
    print("CREATING BALANCED DATASET WITH SMOTE-TOMEK")
    print("="*80)
    
    # Load data
    ecg, pcg, clinical, labels = load_cleaned_data()
    
    # Print original distribution
    print_distribution(labels, "Original Distribution")
    
    # Step 1: Apply SMOTE-Tomek to clinical data
    print("\n" + "="*80)
    print("STEP 1: Applying SMOTE-Tomek to Clinical Data")
    print("="*80)
    
    smt = SMOTETomek(
        smote=SMOTE(
            sampling_strategy='auto',
            k_neighbors=5,
            random_state=42
        ),
        random_state=42
    )
    
    clinical_balanced, labels_balanced = smt.fit_resample(clinical, labels)
    
    print(f"SMOTE-Tomek completed")
    print_distribution(labels_balanced, "Balanced Distribution")
    
    # Step 2: Map to ECG/PCG
    print("\n" + "="*80)
    print("STEP 2: Mapping to ECG/PCG with Augmentation")
    print("="*80)
    
    n_original = len(labels)
    n_balanced = len(labels_balanced)
    n_synthetic = n_balanced - n_original
    
    print(f"Original samples: {n_original}")
    print(f"Balanced samples: {n_balanced}")
    print(f"Synthetic samples to create: {n_synthetic}")
    
    # Create arrays for balanced data
    ecg_balanced = np.zeros((n_balanced, *ecg.shape[1:]), dtype=ecg.dtype)
    pcg_balanced = np.zeros((n_balanced, *pcg.shape[1:]), dtype=pcg.dtype)
    
    # Track which original samples were kept after Tomek cleaning
    # SMOTE-Tomek may remove some original samples
    print("\nMapping samples...")
    
    # For each balanced sample, find the closest original sample
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Compute distances between balanced clinical and original clinical
    print("Computing nearest neighbors for synthetic samples...")
    
    for i in range(n_balanced):
        if i < n_original:
            # Original sample (or kept after Tomek cleaning)
            # Find closest match in original data
            distances = euclidean_distances(
                clinical_balanced[i:i+1], 
                clinical
            )[0]
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < 1e-6:
                # Exact match - use original
                ecg_balanced[i] = ecg[nearest_idx]
                pcg_balanced[i] = pcg[nearest_idx]
            else:
                # Tomek removed this, use nearest with light augmentation
                ecg_balanced[i] = augment_ecg(ecg[nearest_idx], method='jitter')
                pcg_balanced[i] = augment_pcg(pcg[nearest_idx])
        else:
            # Synthetic sample - find nearest original sample of same class
            label = labels_balanced[i]
            class_indices = np.where(labels == label)[0]
            
            # Find closest original sample of same class
            distances = euclidean_distances(
                clinical_balanced[i:i+1],
                clinical[class_indices]
            )[0]
            nearest_in_class = class_indices[np.argmin(distances)]
            
            # Apply augmentation to create diverse synthetic sample
            ecg_balanced[i] = augment_ecg(ecg[nearest_in_class], method='combined')
            pcg_balanced[i] = augment_pcg(pcg[nearest_in_class])
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{n_balanced} samples...")
    
    print(f"Completed mapping all {n_balanced} samples")
    
    # Step 3: Save balanced dataset
    print("\n" + "="*80)
    print("STEP 3: Saving Balanced Dataset")
    print("="*80)
    
    output_path = Path("balanced_data")
    output_path.mkdir(exist_ok=True)
    
    print("Saving arrays...")
    np.save(output_path / "ecg_balanced.npy", ecg_balanced)
    np.save(output_path / "pcg_balanced.npy", pcg_balanced)
    np.save(output_path / "clinical_balanced.npy", clinical_balanced)
    np.save(output_path / "labels_balanced.npy", labels_balanced)
    
    # Save metadata
    metadata = {
        "n_samples": int(n_balanced),
        "n_original": int(n_original),
        "n_synthetic": int(n_synthetic),
        "class_distribution": {
            int(cls): int(count) 
            for cls, count in Counter(labels_balanced).items()
        },
        "shapes": {
            "ecg": list(ecg_balanced.shape),
            "pcg": list(pcg_balanced.shape),
            "clinical": list(clinical_balanced.shape)
        },
        "method": "SMOTE-Tomek",
        "augmentation": "ECG: jitter+scaling+shift, PCG: freq+time masking"
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved to {output_path}/")
    print(f"  - ecg_balanced.npy")
    print(f"  - pcg_balanced.npy")
    print(f"  - clinical_balanced.npy")
    print(f"  - labels_balanced.npy")
    print(f"  - metadata.json")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original dataset: {n_original} samples")
    print(f"Balanced dataset: {n_balanced} samples (+{n_synthetic} synthetic)")
    print(f"Size increase: {100 * (n_balanced - n_original) / n_original:.1f}%")
    print("\nClass distribution (before → after):")
    
    orig_dist = Counter(labels)
    bal_dist = Counter(labels_balanced)
    
    for cls in sorted(set(labels)):
        orig_count = orig_dist[cls]
        bal_count = bal_dist[cls]
        orig_pct = 100 * orig_count / len(labels)
        bal_pct = 100 * bal_count / len(labels_balanced)
        print(f"  Class {cls}: {orig_count:5d} ({orig_pct:5.2f}%) → {bal_count:5d} ({bal_pct:5.2f}%)")
    
    print("\nBalanced dataset created successfully!")
    print("="*80)


if __name__ == "__main__":
    try:
        create_balanced_dataset()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
