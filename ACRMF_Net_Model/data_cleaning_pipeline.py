"""
Comprehensive Data Cleaning Pipeline
=====================================
Cleans and prepares multimodal cardiac data for optimal training.

Issues to address:
1. Missing values in clinical, ECG, PCG data
2. Outliers and anomalies
3. Class imbalance
4. Data normalization/standardization
5. Feature quality assessment
6. Sample alignment across modalities

Goal: Clean, high-quality dataset ready for training
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from config.config import Config
from data.loaders.clinical_loader import ClinicalDataLoader
from data.loaders.ecg_loader import ECGDataLoader
from data.loaders.pcg_loader import PCGDataLoader

print("\n" + "="*80)
print("DATA CLEANING & PREPARATION PIPELINE")
print("="*80)


def analyze_missing_values(data, name="Data"):
    """Analyze missing values in dataset"""
    print(f"\n--- {name} Missing Values Analysis ---")
    
    if len(data.shape) == 1:
        missing_count = np.isnan(data).sum()
        missing_pct = (missing_count / len(data)) * 100
        print(f"Missing values: {missing_count} ({missing_pct:.2f}%)")
        return missing_count, missing_pct
    
    # For 2D data
    total_values = data.size
    missing_count = np.isnan(data).sum()
    missing_pct = (missing_count / total_values) * 100
    
    print(f"Total values: {total_values}")
    print(f"Missing values: {missing_count} ({missing_pct:.2f}%)")
    
    # Per-feature analysis
    if data.shape[1] < 50:  # Only for clinical features
        missing_per_feature = np.isnan(data).sum(axis=0)
        print("\nMissing per feature:")
        for i, count in enumerate(missing_per_feature):
            if count > 0:
                pct = (count / data.shape[0]) * 100
                print(f"  Feature {i}: {count} ({pct:.2f}%)")
    
    # Per-sample analysis
    missing_per_sample = np.isnan(data).sum(axis=1)
    samples_with_missing = (missing_per_sample > 0).sum()
    print(f"\nSamples with missing values: {samples_with_missing} ({samples_with_missing/len(data)*100:.2f}%)")
    print(f"Max missing per sample: {missing_per_sample.max()}")
    print(f"Avg missing per sample: {missing_per_sample.mean():.2f}")
    
    return missing_count, missing_pct


def detect_outliers(data, name="Data", method='iqr', threshold=3):
    """Detect outliers in data"""
    print(f"\n--- {name} Outlier Detection ---")
    
    if method == 'iqr':
        Q1 = np.nanpercentile(data, 25, axis=0)
        Q3 = np.nanpercentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        outlier_count = outliers.sum()
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
        outliers = z_scores > threshold
        outlier_count = outliers.sum()
    
    outlier_pct = (outlier_count / data.size) * 100
    print(f"Method: {method}")
    print(f"Outliers detected: {outlier_count} ({outlier_pct:.2f}%)")
    
    # Per-sample outlier count
    outliers_per_sample = outliers.sum(axis=1)
    samples_with_outliers = (outliers_per_sample > 0).sum()
    print(f"Samples with outliers: {samples_with_outliers} ({samples_with_outliers/len(data)*100:.2f}%)")
    
    return outliers, outlier_count


def clean_clinical_data(clinical_data, labels):
    """Clean clinical data comprehensively"""
    print("\n" + "="*80)
    print("CLEANING CLINICAL DATA")
    print("="*80)
    
    cleaned_data = clinical_data.copy()
    
    # Step 1: Analyze missing values
    missing_count, missing_pct = analyze_missing_values(cleaned_data, "Clinical")
    
    # Step 2: Handle missing values
    print("\n--- Imputing Missing Values ---")
    
    if missing_count > 0:
        # Try multiple strategies
        strategies = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'knn': KNNImputer(n_neighbors=5)
        }
        
        print("\nTesting imputation strategies:")
        best_strategy = None
        best_score = -np.inf
        
        for name, imputer in strategies.items():
            try:
                imputed = imputer.fit_transform(cleaned_data)
                # Score = negative variance of imputed values (prefer stable imputation)
                score = -np.var(imputed)
                print(f"  {name}: score = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_strategy = name
                    cleaned_data = imputed
            except Exception as e:
                print(f"  {name}: failed ({e})")
        
        print(f"\n[OK] Using {best_strategy} imputation")
        
        # Verify no more missing values
        remaining_missing = np.isnan(cleaned_data).sum()
        print(f"[OK] Remaining missing values: {remaining_missing}")
    
    # Step 3: Detect and handle outliers
    outliers, outlier_count = detect_outliers(cleaned_data, "Clinical", method='iqr')
    
    if outlier_count > 0:
        print(f"\nHandling {outlier_count} outliers...")
        # Clip outliers to reasonable bounds (IQR method)
        Q1 = np.percentile(cleaned_data, 25, axis=0)
        Q3 = np.percentile(cleaned_data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        cleaned_data = np.clip(cleaned_data, lower_bound, upper_bound)
        print("[OK] Outliers clipped to IQR bounds")
    
    # Step 4: Normalize features
    print("\n--- Normalizing Features ---")
    scaler = RobustScaler()  # Robust to outliers
    cleaned_data = scaler.fit_transform(cleaned_data)
    print("[OK] Applied RobustScaler normalization")
    
    # Step 5: Check for duplicate samples
    print("\n--- Checking for Duplicates ---")
    unique_samples = np.unique(cleaned_data, axis=0)
    duplicates = len(cleaned_data) - len(unique_samples)
    if duplicates > 0:
        print(f"[WARN]  Found {duplicates} duplicate samples")
        # Keep unique samples
        unique_indices = np.unique(cleaned_data, axis=0, return_index=True)[1]
        cleaned_data = cleaned_data[unique_indices]
        labels = labels[unique_indices]
        print(f"[OK] Removed duplicates. Remaining: {len(cleaned_data)} samples")
    else:
        print("[OK] No duplicates found")
    
    return cleaned_data, labels


def clean_signal_data(signal_data, labels, name="Signal"):
    """Clean ECG/PCG signal data"""
    print("\n" + "="*80)
    print(f"CLEANING {name.upper()} DATA")
    print("="*80)
    
    cleaned_data = signal_data.copy()
    
    # Step 1: Analyze missing values
    missing_count, missing_pct = analyze_missing_values(cleaned_data, name)
    
    # Step 2: Handle missing values
    if missing_count > 0:
        print(f"\nHandling missing values in {name}...")
        
        # For signals, forward/backward fill is often better
        if len(cleaned_data.shape) == 2:
            # Fill missing with interpolation
            for i in range(len(cleaned_data)):
                sample = cleaned_data[i]
                if np.isnan(sample).any():
                    # Linear interpolation
                    nans = np.isnan(sample)
                    x = np.arange(len(sample))
                    sample[nans] = np.interp(x[nans], x[~nans], sample[~nans])
                    cleaned_data[i] = sample
        
        remaining_missing = np.isnan(cleaned_data).sum()
        print(f"[OK] Missing values after interpolation: {remaining_missing}")
        
        # If still missing, use median
        if remaining_missing > 0:
            median_value = np.nanmedian(cleaned_data)
            cleaned_data = np.nan_to_num(cleaned_data, nan=median_value)
            print(f"[OK] Filled remaining with median: {median_value:.4f}")
    
    # Step 3: Detect corrupted signals
    print("\n--- Detecting Corrupted Signals ---")
    
    # Check for flat signals (all same value)
    if len(cleaned_data.shape) == 2:
        signal_variance = np.var(cleaned_data, axis=1)
        flat_signals = signal_variance < 1e-6
        flat_count = flat_signals.sum()
        
        if flat_count > 0:
            print(f"[WARN]  Found {flat_count} flat signals (zero variance)")
            # Remove flat signals
            cleaned_data = cleaned_data[~flat_signals]
            labels = labels[~flat_signals]
            print(f"[OK] Removed flat signals. Remaining: {len(cleaned_data)} samples")
    
    # Step 4: Detect extreme outlier signals
    print("\n--- Detecting Extreme Signals ---")
    signal_means = np.mean(np.abs(cleaned_data), axis=1)
    mean_threshold = np.percentile(signal_means, 99)
    extreme_signals = signal_means > mean_threshold
    extreme_count = extreme_signals.sum()
    
    if extreme_count > 0:
        print(f"[WARN]  Found {extreme_count} extreme signals (>99th percentile)")
        print("Keeping them (may be valid cardiac events)")
    
    # Step 5: Normalize signals
    print("\n--- Normalizing Signals ---")
    # Normalize each signal independently
    if len(cleaned_data.shape) == 2:
        means = cleaned_data.mean(axis=1, keepdims=True)
        stds = cleaned_data.std(axis=1, keepdims=True)
        stds[stds < 1e-6] = 1.0  # Avoid division by zero
        cleaned_data = (cleaned_data - means) / stds
        print("[OK] Applied per-signal z-score normalization")
    else:
        # For 3D data (channels, time)
        scaler = StandardScaler()
        original_shape = cleaned_data.shape
        cleaned_data = cleaned_data.reshape(len(cleaned_data), -1)
        cleaned_data = scaler.fit_transform(cleaned_data)
        cleaned_data = cleaned_data.reshape(original_shape)
        print("[OK] Applied StandardScaler normalization")
    
    return cleaned_data, labels


def align_modalities(clinical_data, ecg_data, pcg_data, clinical_labels, ecg_labels, pcg_labels):
    """Align samples across all three modalities"""
    print("\n" + "="*80)
    print("ALIGNING MODALITIES")
    print("="*80)
    
    print(f"\nBefore alignment:")
    print(f"  Clinical: {len(clinical_data)} samples")
    print(f"  ECG:      {len(ecg_data)} samples")
    print(f"  PCG:      {len(pcg_data)} samples")
    
    # Check if labels match
    min_samples = min(len(clinical_data), len(ecg_data), len(pcg_data))
    
    # Truncate to minimum length (simple alignment)
    clinical_data = clinical_data[:min_samples]
    ecg_data = ecg_data[:min_samples]
    pcg_data = pcg_data[:min_samples]
    clinical_labels = clinical_labels[:min_samples]
    ecg_labels = ecg_labels[:min_samples]
    pcg_labels = pcg_labels[:min_samples]
    
    # Verify labels match
    if not (np.array_equal(clinical_labels, ecg_labels) and np.array_equal(ecg_labels, pcg_labels)):
        print("[WARN]  Labels don't match across modalities!")
        print("Using clinical labels as ground truth...")
        labels = clinical_labels
    else:
        labels = clinical_labels
        print("[OK] Labels match across all modalities")
    
    print(f"\nAfter alignment:")
    print(f"  All modalities: {min_samples} samples")
    
    return clinical_data, ecg_data, pcg_data, labels


def analyze_class_distribution(labels):
    """Analyze class distribution"""
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\nTotal samples: {total}")
    print(f"Number of classes: {len(unique)}")
    print(f"\nDistribution:")
    
    class_info = []
    for cls, count in zip(unique, counts):
        pct = (count / total) * 100
        class_info.append({
            'class': int(cls),
            'count': int(count),
            'percentage': float(pct)
        })
        print(f"  Class {int(cls)}: {count:4d} samples ({pct:5.2f}%)")
    
    # Calculate imbalance ratio
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 10:
        print("[WARN]  SEVERE imbalance detected!")
    elif imbalance_ratio > 3:
        print("[WARN]  MODERATE imbalance detected")
    else:
        print("[OK] Classes are relatively balanced")
    
    return class_info, imbalance_ratio


def save_cleaned_data(clinical_data, ecg_data, pcg_data, labels, class_info, stats):
    """Save cleaned data to disk"""
    print("\n" + "="*80)
    print("SAVING CLEANED DATA")
    print("="*80)
    
    output_dir = Path('cleaned_data')
    output_dir.mkdir(exist_ok=True)
    
    # Save numpy arrays
    np.save(output_dir / 'clinical_cleaned.npy', clinical_data)
    np.save(output_dir / 'ecg_cleaned.npy', ecg_data)
    np.save(output_dir / 'pcg_cleaned.npy', pcg_data)
    np.save(output_dir / 'labels_cleaned.npy', labels)
    
    print(f"[OK] Saved data arrays to {output_dir}/")
    
    # Save metadata
    metadata = {
        'num_samples': int(len(labels)),
        'num_classes': int(len(np.unique(labels))),
        'clinical_features': int(clinical_data.shape[1]),
        'ecg_shape': list(ecg_data.shape[1:]),
        'pcg_shape': list(pcg_data.shape[1:]),
        'class_distribution': class_info,
        'cleaning_stats': stats
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Saved metadata to {output_dir}/metadata.json")
    
    return output_dir


def visualize_cleaning_results(clinical_before, clinical_after, labels, output_dir):
    """Visualize the impact of cleaning"""
    print("\n--- Generating Visualizations ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Data Cleaning Results', fontsize=16, fontweight='bold')
    
    # 1. Missing values before/after
    ax = axes[0, 0]
    missing_before = np.isnan(clinical_before).sum(axis=0)
    missing_after = np.isnan(clinical_after).sum(axis=0)
    x = np.arange(len(missing_before))
    ax.bar(x - 0.2, missing_before, width=0.4, label='Before', alpha=0.7)
    ax.bar(x + 0.2, missing_after, width=0.4, label='After', alpha=0.7)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Missing Count')
    ax.set_title('Missing Values: Before vs After')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution before/after (first feature)
    ax = axes[0, 1]
    ax.hist(clinical_before[:, 0], bins=30, alpha=0.5, label='Before', density=True)
    ax.hist(clinical_after[:, 0], bins=30, alpha=0.5, label='After', density=True)
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Density')
    ax.set_title('Feature 0 Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Class distribution
    ax = axes[0, 2]
    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    ax.bar(unique, counts, color=colors, alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    ax.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap (before)
    ax = axes[1, 0]
    # Remove missing for correlation
    valid_before = clinical_before[~np.isnan(clinical_before).any(axis=1)]
    if len(valid_before) > 0:
        corr_before = np.corrcoef(valid_before[:, :min(10, clinical_before.shape[1])].T)
        im = ax.imshow(corr_before, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Feature Correlation (Before)')
        plt.colorbar(im, ax=ax)
    
    # 5. Correlation heatmap (after)
    ax = axes[1, 1]
    corr_after = np.corrcoef(clinical_after[:, :min(10, clinical_after.shape[1])].T)
    im = ax.imshow(corr_after, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_title('Feature Correlation (After)')
    plt.colorbar(im, ax=ax)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
    CLEANING SUMMARY
    
    Samples: {len(labels)}
    Classes: {len(unique)}
    Features: {clinical_after.shape[1]}
    
    Before:
      Missing: {np.isnan(clinical_before).sum()}
      
    After:
      Missing: {np.isnan(clinical_after).sum()}
      
    [OK] Data is clean and ready!
    """
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cleaning_results.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved visualization to {output_dir}/cleaning_results.png")
    plt.close()


def main():
    """Main cleaning pipeline"""
    
    # Load raw data
    print("\nLoading raw data...")
    cfg = Config()
    
    clinical_loader = ClinicalDataLoader(cfg.clinical_csv)
    ecg_loader = ECGDataLoader(cfg.ptb_xl_dir)
    pcg_loader = PCGDataLoader(cfg.pcg_archive_dir)
    
    clinical_data_raw, clinical_labels, _ = clinical_loader.load()
    
    print("\n[WARN]  ECG and PCG loading may take several minutes...")
    print("Loading ECG data...")
    ecg_data_raw, ecg_labels, _ = ecg_loader.load()
    
    print("Loading PCG data...")
    pcg_data_raw, pcg_labels, _ = pcg_loader.load()
    
    print("\n[OK] Raw data loaded")
    
    # Store original for comparison
    clinical_original = clinical_data_raw.copy()
    
    # Clean each modality
    clinical_data_clean, clinical_labels_clean = clean_clinical_data(clinical_data_raw, clinical_labels)
    ecg_data_clean, ecg_labels_clean = clean_signal_data(ecg_data_raw, ecg_labels, "ECG")
    pcg_data_clean, pcg_labels_clean = clean_signal_data(pcg_data_raw, pcg_labels, "PCG")
    
    # Align modalities
    clinical_final, ecg_final, pcg_final, labels_final = align_modalities(
        clinical_data_clean, ecg_data_clean, pcg_data_clean,
        clinical_labels_clean, ecg_labels_clean, pcg_labels_clean
    )
    
    # Analyze final distribution
    class_info, imbalance_ratio = analyze_class_distribution(labels_final)
    
    # Collect statistics
    stats = {
        'clinical_missing_before': int(np.isnan(clinical_original).sum()),
        'clinical_missing_after': int(np.isnan(clinical_final).sum()),
        'samples_final': int(len(labels_final)),
        'imbalance_ratio': float(imbalance_ratio)
    }
    
    # Save cleaned data
    output_dir = save_cleaned_data(clinical_final, ecg_final, pcg_final, labels_final, class_info, stats)
    
    # Visualize
    visualize_cleaning_results(clinical_original, clinical_final, labels_final, output_dir)
    
    # Final report
    print("\n" + "="*80)
    print("[OK] DATA CLEANING COMPLETE!")
    print("="*80)
    print(f"\nCleaned data saved to: {output_dir}/")
    print(f"\nFiles:")
    print(f"   clinical_cleaned.npy")
    print(f"   ecg_cleaned.npy")
    print(f"   pcg_cleaned.npy")
    print(f"   labels_cleaned.npy")
    print(f"   metadata.json")
    print(f"   cleaning_results.png")
    
    print(f"\nNext step: Train model using cleaned data")
    print(f"  py train_with_cleaned_data.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
