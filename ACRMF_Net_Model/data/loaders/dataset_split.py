"""
Module 8: Dataset Split Module
Stage 2 - Dataset Preparation

Handles train/validation/test splitting with stratification
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging

logger = logging.getLogger("ACRMF-Net")


class DatasetSplitter:
    """
    Handles dataset splitting with stratification for multimodal data
    
    Ensures:
    - Stratified splits (maintain class distribution)
    - Reproducible splits (random seed)
    - Support for cross-validation
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True
    ):
        """
        Initialize Dataset Splitter
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            stratify: Whether to stratify by labels
        """
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.stratify = stratify
        
        # Store split indices
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        logger.info(f"[Stats] DatasetSplitter initialized: train={train_ratio:.1%}, "
                   f"val={val_ratio:.1%}, test={test_ratio:.1%}")
    
    def split(
        self,
        labels: np.ndarray,
        indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train/val/test sets
        
        Args:
            labels: Label array (N,)
            indices: Optional array of indices to split (if None, use range)
            
        Returns:
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices
        """
        if indices is None:
            indices = np.arange(len(labels))
        
        logger.info(f"Splitting {len(labels)} samples...")
        
        # Stratify labels if requested
        stratify_labels = labels if self.stratify else None
        
        # First split: separate test set
        temp_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=labels[indices] if self.stratify else None
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_indices, val_indices = train_test_split(
            temp_indices,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=labels[temp_indices] if self.stratify else None
        )
        
        # Store indices
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        # Log split statistics
        logger.info(f"  Train: {len(train_indices)} samples ({len(train_indices)/len(labels)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_indices)} samples ({len(val_indices)/len(labels)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_indices)} samples ({len(test_indices)/len(labels)*100:.1f}%)")
        
        if self.stratify:
            self._log_class_distribution(labels, train_indices, val_indices, test_indices)
        
        return train_indices, val_indices, test_indices
    
    def _log_class_distribution(
        self,
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray
    ):
        """Log class distribution for each split"""
        logger.info("  Class distribution:")
        
        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            split_labels = labels[idx]
            class_0 = np.sum(split_labels == 0)
            class_1 = np.sum(split_labels == 1)
            total = len(split_labels)
            
            logger.info(f"    {name}: Class 0={class_0} ({class_0/total*100:.1f}%), "
                       f"Class 1={class_1} ({class_1/total*100:.1f}%)")
    
    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get stored split indices"""
        if self.train_indices is None:
            raise ValueError("Dataset not split yet. Call split() first.")
        
        return self.train_indices, self.val_indices, self.test_indices
    
    def create_cross_validation_splits(
        self,
        labels: np.ndarray,
        n_folds: int = 5,
        indices: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits
        
        Args:
            labels: Label array
            n_folds: Number of folds
            indices: Optional indices to split
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if indices is None:
            indices = np.arange(len(labels))
        
        logger.info(f"Creating {n_folds}-fold cross-validation splits...")
        
        if self.stratify:
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            splits = list(kfold.split(indices, labels[indices]))
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            splits = list(kfold.split(indices))
        
        # Convert to absolute indices
        cv_splits = [(indices[train_idx], indices[val_idx]) for train_idx, val_idx in splits]
        
        logger.info(f"  Created {n_folds} folds")
        for i, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(f"    Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        return cv_splits


class MultimodalDatasetSplitter:
    """
    Dataset splitter for multimodal data (Clinical + ECG + PCG)
    
    Ensures all modalities are split consistently using the same indices
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True
    ):
        """Initialize multimodal dataset splitter"""
        self.splitter = DatasetSplitter(
            train_ratio, val_ratio, test_ratio, random_seed, stratify
        )
        
        logger.info(" MultimodalDatasetSplitter initialized")
    
    def split(
        self,
        clinical_data: np.ndarray,
        ecg_data: np.ndarray,
        pcg_data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split multimodal dataset
        
        Args:
            clinical_data: Clinical features (N, 13)
            ecg_data: ECG signals (N, seq_len, 12)
            pcg_data: PCG signals (N, seq_len)
            labels: Labels (N,)
            
        Returns:
            Dictionary with train/val/test splits for each modality
        """
        # Validate shapes
        n_samples = len(labels)
        assert clinical_data.shape[0] == n_samples, "Clinical data size mismatch"
        assert ecg_data.shape[0] == n_samples, "ECG data size mismatch"
        assert pcg_data.shape[0] == n_samples, "PCG data size mismatch"
        
        logger.info(f"Splitting multimodal dataset with {n_samples} samples...")
        
        # Get split indices
        train_idx, val_idx, test_idx = self.splitter.split(labels)
        
        # Split each modality using the same indices
        splits = {
            'train': {
                'clinical': clinical_data[train_idx],
                'ecg': ecg_data[train_idx],
                'pcg': pcg_data[train_idx],
                'labels': labels[train_idx],
                'indices': train_idx
            },
            'val': {
                'clinical': clinical_data[val_idx],
                'ecg': ecg_data[val_idx],
                'pcg': pcg_data[val_idx],
                'labels': labels[val_idx],
                'indices': val_idx
            },
            'test': {
                'clinical': clinical_data[test_idx],
                'ecg': ecg_data[test_idx],
                'pcg': pcg_data[test_idx],
                'labels': labels[test_idx],
                'indices': test_idx
            }
        }
        
        logger.info("[OK] Multimodal dataset split complete")
        self._log_split_shapes(splits)
        
        return splits
    
    def _log_split_shapes(self, splits: Dict):
        """Log shapes of each split"""
        for split_name, split_data in splits.items():
            logger.info(f"  {split_name.capitalize()} split:")
            logger.info(f"    Clinical: {split_data['clinical'].shape}")
            logger.info(f"    ECG:      {split_data['ecg'].shape}")
            logger.info(f"    PCG:      {split_data['pcg'].shape}")
            logger.info(f"    Labels:   {split_data['labels'].shape}")


def split_dataset(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to split dataset
    
    Args:
        labels: Label array
        train_ratio: Training ratio
        val_ratio: Validation ratio
        test_ratio: Test ratio
        random_seed: Random seed
        stratify: Stratify by labels
        
    Returns:
        train_indices, val_indices, test_indices
    """
    splitter = DatasetSplitter(train_ratio, val_ratio, test_ratio, random_seed, stratify)
    return splitter.split(labels)


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import setup_logging
    
    # Setup logging
    setup_logging(log_level="INFO", file_output=False)
    
    print(f"\n{'='*80}")
    print("Testing Dataset Split Module (Module 8)")
    print(f"{'='*80}\n")
    
    # Create synthetic data
    n_samples = 1000
    labels = np.random.randint(0, 2, size=n_samples)
    
    clinical_data = np.random.randn(n_samples, 13)
    ecg_data = np.random.randn(n_samples, 1000, 12)
    pcg_data = np.random.randn(n_samples, 2000)
    
    print(f"Created synthetic multimodal dataset:")
    print(f"  Clinical: {clinical_data.shape}")
    print(f"  ECG:      {ecg_data.shape}")
    print(f"  PCG:      {pcg_data.shape}")
    print(f"  Labels:   {labels.shape}")
    print()
    
    # Test basic splitting
    print("Test 1: Basic Dataset Splitting")
    print("-" * 80)
    splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_idx, val_idx, test_idx = splitter.split(labels)
    print()
    
    # Test multimodal splitting
    print("Test 2: Multimodal Dataset Splitting")
    print("-" * 80)
    mm_splitter = MultimodalDatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    splits = mm_splitter.split(clinical_data, ecg_data, pcg_data, labels)
    print()
    
    # Test cross-validation
    print("Test 3: Cross-Validation Splits")
    print("-" * 80)
    cv_splits = splitter.create_cross_validation_splits(labels, n_folds=5)
    print()
    
    # Verify no overlap
    print("Verification:")
    print("-" * 80)
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    assert len(train_set & val_set) == 0, "Train and val overlap!"
    assert len(train_set & test_set) == 0, "Train and test overlap!"
    assert len(val_set & test_set) == 0, "Val and test overlap!"
    print("[OK] No overlap between splits")
    
    assert len(train_set) + len(val_set) + len(test_set) == n_samples, "Not all samples accounted for!"
    print("[OK] All samples accounted for")
    
    print(f"\n{'='*80}")
    print("[OK] Module 8: Dataset Split Module - Test Complete")
    print(f"{'='*80}\n")
