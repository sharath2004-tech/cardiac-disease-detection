"""
Module 9: Clinical Preprocessor
Stage 3 - Data Preprocessing

Preprocesses clinical features for neural network input
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger("ACRMF-Net")


class ClinicalPreprocessor:
    """
    Preprocesses clinical features
    
    Features:
    - Normalization/Standardization
    - Outlier handling
    - Feature scaling
    - Transformation persistence for test data
    """
    
    def __init__(
        self,
        scaling_method: str = "standard",  # "standard", "minmax", "robust"
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0,  # Standard deviations
        clip_values: bool = True
    ):
        """
        Initialize Clinical Preprocessor
        
        Args:
            scaling_method: Method for feature scaling
            handle_outliers: Whether to handle outliers
            outlier_threshold: Threshold for outlier detection (std devs)
            clip_values: Whether to clip outliers instead of removing
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.clip_values = clip_values
        
        # Initialize scaler
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        self.is_fitted = False
        self.feature_stats = {}
        
        logger.info(f" ClinicalPreprocessor initialized: {scaling_method} scaling")
    
    def fit(self, features: np.ndarray) -> 'ClinicalPreprocessor':
        """
        Fit preprocessor on training data
        
        Args:
            features: Clinical features (N, num_features)
            
        Returns:
            self
        """
        logger.info(f"Fitting clinical preprocessor on {len(features)} samples...")
        
        # Compute feature statistics
        self.feature_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0)
        }
        
        # Fit scaler
        self.scaler.fit(features)
        self.is_fitted = True
        
        logger.info(f"  [OK] Preprocessor fitted")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features
        
        Args:
            features: Clinical features (N, num_features)
            
        Returns:
            Preprocessed features (N, num_features)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        features = features.copy()
        
        # Handle outliers
        if self.handle_outliers:
            features = self._handle_outliers(features)
        
        # Scale features
        features = self.scaler.transform(features)
        
        return features.astype(np.float32)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            features: Clinical features (N, num_features)
            
        Returns:
            Preprocessed features (N, num_features)
        """
        self.fit(features)
        return self.transform(features)
    
    def _handle_outliers(self, features: np.ndarray) -> np.ndarray:
        """Handle outliers using z-score method"""
        if not self.is_fitted:
            return features
        
        # Compute z-scores
        z_scores = np.abs((features - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-8))
        
        if self.clip_values:
            # Clip outliers to threshold
            mask = z_scores > self.outlier_threshold
            if np.any(mask):
                num_outliers = np.sum(mask)
                logger.debug(f"  Clipping {num_outliers} outlier values")
                
                # Clip to mean ± threshold * std
                lower_bound = self.feature_stats['mean'] - self.outlier_threshold * self.feature_stats['std']
                upper_bound = self.feature_stats['mean'] + self.outlier_threshold * self.feature_stats['std']
                features = np.clip(features, lower_bound, upper_bound)
        
        return features
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Inverse transform (denormalize)
        
        Args:
            features: Normalized features (N, num_features)
            
        Returns:
            Original scale features (N, num_features)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
        
        return self.scaler.inverse_transform(features)
    
    def get_feature_importance_stats(self) -> Dict:
        """
        Get feature statistics for importance analysis
        
        Returns:
            Dictionary with feature statistics
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
        
        stats = {}
        for key, values in self.feature_stats.items():
            stats[key] = values.tolist()
        
        return stats


def preprocess_clinical_data(
    train_features: np.ndarray,
    val_features: Optional[np.ndarray] = None,
    test_features: Optional[np.ndarray] = None,
    scaling_method: str = "standard",
    handle_outliers: bool = True
) -> Tuple[np.ndarray, ...]:
    """
    Convenience function to preprocess clinical data
    
    Args:
        train_features: Training features
        val_features: Validation features (optional)
        test_features: Test features (optional)
        scaling_method: Scaling method
        handle_outliers: Handle outliers
        
    Returns:
        Tuple of preprocessed features
    """
    preprocessor = ClinicalPreprocessor(
        scaling_method=scaling_method,
        handle_outliers=handle_outliers
    )
    
    # Fit on training data
    train_preprocessed = preprocessor.fit_transform(train_features)
    
    results = [train_preprocessed]
    
    # Transform validation data
    if val_features is not None:
        val_preprocessed = preprocessor.transform(val_features)
        results.append(val_preprocessed)
    
    # Transform test data
    if test_features is not None:
        test_preprocessed = preprocessor.transform(test_features)
        results.append(test_preprocessed)
    
    logger.info("[OK] Clinical data preprocessing complete")
    
    return tuple(results) if len(results) > 1 else results[0]


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
    print("Testing Clinical Preprocessor (Module 9)")
    print(f"{'='*80}\n")
    
    # Create synthetic clinical data
    np.random.seed(42)
    n_samples = 1000
    n_features = 13
    
    # Simulate features with different scales
    train_features = np.random.randn(n_samples, n_features) * np.array([10, 1, 3, 20, 50, 1, 2, 30, 1, 2, 2, 3, 3])
    train_features += np.array([50, 0.5, 1, 120, 200, 0.3, 1, 140, 0.2, 1, 1, 1, 2])
    
    # Add some outliers
    train_features[0, 0] = 200  # Age outlier
    train_features[5, 4] = 800  # Cholesterol outlier
    
    val_features = np.random.randn(200, n_features) * np.array([10, 1, 3, 20, 50, 1, 2, 30, 1, 2, 2, 3, 3])
    val_features += np.array([50, 0.5, 1, 120, 200, 0.3, 1, 140, 0.2, 1, 1, 1, 2])
    
    print("Original data:")
    print(f"  Train shape: {train_features.shape}")
    print(f"  Val shape:   {val_features.shape}")
    print(f"  Train mean:  {np.mean(train_features, axis=0)[:3]}")
    print(f"  Train std:   {np.std(train_features, axis=0)[:3]}")
    print()
    
    # Test StandardScaler
    print("Test 1: StandardScaler")
    print("-" * 80)
    preprocessor = ClinicalPreprocessor(scaling_method="standard", handle_outliers=True)
    train_scaled = preprocessor.fit_transform(train_features)
    val_scaled = preprocessor.transform(val_features)
    
    print(f"  Scaled train mean: {np.mean(train_scaled, axis=0)[:3]}")
    print(f"  Scaled train std:  {np.std(train_scaled, axis=0)[:3]}")
    print(f"  Scaled val mean:   {np.mean(val_scaled, axis=0)[:3]}")
    print()
    
    # Test MinMaxScaler
    print("Test 2: MinMaxScaler")
    print("-" * 80)
    train_scaled, val_scaled = preprocess_clinical_data(
        train_features, val_features,
        scaling_method="minmax",
        handle_outliers=True
    )
    
    print(f"  Scaled train min: {np.min(train_scaled, axis=0)[:3]}")
    print(f"  Scaled train max: {np.max(train_scaled, axis=0)[:3]}")
    print()
    
    # Test inverse transform
    print("Test 3: Inverse Transform")
    print("-" * 80)
    preprocessor = ClinicalPreprocessor(scaling_method="standard")
    train_scaled = preprocessor.fit_transform(train_features)
    train_reconstructed = preprocessor.inverse_transform(train_scaled)
    
    reconstruction_error = np.mean(np.abs(train_features - train_reconstructed))
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    print()
    
    # Feature statistics
    print("Feature Statistics:")
    print("-" * 80)
    stats = preprocessor.get_feature_importance_stats()
    print(f"  Feature means: {np.array(stats['mean'])[:3]}")
    print(f"  Feature stds:  {np.array(stats['std'])[:3]}")
    
    print(f"\n{'='*80}")
    print("[OK] Module 9: Clinical Preprocessor - Test Complete")
    print(f"{'='*80}\n")
