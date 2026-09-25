"""
Module 5: Clinical Dataset Loader
Stage 2 - Dataset Preparation

Loads UCI Heart Disease dataset and extracts clinical features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger("ACRMF-Net")


class ClinicalDataLoader:
    """
    Loads and processes clinical features from UCI Heart Disease dataset
    
    Features (13 total):
    1. age: Age in years
    2. sex: Sex (1 = male; 0 = female)
    3. cp: Chest pain type (0-3)
    4. trestbps: Resting blood pressure (mm Hg)
    5. chol: Serum cholesterol (mg/dl)
    6. fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    7. restecg: Resting ECG results (0-2)
    8. thalach: Maximum heart rate achieved
    9. exang: Exercise induced angina (1 = yes; 0 = no)
    10. oldpeak: ST depression induced by exercise
    11. slope: Slope of peak exercise ST segment (0-2)
    12. ca: Number of major vessels colored by fluoroscopy (0-3)
    13. thal: Thalassemia (0-3)
    
    Target: 0 = no disease, 1-4 = disease (converted to binary: 0/1)
    """
    
    def __init__(
        self,
        csv_path: Path,
        handle_missing: str = "mean",  # "mean", "median", "drop", "forward_fill"
        normalize: bool = False
    ):
        """
        Initialize Clinical Data Loader
        
        Args:
            csv_path: Path to heart_disease_uci.csv
            handle_missing: Strategy for handling missing values
            normalize: Whether to normalize features (done in preprocessing stage)
        """
        self.csv_path = Path(csv_path)
        self.handle_missing = handle_missing
        self.normalize = normalize
        
        # Feature names
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        self.target_name = 'num'  # Disease indicator (0-4)
        
        # Data storage
        self.data = None
        self.features = None
        self.labels = None
        self.patient_ids = None
        
        logger.info(f" ClinicalDataLoader initialized with {csv_path}")
        
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load clinical data from CSV
        
        Returns:
            features: Clinical features array (N, 13)
            labels: Binary labels (N,) - 0: healthy, 1: disease
            patient_ids: Patient identifiers (N,)
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Clinical dataset not found at {self.csv_path}")
        
        logger.info(f"Loading clinical data from {self.csv_path}...")
        
        # Load CSV
        self.data = pd.read_csv(self.csv_path)
        
        logger.info(f"  Loaded {len(self.data)} patient records")
        
        # Extract patient IDs (if available, otherwise create sequential IDs)
        if 'id' in self.data.columns:
            self.patient_ids = self.data['id'].values
        else:
            self.patient_ids = np.arange(len(self.data))
        
        # Extract features
        if all(feat in self.data.columns for feat in self.feature_names):
            feature_data = self.data[self.feature_names]
        else:
            # Try to infer features (all columns except target)
            target_cols = ['num', 'target', 'label', 'id', 'dataset']
            feature_cols = [col for col in self.data.columns if col not in target_cols]
            feature_data = self.data[feature_cols]
            self.feature_names = feature_cols
            logger.warning(f"Using inferred features: {self.feature_names}")
        
        # Convert to numeric, coercing errors to NaN
        for col in feature_data.columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
        
        self.features = feature_data.values
        
        # Extract labels
        if self.target_name in self.data.columns:
            raw_labels = self.data[self.target_name].values
        elif 'target' in self.data.columns:
            raw_labels = self.data['target'].values
        elif 'label' in self.data.columns:
            raw_labels = self.data['label'].values
        else:
            raise ValueError("No target column found in dataset")
        
        # Convert to binary: 0 = no disease, 1 = disease
        self.labels = (raw_labels > 0).astype(np.int64)
        
        logger.info(f"  Features shape: {self.features.shape}")
        logger.info(f"  Labels shape: {self.labels.shape}")
        logger.info(f"  Class distribution: Healthy={np.sum(self.labels == 0)}, Disease={np.sum(self.labels == 1)}")
        
        # Handle missing values
        if np.any(pd.isna(self.features)):
            logger.warning(f"  Found {np.sum(pd.isna(self.features))} missing values")
            self.features = self._handle_missing_values(self.features)
            logger.info(f"  Missing values handled using '{self.handle_missing}' strategy")
        
        return self.features, self.labels, self.patient_ids
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in features"""
        features_df = pd.DataFrame(features, columns=self.feature_names)
        
        if self.handle_missing == "mean":
            features_df = features_df.fillna(features_df.mean())
        elif self.handle_missing == "median":
            features_df = features_df.fillna(features_df.median())
        elif self.handle_missing == "forward_fill":
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        elif self.handle_missing == "drop":
            features_df = features_df.dropna()
            logger.warning(f"  Dropped {len(features) - len(features_df)} rows with missing values")
        else:
            raise ValueError(f"Unknown missing value strategy: {self.handle_missing}")
        
        # Convert all columns to numeric, coercing errors
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(features_df.mean())
        
        return features_df.values
    
    def get_feature_info(self) -> Dict[str, Dict]:
        """
        Get information about each clinical feature
        
        Returns:
            Dictionary with feature statistics
        """
        if self.features is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        info = {}
        for i, name in enumerate(self.feature_names):
            feature_data = self.features[:, i]
            info[name] = {
                'index': i,
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'missing_count': int(np.sum(pd.isna(feature_data)))
            }
        
        return info
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.features is None or self.labels is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        stats = {
            'num_samples': len(self.features),
            'num_features': self.features.shape[1],
            'num_healthy': np.sum(self.labels == 0),
            'num_disease': np.sum(self.labels == 1),
            'class_ratio': np.sum(self.labels == 1) / len(self.labels),
            'feature_names': self.feature_names,
            'missing_values': np.sum(pd.isna(self.features))
        }
        
        return stats
    
    def summary(self):
        """Print dataset summary"""
        if self.features is None:
            logger.warning("No data loaded yet. Call load() first.")
            return
        
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("Clinical Dataset Summary")
        print("=" * 80)
        print(f"Total Samples:    {stats['num_samples']}")
        print(f"Features:         {stats['num_features']}")
        if len(stats['feature_names']) > 5:
            print(f"Feature Names:    {', '.join(str(f) for f in stats['feature_names'][:5])}...")
        else:
            print(f"Feature Names:    {', '.join(str(f) for f in stats['feature_names'])}")
        print(f"\nClass Distribution:")
        print(f"  Healthy:        {stats['num_healthy']} ({(1-stats['class_ratio'])*100:.1f}%)")
        print(f"  Disease:        {stats['num_disease']} ({stats['class_ratio']*100:.1f}%)")
        print(f"\nMissing Values:   {stats['missing_values']}")
        print("=" * 80 + "\n")
    
    def get_patient_data(self, patient_id: int) -> Dict:
        """
        Get data for a specific patient
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with patient data
        """
        if self.features is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        idx = np.where(self.patient_ids == patient_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Patient {patient_id} not found")
        
        idx = idx[0]
        
        patient_data = {
            'patient_id': patient_id,
            'features': dict(zip(self.feature_names, self.features[idx])),
            'label': int(self.labels[idx]),
            'diagnosis': 'Disease' if self.labels[idx] == 1 else 'Healthy'
        }
        
        return patient_data


def load_clinical_data(
    csv_path: Path,
    handle_missing: str = "mean",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load clinical data
    
    Args:
        csv_path: Path to clinical CSV file
        handle_missing: Strategy for handling missing values
        verbose: Print summary
        
    Returns:
        features: Clinical features (N, 13)
        labels: Binary labels (N,)
        patient_ids: Patient IDs (N,)
    """
    loader = ClinicalDataLoader(csv_path, handle_missing=handle_missing)
    features, labels, patient_ids = loader.load()
    
    if verbose:
        loader.summary()
    
    return features, labels, patient_ids


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import setup_logging
    
    # Setup logging
    setup_logging(log_level="INFO", file_output=False)
    
    # Test data loading
    csv_path = Path(__file__).parent.parent.parent.parent / "heart_disease_uci.csv"
    
    print(f"\n{'='*80}")
    print("Testing Clinical Dataset Loader (Module 5)")
    print(f"{'='*80}\n")
    
    # Load data
    loader = ClinicalDataLoader(csv_path, handle_missing="mean")
    features, labels, patient_ids = loader.load()
    
    # Show summary
    loader.summary()
    
    # Show feature info
    print("\nFeature Statistics:")
    print("-" * 80)
    feature_info = loader.get_feature_info()
    for name, info in list(feature_info.items())[:5]:  # Show first 5
        print(f"{name:12s}: mean={info['mean']:6.2f}, std={info['std']:6.2f}, "
              f"range=[{info['min']:6.2f}, {info['max']:6.2f}]")
    print("...")
    
    # Show sample patient
    print("\nSample Patient Data:")
    print("-" * 80)
    patient_data = loader.get_patient_data(patient_ids[0])
    print(f"Patient ID: {patient_data['patient_id']}")
    print(f"Diagnosis:  {patient_data['diagnosis']}")
    print(f"Features:   {list(patient_data['features'].items())[:3]}...")
    
    print(f"\n{'='*80}")
    print("[OK] Module 5: Clinical Dataset Loader - Test Complete")
    print(f"{'='*80}\n")
