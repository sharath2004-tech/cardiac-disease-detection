"""
Module 6: ECG Dataset Loader
Stage 2 - Dataset Preparation

Loads PTB-XL ECG dataset (12-lead electrocardiography)
"""

import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
import ast

logger = logging.getLogger("ACRMF-Net")


class ECGDataLoader:
    """
    Loads PTB-XL ECG dataset
    
    Dataset Info:
    - 21,837 ECG recordings
    - 12-lead ECG signals
    - 100 Hz and 500 Hz sampling rates available
    - Multiple cardiac conditions annotated
    
    Signal shape: (sequence_length, 12) for 12 leads
    """
    
    def __init__(
        self,
        ptb_xl_dir: Path,
        sampling_rate: int = 100,  # 100 Hz or 500 Hz
        target_length: int = 1000,  # Target sequence length
        use_diagnostic_superclass: bool = True
    ):
        """
        Initialize ECG Data Loader
        
        Args:
            ptb_xl_dir: Path to PTB-XL dataset directory
            sampling_rate: Sampling rate (100 or 500 Hz)
            target_length: Target sequence length after resampling
            use_diagnostic_superclass: Use superclass labels (NORM, MI, STTC, CD, HYP)
        """
        self.ptb_xl_dir = Path(ptb_xl_dir)
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.use_diagnostic_superclass = use_diagnostic_superclass
        
        # Paths
        self.metadata_path = self.ptb_xl_dir / "ptbxl_database.csv"
        self.scp_statements_path = self.ptb_xl_dir / "scp_statements.csv"
        
        # Data storage
        self.metadata = None
        self.scp_codes = None
        self.ecg_signals = []
        self.labels = []
        self.record_ids = []
        
        # Label mapping for binary classification
        self.label_map = {
            'NORM': 0,  # Normal
            'MI': 1,    # Myocardial Infarction
            'STTC': 1,  # ST/T Change
            'CD': 1,    # Conduction Disturbance
            'HYP': 1    # Hypertrophy
        }
        
        logger.info(f"[Stats] ECGDataLoader initialized: {sampling_rate}Hz, length={target_length}")
    
    def load_metadata(self):
        """Load PTB-XL metadata"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
        
        logger.info(f"Loading ECG metadata from {self.metadata_path}...")
        
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata['scp_codes'] = self.metadata['scp_codes'].apply(lambda x: ast.literal_eval(x))
        
        logger.info(f"  Loaded metadata for {len(self.metadata)} ECG recordings")
        
        # Load SCP code descriptions
        if self.scp_statements_path.exists():
            self.scp_codes = pd.read_csv(self.scp_statements_path, index_col=0)
        
        return self.metadata
    
    def load(
        self,
        max_samples: Optional[int] = None,
        filter_labels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load ECG signals and labels
        
        Args:
            max_samples: Maximum number of samples to load (None = all)
            filter_labels: Filter by specific diagnostic labels
            
        Returns:
            signals: ECG signals array (N, target_length, 12)
            labels: Binary labels (N,) - 0: normal, 1: abnormal
            record_ids: Record identifiers (N,)
        """
        if self.metadata is None:
            self.load_metadata()
        
        logger.info(f"Loading ECG signals...")
        
        # Filter metadata if needed
        metadata_subset = self.metadata.copy()
        
        if filter_labels:
            mask = metadata_subset['scp_codes'].apply(
                lambda x: any(label in x for label in filter_labels)
            )
            metadata_subset = metadata_subset[mask]
            logger.info(f"  Filtered to {len(metadata_subset)} records with labels: {filter_labels}")
        
        # Limit samples
        if max_samples:
            metadata_subset = metadata_subset.head(max_samples)
            logger.info(f"  Limited to {max_samples} samples")
        
        # Load signals
        self.ecg_signals = []
        self.labels = []
        self.record_ids = []
        
        num_failed = 0
        
        for idx, row in metadata_subset.iterrows():
            try:
                # Load ECG signal
                signal = self._load_ecg_signal(row)
                
                # Get label
                label = self._get_label(row)
                
                if signal is not None and label is not None:
                    self.ecg_signals.append(signal)
                    self.labels.append(label)
                    self.record_ids.append(row['ecg_id'])
                    
                    if len(self.ecg_signals) % 1000 == 0:
                        logger.info(f"  Loaded {len(self.ecg_signals)} signals...")
                        
            except Exception as e:
                num_failed += 1
                if num_failed <= 5:  # Show first 5 errors
                    logger.warning(f"  Failed to load record {row['ecg_id']}: {e}")
        
        if num_failed > 0:
            logger.warning(f"  Total failed: {num_failed} records")
        
        # Convert to numpy arrays
        signals = np.array(self.ecg_signals, dtype=np.float32)  # (N, target_length, 12)
        labels = np.array(self.labels, dtype=np.int64)          # (N,)
        record_ids = np.array(self.record_ids, dtype=np.int64)  # (N,)
        
        logger.info(f"[OK] Loaded {len(signals)} ECG signals")
        logger.info(f"  Signals shape: {signals.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
        logger.info(f"  Class distribution: Normal={np.sum(labels == 0)}, Abnormal={np.sum(labels == 1)}")
        
        return signals, labels, record_ids
    
    def _load_ecg_signal(self, row: pd.Series) -> Optional[np.ndarray]:
        """Load individual ECG signal"""
        # Construct file path
        if self.sampling_rate == 100:
            filename = self.ptb_xl_dir / row['filename_lr']
        else:
            filename = self.ptb_xl_dir / row['filename_hr']
        
        # Remove .hea extension if present
        filename = str(filename).replace('.hea', '')
        
        # Load with wfdb
        record = wfdb.rdrecord(filename)
        signal = record.p_signal  # Shape: (original_length, 12)
        
        # Resample to target length if needed
        if signal.shape[0] != self.target_length:
            signal = self._resample_signal(signal, self.target_length)
        
        return signal.astype(np.float32)
    
    def _resample_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Resample signal to target length"""
        from scipy.interpolate import interp1d
        
        original_length = signal.shape[0]
        original_indices = np.linspace(0, original_length - 1, original_length)
        target_indices = np.linspace(0, original_length - 1, target_length)
        
        # Resample each lead
        resampled = np.zeros((target_length, signal.shape[1]))
        for lead_idx in range(signal.shape[1]):
            interpolator = interp1d(original_indices, signal[:, lead_idx], kind='linear')
            resampled[:, lead_idx] = interpolator(target_indices)
        
        return resampled
    
    def _get_label(self, row: pd.Series) -> Optional[int]:
        """Extract binary label from SCP codes"""
        scp_codes = row['scp_codes']
        
        if self.use_diagnostic_superclass:
            # Use diagnostic superclass
            if 'diagnostic_superclass' in row and pd.notna(row['diagnostic_superclass']):
                superclass = row['diagnostic_superclass']
                return self.label_map.get(superclass, 1)  # Default to abnormal
        
        # Check for NORM code
        if 'NORM' in scp_codes:
            return 0  # Normal
        else:
            return 1  # Abnormal
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if len(self.ecg_signals) == 0:
            raise ValueError("Data not loaded. Call load() first.")
        
        signals = np.array(self.ecg_signals)
        labels = np.array(self.labels)
        
        stats = {
            'num_samples': len(signals),
            'signal_shape': signals.shape,
            'num_normal': np.sum(labels == 0),
            'num_abnormal': np.sum(labels == 1),
            'class_ratio': np.sum(labels == 1) / len(labels),
            'signal_mean': np.mean(signals),
            'signal_std': np.std(signals),
            'signal_min': np.min(signals),
            'signal_max': np.max(signals)
        }
        
        return stats
    
    def summary(self):
        """Print dataset summary"""
        if len(self.ecg_signals) == 0:
            logger.warning("No data loaded yet. Call load() first.")
            return
        
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("ECG Dataset Summary (PTB-XL)")
        print("=" * 80)
        print(f"Total Samples:    {stats['num_samples']}")
        print(f"Signal Shape:     {stats['signal_shape']}")
        print(f"Sampling Rate:    {self.sampling_rate} Hz")
        print(f"Sequence Length:  {self.target_length}")
        print(f"\nClass Distribution:")
        print(f"  Normal:         {stats['num_normal']} ({(1-stats['class_ratio'])*100:.1f}%)")
        print(f"  Abnormal:       {stats['num_abnormal']} ({stats['class_ratio']*100:.1f}%)")
        print(f"\nSignal Statistics:")
        print(f"  Mean:           {stats['signal_mean']:.4f}")
        print(f"  Std:            {stats['signal_std']:.4f}")
        print(f"  Range:          [{stats['signal_min']:.4f}, {stats['signal_max']:.4f}]")
        print("=" * 80 + "\n")


def load_ecg_data(
    ptb_xl_dir: Path,
    sampling_rate: int = 100,
    target_length: int = 1000,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load ECG data
    
    Args:
        ptb_xl_dir: Path to PTB-XL dataset
        sampling_rate: Sampling rate (100 or 500 Hz)
        target_length: Target sequence length
        max_samples: Maximum samples to load
        verbose: Print summary
        
    Returns:
        signals: ECG signals (N, target_length, 12)
        labels: Binary labels (N,)
        record_ids: Record IDs (N,)
    """
    loader = ECGDataLoader(ptb_xl_dir, sampling_rate, target_length)
    loader.load_metadata()
    signals, labels, record_ids = loader.load(max_samples=max_samples)
    
    if verbose:
        loader.summary()
    
    return signals, labels, record_ids


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
    ptb_xl_dir = Path(__file__).parent.parent.parent.parent / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    
    print(f"\n{'='*80}")
    print("Testing ECG Dataset Loader (Module 6)")
    print(f"{'='*80}\n")
    
    # Load small sample for testing
    loader = ECGDataLoader(ptb_xl_dir, sampling_rate=100, target_length=1000)
    loader.load_metadata()
    
    print(f"Loading first 100 ECG samples for testing...")
    signals, labels, record_ids = loader.load(max_samples=100)
    
    # Show summary
    loader.summary()
    
    # Show sample signal
    print("Sample ECG Signal:")
    print("-" * 80)
    print(f"Record ID:    {record_ids[0]}")
    print(f"Signal shape: {signals[0].shape}")
    print(f"Label:        {labels[0]} ({'Normal' if labels[0] == 0 else 'Abnormal'})")
    print(f"First 5 timesteps (Lead I):")
    print(signals[0][:5, 0])
    
    print(f"\n{'='*80}")
    print("[OK] Module 6: ECG Dataset Loader - Test Complete")
    print(f"{'='*80}\n")
