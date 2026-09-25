"""
Module 10: ECG Preprocessor
Stage 3 - Data Preprocessing

Preprocesses ECG signals for neural network input
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from typing import Tuple, Optional
import logging

logger = logging.getLogger("ACRMF-Net")


class ECGPreprocessor:
    """
    Preprocesses ECG signals
    
    Features:
    - Baseline wander removal
    - Noise filtering (high-pass, low-pass, bandpass)
    - Normalization
    - Optional augmentation
    """
    
    def __init__(
        self,
        sampling_rate: int = 100,  # Hz
        remove_baseline: bool = True,
        apply_bandpass: bool = True,
        lowcut: float = 0.5,  # Hz
        highcut: float = 50.0,  # Hz
        normalize: bool = True,
        normalization_method: str = "zscore"  # "zscore", "minmax", "lead_wise"
    ):
        """
        Initialize ECG Preprocessor
        
        Args:
            sampling_rate: ECG sampling rate in Hz
            remove_baseline: Remove baseline wander
            apply_bandpass: Apply bandpass filter
            lowcut: Lower cutoff frequency for bandpass
            highcut: Upper cutoff frequency for bandpass
            normalize: Normalize signals
            normalization_method: Normalization method
        """
        self.sampling_rate = sampling_rate
        self.remove_baseline = remove_baseline
        self.apply_bandpass = apply_bandpass
        self.lowcut = lowcut
        self.highcut = highcut
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Statistics for normalization (computed from training data)
        self.signal_mean = None
        self.signal_std = None
        self.signal_min = None
        self.signal_max = None
        self.is_fitted = False
        
        logger.info(f"[Fast] ECGPreprocessor initialized: {sampling_rate}Hz, "
                   f"bandpass={apply_bandpass}, normalize={normalize}")
    
    def fit(self, signals: np.ndarray) -> 'ECGPreprocessor':
        """
        Fit preprocessor on training data (compute normalization statistics)
        
        Args:
            signals: ECG signals (N, sequence_length, 12)
            
        Returns:
            self
        """
        logger.info(f"Fitting ECG preprocessor on {len(signals)} signals...")
        
        # Compute statistics
        if self.normalization_method == "lead_wise":
            # Per-lead statistics
            self.signal_mean = np.mean(signals, axis=(0, 1), keepdims=True)  # (1, 1, 12)
            self.signal_std = np.std(signals, axis=(0, 1), keepdims=True)
            self.signal_min = np.min(signals, axis=(0, 1), keepdims=True)
            self.signal_max = np.max(signals, axis=(0, 1), keepdims=True)
        else:
            # Global statistics
            self.signal_mean = np.mean(signals)
            self.signal_std = np.std(signals)
            self.signal_min = np.min(signals)
            self.signal_max = np.max(signals)
        
        self.is_fitted = True
        logger.info(f"  [OK] Preprocessor fitted")
        
        return self
    
    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Transform ECG signals
        
        Args:
            signals: ECG signals (N, sequence_length, 12)
            
        Returns:
            Preprocessed signals (N, sequence_length, 12)
        """
        signals = signals.copy().astype(np.float32)
        
        # Process each sample
        processed = []
        for i in range(len(signals)):
            signal_processed = signals[i]  # (sequence_length, 12)
            
            # Remove baseline wander
            if self.remove_baseline:
                signal_processed = self._remove_baseline_wander(signal_processed)
            
            # Apply bandpass filter
            if self.apply_bandpass:
                signal_processed = self._apply_bandpass_filter(signal_processed)
            
            processed.append(signal_processed)
        
        signals = np.array(processed, dtype=np.float32)
        
        # Normalize
        if self.normalize:
            signals = self._normalize(signals)
        
        return signals
    
    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            signals: ECG signals (N, sequence_length, 12)
            
        Returns:
            Preprocessed signals (N, sequence_length, 12)
        """
        # First apply filtering (doesn't need fitting)
        processed = []
        for i in range(len(signals)):
            signal_processed = signals[i].copy().astype(np.float32)
            
            if self.remove_baseline:
                signal_processed = self._remove_baseline_wander(signal_processed)
            
            if self.apply_bandpass:
                signal_processed = self._apply_bandpass_filter(signal_processed)
            
            processed.append(signal_processed)
        
        signals = np.array(processed, dtype=np.float32)
        
        # Fit normalization statistics
        self.fit(signals)
        
        # Normalize
        if self.normalize:
            signals = self._normalize(signals)
        
        return signals
    
    def _remove_baseline_wander(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Remove baseline wander using median filter
        
        Args:
            ecg_signal: Single ECG signal (sequence_length, 12)
            
        Returns:
            Detrended signal (sequence_length, 12)
        """
        # Use median filter for baseline estimation
        window_size = int(0.2 * self.sampling_rate)  # 200ms window
        if window_size % 2 == 0:
            window_size += 1
        
        detrended = np.zeros_like(ecg_signal)
        for lead in range(ecg_signal.shape[1]):
            baseline = median_filter(ecg_signal[:, lead], size=window_size)
            detrended[:, lead] = ecg_signal[:, lead] - baseline
        
        return detrended
    
    def _apply_bandpass_filter(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter
        
        Args:
            ecg_signal: Single ECG signal (sequence_length, 12)
            
        Returns:
            Filtered signal (sequence_length, 12)
        """
        # Design Butterworth bandpass filter
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Clamp to valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter to each lead
        filtered = np.zeros_like(ecg_signal)
        for lead in range(ecg_signal.shape[1]):
            # Use filtfilt for zero-phase filtering
            filtered[:, lead] = signal.filtfilt(b, a, ecg_signal[:, lead])
        
        return filtered
    
    def _normalize(self, signals: np.ndarray) -> np.ndarray:
        """
        Normalize signals
        
        Args:
            signals: ECG signals (N, sequence_length, 12)
            
        Returns:
            Normalized signals (N, sequence_length, 12)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        if self.normalization_method == "zscore":
            # Z-score normalization
            signals = (signals - self.signal_mean) / (self.signal_std + 1e-8)
        
        elif self.normalization_method == "minmax":
            # Min-max normalization to [-1, 1]
            signals = 2 * (signals - self.signal_min) / (self.signal_max - self.signal_min + 1e-8) - 1
        
        elif self.normalization_method == "lead_wise":
            # Per-lead z-score normalization
            signals = (signals - self.signal_mean) / (self.signal_std + 1e-8)
        
        return signals
    
    def augment(self, signals: np.ndarray, augmentation_prob: float = 0.5) -> np.ndarray:
        """
        Apply data augmentation (optional, for training)
        
        Args:
            signals: ECG signals (N, sequence_length, 12)
            augmentation_prob: Probability of applying augmentation
            
        Returns:
            Augmented signals (N, sequence_length, 12)
        """
        augmented = signals.copy()
        
        for i in range(len(signals)):
            if np.random.rand() < augmentation_prob:
                # Random amplitude scaling (0.9 to 1.1)
                scale = np.random.uniform(0.9, 1.1)
                augmented[i] *= scale
                
                # Random noise addition
                if np.random.rand() < 0.3:
                    noise = np.random.normal(0, 0.01, augmented[i].shape)
                    augmented[i] += noise
        
        return augmented


def preprocess_ecg_data(
    train_signals: np.ndarray,
    val_signals: Optional[np.ndarray] = None,
    test_signals: Optional[np.ndarray] = None,
    sampling_rate: int = 100,
    apply_bandpass: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, ...]:
    """
    Convenience function to preprocess ECG data
    
    Args:
        train_signals: Training ECG signals
        val_signals: Validation ECG signals (optional)
        test_signals: Test ECG signals (optional)
        sampling_rate: Sampling rate
        apply_bandpass: Apply bandpass filter
        normalize: Normalize signals
        
    Returns:
        Tuple of preprocessed signals
    """
    preprocessor = ECGPreprocessor(
        sampling_rate=sampling_rate,
        apply_bandpass=apply_bandpass,
        normalize=normalize
    )
    
    # Fit and transform training data
    train_preprocessed = preprocessor.fit_transform(train_signals)
    
    results = [train_preprocessed]
    
    # Transform validation data
    if val_signals is not None:
        val_preprocessed = preprocessor.transform(val_signals)
        results.append(val_preprocessed)
    
    # Transform test data
    if test_signals is not None:
        test_preprocessed = preprocessor.transform(test_signals)
        results.append(test_preprocessed)
    
    logger.info("[OK] ECG data preprocessing complete")
    
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
    print("Testing ECG Preprocessor (Module 10)")
    print(f"{'='*80}\n")
    
    # Create synthetic ECG data
    np.random.seed(42)
    n_samples = 100
    sequence_length = 1000
    n_leads = 12
    
    # Simulate ECG with baseline wander and noise
    t = np.linspace(0, 10, sequence_length)
    train_signals = np.zeros((n_samples, sequence_length, n_leads))
    
    for i in range(n_samples):
        for lead in range(n_leads):
            # Simulated ECG components
            qrs = 0.5 * np.sin(2 * np.pi * 1.2 * t)  # QRS complex
            baseline = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # Baseline wander
            noise = 0.05 * np.random.randn(sequence_length)  # Noise
            
            train_signals[i, :, lead] = qrs + baseline + noise
    
    val_signals = train_signals[:20].copy()
    
    print("Original data:")
    print(f"  Train shape: {train_signals.shape}")
    print(f"  Val shape:   {val_signals.shape}")
    print(f"  Train mean:  {np.mean(train_signals):.4f}")
    print(f"  Train std:   {np.std(train_signals):.4f}")
    print()
    
    # Test baseline removal and filtering
    print("Test 1: Baseline Removal + Bandpass Filter")
    print("-" * 80)
    preprocessor = ECGPreprocessor(
        sampling_rate=100,
        remove_baseline=True,
        apply_bandpass=True,
        normalize=False
    )
    train_processed = preprocessor.fit_transform(train_signals)
    
    print(f"  Processed mean: {np.mean(train_processed):.4f}")
    print(f"  Processed std:  {np.std(train_processed):.4f}")
    print()
    
    # Test normalization
    print("Test 2: Complete Preprocessing with Normalization")
    print("-" * 80)
    train_processed, val_processed = preprocess_ecg_data(
        train_signals, val_signals,
        sampling_rate=100,
        apply_bandpass=True,
        normalize=True
    )
    
    print(f"  Normalized train mean: {np.mean(train_processed):.4f}")
    print(f"  Normalized train std:  {np.std(train_processed):.4f}")
    print(f"  Normalized val mean:   {np.mean(val_processed):.4f}")
    print()
    
    # Test augmentation
    print("Test 3: Data Augmentation")
    print("-" * 80)
    preprocessor = ECGPreprocessor(sampling_rate=100)
    train_processed = preprocessor.fit_transform(train_signals)
    train_augmented = preprocessor.augment(train_processed, augmentation_prob=1.0)
    
    print(f"  Original mean:   {np.mean(train_processed):.4f}")
    print(f"  Augmented mean:  {np.mean(train_augmented):.4f}")
    print(f"  Difference:      {np.mean(np.abs(train_processed - train_augmented)):.4f}")
    
    print(f"\n{'='*80}")
    print("[OK] Module 10: ECG Preprocessor - Test Complete")
    print(f"{'='*80}\n")
