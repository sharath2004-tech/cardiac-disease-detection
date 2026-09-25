"""
Module 11: PCG Preprocessor
Stage 3 - Data Preprocessing

Preprocesses PCG (heart sound) signals for neural network input
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import logging

logger = logging.getLogger("ACRMF-Net")


class PCGPreprocessor:
    """
    Preprocesses PCG (phonocardiogram) signals
    
    Features:
    - Audio normalization
    - Noise reduction
    - Spectral feature extraction (optional)
    - Signal augmentation (optional)
    """
    
    def __init__(
        self,
        sampling_rate: int = 2000,  # Hz
        apply_filter: bool = True,
        lowcut: float = 25.0,  # Hz (heart sounds typically 25-400Hz)
        highcut: float = 400.0,  # Hz
        normalize: bool = True,
        normalization_method: str = "zscore"  # "zscore", "minmax", "rms"
    ):
        """
        Initialize PCG Preprocessor
        
        Args:
            sampling_rate: PCG sampling rate in Hz
            apply_filter: Apply bandpass filter
            lowcut: Lower cutoff frequency
            highcut: Upper cutoff frequency
            normalize: Normalize signals
            normalization_method: Normalization method
        """
        self.sampling_rate = sampling_rate
        self.apply_filter = apply_filter
        self.lowcut = lowcut
        self.highcut = highcut
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Statistics for normalization
        self.signal_mean = None
        self.signal_std = None
        self.signal_min = None
        self.signal_max = None
        self.signal_rms = None
        self.is_fitted = False
        
        logger.info(f" PCGPreprocessor initialized: {sampling_rate}Hz, "
                   f"filter={apply_filter}, normalize={normalize}")
    
    def fit(self, signals: np.ndarray) -> 'PCGPreprocessor':
        """
        Fit preprocessor on training data
        
        Args:
            signals: PCG signals (N, sequence_length)
            
        Returns:
            self
        """
        logger.info(f"Fitting PCG preprocessor on {len(signals)} signals...")
        
        # Compute statistics
        self.signal_mean = np.mean(signals)
        self.signal_std = np.std(signals)
        self.signal_min = np.min(signals)
        self.signal_max = np.max(signals)
        self.signal_rms = np.sqrt(np.mean(signals ** 2))
        
        self.is_fitted = True
        logger.info(f"  [OK] Preprocessor fitted")
        
        return self
    
    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Transform PCG signals
        
        Args:
            signals: PCG signals (N, sequence_length)
            
        Returns:
            Preprocessed signals (N, sequence_length)
        """
        signals = signals.copy().astype(np.float32)
        
        # Apply bandpass filter
        if self.apply_filter:
            signals = self._apply_bandpass_filter(signals)
        
        # Normalize
        if self.normalize:
            signals = self._normalize(signals)
        
        return signals
    
    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            signals: PCG signals (N, sequence_length)
            
        Returns:
            Preprocessed signals (N, sequence_length)
        """
        signals = signals.copy().astype(np.float32)
        
        # Apply filtering first
        if self.apply_filter:
            signals = self._apply_bandpass_filter(signals)
        
        # Fit normalization statistics
        self.fit(signals)
        
        # Normalize
        if self.normalize:
            signals = self._normalize(signals)
        
        return signals
    
    def _apply_bandpass_filter(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to remove noise outside heart sound range
        
        Args:
            signals: PCG signals (N, sequence_length)
            
        Returns:
            Filtered signals (N, sequence_length)
        """
        # Design Butterworth bandpass filter
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Clamp to valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter to each signal
        filtered = np.zeros_like(signals)
        for i in range(len(signals)):
            filtered[i] = signal.filtfilt(b, a, signals[i])
        
        return filtered
    
    def _normalize(self, signals: np.ndarray) -> np.ndarray:
        """
        Normalize signals
        
        Args:
            signals: PCG signals (N, sequence_length)
            
        Returns:
            Normalized signals (N, sequence_length)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        if self.normalization_method == "zscore":
            # Z-score normalization
            signals = (signals - self.signal_mean) / (self.signal_std + 1e-8)
        
        elif self.normalization_method == "minmax":
            # Min-max normalization to [-1, 1]
            signals = 2 * (signals - self.signal_min) / (self.signal_max - self.signal_min + 1e-8) - 1
        
        elif self.normalization_method == "rms":
            # RMS normalization
            signals = signals / (self.signal_rms + 1e-8)
        
        return signals
    
    def augment(self, signals: np.ndarray, augmentation_prob: float = 0.5) -> np.ndarray:
        """
        Apply data augmentation (optional, for training)
        
        Args:
            signals: PCG signals (N, sequence_length)
            augmentation_prob: Probability of applying augmentation
            
        Returns:
            Augmented signals (N, sequence_length)
        """
        augmented = signals.copy()
        
        for i in range(len(signals)):
            if np.random.rand() < augmentation_prob:
                # Random amplitude scaling (0.9 to 1.1)
                if np.random.rand() < 0.5:
                    scale = np.random.uniform(0.9, 1.1)
                    augmented[i] *= scale
                
                # Random noise addition
                if np.random.rand() < 0.3:
                    noise = np.random.normal(0, 0.01, augmented[i].shape)
                    augmented[i] += noise
                
                # Time shifting (circular shift)
                if np.random.rand() < 0.3:
                    shift = np.random.randint(-50, 50)
                    augmented[i] = np.roll(augmented[i], shift)
        
        return augmented
    
    def extract_spectral_features(
        self,
        signals: np.ndarray,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Extract mel-spectrogram features (optional, for CNN-based models)
        
        Args:
            signals: PCG signals (N, sequence_length)
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
            
        Returns:
            Mel spectrograms (N, n_mels, time_steps)
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not installed. Spectral features not available.")
            return signals
        
        spectrograms = []
        
        for i in range(len(signals)):
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=signals[i],
                sr=self.sampling_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            spectrograms.append(mel_spec_db)
        
        return np.array(spectrograms, dtype=np.float32)


def preprocess_pcg_data(
    train_signals: np.ndarray,
    val_signals: Optional[np.ndarray] = None,
    test_signals: Optional[np.ndarray] = None,
    sampling_rate: int = 2000,
    apply_filter: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, ...]:
    """
    Convenience function to preprocess PCG data
    
    Args:
        train_signals: Training PCG signals
        val_signals: Validation PCG signals (optional)
        test_signals: Test PCG signals (optional)
        sampling_rate: Sampling rate
        apply_filter: Apply bandpass filter
        normalize: Normalize signals
        
    Returns:
        Tuple of preprocessed signals
    """
    preprocessor = PCGPreprocessor(
        sampling_rate=sampling_rate,
        apply_filter=apply_filter,
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
    
    logger.info("[OK] PCG data preprocessing complete")
    
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
    print("Testing PCG Preprocessor (Module 11)")
    print(f"{'='*80}\n")
    
    # Create synthetic PCG data (heart sound simulation)
    np.random.seed(42)
    n_samples = 100
    sequence_length = 2000
    
    # Simulate heart sounds with noise
    t = np.linspace(0, 1, sequence_length)  # 1 second
    train_signals = np.zeros((n_samples, sequence_length))
    
    for i in range(n_samples):
        # Simulate S1 and S2 heart sounds
        s1 = 0.5 * np.exp(-((t - 0.2) ** 2) / 0.001)  # S1 sound
        s2 = 0.3 * np.exp(-((t - 0.6) ** 2) / 0.001)  # S2 sound
        
        # Add harmonics
        harmonics = 0.1 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 200 * t)
        
        # Add noise
        noise = 0.1 * np.random.randn(sequence_length)
        
        train_signals[i] = s1 + s2 + harmonics + noise
    
    val_signals = train_signals[:20].copy()
    
    print("Original data:")
    print(f"  Train shape: {train_signals.shape}")
    print(f"  Val shape:   {val_signals.shape}")
    print(f"  Train mean:  {np.mean(train_signals):.4f}")
    print(f"  Train std:   {np.std(train_signals):.4f}")
    print(f"  Train range: [{np.min(train_signals):.4f}, {np.max(train_signals):.4f}]")
    print()
    
    # Test filtering
    print("Test 1: Bandpass Filtering")
    print("-" * 80)
    preprocessor = PCGPreprocessor(
        sampling_rate=2000,
        apply_filter=True,
        normalize=False
    )
    train_filtered = preprocessor.fit_transform(train_signals)
    
    print(f"  Filtered mean: {np.mean(train_filtered):.4f}")
    print(f"  Filtered std:  {np.std(train_filtered):.4f}")
    print()
    
    # Test complete preprocessing
    print("Test 2: Complete Preprocessing with Normalization")
    print("-" * 80)
    train_processed, val_processed = preprocess_pcg_data(
        train_signals, val_signals,
        sampling_rate=2000,
        apply_filter=True,
        normalize=True
    )
    
    print(f"  Normalized train mean: {np.mean(train_processed):.4f}")
    print(f"  Normalized train std:  {np.std(train_processed):.4f}")
    print(f"  Normalized val mean:   {np.mean(val_processed):.4f}")
    print()
    
    # Test augmentation
    print("Test 3: Data Augmentation")
    print("-" * 80)
    train_augmented = preprocessor.augment(train_processed, augmentation_prob=1.0)
    
    print(f"  Original mean:   {np.mean(train_processed):.4f}")
    print(f"  Augmented mean:  {np.mean(train_augmented):.4f}")
    print(f"  Difference:      {np.mean(np.abs(train_processed - train_augmented)):.4f}")
    print()
    
    # Test RMS normalization
    print("Test 4: RMS Normalization")
    print("-" * 80)
    preprocessor_rms = PCGPreprocessor(
        sampling_rate=2000,
        normalize=True,
        normalization_method="rms"
    )
    train_rms = preprocessor_rms.fit_transform(train_signals)
    
    print(f"  RMS normalized mean: {np.mean(train_rms):.4f}")
    print(f"  RMS normalized std:  {np.std(train_rms):.4f}")
    
    print(f"\n{'='*80}")
    print("[OK] Module 11: PCG Preprocessor - Test Complete")
    print(f"{'='*80}\n")
