"""
Module 12: Data Quality Assessment Module
Stage 3 - Data Preprocessing

Assesses quality of clinical, ECG, and PCG data
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("ACRMF-Net")


class DataQualityAssessor:
    """
    Assesses data quality for multimodal cardiac data
    
    Provides quality scores for:
    - Clinical features (completeness, validity)
    - ECG signals (SNR, artifacts)
    - PCG signals (SNR, quality)
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.5  # Minimum acceptable quality score (0-1)
    ):
        """
        Initialize Quality Assessor
        
        Args:
            min_quality_threshold: Minimum quality threshold for acceptance
        """
        self.min_quality_threshold = min_quality_threshold
        
        logger.info(f"[Search] DataQualityAssessor initialized: threshold={min_quality_threshold}")
    
    def assess_clinical_quality(self, features: np.ndarray) -> np.ndarray:
        """
        Assess clinical feature quality
        
        Args:
            features: Clinical features (N, num_features)
            
        Returns:
            Quality scores (N,) in range [0, 1]
        """
        n_samples, n_features = features.shape
        quality_scores = np.ones(n_samples)
        
        for i in range(n_samples):
            sample = features[i]
            
            # Check for missing values
            missing_ratio = np.sum(np.isnan(sample)) / n_features
            
            # Check for outliers (values beyond 5 std devs)
            mean = np.nanmean(sample)
            std = np.nanstd(sample)
            outliers = np.sum(np.abs(sample - mean) > 5 * std) / n_features
            
            # Compute quality score
            completeness_score = 1.0 - missing_ratio
            validity_score = 1.0 - outliers
            
            quality_scores[i] = 0.7 * completeness_score + 0.3 * validity_score
        
        return quality_scores
    
    def assess_ecg_quality(self, signals: np.ndarray) -> np.ndarray:
        """
        Assess ECG signal quality
        
        Args:
            signals: ECG signals (N, sequence_length, 12)
            
        Returns:
            Quality scores (N,) in range [0, 1]
        """
        n_samples = signals.shape[0]
        quality_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            signal = signals[i]  # (sequence_length, 12)
            
            # Compute quality metrics for each lead
            lead_qualities = []
            
            for lead in range(signal.shape[1]):
                lead_signal = signal[:, lead]
                
                # 1. Signal-to-Noise Ratio (SNR)
                snr = self._compute_snr(lead_signal)
                snr_score = min(snr / 20.0, 1.0)  # Normalize to 0-1 (20dB = good)
                
                # 2. Check for flat line (no signal)
                variance = np.var(lead_signal)
                flat_score = 1.0 if variance > 1e-6 else 0.0
                
                # 3. Check for clipping
                clip_ratio = (np.sum(np.abs(lead_signal) > 0.95 * np.max(np.abs(lead_signal)))) / len(lead_signal)
                clip_score = 1.0 - min(clip_ratio / 0.1, 1.0)  # Penalize if >10% clipped
                
                # Combined score for this lead
                lead_quality = 0.5 * snr_score + 0.3 * flat_score + 0.2 * clip_score
                lead_qualities.append(lead_quality)
            
            # Average quality across leads
            quality_scores[i] = np.mean(lead_qualities)
        
        return quality_scores
    
    def assess_pcg_quality(self, signals: np.ndarray) -> np.ndarray:
        """
        Assess PCG signal quality
        
        Args:
            signals: PCG signals (N, sequence_length)
            
        Returns:
            Quality scores (N,) in range [0, 1]
        """
        n_samples = signals.shape[0]
        quality_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            pcg_signal = signals[i]
            
            # 1. Signal-to-Noise Ratio
            snr = self._compute_snr(pcg_signal)
            snr_score = min(snr / 15.0, 1.0)  # Normalize to 0-1 (15dB = acceptable for PCG)
            
            # 2. Check for silence (no signal)
            rms = np.sqrt(np.mean(pcg_signal ** 2))
            silence_score = 1.0 if rms > 0.01 else 0.0
            
            # 3. Check for saturation/clipping
            max_val = np.max(np.abs(pcg_signal))
            saturation_ratio = np.sum(np.abs(pcg_signal) > 0.95 * max_val) / len(pcg_signal)
            saturation_score = 1.0 - min(saturation_ratio / 0.05, 1.0)
            
            # 4. Spectral energy concentration (heart sounds should be in 25-400 Hz)
            spectral_score = self._compute_spectral_quality(pcg_signal)
            
            # Combined quality score
            quality_scores[i] = (
                0.4 * snr_score +
                0.2 * silence_score +
                0.2 * saturation_score +
                0.2 * spectral_score
            )
        
        return quality_scores
    
    def _compute_snr(self, signal: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio in dB
        
        Args:
            signal: 1D signal
            
        Returns:
            SNR in dB
        """
        # Simple SNR estimation: ratio of signal power to noise power
        # Assume noise is high-frequency component
        
        # Signal power
        signal_power = np.var(signal)
        
        # Estimate noise power from high-frequency components
        # Use difference between adjacent samples as noise estimate
        noise_estimate = np.diff(signal)
        noise_power = np.var(noise_estimate)
        
        # Avoid division by zero
        if noise_power < 1e-10:
            return 50.0  # Very high SNR
        
        snr = 10 * np.log10(signal_power / noise_power)
        return max(snr, 0.0)  # Clip at 0
    
    def _compute_spectral_quality(self, signal: np.ndarray, target_range: Tuple[float, float] = (0.1, 0.5)) -> float:
        """
        Compute spectral quality score
        
        Args:
            signal: 1D signal
            target_range: Target frequency range (normalized by Nyquist)
            
        Returns:
            Quality score [0, 1]
        """
        # Compute FFT
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft) ** 2
        
        # Normalize frequencies
        n = len(signal)
        freqs = np.fft.fftfreq(n)
        
        # Compute energy in target range
        target_mask = (np.abs(freqs) >= target_range[0]) & (np.abs(freqs) <= target_range[1])
        target_energy = np.sum(power_spectrum[target_mask])
        total_energy = np.sum(power_spectrum)
        
        # Ratio of energy in target range
        if total_energy < 1e-10:
            return 0.0
        
        energy_ratio = target_energy / total_energy
        return min(energy_ratio / 0.7, 1.0)  # Normalize (70% in range = perfect)
    
    def assess_multimodal_quality(
        self,
        clinical_features: np.ndarray,
        ecg_signals: np.ndarray,
        pcg_signals: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Assess quality for all modalities
        
        Args:
            clinical_features: Clinical features (N, num_features)
            ecg_signals: ECG signals (N, seq_len, 12)
            pcg_signals: PCG signals (N, seq_len)
            
        Returns:
            Dictionary with quality scores for each modality
        """
        logger.info(f"Assessing data quality for {len(clinical_features)} samples...")
        
        # Assess each modality
        clinical_quality = self.assess_clinical_quality(clinical_features)
        ecg_quality = self.assess_ecg_quality(ecg_signals)
        pcg_quality = self.assess_pcg_quality(pcg_signals)
        
        # Overall quality (average of modalities)
        overall_quality = (clinical_quality + ecg_quality + pcg_quality) / 3.0
        
        # Log statistics
        logger.info(f"  Clinical quality: mean={np.mean(clinical_quality):.3f}, "
                   f"min={np.min(clinical_quality):.3f}")
        logger.info(f"  ECG quality:      mean={np.mean(ecg_quality):.3f}, "
                   f"min={np.min(ecg_quality):.3f}")
        logger.info(f"  PCG quality:      mean={np.mean(pcg_quality):.3f}, "
                   f"min={np.min(pcg_quality):.3f}")
        logger.info(f"  Overall quality:  mean={np.mean(overall_quality):.3f}, "
                   f"min={np.min(overall_quality):.3f}")
        
        # Count low-quality samples
        low_quality = np.sum(overall_quality < self.min_quality_threshold)
        if low_quality > 0:
            logger.warning(f"  {low_quality} samples below quality threshold ({self.min_quality_threshold})")
        
        return {
            'clinical': clinical_quality,
            'ecg': ecg_quality,
            'pcg': pcg_quality,
            'overall': overall_quality
        }
    
    def filter_by_quality(
        self,
        data: Dict[str, np.ndarray],
        quality_scores: np.ndarray,
        min_quality: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Filter data by quality threshold
        
        Args:
            data: Dictionary of data arrays
            quality_scores: Quality scores (N,)
            min_quality: Minimum quality (if None, use self.min_quality_threshold)
            
        Returns:
            Filtered data dictionary
        """
        if min_quality is None:
            min_quality = self.min_quality_threshold
        
        # Find high-quality samples
        high_quality_mask = quality_scores >= min_quality
        num_kept = np.sum(high_quality_mask)
        num_total = len(quality_scores)
        
        logger.info(f"Filtering by quality: kept {num_kept}/{num_total} samples "
                   f"({num_kept/num_total*100:.1f}%)")
        
        # Filter all data
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = value[high_quality_mask]
        
        return filtered_data


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
    print("Testing Data Quality Assessment (Module 12)")
    print(f"{'='*80}\n")
    
    # Create synthetic data with varying quality
    np.random.seed(42)
    n_samples = 100
    
    # Clinical features (some with missing values)
    clinical_features = np.random.randn(n_samples, 13)
    clinical_features[0:10, 0:3] = np.nan  # Add missing values to first 10 samples
    
    # ECG signals (some with low SNR)
    ecg_signals = np.random.randn(n_samples, 1000, 12) * 0.2
    for i in range(n_samples):
        # Add real signal component
        t = np.linspace(0, 10, 1000)
        qrs = np.sin(2 * np.pi * 1.2 * t)
        ecg_signals[i] += qrs[:, np.newaxis] * 0.5
    
    # Make some signals very noisy
    ecg_signals[20:30] = np.random.randn(10, 1000, 12) * 2.0
    
    # PCG signals (some silent)
    pcg_signals = np.random.randn(n_samples, 2000) * 0.1
    for i in range(n_samples):
        # Add heart sound components
        t = np.linspace(0, 1, 2000)
        s1 = 0.5 * np.exp(-((t - 0.2) ** 2) / 0.001)
        s2 = 0.3 * np.exp(-((t - 0.6) ** 2) / 0.001)
        pcg_signals[i] += s1 + s2
    
    # Make some signals silent
    pcg_signals[40:50] = np.random.randn(10, 2000) * 0.001
    
    print("Test Data:")
    print(f"  Clinical: {clinical_features.shape}")
    print(f"  ECG:      {ecg_signals.shape}")
    print(f"  PCG:      {pcg_signals.shape}")
    print()
    
    # Test quality assessment
    print("Test 1: Individual Modality Quality Assessment")
    print("-" * 80)
    assessor = DataQualityAssessor(min_quality_threshold=0.5)
    
    clinical_quality = assessor.assess_clinical_quality(clinical_features)
    ecg_quality = assessor.assess_ecg_quality(ecg_signals)
    pcg_quality = assessor.assess_pcg_quality(pcg_signals)
    
    print(f"  Clinical quality: mean={np.mean(clinical_quality):.3f}, std={np.std(clinical_quality):.3f}")
    print(f"  ECG quality:      mean={np.mean(ecg_quality):.3f}, std={np.std(ecg_quality):.3f}")
    print(f"  PCG quality:      mean={np.mean(pcg_quality):.3f}, std={np.std(pcg_quality):.3f}")
    print()
    
    # Test multimodal assessment
    print("Test 2: Multimodal Quality Assessment")
    print("-" * 80)
    quality_scores = assessor.assess_multimodal_quality(
        clinical_features, ecg_signals, pcg_signals
    )
    print()
    
    # Test filtering
    print("Test 3: Quality-Based Filtering")
    print("-" * 80)
    data = {
        'clinical': clinical_features,
        'ecg': ecg_signals,
        'pcg': pcg_signals,
        'labels': np.random.randint(0, 2, n_samples)
    }
    
    filtered_data = assessor.filter_by_quality(
        data,
        quality_scores['overall'],
        min_quality=0.5
    )
    
    print(f"  Filtered clinical: {filtered_data['clinical'].shape}")
    print(f"  Filtered ECG:      {filtered_data['ecg'].shape}")
    print(f"  Filtered PCG:      {filtered_data['pcg'].shape}")
    
    print(f"\n{'='*80}")
    print("[OK] Module 12: Data Quality Assessment - Test Complete")
    print(f"{'='*80}\n")
