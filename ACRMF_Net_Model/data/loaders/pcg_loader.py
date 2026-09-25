"""
Module 7: PCG Dataset Loader
Stage 2 - Dataset Preparation

Loads CinC Challenge 2016 PCG (Phonocardiogram) dataset - Heart sounds
"""

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger("ACRMF-Net")


class PCGDataLoader:
    """
    Loads CinC Challenge 2016 PCG (heart sound) dataset
    
    Dataset Info:
    - Heart sound recordings from 6 training sets (a-f)
    - Multiple recording devices and locations
    - Binary labels: -1 (normal), 1 (abnormal)
    
    Audio format: WAV files sampled at various rates (resampled to target rate)
    """
    
    def __init__(
        self,
        archive_dir: Path,
        target_sampling_rate: int = 2000,  # Hz
        target_length: int = 2000,  # Number of samples (1 second @ 2000Hz)
        training_sets: List[str] = None  # ['a', 'b', 'c', 'd', 'e', 'f']
    ):
        """
        Initialize PCG Data Loader
        
        Args:
            archive_dir: Path to archive/ directory containing training-a, training-b, etc.
            target_sampling_rate: Target sampling rate for audio
            target_length: Target sequence length
            training_sets: Which training sets to load (default: all)
        """
        self.archive_dir = Path(archive_dir)
        self.target_sampling_rate = target_sampling_rate
        self.target_length = target_length
        
        if training_sets is None:
            training_sets = ['a', 'b', 'c', 'd', 'e', 'f']
        self.training_sets = training_sets
        
        # Data storage
        self.pcg_signals = []
        self.labels = []
        self.record_ids = []
        self.metadata = []
        
        logger.info(f" PCGDataLoader initialized: {target_sampling_rate}Hz, length={target_length}")
        logger.info(f"  Training sets: {training_sets}")
    
    def load(
        self,
        max_samples_per_set: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load PCG signals and labels from all training sets
        
        Args:
            max_samples_per_set: Maximum samples to load per training set
            
        Returns:
            signals: PCG signals array (N, target_length)
            labels: Binary labels (N,) - 0: normal, 1: abnormal
            record_ids: Record identifiers (N,)
        """
        logger.info(f"Loading PCG signals from {len(self.training_sets)} training sets...")
        
        for training_set in self.training_sets:
            logger.info(f"  Loading training set '{training_set}'...")
            self._load_training_set(training_set, max_samples_per_set)
        
        # Convert to numpy arrays
        signals = np.array(self.pcg_signals, dtype=np.float32)  # (N, target_length)
        labels = np.array(self.labels, dtype=np.int64)          # (N,)
        record_ids = self.record_ids                             # List of strings
        
        logger.info(f"[OK] Loaded {len(signals)} PCG signals")
        logger.info(f"  Signals shape: {signals.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
        logger.info(f"  Class distribution: Normal={np.sum(labels == 0)}, Abnormal={np.sum(labels == 1)}")
        
        return signals, labels, record_ids
    
    def _load_training_set(
        self,
        set_name: str,
        max_samples: Optional[int] = None
    ):
        """Load data from a single training set"""
        set_dir = self.archive_dir / f"training-{set_name}"
        
        if not set_dir.exists():
            logger.warning(f"  Training set '{set_name}' not found at {set_dir}")
            return
        
        # Find reference CSV file
        reference_file = set_dir / f"REFERENCE.csv"
        if not reference_file.exists():
            # Try alternative naming
            reference_file = set_dir / f"REFERENCE_{set_name.upper()}.csv"
        
        if not reference_file.exists():
            logger.warning(f"  Reference file not found for set '{set_name}'")
            # Try to load all WAV files without labels
            wav_files = list(set_dir.glob("*.wav"))
            logger.info(f"    Found {len(wav_files)} WAV files (no labels)")
            return
        
        # Load reference CSV
        try:
            reference = pd.read_csv(reference_file, header=None, names=['filename', 'label'])
        except:
            # Try with different delimiter
            reference = pd.read_csv(reference_file, header=None, sep=r'\s+', names=['filename', 'label'])
        
        if max_samples:
            reference = reference.head(max_samples)
        
        logger.info(f"    Loading {len(reference)} files from set '{set_name}'...")
        
        num_failed = 0
        
        for idx, row in reference.iterrows():
            try:
                filename = row['filename']
                label = row['label']
                
                # Construct full path
                wav_path = set_dir / f"{filename}.wav"
                
                # Load audio signal
                signal = self._load_audio(wav_path)
                
                if signal is not None:
                    # Convert label: -1 (normal) -> 0, 1 (abnormal) -> 1
                    binary_label = 0 if label == -1 else 1
                    
                    self.pcg_signals.append(signal)
                    self.labels.append(binary_label)
                    self.record_ids.append(f"{set_name}_{filename}")
                    self.metadata.append({
                        'set': set_name,
                        'filename': filename,
                        'original_label': label
                    })
                    
            except Exception as e:
                num_failed += 1
                if num_failed <= 3:  # Show first 3 errors per set
                    logger.warning(f"    Failed to load {filename}: {e}")
        
        if num_failed > 0:
            logger.warning(f"    Failed: {num_failed} files from set '{set_name}'")
        
        logger.info(f"    [OK] Loaded {len(reference) - num_failed} files from set '{set_name}'")
    
    def _load_audio(self, wav_path: Path) -> Optional[np.ndarray]:
        """Load and preprocess audio signal"""
        if not wav_path.exists():
            return None
        
        # Read WAV file
        sample_rate, audio = wavfile.read(wav_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        
        # Resample if needed
        if sample_rate != self.target_sampling_rate:
            audio = self._resample_audio(audio, sample_rate, self.target_sampling_rate)
        
        # Trim or pad to target length
        if len(audio) > self.target_length:
            # Take center segment
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        elif len(audio) < self.target_length:
            # Pad with zeros
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        original_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """Resample audio to target sampling rate"""
        from scipy.signal import resample
        
        num_samples = int(len(audio) * target_rate / original_rate)
        resampled = resample(audio, num_samples)
        
        return resampled
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if len(self.pcg_signals) == 0:
            raise ValueError("Data not loaded. Call load() first.")
        
        signals = np.array(self.pcg_signals)
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
            'signal_max': np.max(signals),
            'training_sets': list(set([m['set'] for m in self.metadata]))
        }
        
        return stats
    
    def summary(self):
        """Print dataset summary"""
        if len(self.pcg_signals) == 0:
            logger.warning("No data loaded yet. Call load() first.")
            return
        
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("PCG Dataset Summary (CinC Challenge 2016)")
        print("=" * 80)
        print(f"Total Samples:    {stats['num_samples']}")
        print(f"Signal Shape:     {stats['signal_shape']}")
        print(f"Sampling Rate:    {self.target_sampling_rate} Hz")
        print(f"Sequence Length:  {self.target_length}")
        print(f"Training Sets:    {', '.join(stats['training_sets'])}")
        print(f"\nClass Distribution:")
        print(f"  Normal:         {stats['num_normal']} ({(1-stats['class_ratio'])*100:.1f}%)")
        print(f"  Abnormal:       {stats['num_abnormal']} ({stats['class_ratio']*100:.1f}%)")
        print(f"\nSignal Statistics:")
        print(f"  Mean:           {stats['signal_mean']:.4f}")
        print(f"  Std:            {stats['signal_std']:.4f}")
        print(f"  Range:          [{stats['signal_min']:.4f}, {stats['signal_max']:.4f}]")
        print("=" * 80 + "\n")


def load_pcg_data(
    archive_dir: Path,
    target_sampling_rate: int = 2000,
    target_length: int = 2000,
    training_sets: List[str] = None,
    max_samples_per_set: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to load PCG data
    
    Args:
        archive_dir: Path to archive/ directory
        target_sampling_rate: Target sampling rate
        target_length: Target sequence length
        training_sets: Which sets to load
        max_samples_per_set: Max samples per set
        verbose: Print summary
        
    Returns:
        signals: PCG signals (N, target_length)
        labels: Binary labels (N,)
        record_ids: Record IDs (List of strings)
    """
    loader = PCGDataLoader(
        archive_dir,
        target_sampling_rate,
        target_length,
        training_sets
    )
    signals, labels, record_ids = loader.load(max_samples_per_set=max_samples_per_set)
    
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
    archive_dir = Path(__file__).parent.parent.parent.parent / "archive"
    
    print(f"\n{'='*80}")
    print("Testing PCG Dataset Loader (Module 7)")
    print(f"{'='*80}\n")
    
    # Load small sample for testing (first 50 from each set)
    loader = PCGDataLoader(
        archive_dir,
        target_sampling_rate=2000,
        target_length=2000,
        training_sets=['a', 'b']  # Load just sets 'a' and 'b' for testing
    )
    
    print(f"Loading first 50 PCG samples per set for testing...")
    signals, labels, record_ids = loader.load(max_samples_per_set=50)
    
    # Show summary
    loader.summary()
    
    # Show sample signal
    if len(signals) > 0:
        print("Sample PCG Signal:")
        print("-" * 80)
        print(f"Record ID:    {record_ids[0]}")
        print(f"Signal shape: {signals[0].shape}")
        print(f"Label:        {labels[0]} ({'Normal' if labels[0] == 0 else 'Abnormal'})")
        print(f"First 10 samples:")
        print(signals[0][:10])
    
    print(f"\n{'='*80}")
    print("[OK] Module 7: PCG Dataset Loader - Test Complete")
    print(f"{'='*80}\n")
