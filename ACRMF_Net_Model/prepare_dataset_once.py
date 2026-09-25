"""
Prepare and save dataset ONCE - never reload again
Run this ONCE, then use fast_train.py for all training
"""

import sys
import numpy as np
from pathlib import Path
import pandas as pd
import wfdb
import ast
from scipy.io import wavfile
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import setup_logging

logger = setup_logging(log_level="INFO", file_output=True)


def load_ptbxl_all():
    """Load ALL PTB-XL data"""
    
    logger.info("[Stats] Loading Complete PTB-XL Database...")
    
    data_dir = Path('../ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/')
    
    if not data_dir.exists():
        logger.error(f"PTB-XL not found at {data_dir}")
        return None, None, None
    
    # Load metadata
    df = pd.read_csv(data_dir / 'ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load diagnostic mapping
    agg_df = pd.read_csv(data_dir / 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    # Map to superclasses
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    
    # Filter to 5 classes
    superclass_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    df['label'] = df['diagnostic_superclass'].apply(
        lambda x: superclass_map[x[0]] if len(x) == 1 and x[0] in superclass_map else -1
    )
    df = df[df.label != -1]
    
    # Keep only samples with age and sex
    df = df.dropna(subset=['age', 'sex'])
    
    logger.info(f"   Total PTB-XL samples: {len(df)}")
    for name, idx in superclass_map.items():
        count = (df.label == idx).sum()
        logger.info(f"      {name}: {count}")
    
    clinical_features = []
    ecg_signals = []
    labels = []
    
    logger.info("   Loading ECG signals...")
    total_samples = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            logger.info(f"      Progress: {i}/{total_samples} ({100*i/total_samples:.1f}%)")
        try:
            # Clinical features
            clinical = [
                float(row['age']),
                float(row['sex']),
                0, 120, 0, 0, 0, 70, 0, 0, 0, 0, 0
            ]
            
            # Load ECG
            filename = row['filename_lr']
            signal_data, _ = wfdb.rdsamp(str(data_dir / filename))
            
            # Resample to 1000
            if signal_data.shape[0] != 1000:
                from scipy.signal import resample
                signal_data = resample(signal_data, 1000, axis=0)
            
            # Normalize
            signal_data = (signal_data - signal_data.mean()) / (signal_data.std() + 1e-8)
            
            clinical_features.append(clinical)
            ecg_signals.append(signal_data)
            labels.append(row['label'])
            
        except:
            continue
    
    clinical_features = np.array(clinical_features, dtype=np.float32)
    ecg_signals = np.array(ecg_signals, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    logger.info(f"[PASS] PTB-XL loaded: {len(labels)} samples")
    
    return clinical_features, ecg_signals, labels


def load_physionet_pcg_all():
    """Load ALL PhysioNet PCG"""
    
    logger.info("\n[Stats] Loading ALL PhysioNet PCG...")
    
    data_dir = Path('../archive/')
    training_sets = ['training-a', 'training-b', 'training-c', 
                     'training-d', 'training-e', 'training-f']
    
    pcg_spectrograms = []
    pcg_labels = []
    
    for training_set in training_sets:
        folder = data_dir / training_set
        if not folder.exists():
            continue
        
        wav_files = list(folder.glob('*.wav'))
        logger.info(f"   {training_set}: {len(wav_files)} files")
        
        for i, wav_file in enumerate(wav_files):
            if i % 200 == 0:
                logger.info(f"      Progress: {i}/{len(wav_files)}")
            try:
                sr, audio = wavfile.read(str(wav_file))
                
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Create spectrogram
                f, t, Sxx = signal.spectrogram(audio, sr, nperseg=256)
                
                # Resize to 128x128
                from scipy.ndimage import zoom
                zoom_factors = (128 / Sxx.shape[0], 128 / Sxx.shape[1])
                Sxx_resized = zoom(Sxx, zoom_factors, order=1)
                
                # Normalize
                Sxx_resized = (Sxx_resized - Sxx_resized.mean()) / (Sxx_resized.std() + 1e-8)
                
                pcg_spectrograms.append(Sxx_resized)
                
                # Label
                ref_file = folder / "REFERENCE.csv"
                label = 0
                if ref_file.exists():
                    try:
                        ref_df = pd.read_csv(ref_file, header=None)
                        filename = wav_file.stem
                        label_row = ref_df[ref_df[0] == filename]
                        if len(label_row) > 0:
                            label = 1 if label_row.iloc[0, 1] == 1 else 0
                    except:
                        pass
                
                pcg_labels.append(label)
                
            except:
                continue
    
    pcg_spectrograms = np.array(pcg_spectrograms, dtype=np.float32)
    pcg_labels = np.array(pcg_labels, dtype=np.int64)
    
    logger.info(f"[PASS] PCG loaded: {len(pcg_spectrograms)} samples")
    
    return pcg_spectrograms, pcg_labels


def replicate_to_match(data, target_size):
    """Replicate data to match target size"""
    current_size = len(data)
    if current_size >= target_size:
        return data[:target_size]
    
    repeats = target_size // current_size
    remainder = target_size % current_size
    
    replicated = np.tile(data, (repeats, *[1]*(data.ndim-1)))
    
    if remainder > 0:
        replicated = np.concatenate([replicated, data[:remainder]], axis=0)
    
    return replicated


def align_datasets(clinical, ecg, ecg_labels, pcg_spectrograms, pcg_labels):
    """Align all datasets"""
    
    logger.info("\n[Stats] Aligning Datasets...")
    
    max_size = max(len(ecg), len(pcg_spectrograms))
    logger.info(f"   Target size: {max_size}")
    
    # Replicate if needed
    if len(ecg) < max_size:
        logger.info(f"   Replicating ECG/Clinical from {len(ecg)} to {max_size}")
        clinical = replicate_to_match(clinical, max_size)
        ecg = replicate_to_match(ecg, max_size)
        ecg_labels = replicate_to_match(ecg_labels, max_size)
    
    # Separate PCG
    normal_pcg = pcg_spectrograms[pcg_labels == 0]
    abnormal_pcg = pcg_spectrograms[pcg_labels == 1]
    
    logger.info(f"   PCG - Normal: {len(normal_pcg)}, Abnormal: {len(abnormal_pcg)}")
    
    # Align PCG to ECG
    aligned_pcg = []
    np.random.seed(42)
    
    logger.info("   Aligning PCG to ECG labels...")
    for i, label in enumerate(ecg_labels):
        if i % 2000 == 0:
            logger.info(f"      Progress: {i}/{len(ecg_labels)}")
        if label == 0 and len(normal_pcg) > 0:
            idx = np.random.randint(0, len(normal_pcg))
            aligned_pcg.append(normal_pcg[idx])
        elif len(abnormal_pcg) > 0:
            idx = np.random.randint(0, len(abnormal_pcg))
            aligned_pcg.append(abnormal_pcg[idx])
        else:
            idx = np.random.randint(0, len(pcg_spectrograms))
            aligned_pcg.append(pcg_spectrograms[idx])
    
    aligned_pcg = np.array(aligned_pcg, dtype=np.float32)
    
    # Standardize clinical
    clinical = (clinical - clinical.mean(axis=0)) / (clinical.std(axis=0) + 1e-8)
    
    logger.info(f"[PASS] Final aligned dataset: {len(ecg_labels)} samples")
    
    return clinical, ecg, aligned_pcg, ecg_labels


def main():
    """Prepare and save dataset once"""
    
    logger.info("="*80)
    logger.info("PREPARING DATASET - RUN ONCE")
    logger.info("="*80 + "\n")
    
    # Load all data
    clinical, ecg, ecg_labels = load_ptbxl_all()
    if clinical is None:
        logger.error("Failed to load PTB-XL")
        return
    
    pcg_spectrograms, pcg_labels = load_physionet_pcg_all()
    
    # Align
    clinical, ecg, pcg, labels = align_datasets(
        clinical, ecg, ecg_labels, pcg_spectrograms, pcg_labels
    )
    
    # Save
    logger.info("\n[Save] Saving preprocessed dataset...")
    np.savez_compressed(
        'preprocessed_dataset_full.npz',
        clinical=clinical,
        ecg=ecg,
        pcg=pcg,
        labels=labels
    )
    
    logger.info(f"\n[PASS] Dataset saved to: preprocessed_dataset_full.npz")
    logger.info(f"   Total samples: {len(labels)}")
    logger.info(f"   Clinical: {clinical.shape}")
    logger.info(f"   ECG: {ecg.shape}")
    logger.info(f"   PCG: {pcg.shape}")
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"\n[Stats] Class Distribution:")
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    for cls, cnt, name in zip(unique, counts, class_names):
        logger.info(f"   {name}: {cnt} ({cnt/len(labels)*100:.1f}%)")
    
    logger.info("\n" + "="*80)
    logger.info("[PASS] DATASET PREPARATION COMPLETE!")
    logger.info("Now you can train multiple times with: python fast_train.py")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
