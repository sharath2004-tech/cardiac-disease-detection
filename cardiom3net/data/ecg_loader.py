"""
PTB-XL ECG data loader with multi-task label extraction.
Produces: ecg_array, binary_labels, disease_labels, severity_labels
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


def find_ptbxl_root():
    candidates = [
        Path('/kaggle/input/ptb-xl-a-large-publicly-available-ecg-dataset/'
             'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'),
        Path('/kaggle/input/ptb-xl-a-large-publicly-available-ecg-dataset'),
        Path('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'),
        Path.cwd() / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
        Path.cwd() / 'cardiac-disease-detection' /
        'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
    ]
    for c in candidates:
        if (c / 'ptbxl_database.csv').exists():
            return c
    for root in [Path('/kaggle/input'), Path.cwd()]:
        if root.exists():
            for csv in root.rglob('ptbxl_database.csv'):
                return csv.parent
    return None


def normalize_ecg(signal):
    signal = signal.astype(np.float32)
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-6
    return (signal - mean) / std


DISEASE_MAP = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}


def load_ptbxl(max_records=21837, use_100hz=True):
    """
    Load PTB-XL and create multi-task labels.

    Returns dict with:
        ecg_array:       (N, 12, T)  float32
        binary_labels:   (N,)  0=Normal, 1=Disease
        disease_labels:  (N,)  0-4 (NORM/MI/STTC/CD/HYP)
        severity_labels: (N,)  0-2 (mild/moderate/severe from scp confidence)
        meta_df:         clinical metadata from PTB-XL
        domain_labels:   (N,)  derived from recording device for domain adaptation
    """
    dataset_root = find_ptbxl_root()
    if dataset_root is None:
        return None

    print(f"  PTB-XL root: {dataset_root}")
    df = pd.read_csv(dataset_root / 'ptbxl_database.csv', index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    scp_df = pd.read_csv(dataset_root / 'scp_statements.csv', index_col=0)
    scp_df = scp_df[scp_df['diagnostic'] == 1]

    # ── Multi-task label extraction ───────────────────────────────────
    def get_labels(scp_codes):
        classes = []
        max_confidence = 0.0
        for code, conf in scp_codes.items():
            if code in scp_df.index:
                classes.append(scp_df.loc[code, 'diagnostic_class'])
                max_confidence = max(max_confidence, float(conf))
        classes = sorted(set(classes))

        # Binary: Normal vs Disease
        if 'NORM' in classes:
            binary = 0
        elif len(classes) > 0:
            binary = 1
        else:
            return None

        # Disease class: primary diagnosis (most severe / first non-NORM)
        if 'NORM' in classes and len(classes) == 1:
            disease = DISEASE_MAP['NORM']
        else:
            non_norm = [c for c in classes if c != 'NORM']
            disease = DISEASE_MAP.get(non_norm[0], 0) if non_norm else 0

        # Severity from confidence score: <34 mild, 34-67 moderate, >67 severe
        if binary == 0:
            severity = 0  # normal = mild (no disease)
        elif max_confidence < 34:
            severity = 0  # mild
        elif max_confidence < 67:
            severity = 1  # moderate
        else:
            severity = 2  # severe

        return binary, disease, severity

    records = []
    for ecg_id, row in df.iterrows():
        result = get_labels(row['scp_codes'])
        if result is not None:
            records.append((ecg_id, *result))

    label_df = pd.DataFrame(records, columns=['ecg_id', 'binary', 'disease', 'severity'])
    label_df = label_df.set_index('ecg_id')
    df = df.loc[label_df.index]

    # Balanced sampling
    normal_ids = label_df[label_df['binary'] == 0].index
    disease_ids = label_df[label_df['binary'] == 1].index
    per_class = max_records // 2
    rng = np.random.RandomState(42)
    sampled_ids = np.concatenate([
        rng.choice(normal_ids, size=min(len(normal_ids), per_class), replace=False),
        rng.choice(disease_ids, size=min(len(disease_ids), per_class), replace=False),
    ])
    rng.shuffle(sampled_ids)
    df = df.loc[sampled_ids]
    label_df = label_df.loc[sampled_ids]

    # Resolution
    if use_100hz:
        record_field = 'filename_lr'
        target_length = 1000
    else:
        use_hr = 'filename_hr' in df.columns and (dataset_root / f"{df.iloc[0]['filename_hr']}.hea").exists()
        record_field = 'filename_hr' if use_hr else 'filename_lr'
        target_length = 5000 if use_hr else 1000

    print(f"  Sampling rate: {'100' if use_100hz else '500'} Hz → {target_length} timesteps")

    # ── Load ECG signals ──────────────────────────────────────────────
    ecg_signals, binary_list, disease_list, severity_list = [], [], [], []
    loaded_ids, skipped = [], 0

    # Domain labels: derive from strat_fold (1-10) as proxy for recording site
    domain_list = []

    for ecg_id in sampled_ids:
        row = df.loc[ecg_id]
        try:
            signal, _ = wfdb.rdsamp(str(dataset_root / row[record_field]))
            signal = normalize_ecg(signal)
            if signal.shape[0] >= target_length:
                signal = signal[:target_length]
            else:
                pad = np.zeros((target_length - signal.shape[0], signal.shape[1]), np.float32)
                signal = np.vstack([signal, pad])
            ecg_signals.append(signal.T.astype(np.float32))
            binary_list.append(label_df.loc[ecg_id, 'binary'])
            disease_list.append(label_df.loc[ecg_id, 'disease'])
            severity_list.append(label_df.loc[ecg_id, 'severity'])
            loaded_ids.append(ecg_id)
            # Use strat_fold (1-10) as domain proxy → map to 0-2 (3 domains)
            fold = int(row.get('strat_fold', 1))
            domain_list.append(fold % 3)
        except Exception:
            skipped += 1

    ecg_array = np.stack(ecg_signals)
    meta_cols = [c for c in ['age', 'sex', 'height', 'weight'] if c in df.columns]
    meta_df = df.loc[loaded_ids, meta_cols].apply(pd.to_numeric, errors='coerce')

    result = {
        'ecg_array': ecg_array,
        'binary_labels': np.array(binary_list, dtype=np.int64),
        'disease_labels': np.array(disease_list, dtype=np.int64),
        'severity_labels': np.array(severity_list, dtype=np.int64),
        'domain_labels': np.array(domain_list, dtype=np.int64),
        'meta_df': meta_df,
        'meta_cols': meta_cols,
        'signal_length': target_length,
        'num_leads': ecg_array.shape[1],
        'loaded_ids': loaded_ids,
    }

    print(f"  Loaded: {len(loaded_ids)} records | skipped: {skipped}")
    print(f"  ECG shape: {ecg_array.shape}")
    print(f"  Binary dist: {np.bincount(result['binary_labels'])}")
    print(f"  Disease dist: {np.bincount(result['disease_labels'], minlength=5)}")
    print(f"  Severity dist: {np.bincount(result['severity_labels'], minlength=3)}")
    return result
