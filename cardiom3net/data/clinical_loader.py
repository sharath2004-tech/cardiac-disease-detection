"""
UCI Heart Disease clinical data loader.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def find_uci_csv():
    candidates = [
        Path('/kaggle/input/heart-disease-data/heart_disease_uci.csv'),
        Path('heart_disease_uci.csv'),
        Path.cwd() / 'heart_disease_uci.csv',
        Path.cwd() / 'cardiac-disease-detection' / 'heart_disease_uci.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    for root in [Path('/kaggle/input'), Path.cwd()]:
        if root.exists():
            for csv in root.rglob('heart_disease_uci.csv'):
                return csv
    return None


def load_uci_clinical(feature_cols, num_ecg_samples):
    """
    Load UCI Heart Disease CSV and cyclically map rows to ECG samples.
    Returns: clinical_array (N, D), feature_names list
    """
    uci_path = find_uci_csv()
    if uci_path is None:
        return None, None

    uci_df = pd.read_csv(uci_path)
    lbl_col = 'num' if 'num' in uci_df.columns else 'target'
    uci_df[lbl_col] = (uci_df[lbl_col] > 0).astype(int)

    available_cols = [c for c in feature_cols if c in uci_df.columns]
    feat_df = uci_df[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Cyclic mapping: ecg[i] → uci[i % len(uci)]
    idx = np.arange(num_ecg_samples) % len(feat_df)
    clinical_array = feat_df.values[idx].astype(np.float32)

    print(f"  UCI clinical: {len(feat_df)} rows → cyclic-mapped to {num_ecg_samples}")
    print(f"  Features ({len(available_cols)}): {available_cols}")
    return clinical_array, available_cols
