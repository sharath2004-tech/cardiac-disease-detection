"""
UCI Heart Disease clinical feature loader.

Produces a rich clinical feature vector by combining:
  - PTB-XL demographic metadata (age, sex) — real 1:1 per ECG record
  - UCI Heart Disease diagnostic features (cholesterol, BP, chest pain, etc.)
    — label-stratified sampling so Normal ECGs get UCI-Normal rows and
      Disease ECGs get UCI-Disease rows.  Sampling is with replacement.

Final feature order (14 dims):
  [age, sex_enc, cp_enc, trestbps, chol, fbs_enc, restecg_enc,
   thalch, exang_enc, oldpeak, slope_enc, ca, thal_enc, height_norm]
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ── Column encoding helpers ───────────────────────────────────────────────────

_CP_MAP      = {'typical angina': 0, 'atypical angina': 1,
                'non-anginal': 2,    'asymptomatic': 3}
_RESTECG_MAP = {'normal': 0, 'lv hypertrophy': 1, 'st-t abnormality': 2}
_SLOPE_MAP   = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
_THAL_MAP    = {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
_BOOL_MAP    = {True: 1, False: 0, 'TRUE': 1, 'FALSE': 0,
                'true': 1, 'false': 0, 1: 1, 0: 0}
_SEX_MAP     = {'Male': 1, 'male': 1, 'M': 1, 'm': 1,
                'Female': 0, 'female': 0, 'F': 0, 'f': 0}


def _encode_col(series, mapping, default=0):
    return series.map(lambda v: mapping.get(v, default)).astype(float)


def _find_uci_csv():
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


def _prepare_uci_df(uci_path):
    """
    Load and encode the UCI CSV into a clean float32 DataFrame.

    Output columns (14):
        age, sex_enc, cp_enc, trestbps, chol, fbs_enc, restecg_enc,
        thalch, exang_enc, oldpeak, slope_enc, ca, thal_enc, height_norm
    'height_norm' is set to 0.0 (not in UCI; placeholder for PTB-XL age-height
    correlation when real height is unavailable).
    """
    df = pd.read_csv(uci_path)

    lbl_col = 'num' if 'num' in df.columns else 'target'
    df['binary_label'] = (df[lbl_col] > 0).astype(int)

    out = pd.DataFrame()
    out['age']         = pd.to_numeric(df['age'],      errors='coerce').fillna(50)
    out['sex_enc']     = _encode_col(df['sex'],    _SEX_MAP)
    out['cp_enc']      = _encode_col(df['cp'],     _CP_MAP)
    trestbps           = pd.to_numeric(df['trestbps'], errors='coerce')
    out['trestbps']    = trestbps.where(trestbps > 0).fillna(130)            # 0 → NaN → median

    chol               = pd.to_numeric(df['chol'], errors='coerce')
    out['chol']        = chol.where(chol > 0).fillna(200)                    # 0 encoded as missing in UCI

    out['fbs_enc']     = df['fbs'].map(_BOOL_MAP).fillna(0).astype(float)
    out['restecg_enc'] = _encode_col(df['restecg'], _RESTECG_MAP)

    thalch             = pd.to_numeric(df['thalch'], errors='coerce')
    out['thalch']      = thalch.where(thalch > 0).fillna(150)

    out['exang_enc']   = df['exang'].map(_BOOL_MAP).fillna(0).astype(float)

    oldpeak            = pd.to_numeric(df['oldpeak'], errors='coerce')
    out['oldpeak']     = oldpeak.clip(lower=0).fillna(0)                     # clamp negatives; fill NaN
    out['slope_enc']   = df['slope'].map(lambda v: _SLOPE_MAP.get(v, 1)).astype(float)  # NaN → 1 ('flat', the mode)
    out['ca']          = pd.to_numeric(df['ca'],       errors='coerce').fillna(0)
    out['thal_enc']    = _encode_col(df['thal'],     _THAL_MAP)
    out['height_norm'] = 0.0   # placeholder
    out['binary_label'] = df['binary_label'].values

    return out.reset_index(drop=True)


def build_augmented_clinical(ptbxl_age, ptbxl_sex,
                              ptbxl_height, ptbxl_binary_labels,
                              rng=None):
    """
    Build a rich 14-dim clinical feature matrix for every PTB-XL record by:
      1. Keeping PTB-XL age and sex (real 1:1 match).
      2. Label-stratified sampling from UCI for the remaining 12 features
         (chol, BP, chest-pain type, etc.).  Normal PTB-XL ECGs sample from
         UCI-Normal rows; Disease ECGs sample from UCI-Disease rows.

    Args:
        ptbxl_age           : (N,) float  — age from PTB-XL (may have NaN)
        ptbxl_sex           : (N,) float  — sex from PTB-XL (may have NaN)
        ptbxl_height        : (N,) float  — height from PTB-XL (may have NaN)
        ptbxl_binary_labels : (N,) int    — 0=Normal, 1=Disease from PTB-XL
        rng                 : np.random.Generator (for reproducibility)

    Returns:
        clinical_array : (N, 14) float32
        feature_names  : list[str] of length 14
    """
    if rng is None:
        rng = np.random.default_rng(42)

    uci_path = _find_uci_csv()
    if uci_path is None:
        raise FileNotFoundError(
            "heart_disease_uci.csv not found. "
            "Place it in the project root directory."
        )

    uci_df    = _prepare_uci_df(uci_path)
    n_normal  = (uci_df['binary_label'] == 0).sum()
    n_disease = (uci_df['binary_label'] == 1).sum()
    print(f"  UCI clinical: {len(uci_df)} rows "
          f"({n_normal} normal, {n_disease} disease)")

    uci_norm = uci_df[uci_df['binary_label'] == 0].drop(columns='binary_label')
    uci_dis  = uci_df[uci_df['binary_label'] == 1].drop(columns='binary_label')

    feature_names = list(uci_norm.columns)   # 14 cols
    N             = len(ptbxl_binary_labels)
    out           = np.zeros((N, len(feature_names)), dtype=np.float32)

    # Fill base columns from UCI via label-stratified sampling
    is_dis  = ptbxl_binary_labels.astype(bool)
    idx_n   = rng.integers(0, len(uci_norm), size=N)
    idx_d   = rng.integers(0, len(uci_dis),  size=N)

    uci_n_vals = uci_norm.values.astype(np.float32)
    uci_d_vals = uci_dis.values.astype(np.float32)

    out[~is_dis] = uci_n_vals[idx_n[~is_dis]]
    out[is_dis]  = uci_d_vals[idx_d[is_dis]]

    # Override age, sex, height with real PTB-XL values where available
    age_col  = feature_names.index('age')
    sex_col  = feature_names.index('sex_enc')
    ht_col   = feature_names.index('height_norm')

    age_arr = np.asarray(ptbxl_age,    dtype=np.float32)
    sex_arr = np.asarray(ptbxl_sex,    dtype=np.float32)
    ht_arr  = np.asarray(ptbxl_height, dtype=np.float32)

    valid_age = np.isfinite(age_arr)
    valid_sex = np.isfinite(sex_arr)
    valid_ht  = np.isfinite(ht_arr)

    out[valid_age, age_col] = age_arr[valid_age]
    out[valid_sex, sex_col] = sex_arr[valid_sex]
    out[valid_ht,  ht_col]  = ht_arr[valid_ht] / 200.0  # normalise to ~[0.75, 1.0] range

    print(f"  Augmented clinical shape: {out.shape}")
    print(f"  Features: {feature_names}")
    print(f"  PTB-XL real-values used — age: {valid_age.sum()}, "
          f"sex: {valid_sex.sum()}, height: {valid_ht.sum()}")

    return out, feature_names


# ── Legacy shim (used in older scripts) ──────────────────────────────────────

def load_uci_clinical(feature_cols, num_ecg_samples):
    """Backward-compatible cyclic loader (kept for compatibility)."""
    uci_path = _find_uci_csv()
    if uci_path is None:
        return None, None

    uci_df  = pd.read_csv(uci_path)
    lbl_col = 'num' if 'num' in uci_df.columns else 'target'
    uci_df[lbl_col] = (uci_df[lbl_col] > 0).astype(int)

    available_cols = [c for c in feature_cols if c in uci_df.columns]
    feat_df = uci_df[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    idx             = np.arange(num_ecg_samples) % len(feat_df)
    clinical_array  = feat_df.values[idx].astype(np.float32)

    print(f"  UCI clinical (legacy): {len(feat_df)} rows -> cyclic-mapped to {num_ecg_samples}")
    return clinical_array, available_cols

