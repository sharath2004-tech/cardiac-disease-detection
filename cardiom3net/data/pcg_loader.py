"""
PCG Loader: PhysioNet CinC 2016 Heart Sound Dataset.

Archive layout expected:
  archive/
    training-a/
      a0001.wav, a0001.hea, ...
      REFERENCE.csv   (format: record_name,label  where 1=Abnormal, -1=Normal)
    training-b/ ... training-f/

Output: log-mel spectrograms (N, 1, PCG_N_MELS, PCG_T_FRAMES) and binary labels (N,).

Cross-dataset assignment helpers:
  build_pcg_pool()     -- split CinC records into normal / abnormal pools
  assign_pcg_to_ecg()  -- label-matched sampling for PTB-XL records
"""

import csv
import os
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# ── PCG processing constants ──────────────────────────────────────────────────
PCG_TARGET_SR   = 2000      # 2 kHz — native sampling rate of CinC recordings
PCG_DURATION_SEC = 5.0      # Fixed clip / pad length (seconds)
PCG_N_SAMPLES   = int(PCG_TARGET_SR * PCG_DURATION_SEC)  # 10 000 samples
PCG_N_FFT       = 200       # 100 ms STFT window at 2 kHz
PCG_HOP_LEN     = 100       # 50 ms hop
PCG_N_MELS      = 64

# T_FRAMES = (PCG_N_SAMPLES - PCG_N_FFT) // PCG_HOP_LEN + 1  =  99
PCG_T_FRAMES = (PCG_N_SAMPLES - PCG_N_FFT) // PCG_HOP_LEN + 1

_TRAINING_SUBSETS = [
    'training-a', 'training-b', 'training-c',
    'training-d', 'training-e', 'training-f',
]


# ── Mel filterbank (pure numpy, no librosa required) ─────────────────────────

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, fmin=20.0, fmax=None):
    """Triangular mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    if fmax is None:
        fmax = sr / 2.0
    n_freqs  = n_fft // 2 + 1
    mel_pts  = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts   = _mel_to_hz(mel_pts)
    bin_idx  = np.clip(np.floor(hz_pts * n_fft / sr).astype(int), 0, n_freqs - 1)

    fbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bin_idx[m - 1], bin_idx[m], bin_idx[m + 1]
        if mid > lo:
            fbank[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fbank[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
        if mid < n_freqs:
            fbank[m - 1, mid] = 1.0
    return fbank


# Module-level filterbank cache (computed once)
_FILTERBANK = None


def _get_filterbank():
    global _FILTERBANK
    if _FILTERBANK is None:
        _FILTERBANK = _mel_filterbank(PCG_TARGET_SR, PCG_N_FFT, PCG_N_MELS)
    return _FILTERBANK


def _log_mel_spectrogram(signal):
    """
    Compute log-mel spectrogram for a 1-D PCG signal of length PCG_N_SAMPLES.

    Returns:
        float32 ndarray of shape (PCG_N_MELS, PCG_T_FRAMES)
    """
    # Frame via sliding window  (T_FRAMES, PCG_N_FFT)
    frames = np.lib.stride_tricks.sliding_window_view(signal, PCG_N_FFT)[::PCG_HOP_LEN]
    frames = frames[:PCG_T_FRAMES].copy().astype(np.float32)

    # Hann window
    frames *= np.hanning(PCG_N_FFT).astype(np.float32)

    # Power spectrum  (n_fft//2+1, T_FRAMES)
    fft_out = np.fft.rfft(frames, n=PCG_N_FFT, axis=1)
    power   = (np.abs(fft_out) ** 2).T

    # Project to mel scale and take log  (PCG_N_MELS, T_FRAMES)
    mel_spec = _get_filterbank() @ power
    return np.log(mel_spec + 1e-9)


# ── WAV loading helpers ───────────────────────────────────────────────────────

def _load_wav_mono(path):
    """
    Load a WAV file as float32 mono.

    Returns:
        (signal: ndarray float32, sr: int)  or  (None, 0) on failure.
    """
    try:
        sr, data = wavfile.read(path)
    except Exception:
        return None, 0

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        data = data.astype(np.float32) / scale
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
        peak = np.abs(data).max()
        if peak > 1.0:
            data /= peak
    else:
        return None, 0

    if data.ndim > 1:
        data = data[:, 0]   # mono: take first channel

    return data, int(sr)


def _resample_to_target(signal, orig_sr):
    """Rational-ratio resampling to PCG_TARGET_SR."""
    if orig_sr == PCG_TARGET_SR:
        return signal
    g   = gcd(int(orig_sr), PCG_TARGET_SR)
    up  = PCG_TARGET_SR // g
    dn  = orig_sr // g
    return resample_poly(signal, up, dn).astype(np.float32)


def _pad_or_trim(signal):
    """Clip or zero-pad to exactly PCG_N_SAMPLES."""
    n = len(signal)
    if n >= PCG_N_SAMPLES:
        return signal[:PCG_N_SAMPLES]
    return np.pad(signal, (0, PCG_N_SAMPLES - n)).astype(np.float32)


# ── Label loading ─────────────────────────────────────────────────────────────

def _load_subset_labels(subset_dir):
    """
    Read REFERENCE.csv in subset_dir.

    Returns:
        dict {record_name: 0-or-1}  (0 = Normal, 1 = Abnormal)
    """
    ref = os.path.join(subset_dir, 'REFERENCE.csv')
    labels = {}
    if not os.path.isfile(ref):
        return labels
    with open(ref, newline='') as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            name, raw = row[0].strip(), row[1].strip()
            try:
                # CinC 2016 convention: 1 = Abnormal, -1 = Normal
                labels[name] = 1 if int(raw) == 1 else 0
            except ValueError:
                pass
    return labels


# ── Main loader ───────────────────────────────────────────────────────────────

def load_pcg_cinc2016(archive_dir, cache_path=None, max_records=None):
    """
    Load and preprocess all PCG recordings from the CinC 2016 archive.

    Args:
        archive_dir  : Path to the 'archive/' directory.
        cache_path   : Optional .npz path.  If it exists the cache is returned
                       directly; otherwise it is written after loading.
        max_records  : Optional integer cap (quick tests).

    Returns:
        dict with keys:
            'spectrograms'  (N, 1, PCG_N_MELS, PCG_T_FRAMES) float32
            'binary_labels' (N,) int64 — 0 = Normal, 1 = Abnormal
            'n_normal'      int
            'n_abnormal'    int
            'record_names'  list[str]
        or None if archive_dir does not exist.
    """
    if not os.path.isdir(archive_dir):
        print(f"  [PCG] Archive directory not found: {archive_dir}")
        return None

    # ── Cache hit ────────────────────────────────────────────────────
    if cache_path and os.path.isfile(cache_path):
        print(f"  [PCG] Loading from cache: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {
            'spectrograms':  d['spectrograms'],
            'binary_labels': d['binary_labels'],
            'n_normal':      int(d['n_normal']),
            'n_abnormal':    int(d['n_abnormal']),
            'record_names':  list(d['record_names']),
        }

    # ── Load from disk ───────────────────────────────────────────────
    specs, labels, names = [], [], []
    n_skipped = 0

    for subset in _TRAINING_SUBSETS:
        subset_dir = os.path.join(archive_dir, subset)
        if not os.path.isdir(subset_dir):
            continue
        ref = _load_subset_labels(subset_dir)
        for name, lbl in sorted(ref.items()):
            wav_path = os.path.join(subset_dir, name + '.wav')
            if not os.path.isfile(wav_path):
                n_skipped += 1
                continue
            sig, sr = _load_wav_mono(wav_path)
            if sig is None or len(sig) == 0:
                n_skipped += 1
                continue
            sig  = _resample_to_target(sig, sr)
            sig  = _pad_or_trim(sig)
            spec = _log_mel_spectrogram(sig)        # (PCG_N_MELS, PCG_T_FRAMES)
            specs.append(spec[np.newaxis])          # (1, PCG_N_MELS, PCG_T_FRAMES)
            labels.append(lbl)
            names.append(name)
            if max_records and len(specs) >= max_records:
                break
        if max_records and len(specs) >= max_records:
            break

    if not specs:
        print("  [PCG] No valid recordings found.")
        return None

    spec_arr  = np.stack(specs,  axis=0).astype(np.float32)   # (N, 1, n_mels, T)
    label_arr = np.array(labels, dtype=np.int64)

    # Per-channel z-score normalisation across the whole dataset
    mean = spec_arr.mean(axis=(0, 2, 3), keepdims=True)
    std  = spec_arr.std(axis=(0, 2, 3),  keepdims=True) + 1e-8
    spec_arr = (spec_arr - mean) / std

    n_norm = int((label_arr == 0).sum())
    n_abn  = int((label_arr == 1).sum())

    print(f"  [PCG] Loaded {len(spec_arr)} recordings: "
          f"{n_norm} Normal, {n_abn} Abnormal  ({n_skipped} skipped)")
    print(f"  [PCG] Spectrogram per record: (1, {PCG_N_MELS}, {PCG_T_FRAMES})")

    result = {
        'spectrograms':  spec_arr,
        'binary_labels': label_arr,
        'n_normal':      n_norm,
        'n_abnormal':    n_abn,
        'record_names':  names,
    }

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(
            cache_path,
            spectrograms  = spec_arr,
            binary_labels = label_arr,
            n_normal      = np.array(n_norm),
            n_abnormal    = np.array(n_abn),
            record_names  = np.array(names),
        )
        print(f"  [PCG] Cache saved: {cache_path}")

    return result


# ── Cross-dataset pairing helpers ─────────────────────────────────────────────

def build_pcg_pool(pcg_data):
    """
    Split PCG spectrograms into Normal and Abnormal pools.

    Returns:
        normal_pool   (M0, 1, PCG_N_MELS, PCG_T_FRAMES) float32
        abnormal_pool (M1, 1, PCG_N_MELS, PCG_T_FRAMES) float32
    """
    specs  = pcg_data['spectrograms']
    labels = pcg_data['binary_labels']
    return specs[labels == 0].copy(), specs[labels == 1].copy()


def assign_pcg_to_ecg(ptbxl_binary_labels, normal_pool, abnormal_pool, rng=None):
    """
    Cross-dataset PCG assignment.

    For each PTB-XL record, randomly draw a PCG spectrogram from the CinC pool
    whose binary label matches (Normal <-> Normal, Disease <-> Abnormal).
    Sampling is with replacement so pool size does not constrain dataset size.

    Clinical rationale: patients with cardiac disease are significantly more
    likely to present with abnormal heart sounds; this label-conditioned
    pairing preserves that clinically meaningful correlation.

    Args:
        ptbxl_binary_labels : (N,) int — 0 = Normal, 1 = Disease
        normal_pool         : (M0, 1, PCG_N_MELS, PCG_T_FRAMES) float32
        abnormal_pool       : (M1, 1, PCG_N_MELS, PCG_T_FRAMES) float32
        rng                 : np.random.Generator for reproducibility

    Returns:
        pcg_array : (N, 1, PCG_N_MELS, PCG_T_FRAMES) float32
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N         = len(ptbxl_binary_labels)
    shape_out = (N,) + normal_pool.shape[1:]
    pcg_out   = np.empty(shape_out, dtype=np.float32)

    is_abn    = ptbxl_binary_labels.astype(bool)

    if (~is_abn).any():
        idx_n = rng.integers(0, len(normal_pool), size=N)
        pcg_out[~is_abn] = normal_pool[idx_n[~is_abn]]

    if is_abn.any():
        idx_a = rng.integers(0, len(abnormal_pool), size=N)
        pcg_out[is_abn] = abnormal_pool[idx_a[is_abn]]

    return pcg_out
