"""
ECG data augmentation for self-supervised and supervised training.
"""
import numpy as np
import torch


def time_crop(signal, crop_ratio=0.9):
    """Randomly crop a portion of the signal along time axis."""
    length = signal.shape[-1]
    crop_len = int(length * crop_ratio)
    start = np.random.randint(0, length - crop_len + 1)
    cropped = signal[..., start:start + crop_len]
    # Pad back to original length
    if cropped.shape[-1] < length:
        pad = np.zeros((*signal.shape[:-1], length - cropped.shape[-1]), dtype=signal.dtype)
        cropped = np.concatenate([cropped, pad], axis=-1)
    return cropped


def add_gaussian_noise(signal, std=0.02):
    noise = np.random.normal(0, std, signal.shape).astype(signal.dtype)
    return signal + noise


def time_scaling(signal, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return signal * scale


def lead_dropout(signal, drop_prob=0.1):
    """Randomly zero out entire leads."""
    mask = (np.random.rand(signal.shape[0], 1) > drop_prob).astype(signal.dtype)
    return signal * mask


def random_augment(signal):
    """Apply a random combination of augmentations for SimCLR."""
    aug = signal.copy()
    if np.random.rand() > 0.3:
        aug = time_crop(aug)
    if np.random.rand() > 0.3:
        aug = add_gaussian_noise(aug)
    if np.random.rand() > 0.5:
        aug = time_scaling(aug)
    if np.random.rand() > 0.7:
        aug = lead_dropout(aug)
    return aug


class SimCLRAugmentation:
    """Generate two augmented views for contrastive learning."""
    def __call__(self, signal):
        view1 = random_augment(signal)
        view2 = random_augment(signal)
        return view1, view2
