"""
Stage 2: Dataset Loaders
Modules 5-8: Clinical, ECG, PCG loaders and dataset split
"""

from .clinical_loader import ClinicalDataLoader, load_clinical_data
from .ecg_loader import ECGDataLoader, load_ecg_data
from .pcg_loader import PCGDataLoader, load_pcg_data
from .dataset_split import DatasetSplitter, MultimodalDatasetSplitter, split_dataset

__all__ = [
    # Module 5: Clinical
    'ClinicalDataLoader',
    'load_clinical_data',
    # Module 6: ECG
    'ECGDataLoader',
    'load_ecg_data',
    # Module 7: PCG
    'PCGDataLoader',
    'load_pcg_data',
    # Module 8: Dataset Split
    'DatasetSplitter',
    'MultimodalDatasetSplitter',
    'split_dataset'
]
