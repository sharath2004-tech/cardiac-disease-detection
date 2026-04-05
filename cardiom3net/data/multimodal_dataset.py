"""
Unified multimodal PyTorch datasets for CardioM3Net.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import SimCLRAugmentation, random_augment


class MultimodalCardiacDataset(Dataset):
    """Dataset for multi-task supervised training."""
    def __init__(self, ecg, clinical, binary_labels, disease_labels,
                 severity_labels, domain_labels=None, augment=False):
        self.ecg = torch.tensor(ecg, dtype=torch.float32)
        self.clinical = torch.tensor(clinical, dtype=torch.float32)
        self.binary = torch.tensor(binary_labels, dtype=torch.long)
        self.disease = torch.tensor(disease_labels, dtype=torch.long)
        self.severity = torch.tensor(severity_labels, dtype=torch.long)
        self.domain = torch.tensor(domain_labels, dtype=torch.long) if domain_labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, idx):
        ecg = self.ecg[idx]
        if self.augment:
            ecg_np = ecg.numpy()
            ecg_np = random_augment(ecg_np)
            ecg = torch.tensor(ecg_np, dtype=torch.float32)

        sample = {
            'ecg': ecg,
            'clinical': self.clinical[idx],
            'binary': self.binary[idx],
            'disease': self.disease[idx],
            'severity': self.severity[idx],
        }
        if self.domain is not None:
            sample['domain'] = self.domain[idx]
        return sample


class SimCLRDataset(Dataset):
    """Dataset for self-supervised contrastive pretraining."""
    def __init__(self, ecg_array):
        self.ecg = ecg_array  # numpy (N, leads, T)
        self.augment = SimCLRAugmentation()

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        signal = self.ecg[idx]
        view1, view2 = self.augment(signal)
        return (torch.tensor(view1, dtype=torch.float32),
                torch.tensor(view2, dtype=torch.float32))


class MAMLTaskDataset:
    """
    Generates N-way K-shot episodes for MAML training.
    Samples from disease_labels (5 classes).
    """
    def __init__(self, ecg, clinical, disease_labels, n_way=5, k_shot=5, q_query=15):
        self.ecg = ecg
        self.clinical = clinical
        self.disease_labels = disease_labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Index by class
        self.class_indices = {}
        for cls in range(5):
            self.class_indices[cls] = np.where(disease_labels == cls)[0]

        # Only use classes with enough samples
        self.available_classes = [c for c in self.class_indices
                                  if len(self.class_indices[c]) >= k_shot + q_query]

    def sample_episode(self):
        """Returns support set and query set for one MAML episode."""
        if len(self.available_classes) < self.n_way:
            n_way = len(self.available_classes)
        else:
            n_way = self.n_way

        chosen_classes = np.random.choice(self.available_classes, size=n_way, replace=False)

        support_ecg, support_clin, support_labels = [], [], []
        query_ecg, query_clin, query_labels = [], [], []

        for new_label, cls in enumerate(chosen_classes):
            indices = np.random.choice(
                self.class_indices[cls], size=self.k_shot + self.q_query, replace=False
            )
            s_idx = indices[:self.k_shot]
            q_idx = indices[self.k_shot:]

            support_ecg.append(self.ecg[s_idx])
            support_clin.append(self.clinical[s_idx])
            support_labels.extend([new_label] * self.k_shot)

            query_ecg.append(self.ecg[q_idx])
            query_clin.append(self.clinical[q_idx])
            query_labels.extend([new_label] * self.q_query)

        return {
            'support_ecg': torch.tensor(np.concatenate(support_ecg), dtype=torch.float32),
            'support_clinical': torch.tensor(np.concatenate(support_clin), dtype=torch.float32),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_ecg': torch.tensor(np.concatenate(query_ecg), dtype=torch.float32),
            'query_clinical': torch.tensor(np.concatenate(query_clin), dtype=torch.float32),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'n_way': n_way,
        }
