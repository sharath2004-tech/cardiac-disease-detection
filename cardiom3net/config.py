"""
CardioM3Net — Global Configuration
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Seeds & Device ────────────────────────────────────────────────
    seed: int = 42

    # ── Data ──────────────────────────────────────────────────────────
    max_ecg_records: int = 21837
    ecg_sampling_rate: int = 100        # 100 or 500 Hz
    ecg_length: int = 1000              # 1000 for 100Hz, 5000 for 500Hz
    num_leads: int = 12
    test_size: float = 0.2
    # PTB-XL clinical features — real 1:1 matched per ECG patient record.
    # age/sex: always present. height/weight: ~40-57% missing (median-imputed).
    # nurse/site: few missing. device_code: always present (label-encoded string).
    ptbxl_clinical_cols: List[str] = field(default_factory=lambda: [
        'age', 'sex', 'height', 'weight', 'nurse', 'site', 'device_code'
    ])

    # Multi-task label config
    disease_classes: List[str] = field(default_factory=lambda: [
        'NORM', 'MI', 'STTC', 'CD', 'HYP'
    ])
    num_disease_classes: int = 5
    severity_bins: int = 3              # mild / moderate / severe

    # ── Model Architecture ────────────────────────────────────────────
    ecg_embed_dim: int = 256
    clinical_embed_dim: int = 64
    fusion_dim: int = 256
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 512
    dropout: float = 0.3

    # ── Self-Supervised Pretraining ───────────────────────────────────
    ssl_epochs: int = 20
    ssl_batch_size: int = 64
    ssl_lr: float = 3e-4
    ssl_temperature: float = 0.07
    ssl_projection_dim: int = 128

    # ── Supervised Training ───────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3

    # Multi-task loss weights
    lambda_binary: float = 1.0
    lambda_disease: float = 1.0
    lambda_severity: float = 0.5

    # ── MAML Meta-Learning ────────────────────────────────────────────
    maml_inner_lr: float = 0.01
    maml_outer_lr: float = 1e-4
    maml_inner_steps: int = 3
    maml_episodes: int = 100
    maml_n_way: int = 5
    maml_k_shot: int = 5
    maml_q_query: int = 15

    # ── Domain Adaptation ─────────────────────────────────────────────
    da_epochs: int = 15
    da_lambda: float = 0.1              # GRL weight
    da_lr: float = 5e-4

    # ── Federated Learning ────────────────────────────────────────────
    fed_rounds: int = 5
    fed_clients: int = 3
    fed_local_epochs: int = 2

    # ── PCG (CinC 2016 heart sound dataset) ──────────────────────────
    pcg_archive_dir:  str  = "archive"
    pcg_cache_path:   str  = "cardiom3net_results/pcg_cache.npz"
    pcg_embed_dim:    int  = 128
    use_pcg:          bool = True

    # ── Output ────────────────────────────────────────────────────────
    output_dir: str = "cardiom3net_results"
