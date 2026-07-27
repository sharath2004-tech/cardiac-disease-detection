"""
Module 2: Configuration Module
ACRMF-Net Configuration Settings
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class ACRMFConfig:
    """
    Complete configuration for ACRMF-Net implementation
    Following the 13-stage roadmap architecture
    """
    
    # ============================================================================
    # PROJECT PATHS
    # ============================================================================
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "datasets")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "checkpoints")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "logs")
    figure_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "figures")
    metrics_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "metrics")
    
    # Dataset paths
    ptb_xl_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    pcg_archive_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "archive")
    clinical_csv: Path = field(default_factory=lambda: Path(__file__).parent.parent / "heart_disease_uci.csv")
    
    # ============================================================================
    # STAGE 4: CLINICAL ENCODER (Module 13)
    # ============================================================================
    clinical_input_dim: int = 13  # Number of clinical features
    clinical_hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 128])
    clinical_output_dim: int = 128  # (B, 128) as per roadmap
    clinical_dropout: float = 0.3
    clinical_batch_norm: bool = True
    
    # ============================================================================
    # STAGE 5: ECG ENCODER (Module 14)
    # ============================================================================
    ecg_input_channels: int = 12  # 12-lead ECG
    ecg_sequence_length: int = 1000  # Time steps
    ecg_base_filters: int = 64
    ecg_output_dim: int = 128  # (B, 128) as per roadmap
    ecg_num_blocks: int = 4
    ecg_dropout: float = 0.2
    
    # ECG preprocessing
    ecg_sampling_rate: int = 100  # Hz
    ecg_signal_length: int = 1000  # 10 seconds @ 100Hz
    
    # ============================================================================
    # STAGE 6: PCG ENCODER (Module 15)
    # ============================================================================
    pcg_input_channels: int = 1  # Mono audio
    pcg_sequence_length: int = 2000  # Time steps
    pcg_base_filters: int = 32
    pcg_output_dim: int = 128  # (B, 128) as per roadmap
    pcg_num_blocks: int = 3
    pcg_dropout: float = 0.2
    
    # PCG preprocessing
    pcg_sampling_rate: int = 2000  # Hz
    pcg_n_mels: int = 64
    pcg_n_fft: int = 1024
    pcg_hop_length: int = 512
    
    # ============================================================================
    # STAGE 7: RELIABILITY ESTIMATION NETWORK - REN (Module 16)
    # ============================================================================
    ren_hidden_dim: int = 256
    ren_output_dim: int = 1  # Reliability score per modality (Rc, Re, Rp)
    ren_num_layers: int = 3
    ren_dropout: float = 0.3
    
    # ============================================================================
    # STAGE 8: CONFIDENCE ESTIMATION NETWORK - CEN (Module 17)
    # ============================================================================
    cen_hidden_dim: int = 256
    cen_output_dim: int = 1  # Confidence score per modality (Cc, Ce, Cp, Cf)
    cen_num_layers: int = 3
    cen_dropout: float = 0.3
    
    # ============================================================================
    # STAGE 9: ADAPTIVE WEIGHT GENERATOR - AWG (Module 18)
    # ============================================================================
    awg_hidden_dim: int = 128
    awg_num_modalities: int = 3  # Clinical, ECG, PCG
    awg_temperature: float = 1.0  # Softmax temperature
    awg_dropout: float = 0.2
    # Constraint: Wc + We + Wp = 1
    
    # ============================================================================
    # STAGE 10: ACRMF FUSION MODULE (Module 19) - MAIN CONTRIBUTION
    # ============================================================================
    fusion_method: str = "acrmf"  # Adaptive Clinically-aware Reliability-aware Multimodal Fusion
    fusion_hidden_dim: int = 256
    fusion_output_dim: int = 128  # (B, 128) fused feature
    fusion_attention_heads: int = 4
    fusion_dropout: float = 0.2
    
    # ============================================================================
    # STAGE 11: DISEASE PREDICTION (Module 20-21)
    # ============================================================================
    num_classes: int = 2  # Binary classification (Normal/Abnormal)
    decision_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    output_activation: str = "sigmoid"  # For binary classification
    
    # ============================================================================
    # STAGE 12: TRAINING CONFIGURATION (Module 22-25)
    # ============================================================================
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    
    # Loss weights (Module 22: Composite Loss)
    classification_loss_weight: float = 1.0
    reliability_loss_weight: float = 0.3
    confidence_loss_weight: float = 0.2
    fusion_diversity_weight: float = 0.1
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"  # Options: "cosine", "step", "plateau"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Gradient clipping
    gradient_clip_value: float = 1.0
    
    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================
    # Dataset split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Data quality thresholds
    min_quality_threshold: float = 0.5  # Minimum acceptable signal quality
    
    # ============================================================================
    # STAGE 13: EVALUATION CONFIGURATION (Module 26-30)
    # ============================================================================
    # Metrics to compute
    compute_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr",
        "sensitivity", "specificity", "confusion_matrix"
    ])
    
    # Explainability
    use_shap: bool = True
    use_gradcam: bool = True
    shap_background_samples: int = 100
    
    # Visualization
    save_figures: bool = True
    figure_dpi: int = 300
    figure_format: str = "png"
    
    # Statistical tests
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000
    
    # ============================================================================
    # HARDWARE CONFIGURATION
    # ============================================================================
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    
    # ============================================================================
    # REPRODUCIBILITY
    # ============================================================================
    random_seed: int = 42
    deterministic: bool = True
    
    # ============================================================================
    # LOGGING
    # ============================================================================
    log_level: str = "INFO"
    log_interval: int = 10  # Log every N batches
    save_checkpoint_interval: int = 5  # Save every N epochs
    
    # Experiment tracking
    experiment_name: str = "acrmf_net_experiment"
    run_name: Optional[str] = None
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    def __post_init__(self):
        """Create necessary directories after initialization"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
    def validate(self):
        """Validate configuration settings"""
        # Check split ratios sum to 1
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total_ratio - 1.0) < 1e-6, f"Split ratios must sum to 1, got {total_ratio}"
        
        # Check output dimensions match
        assert self.clinical_output_dim == self.ecg_output_dim == self.pcg_output_dim, \
            "All encoder output dimensions must match for fusion"
        
        # Check dataset paths exist
        if not self.ptb_xl_dir.exists():
            print(f"Warning: PTB-XL dataset not found at {self.ptb_xl_dir}")
        if not self.pcg_archive_dir.exists():
            print(f"Warning: PCG dataset not found at {self.pcg_archive_dir}")
        if not self.clinical_csv.exists():
            print(f"Warning: Clinical CSV not found at {self.clinical_csv}")
        
        print("✓ Configuration validated successfully")
        
    def summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("ACRMF-Net Configuration Summary")
        print("=" * 80)
        print(f"\n📁 Paths:")
        print(f"  Project Root: {self.project_root}")
        print(f"  Results Dir:  {self.results_dir}")
        print(f"\n🏗️  Architecture:")
        print(f"  Clinical Encoder: {self.clinical_input_dim} → {self.clinical_output_dim}")
        print(f"  ECG Encoder:      {self.ecg_input_channels}×{self.ecg_sequence_length} → {self.ecg_output_dim}")
        print(f"  PCG Encoder:      {self.pcg_input_channels}×{self.pcg_sequence_length} → {self.pcg_output_dim}")
        print(f"  Fusion Output:    {self.fusion_output_dim}")
        print(f"  Classes:          {self.num_classes}")
        print(f"\n🎯 Training:")
        print(f"  Batch Size:       {self.batch_size}")
        print(f"  Epochs:           {self.num_epochs}")
        print(f"  Learning Rate:    {self.learning_rate}")
        print(f"  Optimizer:        {self.optimizer}")
        print(f"\n📊 Data Split:")
        print(f"  Train:            {self.train_ratio*100:.1f}%")
        print(f"  Validation:       {self.val_ratio*100:.1f}%")
        print(f"  Test:             {self.test_ratio*100:.1f}%")
        print(f"\n💾 Hardware:")
        print(f"  Device:           {self.device}")
        print(f"  Mixed Precision:  {self.use_amp}")
        print(f"  Workers:          {self.num_workers}")
        print("=" * 80)


def get_config(**kwargs) -> ACRMFConfig:
    """
    Get configuration with optional overrides
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        ACRMFConfig instance
    """
    config = ACRMFConfig(**kwargs)
    config.validate()
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    config.summary()
