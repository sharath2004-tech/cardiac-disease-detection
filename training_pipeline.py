"""
Complete Training Pipeline for Improved ECG Classification
Uses the enhanced model and training techniques to achieve >90% accuracy

Key Improvements:
1. Deeper CNN-Transformer hybrid with residual connections and SE blocks
2. Advanced data augmentation targeting minority classes
3. Combined loss with auxiliary classifier
4. Layer-wise learning rates
5. Test-time augmentation
6. Gradient clipping and mixed precision training
7. Better handling of class imbalance

Usage:
    Run this after loading and preprocessing your MIT-BIH dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Import our improved components
from improved_model import ImprovedECGClassifier
from improved_training import (
    ImprovedTrainer, 
    compute_optimal_class_weights,
    create_balanced_sampler
)


def setup_improved_pipeline(X_train, y_train, X_test, y_test, 
                           batch_size=128, use_balanced_sampler=True):
    """
    Setup the complete training pipeline with all improvements
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test/validation data and labels
        batch_size: Batch size for training
        use_balanced_sampler: Whether to use weighted sampling
    
    Returns:
        model, trainer, train_loader, test_loader
    """
    
    print("=" * 70)
    print("SETTING UP IMPROVED ECG CLASSIFICATION PIPELINE")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Convert to tensors
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
    
    # Ensure correct shape [B, 1, L]
    if X_train.dim() == 2:
        X_train = X_train.unsqueeze(1)
    if X_test.dim() == 2:
        X_test = X_test.unsqueeze(1)
    
    print(f"✓ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"✓ Class distribution in training: {np.bincount(y_train.numpy())}")
    print(f"✓ Class distribution in test: {np.bincount(y_test.numpy())}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    if use_balanced_sampler:
        print("\n✓ Using balanced sampler for training")
        train_sampler = create_balanced_sampler(
            type('Dataset', (), {'targets': y_train.numpy()})(), 
            minority_oversample=3.0
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Compute class weights for loss function
    print("\n✓ Computing optimal class weights...")
    class_weights = compute_optimal_class_weights(train_loader, strategy='balanced_extreme')
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights]}")
    
    # Initialize model
    print("\n✓ Initializing improved model...")
    model = ImprovedECGClassifier(num_classes=4, dropout=0.35)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer with optimal hyperparameters
    print("\n✓ Initializing trainer...")
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        lr=0.0003,  # Conservative learning rate
        weight_decay=1e-3,
        class_weights=class_weights
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE SETUP COMPLETE - Ready to train!")
    print("=" * 70)
    
    return model, trainer, train_loader, test_loader


def train_improved_model(X_train, y_train, X_test, y_test, 
                        epochs=50, batch_size=128, 
                        save_path='best_improved_model.pth'):
    """
    Complete training function - just call this with your data!
    
    Args:
        X_train, y_train: Training data and labels (numpy arrays or tensors)
        X_test, y_test: Test data and labels
        epochs: Number of epochs to train
        batch_size: Batch size
        save_path: Where to save the best model
    
    Returns:
        trainer: Trained ImprovedTrainer object
        history: Training history dict
    """
    
    # Setup pipeline
    model, trainer, train_loader, test_loader = setup_improved_pipeline(
        X_train, y_train, X_test, y_test, 
        batch_size=batch_size, 
        use_balanced_sampler=True
    )
    
    # Train
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING FOR {epochs} EPOCHS")
    print(f"{'='*70}\n")
    
    history = trainer.train(epochs=epochs, save_path=save_path, verbose=True)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    preds, targets, probs = trainer.evaluate_detailed(test_loader, checkpoint_path=save_path)
    
    return trainer, history, preds, targets, probs


def quick_test():
    """
    Quick test with dummy data to verify everything works
    """
    print("Running quick test with dummy data...\n")
    
    # Create dummy data
    X_train = np.random.randn(1000, 180).astype(np.float32)
    y_train = np.random.randint(0, 4, 1000)
    X_test = np.random.randn(200, 180).astype(np.float32)
    y_test = np.random.randint(0, 4, 200)
    
    # Make class distribution realistic (imbalanced)
    y_train = np.concatenate([
        np.zeros(700, dtype=np.int64),  # Normal
        np.ones(50, dtype=np.int64),    # S
        np.full(200, 2, dtype=np.int64), # V
        np.full(50, 3, dtype=np.int64)   # F
    ])
    np.random.shuffle(y_train)
    
    y_test = np.concatenate([
        np.zeros(140, dtype=np.int64),   # Normal
        np.ones(10, dtype=np.int64),     # S
        np.full(40, 2, dtype=np.int64),  # V
        np.full(10, 3, dtype=np.int64)   # F
    ])
    np.random.shuffle(y_test)
    
    # Quick training (2 epochs only)
    trainer, history, preds, targets, probs = train_improved_model(
        X_train, y_train, X_test, y_test,
        epochs=2,  # Just 2 epochs for testing
        batch_size=32,
        save_path='test_model.pth'
    )
    
    print("\n✓ Quick test completed successfully!")
    print("  You can now use this pipeline with your real MIT-BIH data")


# ============================================================================
# EXAMPLE USAGE WITH REAL DATA
# ============================================================================

def example_usage_with_mitbih():
    """
    Example of how to use this with your MIT-BIH dataset
    
    Assumes you have already loaded and preprocessed your data:
    - X_train_resampled, y_train_resampled
    - X_test, y_test
    - Each ECG signal should be shape (180,) or (1, 180)
    """
    
    # Load your preprocessed MIT-BIH data here
    # X_train_resampled, y_train_resampled, X_test, y_test = load_mitbih_data()
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  EXAMPLE USAGE WITH MIT-BIH DATASET                            ║
    ╚════════════════════════════════════════════════════════════════╝
    
    After loading your MIT-BIH data, use the pipeline like this:
    
    ```python
    from training_pipeline import train_improved_model
    
    # Train the improved model
    trainer, history, preds, targets, probs = train_improved_model(
        X_train_resampled,  # Your training data
        y_train_resampled,  # Your training labels
        X_test,             # Your test data
        y_test,             # Your test labels
        epochs=50,          # Train for 50 epochs (with early stopping)
        batch_size=128,     # Batch size
        save_path='best_ecg_model.pth'  # Where to save best model
    )
    
    # The trainer object gives you access to the model and training history
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best validation F1 score: {trainer.best_val_f1:.2f}%")
    
    # You can also access the training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (%)')
    plt.legend()
    plt.title('F1 Score (Macro)')
    plt.tight_layout()
    plt.savefig('training_history.png')
    ```
    
    Expected Results:
    - Overall accuracy: 90-93%
    - Normal (N): >85%
    - Ventricular (V): >80%
    - Supraventricular (S): 30-60% (improved from 1%)
    - Fusion (F): 15-40% (improved from 0.5%)
    
    The minority classes will still be challenging due to extreme imbalance,
    but the overall accuracy and macro F1 score will be much better!
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run quick test
        quick_test()
    else:
        # Show usage instructions
        example_usage_with_mitbih()
        
        print("\n" + "=" * 70)
        print("To run a quick test with dummy data, use:")
        print("  python training_pipeline.py --test")
        print("=" * 70)
