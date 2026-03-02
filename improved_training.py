import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
import time
from collections import Counter

from improved_model import ImprovedECGClassifier, CombinedLoss, apply_augmentation, ECGAugmentation


class ImprovedTrainer:
    """
    Enhanced training pipeline with:
    - Gradient clipping
    - Mixed precision training
    - Advanced learning rate scheduling
    - Test-time augmentation
    - Model ensemble
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 lr=0.0003, weight_decay=1e-3, class_weights=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function with class weights
        if class_weights is not None:
            alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
        else:
            alpha = None
        
        self.criterion = CombinedLoss(alpha=alpha, gamma=2.5, aux_weight=0.3)
        
        # Optimizer with layer-wise learning rates
        self.optimizer = optim.AdamW([
            {'params': model.stem.parameters(), 'lr': lr * 0.5},
            {'params': model.res_block1.parameters(), 'lr': lr * 0.7},
            {'params': model.res_block2.parameters(), 'lr': lr * 0.8},
            {'params': model.branch1.parameters(), 'lr': lr},
            {'params': model.branch2.parameters(), 'lr': lr},
            {'params': model.branch3.parameters(), 'lr': lr},
            {'params': model.merge.parameters(), 'lr': lr},
            {'params': model.res_block3.parameters(), 'lr': lr},
            {'params': model.to_transformer.parameters(), 'lr': lr * 1.2},
            {'params': model.transformer1.parameters(), 'lr': lr * 1.2},
            {'params': model.transformer2.parameters(), 'lr': lr * 1.2},
            {'params': model.transformer3.parameters(), 'lr': lr * 1.2},
            {'params': model.transformer4.parameters(), 'lr': lr * 1.2},
            {'params': model.classifier.parameters(), 'lr': lr * 1.5},
            {'params': model.aux_classifier.parameters(), 'lr': lr * 1.5},
        ], weight_decay=weight_decay)
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Gradient scaler for mixed precision (optional)
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience = 0
        self.max_patience = 12
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1': []
        }
    
    def train_epoch(self, epoch, warmup_epochs=5):
        """Train for one epoch with warmup"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Warmup learning rate
        base_lr = self.optimizer.param_groups[0]['lr']
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply augmentation on-the-fly for minority classes
            # (This is done here for demonstration; ideally in dataset)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision (optional)
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    main_output, aux_output = self.model(data)
                    loss, main_loss, aux_loss = self.criterion(main_output, aux_output, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                main_output, aux_output = self.model(data)
                loss, main_loss, aux_loss = self.criterion(main_output, aux_output, target)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = main_output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, use_tta=False):
        """Validate with optional test-time augmentation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if use_tta:
                    # Test-time augmentation
                    outputs = []
                    
                    # Original
                    output = self.model(data)
                    outputs.append(output)
                    
                    # Augmented versions
                    aug = ECGAugmentation()
                    for _ in range(3):
                        aug_data = data.clone()
                        if np.random.random() < 0.5:
                            aug_data = aug.add_noise(aug_data, noise_level=0.005)
                        if np.random.random() < 0.5:
                            aug_data = aug.time_shift(aug_data, max_shift=5)
                        
                        output = self.model(aug_data)
                        outputs.append(output)
                    
                    # Average predictions
                    output = torch.stack(outputs).mean(0)
                else:
                    output = self.model(data)
                
                # For validation loss, we don't have aux output in eval mode
                # So we use a simple cross-entropy
                loss = nn.CrossEntropyLoss()(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Compute F1 score (macro for minority classes)
        f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
        
        return avg_loss, accuracy, f1_macro, np.array(all_preds), np.array(all_targets)
    
    def train(self, epochs=50, save_path='best_model.pth', verbose=True):
        """Main training loop"""
        print("=" * 60)
        print("TRAINING IMPROVED ECG CLASSIFIER")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch, warmup_epochs=5)
            
            # Validate (use TTA for final epochs)
            use_tta = epoch >= epochs - 5
            val_loss, val_acc, val_f1, preds, targets = self.validate(use_tta=use_tta)
            
            # Learning rate scheduling
            if epoch >= 5:  # After warmup
                self.scheduler.step()
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Save best model based on F1 score (better for imbalanced data)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.patience = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, save_path)
                
                if verbose:
                    print(f"✓ Saved new best model (Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%)")
            else:
                self.patience += 1
            
            # Logging
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            if verbose:
                # Compute per-class accuracy
                if (epoch + 1) % 3 == 0 or epoch < 5:
                    class_names = ['N', 'S', 'V', 'F']
                    class_accs = []
                    for i in range(4):
                        mask = targets == i
                        if mask.sum() > 0:
                            class_acc = (preds[mask] == i).sum() / mask.sum() * 100
                            class_accs.append(f"{class_names[i]}:{class_acc:.1f}%")
                    
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | F1: {val_f1:.2f}% | "
                          f"Best F1: {self.best_val_f1:.2f}% | LR: {lr:.6f} | Time: {epoch_time:.1f}s")
                    print(f"  Per-class: {' | '.join(class_accs)}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | F1: {val_f1:.2f}% | "
                          f"Best F1: {self.best_val_f1:.2f}% | LR: {lr:.6f}")
            
            # Early stopping
            if self.patience >= self.max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("\n" + "=" * 60)
        print(f"🎯 Training Complete!")
        print(f"   Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"   Best Validation F1 Score: {self.best_val_f1:.2f}%")
        print("=" * 60)
        
        return self.history
    
    def evaluate_detailed(self, test_loader, checkpoint_path='best_model.pth'):
        """Detailed evaluation with classification report"""
        # Load best model
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Test-time augmentation for better predictions
                outputs = []
                for _ in range(5):  # 5 augmented versions
                    if _ == 0:
                        aug_data = data
                    else:
                        aug = ECGAugmentation()
                        aug_data = data.clone()
                        aug_data = aug.add_noise(aug_data, noise_level=0.005)
                        aug_data = aug.time_shift(aug_data, max_shift=5)
                    
                    output = self.model(aug_data)
                    outputs.append(output)
                
                # Average predictions from all augmentations
                output = torch.stack(outputs).mean(0)
                probs = torch.softmax(output, dim=1)
                
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        class_names = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)']
        
        print("\n" + "=" * 70)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        print("\n" + "=" * 70)
        print("PER-CLASS PERFORMANCE")
        print("=" * 70)
        for i, name in enumerate(['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)']):
            total = cm[i].sum()
            correct = cm[i, i]
            acc = 100.0 * correct / total if total > 0 else 0
            status = "✓ GOOD" if acc > 70 else ("⚠ FAIR" if acc > 40 else "❌ POOR")
            print(f"{name:24s}: {acc:6.2f}% ({correct:5d}/{total:5d}) {status}")
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def create_balanced_sampler(dataset, minority_oversample=3.0):
    """
    Create a weighted sampler to balance classes during training
    Oversamples minority classes more aggressively
    """
    targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    
    class_counts = Counter(targets)
    print(f"\nClass distribution: {dict(class_counts)}")
    
    # Compute weights
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Boost minority classes even more
    max_weight = max(class_weights.values())
    for cls in [1, 3]:  # S and F classes
        if cls in class_weights:
            class_weights[cls] = max_weight * minority_oversample
    
    # Assign weight to each sample
    sample_weights = [class_weights[target] for target in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def compute_optimal_class_weights(train_loader, strategy='balanced_extreme'):
    """
    Compute optimal class weights for loss function
    """
    # Count class distribution
    all_targets = []
    for _, targets in train_loader:
        all_targets.extend(targets.numpy())
    
    counts = Counter(all_targets)
    print(f"Training class counts: {dict(counts)}")
    
    if strategy == 'inverse':
        # Inverse frequency
        total = len(all_targets)
        weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    
    elif strategy == 'balanced_extreme':
        # Extreme weights for very rare classes
        total = len(all_targets)
        weights = {}
        for cls, count in counts.items():
            base_weight = total / (len(counts) * count)
            if cls in [1, 3]:  # S and F - very rare
                weights[cls] = base_weight * 5.0  # 5x boost
            elif cls == 2:  # V - moderately rare
                weights[cls] = base_weight * 1.5
            else:  # N - common
                weights[cls] = base_weight * 0.5
    
    # Convert to list in class order
    weight_list = [weights[i] for i in range(len(counts))]
    
    print(f"Computed class weights: {weight_list}")
    
    return weight_list


if __name__ == "__main__":
    # This script is meant to be imported
    # See training_pipeline.py for usage example
    print("Improved training utilities loaded successfully!")
