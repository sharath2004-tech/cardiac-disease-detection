"""
Debug script to test model and loss on a small batch
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# Load data
print("Loading data...")
data = np.load('preprocessed_dataset_full.npz')
clinical = torch.FloatTensor(data['clinical'][:64])
ecg = torch.FloatTensor(data['ecg'][:64])
pcg = torch.FloatTensor(data['pcg'][:64])
labels = torch.LongTensor(data['labels'][:64])

print(f"Data shapes:")
print(f"  Clinical: {clinical.shape}")
print(f"  ECG: {ecg.shape}")
print(f"  PCG: {pcg.shape}")
print(f"  Labels: {labels.shape}")
print(f"\nLabel statistics:")
print(f"  Min: {labels.min().item()}")
print(f"  Max: {labels.max().item()}")
print(f"  Unique: {torch.unique(labels).tolist()}")

# Create dataset
dataset = TensorDataset(clinical, ecg, pcg, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Initialize model
print("\nInitializing model...")
from models.acrmf import ACRMFNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = ACRMFNet(
    clinical_input_dim=13,
    ecg_input_dim=1000,
    pcg_input_dim=1000,
    embedding_dim=128,
    num_classes=5,
    dropout=0.3,
    use_ren=True,
    use_cen=True,
    use_awg=True
).to(device)

print("Model initialized successfully")

# Initialize loss
print("\nInitializing loss...")
from losses.composite_loss import CompositeLoss

criterion = CompositeLoss(
    num_classes=5,
    lambda_cls=1.0,
    lambda_rel=0.1,
    lambda_conf=0.1,
    lambda_fusion=0.05,
    lambda_consist=0.1,
    label_smoothing=0.0
)
print("Loss initialized successfully")

# Test forward pass
print("\nTesting forward pass...")
model.eval()

for batch_idx, (clinical_b, ecg_b, pcg_b, labels_b) in enumerate(loader):
    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Clinical: {clinical_b.shape}")
    print(f"  ECG: {ecg_b.shape}")
    print(f"  PCG: {pcg_b.shape}")
    print(f"  Labels: {labels_b.shape}, values: {labels_b.tolist()}")
    
    # Move to device
    clinical_b = clinical_b.to(device)
    ecg_b = ecg_b.to(device)
    pcg_b = pcg_b.to(device)
    labels_b = labels_b.to(device)
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(clinical_b, ecg_b, pcg_b)
        
        print(f"  Forward pass successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        print(f"  Fused logits shape: {outputs['fused_logits'].shape}")
        
        # Test loss computation
        try:
            loss = criterion(outputs, labels_b)
            print(f"  Loss computation successful: {loss.item():.4f}")
        except Exception as e:
            print(f"  ERROR in loss computation: {e}")
            print(f"  Fused logits: {outputs['fused_logits'].shape}, dtype: {outputs['fused_logits'].dtype}")
            print(f"  Labels: {labels_b.shape}, dtype: {labels_b.dtype}")
            print(f"  Labels range: [{labels_b.min().item()}, {labels_b.max().item()}]")
            import traceback
            traceback.print_exc()
            break
            
    except Exception as e:
        print(f"  ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        break
    
    if batch_idx >= 2:  # Test first 3 batches
        break

print("\nDebug complete!")
