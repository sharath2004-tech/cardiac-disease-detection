"""
Evaluate any checkpoint with Test-Time Augmentation
Use this to get the true accuracy of your models
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from test_complete_model import ACRMFNet


def test_time_augmentation(model, clinical, ecg, pcg, n_aug=7, device='cuda'):
    """Perform TTA for more robust predictions"""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        # Original
        outputs = model(clinical, ecg, pcg)
        all_probs.append(outputs['probabilities'])
        
        # Augmented versions
        for _ in range(n_aug - 1):
            # Small noise augmentation
            clinical_aug = clinical + torch.randn_like(clinical) * 0.02
            ecg_aug = ecg + torch.randn_like(ecg) * 0.01
            pcg_aug = pcg + torch.randn_like(pcg) * 0.005
            
            # Clip to reasonable ranges
            clinical_aug = torch.clamp(clinical_aug, -5, 5)
            ecg_aug = torch.clamp(ecg_aug, -5, 5)
            pcg_aug = torch.clamp(pcg_aug, -5, 5)
            
            outputs = model(clinical_aug, ecg_aug, pcg_aug)
            all_probs.append(outputs['probabilities'])
    
    # Average predictions
    avg_probs = torch.stack(all_probs).mean(dim=0)
    predictions = avg_probs.argmax(dim=1)
    
    return predictions, avg_probs


def evaluate_model(checkpoint_path, data_path='preprocessed_dataset_full.npz', 
                   n_aug=7, batch_size=32, embedding_dim=256):
    """Evaluate a model checkpoint with TTA"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"TTA Augmentations: {n_aug}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    if not Path(data_path).exists():
        print(f"ERROR: {data_path} not found!")
        return
    
    data = np.load(data_path)
    clinical = data['clinical']
    ecg = data['ecg']
    pcg = data['pcg']
    labels = data['labels']
    print(f"Loaded {len(labels)} samples\n")
    
    # Split (same as training)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    model = ACRMFNet(num_classes=5, embedding_dim=embedding_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"Checkpoint val accuracy: {checkpoint['val_acc']:.2f}%")
    if 'tta_acc' in checkpoint:
        print(f"Checkpoint TTA accuracy: {checkpoint['tta_acc']:.2f}%\n")
    else:
        print()
    
    # Evaluate on validation set
    print("Evaluating on validation set...\n")
    model.eval()
    
    # Without TTA
    print("1. Standard Evaluation (no TTA):")
    correct_std = 0
    class_correct_std = [0] * 5
    class_total = [0] * 5
    
    val_clinical = torch.FloatTensor(clinical[val_idx])
    val_ecg = torch.FloatTensor(ecg[val_idx])
    val_pcg = torch.FloatTensor(pcg[val_idx])
    val_labels = labels[val_idx]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(val_idx), batch_size), desc="Standard"):
            batch_clinical = val_clinical[i:i+batch_size].to(device)
            batch_ecg = val_ecg[i:i+batch_size].to(device)
            batch_pcg = val_pcg[i:i+batch_size].to(device)
            batch_labels = val_labels[i:i+batch_size]
            
            outputs = model(batch_clinical, batch_ecg, batch_pcg)
            preds = outputs['predictions'].cpu().numpy()
            
            for j, pred in enumerate(preds):
                label = batch_labels[j]
                class_total[label] += 1
                if pred == label:
                    correct_std += 1
                    class_correct_std[label] += 1
    
    acc_std = 100 * correct_std / len(val_idx)
    class_acc_std = [100 * class_correct_std[i] / max(1, class_total[i]) for i in range(5)]
    
    print(f"   Accuracy: {acc_std:.2f}%")
    print(f"   Per-class: {[f'{a:.1f}%' for a in class_acc_std]}\n")
    
    # With TTA
    print(f"2. Test-Time Augmentation (TTA with {n_aug} augmentations):")
    correct_tta = 0
    class_correct_tta = [0] * 5
    
    with torch.no_grad():
        for i in tqdm(range(0, len(val_idx), batch_size), desc="TTA"):
            batch_clinical = val_clinical[i:i+batch_size].to(device)
            batch_ecg = val_ecg[i:i+batch_size].to(device)
            batch_pcg = val_pcg[i:i+batch_size].to(device)
            batch_labels = val_labels[i:i+batch_size]
            
            preds, probs = test_time_augmentation(
                model, batch_clinical, batch_ecg, batch_pcg, n_aug=n_aug, device=device
            )
            preds = preds.cpu().numpy()
            
            for j, pred in enumerate(preds):
                label = batch_labels[j]
                if pred == label:
                    correct_tta += 1
                    class_correct_tta[label] += 1
    
    acc_tta = 100 * correct_tta / len(val_idx)
    class_acc_tta = [100 * class_correct_tta[i] / max(1, class_total[i]) for i in range(5)]
    
    print(f"   Accuracy: {acc_tta:.2f}%")
    print(f"   Per-class: {[f'{a:.1f}%' for a in class_acc_tta]}\n")
    
    # Compare
    improvement = acc_tta - acc_std
    print(f"{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Standard Accuracy:    {acc_std:.2f}%")
    print(f"TTA Accuracy:         {acc_tta:.2f}%")
    print(f"TTA Improvement:      {improvement:+.2f}%")
    print(f"Target (90%):         {'[PASS] ACHIEVED!' if acc_tta >= 90.0 else f'[FAIL] {90.0 - acc_tta:.2f}% away'}")
    print(f"{'='*80}\n")
    
    # Per-class comparison
    print("Per-Class Breakdown:")
    print(f"{'Class':<10} {'Standard':<12} {'TTA':<12} {'Samples':<10}")
    print(f"{'-'*44}")
    for i in range(5):
        print(f"Class {i}    {class_acc_std[i]:>6.1f}%      {class_acc_tta[i]:>6.1f}%      {class_total[i]:>6}")
    print()
    
    return {
        'standard_acc': acc_std,
        'tta_acc': acc_tta,
        'improvement': improvement,
        'class_acc_std': class_acc_std,
        'class_acc_tta': class_acc_tta,
        'class_totals': class_total
    }


def compare_checkpoints(checkpoint_paths, n_aug=7):
    """Compare multiple checkpoints"""
    print(f"\n{'='*80}")
    print(f"COMPARING MULTIPLE CHECKPOINTS")
    print(f"{'='*80}\n")
    
    results = []
    for cp in checkpoint_paths:
        if not Path(cp).exists():
            print(f"[WARN]  Skipping {cp} (not found)")
            continue
        
        result = evaluate_model(cp, n_aug=n_aug)
        if result:
            results.append({
                'path': cp,
                **result
            })
    
    if not results:
        print("No valid checkpoints found!")
        return
    
    # Summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Checkpoint':<40} {'Standard':<12} {'TTA':<12} {'Gain':<8}")
    print(f"{'-'*72}")
    for r in results:
        name = Path(r['path']).name
        print(f"{name:<40} {r['standard_acc']:>6.2f}%      {r['tta_acc']:>6.2f}%      {r['improvement']:>+5.2f}%")
    
    # Best model
    best = max(results, key=lambda x: x['tta_acc'])
    print(f"\n BEST MODEL:")
    print(f"   {Path(best['path']).name}")
    print(f"   TTA Accuracy: {best['tta_acc']:.2f}%")
    print(f"   {'[PASS] Target achieved!' if best['tta_acc'] >= 90.0 else '[FAIL] Below 90% target'}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model with TTA')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--checkpoints', type=str, nargs='+', help='Multiple checkpoint paths')
    parser.add_argument('--n_aug', type=int, default=7, help='Number of TTA augmentations')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Model embedding dimension')
    
    args = parser.parse_args()
    
    if args.checkpoints:
        compare_checkpoints(args.checkpoints, n_aug=args.n_aug)
    elif args.checkpoint:
        evaluate_model(args.checkpoint, n_aug=args.n_aug, embedding_dim=args.embedding_dim)
    else:
        # Auto-find checkpoints
        checkpoint_dirs = ['checkpoints_90plus', 'checkpoints_fixed', 'checkpoints']
        found = []
        
        for dir_name in checkpoint_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                checkpoints = list(dir_path.glob('*.pth'))
                if checkpoints:
                    # Get the latest one
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    found.append(str(latest))
        
        if found:
            print("No checkpoint specified. Evaluating latest checkpoints found:")
            for cp in found:
                print(f"  - {cp}")
            print()
            compare_checkpoints(found, n_aug=args.n_aug)
        else:
            print("No checkpoint specified and none found automatically!")
            print("\nUsage:")
            print("  python evaluate_with_tta.py --checkpoint path/to/model.pth")
            print("  python evaluate_with_tta.py --checkpoints model1.pth model2.pth model3.pth")
            print("\nOptions:")
            print("  --n_aug N         Number of TTA augmentations (default: 7)")
            print("  --embedding_dim D Model embedding dimension (default: 256)")
