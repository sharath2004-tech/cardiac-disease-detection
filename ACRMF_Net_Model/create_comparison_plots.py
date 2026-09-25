"""
Comprehensive Experiment Comparison Plots Generator
===================================================
Creates detailed comparison visualizations for all experiments

Usage:
    py create_comparison_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import json
from collections import defaultdict

print("\n" + "="*80)
print("EXPERIMENT COMPARISON PLOTS GENERATOR")
print("="*80)

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
output_dir = Path('experiments/comparison_plots')
output_dir.mkdir(exist_ok=True)

print(f"\nOutput directory: {output_dir}")

# Find all experiment directories
exp_dirs = sorted([d for d in Path('experiments').iterdir() 
                   if d.is_dir() and d.name.startswith('Exp')])

print(f"\nFound {len(exp_dirs)} experiments:")
for exp_dir in exp_dirs:
    print(f"  - {exp_dir.name}")

# Load all experiment data
experiments = {}

for exp_dir in exp_dirs:
    model_path = exp_dir / 'best_model.pth'
    
    if not model_path.exists():
        print(f"\n[WARN] No model found in {exp_dir.name}, skipping...")
        continue
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        exp_name = exp_dir.name
        val_metrics = checkpoint.get('val_metrics', {})
        history = checkpoint.get('history', {})
        
        experiments[exp_name] = {
            'name': exp_name,
            'epoch': checkpoint.get('epoch', 0),
            'val_acc': val_metrics.get('accuracy', 0),
            'val_f1': val_metrics.get('macro_f1', 0),
            'val_weighted_f1': val_metrics.get('weighted_f1', 0),
            'per_class_acc': val_metrics.get('per_class_accuracy', []),
            'per_class_f1': val_metrics.get('per_class_f1', []),
            'history': history,
            'train_acc': checkpoint.get('train_acc', 0)
        }
        
        print(f"  [OK] Loaded {exp_name}: {val_metrics.get('accuracy', 0):.2f}% accuracy")
        
    except Exception as e:
        print(f"  [ERROR] Failed to load {exp_dir.name}: {e}")

if not experiments:
    print("\n[ERROR] No experiments loaded!")
    exit(1)

print(f"\n[OK] Successfully loaded {len(experiments)} experiments")

# Sort experiments by name
exp_names = sorted(experiments.keys())

print("\n" + "="*80)
print("GENERATING COMPARISON PLOTS")
print("="*80)

# ============================================================================
# Plot 1: Overall Accuracy Comparison
# ============================================================================
print("\n1. Creating overall accuracy comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

val_accs = [experiments[name]['val_acc'] for name in exp_names]
train_accs = [experiments[name]['train_acc'] for name in exp_names]

x = np.arange(len(exp_names))
width = 0.35

# Color code by performance
colors_val = ['green' if a >= 90 else 'orange' if a >= 80 else 'red' for a in val_accs]
colors_train = ['lightgreen' if a >= 90 else 'lightyellow' if a >= 80 else 'lightcoral' for a in train_accs]

bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', 
               color=colors_train, alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, val_accs, width, label='Validation Accuracy',
               color=colors_val, alpha=0.7, edgecolor='black')

# Add target lines
ax.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% Target', alpha=0.7)
ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% Threshold', alpha=0.7)

ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Overall Accuracy Comparison Across All Experiments', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, rotation=45, ha='right')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
           f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
           f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '01_overall_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 01_overall_accuracy_comparison.png")

# ============================================================================
# Plot 2: F1 Score Comparison
# ============================================================================
print("\n2. Creating F1 score comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

val_f1s = [experiments[name]['val_f1'] for name in exp_names]
val_weighted_f1s = [experiments[name]['val_weighted_f1'] for name in exp_names]

bars1 = ax.bar(x - width/2, val_f1s, width, label='Macro F1', 
               color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, val_weighted_f1s, width, label='Weighted F1',
               color='coral', alpha=0.7, edgecolor='black')

ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax.set_title('F1 Score Comparison Across All Experiments', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, rotation=45, ha='right')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '02_f1_score_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 02_f1_score_comparison.png")

# ============================================================================
# Plot 3: Per-Class Accuracy Heatmap
# ============================================================================
print("\n3. Creating per-class accuracy heatmap...")

# Build per-class accuracy matrix
per_class_data = []
for name in exp_names:
    per_class_acc = experiments[name]['per_class_acc']
    per_class_data.append(per_class_acc)

per_class_matrix = np.array(per_class_data)
num_classes = per_class_matrix.shape[1]

fig, ax = plt.subplots(figsize=(12, len(exp_names) * 0.6 + 2))

im = ax.imshow(per_class_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(len(exp_names)))
ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
ax.set_yticklabels(exp_names)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations
for i in range(len(exp_names)):
    for j in range(num_classes):
        text = ax.text(j, i, f'{per_class_matrix[i, j]:.1f}',
                      ha="center", va="center", color="black", fontweight='bold')

ax.set_title('Per-Class Accuracy Heatmap Across All Experiments', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Experiment', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '03_per_class_accuracy_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 03_per_class_accuracy_heatmap.png")

# ============================================================================
# Plot 4: Per-Class Performance Comparison (Bar Charts)
# ============================================================================
print("\n4. Creating per-class performance bars...")

fig, axes = plt.subplots(1, num_classes, figsize=(4*num_classes, 8))

if num_classes == 1:
    axes = [axes]

for cls_idx in range(num_classes):
    ax = axes[cls_idx]
    
    class_accs = [per_class_matrix[i, cls_idx] for i in range(len(exp_names))]
    colors = ['green' if a >= 80 else 'orange' if a >= 70 else 'red' for a in class_accs]
    
    bars = ax.bar(range(len(exp_names)), class_accs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Experiment', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Class {cls_idx} Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=90, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add value labels
    for bar, acc in zip(bars, class_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{acc:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '04_per_class_performance_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 04_per_class_performance_bars.png")

# ============================================================================
# Plot 5: Training Epochs Comparison
# ============================================================================
print("\n5. Creating training epochs comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

epochs = [experiments[name]['epoch'] for name in exp_names]
colors_epoch = plt.cm.viridis(np.linspace(0, 1, len(exp_names)))

bars = ax.bar(x, epochs, color=colors_epoch, alpha=0.7, edgecolor='black')

ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Epochs', fontsize=13, fontweight='bold')
ax.set_title('Training Epochs Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, epoch in zip(bars, epochs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
           f'{epoch}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '05_training_epochs_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 05_training_epochs_comparison.png")

# ============================================================================
# Plot 6: Best vs Worst Class Analysis
# ============================================================================
print("\n6. Creating best vs worst class analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Best class
ax = axes[0]
best_class_accs = [np.max(per_class_matrix[i]) for i in range(len(exp_names))]
best_class_ids = [np.argmax(per_class_matrix[i]) for i in range(len(exp_names))]

colors_best = ['green' if a >= 90 else 'orange' if a >= 80 else 'red' for a in best_class_accs]
bars = ax.bar(range(len(exp_names)), best_class_accs, color=colors_best, alpha=0.7, edgecolor='black')

ax.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% Target', alpha=0.7)
ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% Threshold', alpha=0.7)

ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
ax.set_ylabel('Best Class Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Best Performing Class per Experiment', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels(exp_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

for bar, acc, cls in zip(bars, best_class_accs, best_class_ids):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
           f'{acc:.1f}%\n(C{cls})', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Worst class
ax = axes[1]
worst_class_accs = [np.min(per_class_matrix[i]) for i in range(len(exp_names))]
worst_class_ids = [np.argmin(per_class_matrix[i]) for i in range(len(exp_names))]

colors_worst = ['green' if a >= 80 else 'orange' if a >= 70 else 'red' for a in worst_class_accs]
bars = ax.bar(range(len(exp_names)), worst_class_accs, color=colors_worst, alpha=0.7, edgecolor='black')

ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% Threshold', alpha=0.7)
ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% Critical', alpha=0.7)

ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
ax.set_ylabel('Worst Class Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Worst Performing Class per Experiment', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels(exp_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

for bar, acc, cls in zip(bars, worst_class_accs, worst_class_ids):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
           f'{acc:.1f}%\n(C{cls})', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '06_best_worst_class_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 06_best_worst_class_analysis.png")

# ============================================================================
# Plot 7: Overall Performance Ranking
# ============================================================================
print("\n7. Creating overall performance ranking...")

fig, ax = plt.subplots(figsize=(12, 10))

# Sort by validation accuracy
sorted_indices = np.argsort(val_accs)[::-1]
sorted_names = [exp_names[i] for i in sorted_indices]
sorted_accs = [val_accs[i] for i in sorted_indices]
sorted_colors = [colors_val[i] for i in sorted_indices]

y_pos = range(len(sorted_names))
bars = ax.barh(y_pos, sorted_accs, color=sorted_colors, alpha=0.7, edgecolor='black')

ax.axvline(x=90, color='green', linestyle='--', linewidth=2, label='90% Target', alpha=0.7)
ax.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='80% Threshold', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names)
ax.set_xlabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Experiments Ranked by Validation Accuracy', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([0, 105])

# Add value labels and rank
for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
           f'#{i+1}: {acc:.2f}%', ha='left', va='center', 
           fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '07_performance_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 07_performance_ranking.png")

# ============================================================================
# Plot 8: Overfitting Analysis (Train-Val Gap)
# ============================================================================
print("\n8. Creating overfitting analysis...")

fig, ax = plt.subplots(figsize=(14, 8))

gaps = [experiments[name]['train_acc'] - experiments[name]['val_acc'] for name in exp_names]
colors_gap = ['green' if g < 5 else 'orange' if g < 10 else 'red' for g in gaps]

bars = ax.bar(x, gaps, color=colors_gap, alpha=0.7, edgecolor='black')

ax.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='5% Warning', alpha=0.7)
ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10% Danger', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
ax.set_ylabel('Train - Validation Gap (%)', fontsize=13, fontweight='bold')
ax.set_title('Overfitting Analysis (Train-Val Accuracy Gap)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    y_pos = height + 0.5 if height > 0 else height - 0.5
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
           f'{gap:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
           fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '08_overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 08_overfitting_analysis.png")

# ============================================================================
# Plot 9: Comprehensive Metrics Dashboard
# ============================================================================
print("\n9. Creating comprehensive metrics dashboard...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Validation Accuracy
ax = fig.add_subplot(gs[0, 0])
ax.bar(range(len(exp_names)), val_accs, color=colors_val, alpha=0.7, edgecolor='black')
ax.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_title('Validation Accuracy', fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels([name.split('_')[0] for name in exp_names], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Accuracy (%)')
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Macro F1
ax = fig.add_subplot(gs[0, 1])
ax.bar(range(len(exp_names)), val_f1s, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title('Macro F1 Score', fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels([name.split('_')[0] for name in exp_names], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('F1 Score')
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Training Epochs
ax = fig.add_subplot(gs[0, 2])
ax.bar(range(len(exp_names)), epochs, color='coral', alpha=0.7, edgecolor='black')
ax.set_title('Training Epochs', fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels([name.split('_')[0] for name in exp_names], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Epochs')
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Best Class Performance
ax = fig.add_subplot(gs[1, 0])
ax.bar(range(len(exp_names)), best_class_accs, color=colors_best, alpha=0.7, edgecolor='black')
ax.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_title('Best Class Accuracy', fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels([name.split('_')[0] for name in exp_names], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Accuracy (%)')
ax.grid(True, alpha=0.3, axis='y')

# Panel 5: Worst Class Performance
ax = fig.add_subplot(gs[1, 1])
ax.bar(range(len(exp_names)), worst_class_accs, color=colors_worst, alpha=0.7, edgecolor='black')
ax.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_title('Worst Class Accuracy', fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels([name.split('_')[0] for name in exp_names], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Accuracy (%)')
ax.grid(True, alpha=0.3, axis='y')

# Panel 6: Train-Val Gap
ax = fig.add_subplot(gs[1, 2])
ax.bar(range(len(exp_names)), gaps, color=colors_gap, alpha=0.7, edgecolor='black')
ax.axhline(y=5, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_title('Overfitting Gap', fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels([name.split('_')[0] for name in exp_names], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Gap (%)')
ax.grid(True, alpha=0.3, axis='y')

# Panel 7: Per-Class Heatmap (simplified)
ax = fig.add_subplot(gs[2, :])
im = ax.imshow(per_class_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels([name.split('_')[0] for name in exp_names], rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(num_classes))
ax.set_yticklabels([f'C{i}' for i in range(num_classes)])
ax.set_title('Per-Class Accuracy Heatmap', fontweight='bold')
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
cbar.set_label('Accuracy (%)', fontweight='bold')

# Add overall title
fig.suptitle('Comprehensive Experiment Metrics Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig(output_dir / '09_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: 09_comprehensive_dashboard.png")

# ============================================================================
# Generate Summary Report
# ============================================================================
print("\n10. Generating summary report...")

summary_path = output_dir / 'comparison_summary.txt'

with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXPERIMENT COMPARISON SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total Experiments: {len(experiments)}\n\n")
    
    # Best experiment
    best_idx = np.argmax(val_accs)
    best_name = exp_names[best_idx]
    
    f.write("BEST EXPERIMENT\n")
    f.write("-"*80 + "\n")
    f.write(f"Name: {best_name}\n")
    f.write(f"Validation Accuracy: {val_accs[best_idx]:.2f}%\n")
    f.write(f"Macro F1 Score: {val_f1s[best_idx]:.4f}\n")
    f.write(f"Training Epochs: {epochs[best_idx]}\n")
    f.write(f"Train-Val Gap: {gaps[best_idx]:.2f}%\n\n")
    
    # All experiments ranking
    f.write("RANKING BY VALIDATION ACCURACY\n")
    f.write("-"*80 + "\n")
    for i, idx in enumerate(sorted_indices):
        name = exp_names[idx]
        acc = val_accs[idx]
        f1 = val_f1s[idx]
        status = "[OK]" if acc >= 90 else "[WARN]" if acc >= 80 else "[FAIL]"
        f.write(f"#{i+1:2d} {status} {name:35s} {acc:6.2f}% (F1: {f1:.4f})\n")
    
    f.write("\n")
    
    # Per-class analysis
    f.write("PER-CLASS ANALYSIS\n")
    f.write("-"*80 + "\n")
    for cls_idx in range(num_classes):
        f.write(f"\nClass {cls_idx}:\n")
        class_accs = per_class_matrix[:, cls_idx]
        best_exp_idx = np.argmax(class_accs)
        worst_exp_idx = np.argmin(class_accs)
        
        f.write(f"  Best:  {exp_names[best_exp_idx]:35s} {class_accs[best_exp_idx]:.2f}%\n")
        f.write(f"  Worst: {exp_names[worst_exp_idx]:35s} {class_accs[worst_exp_idx]:.2f}%\n")
        f.write(f"  Mean:  {np.mean(class_accs):.2f}%\n")
        f.write(f"  Std:   {np.std(class_accs):.2f}%\n")
    
    f.write("\n" + "="*80 + "\n")

print("  [OK] Saved: comparison_summary.txt")

# ============================================================================
# Save JSON results
# ============================================================================
print("\n11. Saving JSON results...")

results = {
    'experiments': {},
    'summary': {
        'total_experiments': len(experiments),
        'best_experiment': {
            'name': best_name,
            'val_acc': float(val_accs[best_idx]),
            'val_f1': float(val_f1s[best_idx])
        }
    }
}

for name in exp_names:
    results['experiments'][name] = {
        'val_acc': float(experiments[name]['val_acc']),
        'val_f1': float(experiments[name]['val_f1']),
        'epoch': int(experiments[name]['epoch']),
        'per_class_acc': [float(x) for x in experiments[name]['per_class_acc']]
    }

json_path = output_dir / 'comparison_results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print("  [OK] Saved: comparison_results.json")

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "="*80)
print("COMPARISON PLOTS GENERATION COMPLETE!")
print("="*80)
print(f"\nGenerated {11} visualization files in: {output_dir}/")
print("\nFiles created:")
print("  1. 01_overall_accuracy_comparison.png")
print("  2. 02_f1_score_comparison.png")
print("  3. 03_per_class_accuracy_heatmap.png")
print("  4. 04_per_class_performance_bars.png")
print("  5. 05_training_epochs_comparison.png")
print("  6. 06_best_worst_class_analysis.png")
print("  7. 07_performance_ranking.png")
print("  8. 08_overfitting_analysis.png")
print("  9. 09_comprehensive_dashboard.png")
print("  10. comparison_summary.txt")
print("  11. comparison_results.json")
print("\n" + "="*80)
print(f"Best Experiment: {best_name} ({val_accs[best_idx]:.2f}% accuracy)")
print("="*80 + "\n")
