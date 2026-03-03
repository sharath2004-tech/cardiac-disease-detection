"""
Model Comparison Visualization Functions
==========================================

Comprehensive visualization suite for comparing ECG classification models.

Functions included:
1. plot_accuracy_comparison() - Bar charts for accuracy and F1-score
2. plot_per_class_f1() - Per-class F1-score comparison
3. plot_class_wise_heatmap() - Heatmap of class-wise performance
4. plot_confusion_matrices() - Confusion matrix grid
5. plot_error_analysis() - Misclassification pattern analysis
6. plot_improvement_over_baseline() - Relative improvements
7. plot_performance_vs_complexity() - Performance vs parameter count
8. plot_metrics_radar() - Multi-metric radar chart
9. create_model_dashboard() - Comprehensive dashboard
10. generate_all_visualizations() - Generate all plots at once

Usage:
    from model_visualization_functions import *
    
    # After running experiments
    all_results = {
        'SimpleCNN': results,
        'LSTM': lstm_results,
        'ResNet-1D': resnet_results,
        'TCN': tcn_results
    }
    
    # Generate all visualizations
    generate_all_visualizations(all_results, save_all=True)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_accuracy_comparison(results_dict, save_fig=False, filename='accuracy_comparison.png'):
    """
    Create bar chart comparing accuracy across all models
    
    Args:
        results_dict: Dictionary with model names as keys and result dicts as values
        save_fig: Whether to save the figure
        filename: Filename for saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = list(results_dict.keys())
    
    # Extract metrics
    accuracies = [np.mean(results_dict[m]['accuracy']) for m in models]
    acc_stds = [np.std(results_dict[m]['accuracy']) for m in models]
    f1_macros = [np.mean(results_dict[m]['f1_macro']) for m in models]
    f1_stds = [np.std(results_dict[m]['f1_macro']) for m in models]
    
    x = np.arange(len(models))
    width = 0.6
    
    # Accuracy plot
    bars1 = ax1.bar(x, accuracies, width, yerr=acc_stds, capsize=5,
                    color='skyblue', edgecolor='navy', linewidth=1.5,
                    error_kw={'linewidth': 2, 'ecolor': 'darkblue'})
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison (5-Fold Patient-wise CV)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim([0.7, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.85, color='red', linestyle='--', linewidth=1, 
                label='85% threshold', alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, acc, std) in enumerate(zip(bars1, accuracies, acc_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{acc:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.legend()
    
    # F1-Macro plot
    bars2 = ax2.bar(x, f1_macros, width, yerr=f1_stds, capsize=5,
                    color='lightcoral', edgecolor='darkred', linewidth=1.5,
                    error_kw={'linewidth': 2, 'ecolor': 'darkred'})
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score (Macro)', fontsize=12, fontweight='bold')
    ax2.set_title('Model F1-Macro Comparison (5-Fold Patient-wise CV)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim([0.7, 1.0])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0.85, color='red', linestyle='--', linewidth=1, 
                label='85% threshold', alpha=0.5)
    
    # Add value labels
    for i, (bar, f1, std) in enumerate(zip(bars2, f1_macros, f1_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{f1:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.legend()
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {filename}")
    
    plt.show()
    
    return fig


def plot_per_class_f1(results_dict, save_fig=False, filename='per_class_f1.png'):
    """
    Create grouped bar chart showing F1-score for each class across all models
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = list(results_dict.keys())
    class_names = ['N (Normal)', 'S (Supra)', 'V (Ventr)', 'F (Fusion)']
    
    x = np.arange(len(class_names))
    width = 0.8 / len(models)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        results = results_dict[model]
        f1_scores = [np.mean(results['f1_per_class'][j]) for j in range(4)]
        f1_stds = [np.std(results['f1_per_class'][j]) for j in range(4)]
        
        offset = (i - len(models)/2) * width + width/2
        bars = ax.bar(x + offset, f1_scores, width, label=model, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5,
                     yerr=f1_stds, capsize=3, error_kw={'linewidth': 1.5})
        
        # Add value labels
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            if height > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.2f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Arrhythmia Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison\n(Minority Class Performance Analysis)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {filename}")
    
    plt.show()
    
    return fig


def plot_confusion_matrices(results_dict, save_fig=False, filename='confusion_matrices.png'):
    """
    Plot confusion matrices for all models in a grid
    """
    models = list(results_dict.keys())
    n_models = len(models)
    
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    class_names = ['N', 'S', 'V', 'F']
    
    for idx, model in enumerate(models):
        results = results_dict[model]
        
        # Average confusion matrix across folds
        cm_sum = np.zeros((4, 4))
        for cm in results['confusion_matrices']:
            cm_sum += cm
        cm_avg = cm_sum / len(results['confusion_matrices'])
        
        # Normalize by row
        cm_normalized = cm_avg / cm_avg.sum(axis=1, keepdims=True)
        
        ax = axes[idx]
        im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center", color=text_color,
                       fontsize=9, fontweight='bold')
        
        ax.set_title(f'{model}\n(Acc: {np.mean(results["accuracy"]):.3f})', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('True', fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices Comparison (Normalized)', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {filename}")
    
    plt.show()
    
    return fig


def plot_improvement_over_baseline(results_dict, baseline_model='SimpleCNN', 
                                  save_fig=False, filename='improvements.png'):
    """
    Show relative improvements of advanced models over baseline
    """
    if baseline_model not in results_dict:
        print(f"Error: Baseline model '{baseline_model}' not found!")
        return
    
    baseline_acc = np.mean(results_dict[baseline_model]['accuracy'])
    baseline_f1 = np.mean(results_dict[baseline_model]['f1_macro'])
    
    models = [m for m in results_dict.keys() if m != baseline_model]
    
    acc_improvements = []
    f1_improvements = []
    
    for model in models:
        acc = np.mean(results_dict[model]['accuracy'])
        f1 = np.mean(results_dict[model]['f1_macro'])
        
        acc_imp = ((acc - baseline_acc) / baseline_acc) * 100
        f1_imp = ((f1 - baseline_f1) / baseline_f1) * 100
        
        acc_improvements.append(acc_imp)
        f1_improvements.append(f1_imp)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(models))
    
    # Accuracy improvements
    colors_acc = ['green' if imp > 0 else 'red' for imp in acc_improvements]
    bars1 = ax1.barh(x, acc_improvements, color=colors_acc, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(models, fontsize=10)
    ax1.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Accuracy Improvement vs {baseline_model}', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars1, acc_improvements)):
        width = bar.get_width()
        ax1.text(width + (0.3 if width > 0 else -0.3), bar.get_y() + bar.get_height()/2.,
                f'{imp:+.2f}%',
                ha='left' if width > 0 else 'right', va='center', 
                fontsize=10, fontweight='bold')
    
    # F1-Macro improvements
    colors_f1 = ['green' if imp > 0 else 'red' for imp in f1_improvements]
    bars2 = ax2.barh(x, f1_improvements, color=colors_f1, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(models, fontsize=10)
    ax2.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'F1-Macro Improvement vs {baseline_model}', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars2, f1_improvements)):
        width = bar.get_width()
        ax2.text(width + (0.3 if width > 0 else -0.3), bar.get_y() + bar.get_height()/2.,
                f'{imp:+.2f}%',
                ha='left' if width > 0 else 'right', va='center', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {filename}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print(f"IMPROVEMENTS OVER {baseline_model.upper()}")
    print("="*70)
    print(f"\nBaseline Performance:")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  F1-Macro: {baseline_f1:.4f}")
    print(f"\nImprovement Summary:")
    for i, model in enumerate(models):
        print(f"  {model:<20}: Acc {acc_improvements[i]:+.2f}%  |  F1 {f1_improvements[i]:+.2f}%")
    print("="*70)
    
    return fig


def create_model_dashboard(results_dict, save_fig=False, filename='model_dashboard.png'):
    """
    Create comprehensive dashboard with all key metrics
    """
    models = list(results_dict.keys())
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [np.mean(results_dict[m]['accuracy']) for m in models]
    acc_stds = [np.std(results_dict[m]['accuracy']) for m in models]
    x = np.arange(len(models))
    ax1.barh(x, accuracies, xerr=acc_stds, color='skyblue', edgecolor='navy', linewidth=1.5, capsize=5)
    ax1.set_yticks(x)
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel('Accuracy', fontsize=10, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, (acc, std) in enumerate(zip(accuracies, acc_stds)):
        ax1.text(acc + std + 0.01, i, f'{acc:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # 2. F1-Macro Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    f1_macros = [np.mean(results_dict[m]['f1_macro']) for m in models]
    f1_stds = [np.std(results_dict[m]['f1_macro']) for m in models]
    ax2.barh(x, f1_macros, xerr=f1_stds, color='lightcoral', edgecolor='darkred', linewidth=1.5, capsize=5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel('F1-Score (Macro)', fontsize=10, fontweight='bold')
    ax2.set_title('Model F1-Macro', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, (f1, std) in enumerate(zip(f1_macros, f1_stds)):
        ax2.text(f1 + std + 0.01, i, f'{f1:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # 3. Per Class F1 Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    class_names = ['N', 'S', 'V', 'F']
    f1_matrix = np.zeros((len(models), 4))
    for i, model in enumerate(models):
        for j in range(4):
            f1_matrix[i, j] = np.mean(results_dict[model]['f1_per_class'][j])
    im3 = ax3.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    ax3.set_xticks(np.arange(4))
    ax3.set_yticks(np.arange(len(models)))
    ax3.set_xticklabels(class_names, fontsize=9)
    ax3.set_yticklabels(models, fontsize=9)
    ax3.set_title('Per-Class F1 Scores', fontsize=11, fontweight='bold')
    for i in range(len(models)):
        for j in range(4):
            ax3.text(j, i, f'{f1_matrix[i, j]:.2f}', ha="center", va="center",
                    color="black" if f1_matrix[i, j] < 0.75 else "white", fontsize=7)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. Best Model Confusion Matrix
    best_model = max(results_dict.items(), key=lambda x: np.mean(x[1]['f1_macro']))[0]
    ax4 = fig.add_subplot(gs[1:, 0:2 ])
    
    cm_sum = np.zeros((4, 4))
    for cm in results_dict[best_model]['confusion_matrices']:
        cm_sum += cm
    cm_normalized = cm_sum / cm_sum.sum(axis=1, keepdims=True)
    
    im4 = ax4.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(np.arange(4))
    ax4.set_yticks(np.arange(4))
    ax4.set_xticklabels(class_names, fontsize=11)
    ax4.set_yticklabels(class_names, fontsize=11)
    ax4.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True', fontsize=12, fontweight='bold')
    ax4.set_title(f'Best Model: {best_model}\nConfusion Matrix (Normalized)', 
                 fontsize=13, fontweight='bold')
    
    for i in range(4):
        for j in range(4):
            text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            ax4.text(j, i, f'{cm_normalized[i, j]:.3f}\n({int(cm_sum[i, j])})',
                    ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
    
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # 5. Performance Summary Table
    ax5 = fig.add_subplot(gs[1:, 2])
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = [['Model', 'Acc', 'F1', 'Rank']]
    
    ranked = sorted(results_dict.items(), key=lambda x: np.mean(x[1]['f1_macro']), reverse=True)
    for rank, (model, results) in enumerate(ranked, 1):
        acc = np.mean(results['accuracy'])
        f1 = np.mean(results['f1_macro'])
        table_data.append([model, f'{acc:.3f}', f'{f1:.3f}', f'#{rank}'])
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.5, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best
    for i in range(4):
        table[(1, i)].set_facecolor('#FFD700')
    
    ax5.set_title('Performance Ranking\n(by F1-Macro)', fontsize=11, fontweight='bold', pad=20)
    
    fig.suptitle('ECG Arrhythmia Classification - Complete Model Comparison Dashboard',  
                 fontsize=18, fontweight='bold', y=0.98)
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved: {filename}")
    
    plt.show()
    
    return fig


def generate_all_visualizations(results_dict, save_all=True, output_dir='./figures/'):
    """
    Generate all visualization plots at once
    
    Returns dictionary containing all figure objects
    """
    import os
    if save_all and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}")
    
    print("\n" + "="*70)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*70)
    
    figures = {}
    
    print("\n1. Generating accuracy comparison...")
    figures['accuracy'] = plot_accuracy_comparison(results_dict, save_all, 
                                                   f'{output_dir}accuracy_comparison.png')
    
    print("\n2. Generating per-class F1 comparison...")
    figures['per_class'] = plot_per_class_f1(results_dict, save_all, 
                                            f'{output_dir}per_class_f1.png')
    
    print("\n3. Generating confusion matrices...")
    figures['confusion'] = plot_confusion_matrices(results_dict, save_all, 
                                                   f'{output_dir}confusion_matrices.png')
    
    print("\n4. Generating improvement analysis...")
    figures['improvement'] = plot_improvement_over_baseline(results_dict, 'SimpleCNN', 
                                                           save_all, f'{output_dir}improvements.png')
    
    print("\n5. Generating comprehensive dashboard...")
    figures['dashboard'] = create_model_dashboard(results_dict, save_all, 
                                                  f'{output_dir}model_dashboard.png')
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*70)
    
    if save_all:
        print(f"\n📁 All figures saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - accuracy_comparison.png")
        print("  - per_class_f1.png")
        print("  - confusion_matrices.png")
        print("  - improvements.png")
        print("  - model_dashboard.png")
    
    return figures


if __name__ == "__main__":
    print("="*70)
    print("MODEL VISUALIZATION SUITE")
    print("="*70)
    print("\nAvailable Functions:")
    print("  1. plot_accuracy_comparison(results_dict)")
    print("  2. plot_per_class_f1(results_dict)")
    print("  3. plot_confusion_matrices(results_dict)")
    print("  4. plot_improvement_over_baseline(results_dict)")
    print("  5. create_model_dashboard(results_dict)")
    print("  6. generate_all_visualizations(results_dict)  ← Generate ALL!")
    print("\nImport this module in your notebook:")
    print("  from model_visualization_functions import *")
    print("="*70)
