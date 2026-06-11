"""Utility functions: seeding, metrics, plotting."""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, classification_report)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("  CPU mode (no CUDA GPU detected)")
    return device


def compute_binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    # Replace NaN/Inf probs (can occur after MAML) with 0.5 (uninformative)
    bad = ~np.isfinite(y_prob)
    if bad.any():
        y_prob = y_prob.copy()
        y_prob[bad] = 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    m = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    m['roc_auc'] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    return m


def compute_multiclass_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def plot_training_curves(history, output_dir, title="CardioM3Net"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title(f'{title} — Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['val_binary_acc'], label='Binary Acc', linewidth=2, color='green')
    axes[0, 1].axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='90% target')
    axes[0, 1].set_title(f'{title} — Binary Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['val_binary_auc'], label='Binary AUC', linewidth=2, color='purple')
    axes[1, 0].set_title(f'{title} — ROC-AUC', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('AUC'); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['val_disease_acc'], label='Disease Acc', linewidth=2, color='orange')
    axes[1, 1].set_title(f'{title} — Disease Classification Acc', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Accuracy'); axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(y_true_binary, y_pred_binary,
                            y_true_disease, y_pred_disease,
                            disease_names, output_dir):
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm1 = confusion_matrix(y_true_binary, y_pred_binary)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Disease'], yticklabels=['Normal', 'Disease'])
    axes[0].set_title('Binary Classification', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    cm2 = confusion_matrix(y_true_disease, y_pred_disease)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=disease_names, yticklabels=disease_names)
    axes[1].set_title('Disease Classification', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(results_dict, output_dir):
    """results_dict: {model_name: (y_true, y_prob)}"""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.title('ROC Curves — CardioM3Net Variants', fontweight='bold')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(comparison_results, output_dir):
    """
    Generate a grouped bar chart comparing all models across key metrics.

    Args:
        comparison_results : OrderedDict / dict {model_name: results_dict}
                             Each results_dict must contain:
                               binary   -> {accuracy, roc_auc, f1}
                               disease  -> {accuracy, f1_weighted}
                               severity -> {accuracy}
        output_dir         : directory to save the figure
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(comparison_results.keys())
    metrics = [
        ('Binary Acc',       lambda r: r['binary']['accuracy']),
        ('Binary AUC',       lambda r: r['binary']['roc_auc']),
        ('Binary F1',        lambda r: r['binary']['f1']),
        ('Disease Acc',      lambda r: r['disease']['accuracy']),
        ('Disease F1(w)',     lambda r: r['disease']['f1_weighted']),
        ('Severity Acc',     lambda r: r['severity']['accuracy']),
    ]

    metric_labels = [m[0] for m in metrics]
    n_models      = len(model_names)
    n_metrics     = len(metrics)

    values = np.zeros((n_models, n_metrics))
    for i, name in enumerate(model_names):
        res = comparison_results[name]
        for j, (_, extractor) in enumerate(metrics):
            try:
                v = extractor(res)
                values[i, j] = v if np.isfinite(v) else 0.0
            except (KeyError, TypeError):
                values[i, j] = 0.0

    # ── Figure: grouped bars ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    x          = np.arange(n_metrics)
    bar_width  = 0.8 / n_models
    # Use a distinguishable palette; last deep-learning proposed model gets a special colour
    _palette = ['#607D8B', '#78909C', '#1976D2', '#43A047', '#FB8C00', '#E53935', '#8E24AA']
    colors = (_palette + list(plt.cm.tab10(np.linspace(0, 0.9, max(0, n_models-len(_palette))))))[:n_models]

    for i, (name, color) in enumerate(zip(model_names, colors)):
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars   = ax.bar(x + offset, values[i], bar_width,
                        label=name, color=color, alpha=0.85, edgecolor='white')
        # Annotate bars
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=6.5,
                        rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Model Comparison — CardioM3Net vs Baselines', fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.4, linewidth=1)

    # Mark proposed model (last entry) with a star in the legend
    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Text table ───────────────────────────────────────────────────
    header = f"{'Model':<42} " + "  ".join(f"{m:>12}" for m in metric_labels)
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, name in enumerate(model_names):
        tag     = " [*]" if 'Proposed' in name else "    "
        row     = f"{name + tag:<42} " + "  ".join(f"{values[i, j]:>12.4f}" for j in range(n_metrics))
        print(row)
    print("-" * len(header))
    print("[*] Proposed model")
    print("=" * len(header))


# ──────────────────────────────────────────────────────────────────────────────
#  Additional visualizations for research papers
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_class_roc(y_true_disease, y_score_disease, class_names, output_dir):
    """One-vs-rest ROC curve for each disease class with individual AUC scores."""
    from sklearn.preprocessing import label_binarize

    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)
    y_true = np.asarray(y_true_disease, dtype=int)
    y_score = np.asarray(y_score_disease, dtype=float)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc_val = auc(fpr, tpr)
        axes[i].plot(fpr, tpr, linewidth=2.5, color='#1976D2',
                     label=f'AUC = {roc_auc_val:.3f}')
        axes[i].fill_between(fpr, tpr, alpha=0.12, color='#1976D2')
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.4)
        axes[i].set_title(f'Class: {name}', fontweight='bold')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].legend(loc='lower right')
        axes[i].grid(True, alpha=0.3)

    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Per-Class ROC Curves (One-vs-Rest)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'per_class_roc.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_precision_recall_curves(results_dict, output_dir):
    """Precision-Recall curves for binary classification across model variants."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    baseline_pos_rate = None
    for name, (y_true, y_prob) in results_dict.items():
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        if baseline_pos_rate is None:
            baseline_pos_rate = y_true.mean()
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, linewidth=2, label=f'{name} (AP={ap:.3f})')

    if baseline_pos_rate is not None:
        ax.axhline(y=baseline_pos_rate, color='k', linestyle='--', alpha=0.5,
                   label=f'Random baseline (P={baseline_pos_rate:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    path = os.path.join(output_dir, 'precision_recall_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_normalized_confusion_matrix(y_true_disease, y_pred_disease, disease_names, output_dir):
    """Side-by-side count and row-normalized confusion matrices for disease classification."""
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true_disease, y_pred_disease)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=disease_names, yticklabels=disease_names)
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1],
                xticklabels=disease_names, yticklabels=disease_names,
                vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Row-Normalized)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_maml_loss(maml_losses, output_dir):
    """MAML meta-training loss over episodes with a smoothed moving-average overlay."""
    if not maml_losses:
        return
    os.makedirs(output_dir, exist_ok=True)

    losses = np.asarray(maml_losses, dtype=float)
    episodes = np.arange(1, len(losses) + 1)
    win = max(5, len(losses) // 10)
    ma = np.convolve(losses, np.ones(win) / win, mode='valid')
    ma_x = np.arange(win, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, losses, alpha=0.3, color='#90CAF9', linewidth=1,
            label='Episode loss')
    ax.plot(ma_x, ma, color='#1565C0', linewidth=2.5,
            label=f'Moving avg (window={win})')
    ax.set_title('MAML Meta-Learning Loss per Episode', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Query Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'maml_loss.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_federated_convergence(fed_history, central_results, output_dir):
    """Round-by-round federated metrics vs central training final performance."""
    if not fed_history:
        return
    os.makedirs(output_dir, exist_ok=True)

    rounds = range(1, len(fed_history) + 1)
    bin_acc = [r['binary']['accuracy'] for r in fed_history]
    bin_auc = [r['binary']['roc_auc'] for r in fed_history]
    dis_acc = [r['disease']['accuracy'] for r in fed_history]

    c_bin_acc = central_results['binary']['accuracy']
    c_bin_auc = central_results['binary']['roc_auc']
    c_dis_acc = central_results['disease']['accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, fed_vals, c_val, label in zip(
        axes,
        [bin_acc, bin_auc, dis_acc],
        [c_bin_acc, c_bin_auc, c_dis_acc],
        ['Binary Accuracy', 'Binary AUC', 'Disease Accuracy'],
    ):
        ax.plot(list(rounds), fed_vals, marker='o', linewidth=2.5,
                color='#43A047', label='Federated', markersize=7)
        ax.axhline(y=c_val, color='#E53935', linestyle='--', linewidth=2,
                   label=f'Central ({c_val:.3f})')
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(rounds))

    plt.suptitle('Federated vs Central Learning Convergence',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'federated_convergence.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_class_metrics_radar(y_true_disease, y_pred_disease, class_names, output_dir):
    """Radar chart comparing per-class precision, recall, and F1-score."""
    from sklearn.metrics import precision_recall_fscore_support

    os.makedirs(output_dir, exist_ok=True)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_disease, y_pred_disease,
        average=None, labels=list(range(len(class_names))), zero_division=0,
    )

    metrics = [('Precision', precision), ('Recall', recall), ('F1', f1)]
    n = len(class_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#1976D2', '#43A047', '#FB8C00']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'polar': True})

    for ax, (metric_name, values), color in zip(axes, metrics, colors):
        vals = values.tolist() + values[:1].tolist()
        ax.plot(angles, vals, linewidth=2.5, color=color)
        ax.fill(angles, vals, alpha=0.2, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(class_names, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=7)
        ax.set_title(metric_name, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Class Metrics Radar — Disease Classification',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'class_metrics_radar.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
