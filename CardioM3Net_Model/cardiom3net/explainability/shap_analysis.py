"""
SHAP-based clinical feature importance via surrogate model.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier


def run_shap_analysis(train_clinical, train_labels, test_clinical,
                      feature_names, output_dir, max_samples=200):
    """Train RF surrogate on clinical features and compute SHAP values."""
    if not SHAP_AVAILABLE:
        print("  SHAP not installed -- skipping. Install with: pip install shap")
        return

    os.makedirs(output_dir, exist_ok=True)
    print("  Computing SHAP values via RF surrogate...")

    surrogate = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    surrogate.fit(train_clinical, train_labels.astype(int))

    n = min(max_samples, len(test_clinical))
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(test_clinical[:n])

    shap_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, max(5, len(feature_names) * 0.6)))
    shap.summary_plot(shap_to_plot, test_clinical[:n],
                      feature_names=feature_names, plot_type='bar',
                      max_display=len(feature_names), show=False)
    plt.suptitle('Clinical Feature Importance (SHAP)', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'shap_clinical.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Detailed summary plot
    plt.figure(figsize=(10, max(5, len(feature_names) * 0.6)))
    shap.summary_plot(shap_to_plot, test_clinical[:n],
                      feature_names=feature_names,
                      max_display=len(feature_names), show=False)
    plt.suptitle('SHAP Feature Impact (detailed)', fontweight='bold', y=1.02)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'shap_clinical_detail.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")


def plot_modality_weights(modality_weights, output_dir):
    """Visualize average modality contribution weights (bimodal or trimodal)."""
    os.makedirs(output_dir, exist_ok=True)
    avg = modality_weights.mean(axis=0)

    # Derive labels and colours from the actual number of modalities
    n = len(avg)
    _labels = ['ECG', 'Clinical', 'PCG']
    _colors = ['#2196F3', '#FF9800', '#4CAF50']
    labels = _labels[:n]
    colors = _colors[:n]

    plt.figure(figsize=(max(5, n * 2), 4))
    bars = plt.bar(labels, avg, color=colors, edgecolor='black')
    for bar, val in zip(bars, avg):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontweight='bold')
    plt.title('Average Modality Contribution Weights', fontweight='bold')
    plt.ylabel('Weight')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(output_dir, 'modality_weights.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_modality_weight_distribution(modality_weights, labels, class_names, output_dir):
    """Violin plot of per-sample modality gate weights grouped by disease class."""
    os.makedirs(output_dir, exist_ok=True)

    modality_weights = np.asarray(modality_weights)
    labels = np.asarray(labels, dtype=int)
    n_modalities = modality_weights.shape[1]
    _labels = ['ECG', 'Clinical', 'PCG']
    _colors = ['#2196F3', '#FF9800', '#4CAF50']
    modality_labels = _labels[:n_modalities]
    modality_colors = _colors[:n_modalities]

    fig, axes = plt.subplots(1, n_modalities, figsize=(5 * n_modalities, 5),
                              sharey=True)
    if n_modalities == 1:
        axes = [axes]

    for m_idx, (ax, m_label, m_color) in enumerate(
        zip(axes, modality_labels, modality_colors)
    ):
        data_per_class = [
            modality_weights[labels == c, m_idx]
            for c in range(len(class_names))
        ]
        positions = [i for i, d in enumerate(data_per_class) if len(d) > 0]
        data_non_empty = [d for d in data_per_class if len(d) > 0]
        labels_non_empty = [class_names[i] for i in positions]

        if data_non_empty:
            vp = ax.violinplot(data_non_empty, positions=positions,
                               showmedians=True, showextrema=True)
            for body in vp['bodies']:
                body.set_facecolor(m_color)
                body.set_alpha(0.55)
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(2)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_non_empty, rotation=30, ha='right')
        ax.set_title(f'{m_label} Gate Weight', fontweight='bold')
        if m_idx == 0:
            ax.set_ylabel('Weight')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Modality Gate Weight Distribution per Disease Class',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'modality_weight_distribution.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
