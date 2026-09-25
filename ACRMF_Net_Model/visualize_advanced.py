"""
Advanced Model Visualization Suite
===================================
Comprehensive 3D and interactive visualizations for trained ACRMF-Net model.

Visualizations:
1. 3D Feature Space (t-SNE, PCA, UMAP)
2. 3D Decision Boundaries
3. Confusion Matrix (Enhanced)
4. Per-Class Performance Radar Chart
5. Learning Curves (Loss, Accuracy, F1)
6. Attention Heatmaps
7. Class Separation Metrics
8. Prediction Confidence Distribution
9. ROC Curves (Multi-class)
10. Feature Importance
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP (optional)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from test_complete_model import ACRMFNet

print("\n" + "="*80)
print("ADVANCED MODEL VISUALIZATION SUITE")
print("="*80)


def load_model_and_data(checkpoint_path='checkpoints/best_model.pth'):
    """Load trained model and data"""
    print("\nLoading model and data...")
    
    # Check if cleaned data exists
    cleaned_data_dir = Path('cleaned_data')
    
    if not cleaned_data_dir.exists():
        print("\n" + "="*80)
        print("ERROR: Cleaned data not found!")
        print("="*80)
        print("\nThe visualization script requires cleaned data.")
        print("\nPlease run data cleaning first:")
        print("  py data_cleaning_pipeline.py")
        print("\nThis will create the cleaned_data/ directory with:")
        print("  - clinical_cleaned.npy")
        print("  - ecg_cleaned.npy")
        print("  - pcg_cleaned.npy")
        print("  - labels_cleaned.npy")
        print("  - metadata.json")
        print("\nAfter cleaning, you can run this visualization script.")
        print("="*80)
        sys.exit(1)
    
    # Load data
    print("Loading dataset...")
    try:
        clinical_data = np.load('cleaned_data/clinical_cleaned.npy')
        ecg_data = np.load('cleaned_data/ecg_cleaned.npy')
        pcg_data = np.load('cleaned_data/pcg_cleaned.npy')
        labels = np.load('cleaned_data/labels_cleaned.npy')
        
        with open('cleaned_data/metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        print(f"\n❌ Error: Missing file - {e}")
        print("\nPlease ensure all cleaned data files exist:")
        print("  py data_cleaning_pipeline.py")
        sys.exit(1)
    
    print(f"[OK] Loaded {len(labels)} samples")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = metadata['num_classes']
    
    # ACRMFNet has fixed architecture, only num_classes is configurable
    model = ACRMFNet(
        num_classes=num_classes,
        embedding_dim=128,
        use_clinical_gating=True,
        use_cross_attention=False
    ).to(device)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Loaded model from {checkpoint_path}")
        print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print(f"[WARN]  No checkpoint found at {checkpoint_path}")
        print("Using randomly initialized model for visualization")
    
    model.eval()
    
    return model, clinical_data, ecg_data, pcg_data, labels, metadata, device


def extract_features(model, clinical_data, ecg_data, pcg_data, device, batch_size=64):
    """Extract features and predictions from model"""
    print("\nExtracting features from model...")
    
    model.eval()
    
    clinical_tensor = torch.FloatTensor(clinical_data).to(device)
    ecg_tensor = torch.FloatTensor(ecg_data).to(device)
    pcg_tensor = torch.FloatTensor(pcg_data).to(device)
    
    all_features = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(clinical_data), batch_size)):
            batch_clinical = clinical_tensor[i:i+batch_size]
            batch_ecg = ecg_tensor[i:i+batch_size]
            batch_pcg = pcg_tensor[i:i+batch_size]
            
            # Get outputs
            outputs = model(batch_clinical, batch_ecg, batch_pcg)
            
            # Try to get intermediate features
            try:
                # This assumes model has a method to get features
                # Modify based on your actual model architecture
                features = model.get_features(batch_clinical, batch_ecg, batch_pcg)
            except:
                # Fallback: use output logits as features
                features = outputs
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_features.append(features.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())
    
    features = np.vstack(all_features)
    predictions = np.concatenate(all_predictions)
    probabilities = np.vstack(all_probabilities)
    
    print(f"[OK] Extracted features: {features.shape}")
    
    return features, predictions, probabilities


def plot_3d_feature_space(features, labels, predictions, method='tsne', save_path='viz_3d_features.png'):
    """3D visualization of feature space"""
    print(f"\nGenerating 3D feature space visualization ({method.upper()})...")
    
    # Reduce to 3D
    if method == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        title = 't-SNE 3D Feature Space'
    elif method == 'pca':
        reducer = PCA(n_components=3, random_state=42)
        title = 'PCA 3D Feature Space'
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = UMAP(n_components=3, random_state=42)
        title = 'UMAP 3D Feature Space'
    else:
        print(f"Method {method} not available, using PCA")
        reducer = PCA(n_components=3, random_state=42)
        title = 'PCA 3D Feature Space'
    
    features_3d = reducer.fit_transform(features)
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Colored by True Labels
    ax1 = fig.add_subplot(221, projection='3d')
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors[i]], label=f'Class {label}', alpha=0.6, s=30)
    
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_zlabel('Component 3')
    ax1.set_title(f'{title} - True Labels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Colored by Predictions
    ax2 = fig.add_subplot(222, projection='3d')
    for i, label in enumerate(unique_labels):
        mask = predictions == label
        ax2.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors[i]], label=f'Pred {label}', alpha=0.6, s=30)
    
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')
    ax2.set_title(f'{title} - Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correct vs Incorrect
    ax3 = fig.add_subplot(223, projection='3d')
    correct = labels == predictions
    ax3.scatter(features_3d[correct, 0], features_3d[correct, 1], features_3d[correct, 2],
               c='green', label='Correct', alpha=0.6, s=30)
    ax3.scatter(features_3d[~correct, 0], features_3d[~correct, 1], features_3d[~correct, 2],
               c='red', label='Incorrect', alpha=0.8, s=50, marker='x')
    
    ax3.set_xlabel('Component 1')
    ax3.set_ylabel('Component 2')
    ax3.set_zlabel('Component 3')
    ax3.set_title('Correct vs Incorrect Predictions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Class Centroids
    ax4 = fig.add_subplot(224, projection='3d')
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroid = features_3d[mask].mean(axis=0)
        
        # Plot samples
        ax4.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors[i]], alpha=0.3, s=20)
        
        # Plot centroid
        ax4.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                   c=[colors[i]], s=200, marker='*', 
                   edgecolors='black', linewidths=2, label=f'Class {label} Center')
    
    ax4.set_xlabel('Component 1')
    ax4.set_ylabel('Component 2')
    ax4.set_zlabel('Component 3')
    ax4.set_title('Feature Space with Class Centroids')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_confusion_matrix_enhanced(labels, predictions, num_classes, save_path='viz_confusion_matrix.png'):
    """Enhanced confusion matrix visualization"""
    print("\nGenerating enhanced confusion matrix...")
    
    cm = confusion_matrix(labels, predictions)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Raw counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - Counts')
    
    # Plot 2: Normalized by true labels (recall)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax,
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - Recall (Row-wise %)')
    
    # Plot 3: Normalized by predictions (precision)
    cm_normalized_pred = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    ax = axes[2]
    sns.heatmap(cm_normalized_pred, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax,
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - Precision (Column-wise %)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_performance_radar(labels, predictions, probabilities, num_classes, save_path='viz_radar_chart.png'):
    """Radar chart for per-class performance metrics"""
    print("\nGenerating performance radar chart...")
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate metrics per class
    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'Confidence': []
    }
    
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() > 0:
            # Accuracy
            acc = (predictions[mask] == cls).sum() / mask.sum()
            metrics['Accuracy'].append(acc)
            
            # Precision, Recall, F1
            prec = precision_score(labels == cls, predictions == cls, zero_division=0)
            rec = recall_score(labels == cls, predictions == cls, zero_division=0)
            f1 = f1_score(labels == cls, predictions == cls, zero_division=0)
            metrics['Precision'].append(prec)
            metrics['Recall'].append(rec)
            metrics['F1-Score'].append(f1)
            
            # Average confidence for this class
            conf = probabilities[mask, cls].mean()
            metrics['Confidence'].append(conf)
    
    # Create radar chart for each class
    fig, axes = plt.subplots(1, num_classes, figsize=(6*num_classes, 6), 
                             subplot_kw=dict(projection='polar'))
    
    if num_classes == 1:
        axes = [axes]
    
    categories = list(metrics.keys())
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for cls_idx in range(num_classes):
        ax = axes[cls_idx]
        
        values = [metrics[cat][cls_idx] for cat in categories]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Class {cls_idx}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_title(f'Class {cls_idx} Performance\nAcc: {metrics["Accuracy"][cls_idx]*100:.1f}%', 
                     fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)
        
        # Add threshold line at 80%
        ax.plot(angles, [0.8]*len(angles), 'r--', linewidth=1, alpha=0.5, label='80% Target')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_confidence_distribution(probabilities, labels, predictions, num_classes, save_path='viz_confidence.png'):
    """Plot prediction confidence distribution"""
    print("\nGenerating confidence distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall confidence distribution
    ax = axes[0, 0]
    max_probs = probabilities.max(axis=1)
    correct = labels == predictions
    
    ax.hist(max_probs[correct], bins=50, alpha=0.6, label='Correct', color='green', density=True)
    ax.hist(max_probs[~correct], bins=50, alpha=0.6, label='Incorrect', color='red', density=True)
    ax.axvline(x=0.8, color='black', linestyle='--', label='80% Threshold')
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Overall Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Confidence by class
    ax = axes[0, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for cls in range(num_classes):
        mask = labels == cls
        class_confidences = probabilities[mask, cls]
        ax.hist(class_confidences, bins=30, alpha=0.5, label=f'Class {cls}', 
                color=colors[cls], density=True)
    
    ax.axvline(x=0.8, color='black', linestyle='--', label='80% Target')
    ax.set_xlabel('Confidence for True Class')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution per Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Confidence vs Accuracy
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins)-1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
        if mask.sum() > 0:
            bin_acc = (labels[mask] == predictions[mask]).mean()
            bin_accuracies.append(bin_acc)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='steelblue')
    ax.plot(bin_centers, bin_centers, 'r--', label='Perfect Calibration')
    ax.set_xlabel('Predicted Confidence')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title('Confidence Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add sample counts as text
    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        if count > 0:
            ax.text(center, bin_accuracies[i] + 0.02, str(count), 
                   ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Per-class confidence box plot
    ax = axes[1, 1]
    confidence_data = []
    labels_list = []
    
    for cls in range(num_classes):
        mask = labels == cls
        class_confidences = probabilities[mask, cls]
        confidence_data.append(class_confidences)
        labels_list.append(f'Class {cls}')
    
    bp = ax.boxplot(confidence_data, labels=labels_list, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(y=0.8, color='red', linestyle='--', label='80% Target')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Distribution per Class (Box Plot)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_roc_curves(labels, probabilities, num_classes, save_path='viz_roc_curves.png'):
    """Plot ROC curves for multi-class classification"""
    print("\nGenerating ROC curves...")
    
    # Binarize labels for multi-class ROC
    labels_bin = label_binarize(labels, classes=range(num_classes))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All ROC curves
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for cls in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, cls], probabilities[:, cls])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[cls], lw=2, 
               label=f'Class {cls} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Classes')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Plot 2: Per-class AUC comparison
    ax = axes[1]
    aucs = []
    for cls in range(num_classes):
        try:
            auc_score = roc_auc_score(labels_bin[:, cls], probabilities[:, cls])
            aucs.append(auc_score)
        except:
            aucs.append(0)
    
    bars = ax.bar(range(num_classes), aucs, color=colors, alpha=0.7)
    ax.axhline(y=0.9, color='green', linestyle='--', label='90% Target')
    ax.axhline(y=0.8, color='orange', linestyle='--', label='80% Threshold')
    ax.set_xlabel('Class')
    ax.set_ylabel('AUC Score')
    ax.set_title('Area Under ROC Curve per Class')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_class_separation(features, labels, num_classes, save_path='viz_class_separation.png'):
    """Analyze and visualize class separation"""
    print("\nAnalyzing class separation...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Inter-class distances
    ax = axes[0, 0]
    centroids = []
    for cls in range(num_classes):
        mask = labels == cls
        centroid = features[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(centroids, metric='euclidean'))
    
    im = ax.imshow(distances, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    ax.set_title('Inter-Class Centroid Distances')
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, f'{distances[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Plot 2: Intra-class variance
    ax = axes[0, 1]
    variances = []
    for cls in range(num_classes):
        mask = labels == cls
        variance = features[mask].var(axis=0).mean()
        variances.append(variance)
    
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    bars = ax.bar(range(num_classes), variances, color=colors, alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Average Feature Variance')
    ax.set_title('Intra-Class Variance (Lower = More Compact)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Silhouette-like metric
    ax = axes[1, 0]
    
    # Calculate separation metric for each class
    separation_scores = []
    for cls in range(num_classes):
        mask = labels == cls
        class_features = features[mask]
        other_features = features[~mask]
        
        if len(class_features) > 0 and len(other_features) > 0:
            # Average distance to own class centroid
            intra_dist = np.linalg.norm(class_features - class_features.mean(axis=0), axis=1).mean()
            
            # Average distance to nearest other class centroid
            other_centroids = [centroids[i] for i in range(num_classes) if i != cls]
            inter_dists = [np.linalg.norm(class_features - cent, axis=1).mean() 
                          for cent in other_centroids]
            inter_dist = min(inter_dists) if inter_dists else 0
            
            # Separation score (higher is better)
            if intra_dist > 0:
                sep_score = (inter_dist - intra_dist) / max(inter_dist, intra_dist)
            else:
                sep_score = 1.0
            
            separation_scores.append(sep_score)
        else:
            separation_scores.append(0)
    
    bars = ax.bar(range(num_classes), separation_scores, color=colors, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', label='Poor Separation')
    ax.set_xlabel('Class')
    ax.set_ylabel('Separation Score')
    ax.set_title('Class Separation Quality (Higher = Better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, separation_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 4: 2D PCA projection with centroids
    ax = axes[1, 1]
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    centroids_2d = pca.transform(centroids)
    
    for cls in range(num_classes):
        mask = labels == cls
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  c=[colors[cls]], alpha=0.4, s=20, label=f'Class {cls}')
        ax.scatter(centroids_2d[cls, 0], centroids_2d[cls, 1],
                  c=[colors[cls]], s=300, marker='*', 
                  edgecolors='black', linewidths=2)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('2D PCA Projection with Centroids')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def create_summary_dashboard(labels, predictions, probabilities, num_classes, save_path='viz_summary_dashboard.png'):
    """Create comprehensive summary dashboard"""
    print("\nCreating summary dashboard...")
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    overall_acc = accuracy_score(labels, predictions)
    overall_f1 = f1_score(labels, predictions, average='macro')
    
    # Title
    fig.suptitle(f'Model Performance Dashboard - Overall Accuracy: {overall_acc*100:.2f}%',
                fontsize=20, fontweight='bold')
    
    # 1. Per-class accuracy bar chart
    ax = fig.add_subplot(gs[0, 0])
    per_class_acc = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() > 0:
            acc = (predictions[mask] == cls).sum() / mask.sum()
            per_class_acc.append(acc * 100)
        else:
            per_class_acc.append(0)
    
    colors = ['green' if acc >= 80 else 'orange' if acc >= 70 else 'red' 
              for acc in per_class_acc]
    bars = ax.bar(range(num_classes), per_class_acc, color=colors, alpha=0.7)
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Target')
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% Target')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Class distribution
    ax = fig.add_subplot(gs[0, 1])
    unique, counts = np.unique(labels, return_counts=True)
    colors_dist = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    ax.bar(unique, counts, color=colors_dist, alpha=0.7)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentages
    total = len(labels)
    for cls, count in zip(unique, counts):
        pct = count / total * 100
        ax.text(cls, count + max(counts)*0.02, f'{pct:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Confusion matrix (compact)
    ax = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.0%', cmap='RdYlGn', ax=ax,
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Recall)', fontsize=14, fontweight='bold')
    
    # 4. Metrics comparison
    ax = fig.add_subplot(gs[1, :])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = []
    
    for cls in range(num_classes):
        cls_metrics = []
        mask = labels == cls
        if mask.sum() > 0:
            acc = (predictions[mask] == cls).sum() / mask.sum()
            prec = precision_score(labels == cls, predictions == cls, zero_division=0)
            rec = recall_score(labels == cls, predictions == cls, zero_division=0)
            f1 = f1_score(labels == cls, predictions == cls, zero_division=0)
            cls_metrics = [acc, prec, rec, f1]
        else:
            cls_metrics = [0, 0, 0, 0]
        metrics_values.append(cls_metrics)
    
    x = np.arange(len(metrics_names))
    width = 0.15
    colors_met = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        offset = width * (i - num_classes/2)
        ax.bar(x + offset, [m*100 for m in metrics_values[i]], width,
               label=f'Class {i}', color=colors_met[i], alpha=0.7)
    
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(ncol=num_classes)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Statistics table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    stats_text = f"""
    OVERALL STATISTICS
    
    Total Samples: {len(labels)}
    Number of Classes: {num_classes}
    
    Overall Accuracy: {overall_acc*100:.2f}%
    Overall Macro F1: {overall_f1:.4f}
    Overall Precision: {precision_score(labels, predictions, average='macro'):.4f}
    Overall Recall: {recall_score(labels, predictions, average='macro'):.4f}
    
    Correct Predictions: {(labels == predictions).sum()} ({(labels == predictions).mean()*100:.2f}%)
    Incorrect Predictions: {(labels != predictions).sum()} ({(labels != predictions).mean()*100:.2f}%)
    
    Classes meeting 80% threshold: {sum(1 for acc in per_class_acc if acc >= 80)} / {num_classes}
    Classes meeting 90% threshold: {sum(1 for acc in per_class_acc if acc >= 90)} / {num_classes}
    
    Average Confidence: {probabilities.max(axis=1).mean():.4f}
    """
    
    ax.text(0.05, 0.5, stats_text, fontsize=14, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def main():
    """Main visualization pipeline"""
    
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}/")
    
    # Load model and data
    model, clinical_data, ecg_data, pcg_data, labels, metadata, device = load_model_and_data()
    
    num_classes = metadata['num_classes']
    
    # Extract features and predictions
    features, predictions, probabilities = extract_features(
        model, clinical_data, ecg_data, pcg_data, device
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. 3D Feature Space (t-SNE)
    plot_3d_feature_space(features, labels, predictions, method='tsne',
                         save_path=output_dir / 'viz_3d_tsne.png')
    
    # 2. 3D Feature Space (PCA)
    plot_3d_feature_space(features, labels, predictions, method='pca',
                         save_path=output_dir / 'viz_3d_pca.png')
    
    # 3. 3D Feature Space (UMAP) if available
    if UMAP_AVAILABLE:
        plot_3d_feature_space(features, labels, predictions, method='umap',
                             save_path=output_dir / 'viz_3d_umap.png')
    
    # 4. Enhanced Confusion Matrix
    plot_confusion_matrix_enhanced(labels, predictions, num_classes,
                                  save_path=output_dir / 'viz_confusion_matrix.png')
    
    # 5. Performance Radar Chart
    plot_performance_radar(labels, predictions, probabilities, num_classes,
                          save_path=output_dir / 'viz_radar_chart.png')
    
    # 6. Confidence Distribution
    plot_confidence_distribution(probabilities, labels, predictions, num_classes,
                                save_path=output_dir / 'viz_confidence.png')
    
    # 7. ROC Curves
    plot_roc_curves(labels, probabilities, num_classes,
                   save_path=output_dir / 'viz_roc_curves.png')
    
    # 8. Class Separation Analysis
    plot_class_separation(features, labels, num_classes,
                         save_path=output_dir / 'viz_class_separation.png')
    
    # 9. Summary Dashboard
    create_summary_dashboard(labels, predictions, probabilities, num_classes,
                            save_path=output_dir / 'viz_summary_dashboard.png')
    
    # Final summary
    print("\n" + "="*80)
    print("[OK] VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated visualizations in: {output_dir}/")
    print("\nFiles created:")
    for viz_file in sorted(output_dir.glob('viz_*.png')):
        print(f"   {viz_file.name}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
