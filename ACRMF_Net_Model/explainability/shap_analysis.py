"""
SHAP (SHapley Additive exPlanations) Analysis for ACRMF-Net
Provides feature importance and model interpretability
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("ACRMF-Net")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class SHAPAnalyzer:
    """
    SHAP Analysis for ACRMF-Net
    
    Provides model interpretability through SHAP values:
    - Feature importance for clinical data
    - Attention weights analysis
    - Modality contribution
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize SHAP Analyzer
        
        Args:
            model: ACRMF-Net model
            device: Device for computation
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("SHAPAnalyzer initialized")
    
    def explain_clinical_features(
        self,
        clinical_data: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        num_samples: int = 100
    ) -> Dict:
        """
        Compute SHAP values for clinical features
        
        Args:
            clinical_data: Clinical input data (B, num_features)
            feature_names: Names of clinical features
            num_samples: Number of background samples
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        logger.info("Computing SHAP values for clinical features...")
        
        # Define model prediction function
        def predict_fn(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                # Create dummy ECG and PCG
                batch_size = x_tensor.shape[0]
                ecg = torch.zeros(batch_size, 1000).to(self.device)
                pcg = torch.zeros(batch_size, 1000).to(self.device)
                
                outputs = self.model(x_tensor, ecg, pcg)
                logits = outputs['fused_logits']
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
        
        # Select background samples
        background = clinical_data[:min(num_samples, len(clinical_data))].cpu().numpy()
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values
        test_samples = clinical_data[:10].cpu().numpy()  # Explain first 10 samples
        shap_values = explainer.shap_values(test_samples)
        
        # Default feature names if not provided
        if feature_names is None:
            num_features = clinical_data.shape[1]
            feature_names = [f"Feature_{i}" for i in range(num_features)]
        
        return {
            'shap_values': shap_values,
            'base_values': explainer.expected_value,
            'feature_names': feature_names,
            'test_samples': test_samples
        }
    
    def plot_feature_importance(
        self,
        shap_results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance from SHAP values
        
        Args:
            shap_results: Results from explain_clinical_features
            save_path: Path to save plot
        """
        shap_values = shap_results['shap_values']
        feature_names = shap_results['feature_names']
        test_samples = shap_results['test_samples']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Summary plot (bar)
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            mean_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        sorted_idx = np.argsort(mean_shap)
        
        axes[0].barh(range(len(feature_names)), mean_shap[sorted_idx])
        axes[0].set_yticks(range(len(feature_names)))
        axes[0].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[0].set_xlabel('Mean |SHAP value|')
        axes[0].set_title('Feature Importance')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Beeswarm-style scatter
        if isinstance(shap_values, list):
            plot_shap = shap_values[0]  # Use first class
        else:
            plot_shap = shap_values
        
        for i, feat_idx in enumerate(sorted_idx[-10:]):  # Top 10 features
            y_vals = plot_shap[:, feat_idx]
            x_vals = test_samples[:, feat_idx]
            axes[1].scatter(x_vals, [i] * len(x_vals), c=y_vals, 
                          cmap='RdBu_r', alpha=0.7, s=50)
        
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([feature_names[i] for i in sorted_idx[-10:]])
        axes[1].set_xlabel('Feature Value')
        axes[1].set_title('Feature Impact (Top 10)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def explain_single_prediction(
        self,
        clinical: torch.Tensor,
        ecg: torch.Tensor,
        pcg: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Explain a single prediction with waterfall plot
        
        Args:
            clinical: Clinical features (1, num_features)
            ecg: ECG signal (1, 1000)
            pcg: PCG signal (1, 1000)
            feature_names: Feature names
            save_path: Path to save plot
            
        Returns:
            Explanation dictionary
        """
        with torch.no_grad():
            outputs = self.model(
                clinical.to(self.device),
                ecg.to(self.device),
                pcg.to(self.device)
            )
            
            logits = outputs['fused_logits']
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        logger.info(f"Prediction: Class {pred_class}, Probability: {pred_prob:.4f}")
        
        return {
            'predicted_class': pred_class,
            'probability': pred_prob,
            'all_probabilities': probs.cpu().numpy()[0],
            'reliability': outputs.get('reliability', {}),
            'confidence': outputs.get('confidence', {}),
            'weights': outputs.get('weights', {})
        }


class ModaliContributionAnalyzer:
    """
    Analyzes contribution of each modality (Clinical, ECG, PCG)
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def analyze_modality_contributions(
        self,
        clinical: torch.Tensor,
        ecg: torch.Tensor,
        pcg: torch.Tensor
    ) -> Dict:
        """
        Analyze how each modality contributes to final prediction
        
        Args:
            clinical: Clinical data (B, num_features)
            ecg: ECG signal (B, 1000)
            pcg: PCG signal (B, 1000)
            
        Returns:
            Contribution analysis
        """
        with torch.no_grad():
            # Full model prediction
            outputs_full = self.model(
                clinical.to(self.device),
                ecg.to(self.device),
                pcg.to(self.device)
            )
            
            # Get individual modality predictions
            clinical_logits = outputs_full.get('clinical_logits')
            ecg_logits = outputs_full.get('ecg_logits')
            pcg_logits = outputs_full.get('pcg_logits')
            fused_logits = outputs_full['fused_logits']
            
            # Convert to probabilities
            clinical_probs = torch.softmax(clinical_logits, dim=1) if clinical_logits is not None else None
            ecg_probs = torch.softmax(ecg_logits, dim=1) if ecg_logits is not None else None
            pcg_probs = torch.softmax(pcg_logits, dim=1) if pcg_logits is not None else None
            fused_probs = torch.softmax(fused_logits, dim=1)
            
            # Get weights
            weights = outputs_full.get('weights', {})
            
        return {
            'clinical_probs': clinical_probs.cpu().numpy() if clinical_probs is not None else None,
            'ecg_probs': ecg_probs.cpu().numpy() if ecg_probs is not None else None,
            'pcg_probs': pcg_probs.cpu().numpy() if pcg_probs is not None else None,
            'fused_probs': fused_probs.cpu().numpy(),
            'weights': {k: v.cpu().numpy() for k, v in weights.items()},
            'reliability': {k: v.cpu().numpy() for k, v in outputs_full.get('reliability', {}).items()},
            'confidence': {k: v.cpu().numpy() for k, v in outputs_full.get('confidence', {}).items()}
        }
    
    def plot_contributions(
        self,
        analysis: Dict,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot modality contributions
        
        Args:
            analysis: Results from analyze_modality_contributions
            class_names: Names of disease classes
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        num_classes = analysis['fused_probs'].shape[1]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        # Plot 1: Probability comparison
        x = np.arange(num_classes)
        width = 0.2
        
        if analysis['clinical_probs'] is not None:
            axes[0, 0].bar(x - width, analysis['clinical_probs'][0], width, label='Clinical', alpha=0.8)
        if analysis['ecg_probs'] is not None:
            axes[0, 0].bar(x, analysis['ecg_probs'][0], width, label='ECG', alpha=0.8)
        if analysis['pcg_probs'] is not None:
            axes[0, 0].bar(x + width, analysis['pcg_probs'][0], width, label='PCG', alpha=0.8)
        
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].set_title('Modality Predictions')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Fusion weights
        weights = analysis['weights']
        if weights:
            weight_names = ['Clinical', 'ECG', 'PCG']
            weight_values = [
                weights['Wc'][0, 0],
                weights['We'][0, 0],
                weights['Wp'][0, 0]
            ]
            axes[0, 1].bar(weight_names, weight_values, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[0, 1].set_ylabel('Weight')
            axes[0, 1].set_title('Fusion Weights')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reliability scores
        reliability = analysis['reliability']
        if reliability:
            rel_names = ['Clinical', 'ECG', 'PCG']
            rel_values = [
                reliability['Rc'][0, 0],
                reliability['Re'][0, 0],
                reliability['Rp'][0, 0]
            ]
            axes[1, 0].bar(rel_names, rel_values, alpha=0.8, color=['#d62728', '#9467bd', '#8c564b'])
            axes[1, 0].set_ylabel('Reliability')
            axes[1, 0].set_title('Reliability Scores (REN)')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Final prediction
        axes[1, 1].bar(x, analysis['fused_probs'][0], alpha=0.8, color='green')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_title('Final Fused Prediction')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(class_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved contribution plot: {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SHAP Analysis for ACRMF-Net")
    print("="*80 + "\n")
    
    print("Note: This is a placeholder example.")
    print("Actual usage requires a trained ACRMF-Net model.")
    print("\nExample usage:")
    print("""
    from explainability.shap_analysis import SHAPAnalyzer
    
    # Initialize analyzer
    analyzer = SHAPAnalyzer(model, device='cuda')
    
    # Explain clinical features
    shap_results = analyzer.explain_clinical_features(
        clinical_data, 
        feature_names=['Age', 'BP', 'Cholesterol', ...]
    )
    
    # Plot feature importance
    analyzer.plot_feature_importance(shap_results, save_path='shap_importance.png')
    """)
    
    print("\n" + "="*80)
