"""
Integrated Gradients for ACRMF-Net
Provides attribution analysis for deep learning models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("ACRMF-Net")


class IntegratedGradients:
    """
    Integrated Gradients attribution method
    
    Computes feature attributions by integrating gradients along
    a path from a baseline to the input.
    
    Reference: Sundararajan et al. "Axiomatic Attribution for Deep Networks" (2017)
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize Integrated Gradients
        
        Args:
            model: ACRMF-Net model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("IntegratedGradients initialized")
    
    def compute_gradients(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gradients with respect to inputs
        
        Args:
            inputs: Tuple of (clinical, ecg, pcg)
            target_class: Target class for attribution
            
        Returns:
            Tuple of gradients for each modality
        """
        clinical, ecg, pcg = inputs
        
        # Ensure requires_grad
        clinical = clinical.clone().requires_grad_(True)
        ecg = ecg.clone().requires_grad_(True)
        pcg = pcg.clone().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(clinical, ecg, pcg)
        logits = outputs['fused_logits']
        
        # Get target class score
        target_score = logits[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        return clinical.grad, ecg.grad, pcg.grad
    
    def integrated_gradients(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_class: int,
        baselines: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        num_steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients
        
        Args:
            inputs: Tuple of (clinical, ecg, pcg)
            target_class: Target class for attribution
            baselines: Baseline inputs (typically zeros)
            num_steps: Number of integration steps
            
        Returns:
            Dictionary with attributions for each modality
        """
        clinical, ecg, pcg = inputs
        
        # Create baselines if not provided
        if baselines is None:
            baselines = (
                torch.zeros_like(clinical),
                torch.zeros_like(ecg),
                torch.zeros_like(pcg)
            )
        
        baseline_clinical, baseline_ecg, baseline_pcg = baselines
        
        # Initialize attribution accumulator
        clinical_attr = torch.zeros_like(clinical)
        ecg_attr = torch.zeros_like(ecg)
        pcg_attr = torch.zeros_like(pcg)
        
        # Compute gradients along the path
        for step in range(num_steps):
            alpha = (step + 1) / num_steps
            
            # Interpolate between baseline and input
            interp_clinical = baseline_clinical + alpha * (clinical - baseline_clinical)
            interp_ecg = baseline_ecg + alpha * (ecg - baseline_ecg)
            interp_pcg = baseline_pcg + alpha * (pcg - baseline_pcg)
            
            # Move to device
            interp_clinical = interp_clinical.to(self.device)
            interp_ecg = interp_ecg.to(self.device)
            interp_pcg = interp_pcg.to(self.device)
            
            # Compute gradients
            grads = self.compute_gradients(
                (interp_clinical, interp_ecg, interp_pcg),
                target_class
            )
            
            # Accumulate gradients
            clinical_attr += grads[0].cpu()
            ecg_attr += grads[1].cpu()
            pcg_attr += grads[2].cpu()
        
        # Average gradients and multiply by input - baseline
        clinical_attr = (clinical - baseline_clinical) * clinical_attr / num_steps
        ecg_attr = (ecg - baseline_ecg) * ecg_attr / num_steps
        pcg_attr = (pcg - baseline_pcg) * pcg_attr / num_steps
        
        logger.info(f"Computed integrated gradients for class {target_class}")
        
        return {
            'clinical': clinical_attr,
            'ecg': ecg_attr,
            'pcg': pcg_attr
        }
    
    def visualize_attributions(
        self,
        attributions: Dict[str, torch.Tensor],
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        feature_names: Optional[list] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize integrated gradients attributions
        
        Args:
            attributions: Attribution dictionary from integrated_gradients
            inputs: Original inputs
            feature_names: Names of clinical features
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Clinical attributions
        clinical_attr = attributions['clinical'][0].numpy()
        
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(len(clinical_attr))]
        
        sorted_idx = np.argsort(np.abs(clinical_attr))
        axes[0, 0].barh(range(len(clinical_attr)), clinical_attr[sorted_idx])
        axes[0, 0].set_yticks(range(len(clinical_attr)))
        axes[0, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[0, 0].set_xlabel('Attribution')
        axes[0, 0].set_title('Clinical Feature Attributions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ECG attributions
        ecg_attr = attributions['ecg'][0].numpy()
        axes[0, 1].plot(ecg_attr, linewidth=0.5, alpha=0.7)
        axes[0, 1].fill_between(range(len(ecg_attr)), 0, ecg_attr, alpha=0.3)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Attribution')
        axes[0, 1].set_title('ECG Signal Attributions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # PCG attributions
        pcg_attr = attributions['pcg'][0].numpy()
        axes[1, 0].plot(pcg_attr, linewidth=0.5, alpha=0.7, color='green')
        axes[1, 0].fill_between(range(len(pcg_attr)), 0, pcg_attr, alpha=0.3, color='green')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Attribution')
        axes[1, 0].set_title('PCG Signal Attributions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Attribution magnitude comparison
        modalities = ['Clinical', 'ECG', 'PCG']
        magnitudes = [
            np.abs(clinical_attr).sum(),
            np.abs(ecg_attr).sum(),
            np.abs(pcg_attr).sum()
        ]
        axes[1, 1].bar(modalities, magnitudes, alpha=0.8, color=['blue', 'orange', 'green'])
        axes[1, 1].set_ylabel('Total Attribution Magnitude')
        axes[1, 1].set_title('Modality Contribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attribution visualization: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def explain_prediction(
        self,
        clinical: torch.Tensor,
        ecg: torch.Tensor,
        pcg: torch.Tensor,
        target_class: Optional[int] = None,
        num_steps: int = 50
    ) -> Dict:
        """
        Complete explanation of a single prediction
        
        Args:
            clinical: Clinical features (1, num_features)
            ecg: ECG signal (1, 1000)
            pcg: PCG signal (1, 1000)
            target_class: Target class (if None, use predicted class)
            num_steps: Number of integration steps
            
        Returns:
            Complete explanation dictionary
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
        
        if target_class is None:
            target_class = pred_class
        
        logger.info(f"Explaining prediction for class {target_class}")
        
        # Compute integrated gradients
        attributions = self.integrated_gradients(
            (clinical, ecg, pcg),
            target_class,
            num_steps=num_steps
        )
        
        return {
            'predicted_class': pred_class,
            'target_class': target_class,
            'probability': pred_prob,
            'all_probabilities': probs.cpu().numpy()[0],
            'attributions': attributions,
            'reliability': {k: v.cpu().numpy() for k, v in outputs.get('reliability', {}).items()},
            'confidence': {k: v.cpu().numpy() for k, v in outputs.get('confidence', {}).items()},
            'weights': {k: v.cpu().numpy() for k, v in outputs.get('weights', {}).items()}
        }


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Integrated Gradients for ACRMF-Net")
    print("="*80 + "\n")
    
    print("Note: This is a placeholder example.")
    print("Actual usage requires a trained ACRMF-Net model.")
    print("\nExample usage:")
    print("""
    from explainability.integrated_gradients import IntegratedGradients
    
    # Initialize
    ig = IntegratedGradients(model, device='cuda')
    
    # Explain a prediction
    explanation = ig.explain_prediction(
        clinical, ecg, pcg,
        target_class=0,
        num_steps=50
    )
    
    # Visualize
    ig.visualize_attributions(
        explanation['attributions'],
        (clinical, ecg, pcg),
        feature_names=['Age', 'BP', ...],
        save_path='ig_attribution.png'
    )
    """)
    
    print("\n" + "="*80)
