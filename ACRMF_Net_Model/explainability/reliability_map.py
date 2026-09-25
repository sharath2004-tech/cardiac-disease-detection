"""
Reliability Map Visualization for ACRMF-Net
Visualizes how reliability scores vary across samples and modalities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("ACRMF-Net")


class ReliabilityMapGenerator:
    """
    Generates reliability maps showing data quality assessment
    across different samples and modalities
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize Reliability Map Generator
        
        Args:
            model: ACRMF-Net model with REN module
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("ReliabilityMapGenerator initialized")
    
    def generate_reliability_map(
        self,
        clinical_data: torch.Tensor,
        ecg_data: torch.Tensor,
        pcg_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Generate reliability map for a batch of samples
        
        Args:
            clinical_data: Clinical features (B, num_features)
            ecg_data: ECG signals (B, 1000)
            pcg_data: PCG signals (B, 1000)
            labels: Optional ground truth labels (B,)
            
        Returns:
            Dictionary with reliability scores and metadata
        """
        with torch.no_grad():
            outputs = self.model(
                clinical_data.to(self.device),
                ecg_data.to(self.device),
                pcg_data.to(self.device)
            )
            
            reliability = outputs.get('reliability', {})
            confidence = outputs.get('confidence', {})
            weights = outputs.get('weights', {})
            
            logits = outputs['fused_logits']
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Move to CPU
            Rc = reliability['Rc'].cpu().numpy() if 'Rc' in reliability else None
            Re = reliability['Re'].cpu().numpy() if 'Re' in reliability else None
            Rp = reliability['Rp'].cpu().numpy() if 'Rp' in reliability else None
            
            Wc = weights['Wc'].cpu().numpy() if 'Wc' in weights else None
            We = weights['We'].cpu().numpy() if 'We' in weights else None
            Wp = weights['Wp'].cpu().numpy() if 'Wp' in weights else None
            
            Cf = confidence['Cf'].cpu().numpy() if 'Cf' in confidence else None
            
            predictions_np = predictions.cpu().numpy()
            probs_np = probs.cpu().numpy()
        
        # Compute correctness if labels provided
        if labels is not None:
            labels_np = labels.cpu().numpy()
            correct = (predictions_np == labels_np).astype(float)
        else:
            correct = None
        
        return {
            'reliability': {
                'clinical': Rc,
                'ecg': Re,
                'pcg': Rp
            },
            'weights': {
                'clinical': Wc,
                'ecg': We,
                'pcg': Wp
            },
            'confidence': Cf,
            'predictions': predictions_np,
            'probabilities': probs_np,
            'correct': correct
        }
    
    def plot_reliability_heatmap(
        self,
        reliability_map: Dict,
        sample_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot reliability heatmap
        
        Args:
            reliability_map: Output from generate_reliability_map
            sample_indices: Indices of samples to plot
            save_path: Path to save plot
        """
        reliability = reliability_map['reliability']
        
        # Stack reliability scores
        Rc = reliability['clinical']
        Re = reliability['ecg']
        Rp = reliability['pcg']
        
        if sample_indices is not None:
            Rc = Rc[sample_indices]
            Re = Re[sample_indices]
            Rp = Rp[sample_indices]
        
        # Create reliability matrix (num_samples x 3)
        rel_matrix = np.hstack([Rc, Re, Rp])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Heatmap
        im = axes[0].imshow(rel_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_yticks([0, 1, 2])
        axes[0].set_yticklabels(['Clinical', 'ECG', 'PCG'])
        axes[0].set_xlabel('Sample Index')
        axes[0].set_title('Reliability Scores Heatmap')
        plt.colorbar(im, ax=axes[0], label='Reliability')
        
        # Plot 2: Violin plot
        data_for_violin = [Rc.flatten(), Re.flatten(), Rp.flatten()]
        positions = [1, 2, 3]
        violin_parts = axes[1].violinplot(
            data_for_violin,
            positions=positions,
            showmeans=True,
            showmedians=True
        )
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(['Clinical', 'ECG', 'PCG'])
        axes[1].set_ylabel('Reliability Score')
        axes[1].set_title('Reliability Distribution')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved reliability heatmap: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_reliability_vs_performance(
        self,
        reliability_map: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot relationship between reliability and prediction performance
        
        Args:
            reliability_map: Output from generate_reliability_map
            save_path: Path to save plot
        """
        if reliability_map['correct'] is None:
            logger.warning("No ground truth labels provided, skipping performance plot")
            return
        
        reliability = reliability_map['reliability']
        correct = reliability_map['correct']
        confidence = reliability_map['confidence']
        
        # Average reliability
        Rc = reliability['clinical']
        Re = reliability['ecg']
        Rp = reliability['pcg']
        avg_reliability = (Rc + Re + Rp) / 3.0
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Reliability vs Correctness
        colors = ['red' if c == 0 else 'green' for c in correct]
        axes[0, 0].scatter(avg_reliability, confidence, c=colors, alpha=0.6, s=50)
        axes[0, 0].set_xlabel('Average Reliability')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_title('Reliability vs Confidence')
        axes[0, 0].legend(['Incorrect', 'Correct'])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Per-modality reliability vs correctness
        axes[0, 1].scatter(Rc, correct, alpha=0.5, label='Clinical', s=30)
        axes[0, 1].scatter(Re, correct + 0.02, alpha=0.5, label='ECG', s=30)
        axes[0, 1].scatter(Rp, correct + 0.04, alpha=0.5, label='PCG', s=30)
        axes[0, 1].set_xlabel('Reliability')
        axes[0, 1].set_ylabel('Correctness (0=wrong, 1=right)')
        axes[0, 1].set_title('Modality Reliability vs Correctness')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reliability distribution for correct vs incorrect
        correct_mask = correct == 1
        incorrect_mask = correct == 0
        
        if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
            axes[1, 0].hist(avg_reliability[correct_mask].flatten(), bins=20, alpha=0.6, 
                           label='Correct', color='green', density=True)
            axes[1, 0].hist(avg_reliability[incorrect_mask].flatten(), bins=20, alpha=0.6, 
                           label='Incorrect', color='red', density=True)
            axes[1, 0].set_xlabel('Average Reliability')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Reliability Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Confidence distribution
        if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
            axes[1, 1].hist(confidence[correct_mask].flatten(), bins=20, alpha=0.6, 
                           label='Correct', color='green', density=True)
            axes[1, 1].hist(confidence[incorrect_mask].flatten(), bins=20, alpha=0.6, 
                           label='Incorrect', color='red', density=True)
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Confidence Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved reliability vs performance plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_weight_evolution(
        self,
        reliability_map: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot how adaptive weights relate to reliability
        
        Args:
            reliability_map: Output from generate_reliability_map
            save_path: Path to save plot
        """
        reliability = reliability_map['reliability']
        weights = reliability_map['weights']
        
        Rc = reliability['clinical'].flatten()
        Re = reliability['ecg'].flatten()
        Rp = reliability['pcg'].flatten()
        
        Wc = weights['clinical'].flatten()
        We = weights['ecg'].flatten()
        Wp = weights['pcg'].flatten()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Clinical
        axes[0].scatter(Rc, Wc, alpha=0.6, s=30)
        axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        axes[0].set_xlabel('Clinical Reliability (Rc)')
        axes[0].set_ylabel('Clinical Weight (Wc)')
        axes[0].set_title('Clinical: Reliability vs Weight')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ECG
        axes[1].scatter(Re, We, alpha=0.6, s=30, color='orange')
        axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        axes[1].set_xlabel('ECG Reliability (Re)')
        axes[1].set_ylabel('ECG Weight (We)')
        axes[1].set_title('ECG: Reliability vs Weight')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # PCG
        axes[2].scatter(Rp, Wp, alpha=0.6, s=30, color='green')
        axes[2].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        axes[2].set_xlabel('PCG Reliability (Rp)')
        axes[2].set_ylabel('PCG Weight (Wp)')
        axes[2].set_title('PCG: Reliability vs Weight')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weight evolution plot: {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Reliability Map Visualization for ACRMF-Net")
    print("="*80 + "\n")
    
    print("Note: This is a placeholder example.")
    print("Actual usage requires a trained ACRMF-Net model.")
    print("\nExample usage:")
    print("""
    from explainability.reliability_map import ReliabilityMapGenerator
    
    # Initialize
    rel_map_gen = ReliabilityMapGenerator(model, device='cuda')
    
    # Generate reliability map
    rel_map = rel_map_gen.generate_reliability_map(
        clinical_data, ecg_data, pcg_data, labels
    )
    
    # Visualize
    rel_map_gen.plot_reliability_heatmap(rel_map, save_path='reliability_heatmap.png')
    rel_map_gen.plot_reliability_vs_performance(rel_map, save_path='reliability_performance.png')
    rel_map_gen.plot_weight_evolution(rel_map, save_path='weight_evolution.png')
    """)
    
    print("\n" + "="*80)
