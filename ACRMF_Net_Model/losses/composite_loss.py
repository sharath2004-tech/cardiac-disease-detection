"""
Composite Loss Function for ACRMF-Net
Combines multiple loss components for end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("ACRMF-Net")


class CompositeLoss(nn.Module):
    """
    Composite Loss Function for ACRMF-Net
    
    Combines multiple loss components:
    1. Classification Loss (Cross-Entropy)
    2. Reliability Loss (encourages meaningful reliability scores)
    3. Confidence Loss (calibration loss)
    4. Fusion Loss (encourages effective weight adaptation)
    5. Consistency Loss (agreement between modalities)
    
    Total Loss = λ1*L_cls + λ2*L_rel + λ3*L_conf + λ4*L_fusion + λ5*L_consist
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        lambda_cls: float = 1.0,
        lambda_rel: float = 0.1,
        lambda_conf: float = 0.1,
        lambda_fusion: float = 0.05,
        lambda_consist: float = 0.1,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize Composite Loss
        
        Args:
            num_classes: Number of disease classes
            lambda_cls: Weight for classification loss
            lambda_rel: Weight for reliability loss
            lambda_conf: Weight for confidence loss
            lambda_fusion: Weight for fusion loss
            lambda_consist: Weight for consistency loss
            label_smoothing: Label smoothing factor
            class_weights: Optional class weights for imbalanced data
        """
        super(CompositeLoss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_rel = lambda_rel
        self.lambda_conf = lambda_conf
        self.lambda_fusion = lambda_fusion
        self.lambda_consist = lambda_consist
        self.label_smoothing = label_smoothing
        
        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        logger.info(f"CompositeLoss initialized")
        logger.info(f"  λ_cls={lambda_cls}, λ_rel={lambda_rel}, λ_conf={lambda_conf}")
        logger.info(f"  λ_fusion={lambda_fusion}, λ_consist={lambda_consist}")
    
    def classification_loss(
        self,
        fused_logits: torch.Tensor,
        labels: torch.Tensor,
        clinical_logits: Optional[torch.Tensor] = None,
        ecg_logits: Optional[torch.Tensor] = None,
        pcg_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute classification loss
        
        Args:
            fused_logits: Final fused predictions (B, num_classes)
            labels: Ground truth labels (B,)
            clinical_logits: Clinical predictions (optional)
            ecg_logits: ECG predictions (optional)
            pcg_logits: PCG predictions (optional)
            
        Returns:
            Classification loss
        """
        # Main loss on fused predictions
        loss = self.ce_loss(fused_logits, labels)
        
        # Optional: Add auxiliary losses for individual modalities
        if clinical_logits is not None:
            loss += 0.3 * self.ce_loss(clinical_logits, labels)
        if ecg_logits is not None:
            loss += 0.3 * self.ce_loss(ecg_logits, labels)
        if pcg_logits is not None:
            loss += 0.3 * self.ce_loss(pcg_logits, labels)
        
        return loss
    
    def reliability_loss(
        self,
        reliability: Dict[str, torch.Tensor],
        fused_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Reliability loss: Encourages higher reliability for correct predictions
        
        Args:
            reliability: Dict with Rc, Re, Rp
            fused_logits: Predictions (B, num_classes)
            labels: Ground truth (B,)
            
        Returns:
            Reliability loss
        """
        # Get predictions
        predictions = torch.argmax(fused_logits, dim=1)
        correct = (predictions == labels).float().unsqueeze(1)  # (B, 1)
        
        # Extract reliability scores
        Rc = reliability['Rc']
        Re = reliability['Re']
        Rp = reliability['Rp']
        
        # Clamp reliability values to [eps, 1-eps] for numerical stability
        eps = 1e-7
        Rc = torch.clamp(Rc, min=eps, max=1.0-eps)
        Re = torch.clamp(Re, min=eps, max=1.0-eps)
        Rp = torch.clamp(Rp, min=eps, max=1.0-eps)
        
        # Average reliability
        avg_reliability = (Rc + Re + Rp) / 3.0
        
        # Loss: Encourage high reliability for correct predictions,
        # low reliability for incorrect predictions
        loss = F.binary_cross_entropy(
            avg_reliability,
            correct,
            reduction='mean'
        )
        
        return loss
    
    def confidence_loss(
        self,
        confidence: Dict[str, torch.Tensor],
        fused_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Confidence calibration loss
        
        Args:
            confidence: Dict with Cc, Ce, Cp, Cf
            fused_logits: Predictions (B, num_classes)
            labels: Ground truth (B,)
            
        Returns:
            Confidence loss
        """
        # Get prediction probabilities
        probs = F.softmax(fused_logits, dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        max_probs = max_probs.unsqueeze(1)  # (B, 1)
        
        # Correctness indicator
        correct = (predictions == labels).float().unsqueeze(1)
        
        # Fused confidence
        Cf = confidence['Cf']
        
        # Clamp confidence to [eps, 1-eps] for numerical stability
        eps = 1e-7
        Cf = torch.clamp(Cf, min=eps, max=1.0-eps)
        
        # Loss 1: Confidence should match prediction probability
        prob_loss = F.mse_loss(Cf, max_probs)
        
        # Loss 2: High confidence for correct, low for incorrect
        calibration_loss = F.binary_cross_entropy(Cf, correct)
        
        loss = 0.5 * prob_loss + 0.5 * calibration_loss
        
        return loss
    
    def fusion_loss(
        self,
        weights: Dict[str, torch.Tensor],
        reliability: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fusion loss: Encourages weights to align with reliability
        
        Args:
            weights: Dict with Wc, We, Wp
            reliability: Dict with Rc, Re, Rp
            
        Returns:
            Fusion loss
        """
        # Extract weights and reliability
        Wc, We, Wp = weights['Wc'], weights['We'], weights['Wp']
        Rc, Re, Rp = reliability['Rc'], reliability['Re'], reliability['Rp']
        
        # Clamp reliability values to [eps, 1-eps] for numerical stability
        eps = 1e-7
        Rc = torch.clamp(Rc, min=eps, max=1.0-eps)
        Re = torch.clamp(Re, min=eps, max=1.0-eps)
        Rp = torch.clamp(Rp, min=eps, max=1.0-eps)
        
        # Normalize reliability to sum to 1
        total_rel = Rc + Re + Rp + 1e-8
        Rc_norm = Rc / total_rel
        Re_norm = Re / total_rel
        Rp_norm = Rp / total_rel
        
        # MSE between weights and normalized reliability
        loss = (
            F.mse_loss(Wc, Rc_norm) +
            F.mse_loss(We, Re_norm) +
            F.mse_loss(Wp, Rp_norm)
        ) / 3.0
        
        return loss
    
    def consistency_loss(
        self,
        clinical_logits: torch.Tensor,
        ecg_logits: torch.Tensor,
        pcg_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency loss: Encourages agreement between modalities
        
        Args:
            clinical_logits: Clinical predictions (B, num_classes)
            ecg_logits: ECG predictions (B, num_classes)
            pcg_logits: PCG predictions (B, num_classes)
            
        Returns:
            Consistency loss
        """
        # Convert to probabilities (detach to avoid affecting gradients)
        eps = 1e-7
        clinical_probs = F.softmax(clinical_logits, dim=1).clamp(min=eps, max=1.0-eps)
        ecg_probs = F.softmax(ecg_logits, dim=1).clamp(min=eps, max=1.0-eps)
        pcg_probs = F.softmax(pcg_logits, dim=1).clamp(min=eps, max=1.0-eps)
        
        # KL divergence between modality predictions
        # F.kl_div expects input in log space and target NOT in log space
        loss_ce = F.kl_div(
            torch.log(clinical_probs),
            ecg_probs.detach(),
            reduction='batchmean'
        )
        loss_cp = F.kl_div(
            torch.log(clinical_probs),
            pcg_probs.detach(),
            reduction='batchmean'
        )
        loss_ep = F.kl_div(
            torch.log(ecg_probs),
            pcg_probs.detach(),
            reduction='batchmean'
        )
        
        loss = (loss_ce + loss_cp + loss_ep) / 3.0
        
        return loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute composite loss
        
        Args:
            outputs: Dictionary containing:
                - fused_logits: Final predictions (B, num_classes)
                - clinical_logits: Clinical predictions (B, num_classes)
                - ecg_logits: ECG predictions (B, num_classes)
                - pcg_logits: PCG predictions (B, num_classes)
                - reliability: Dict with Rc, Re, Rp
                - confidence: Dict with Cc, Ce, Cp, Cf
                - weights: Dict with Wc, We, Wp
            labels: Ground truth labels (B,)
            return_components: If True, return loss components
            
        Returns:
            Total loss (or dict of loss components if return_components=True)
        """
        # Extract outputs
        fused_logits = outputs['fused_logits']
        clinical_logits = outputs.get('clinical_logits')
        ecg_logits = outputs.get('ecg_logits')
        pcg_logits = outputs.get('pcg_logits')
        reliability = outputs.get('reliability', {})
        confidence = outputs.get('confidence', {})
        weights = outputs.get('weights', {})
        
        # Compute individual loss components
        L_cls = self.classification_loss(
            fused_logits, labels,
            clinical_logits, ecg_logits, pcg_logits
        )
        
        # Reliability loss (if reliability scores available)
        if reliability:
            L_rel = self.reliability_loss(reliability, fused_logits, labels)
        else:
            L_rel = torch.tensor(0.0, device=fused_logits.device)
        
        # Confidence loss (if confidence scores available)
        if confidence:
            L_conf = self.confidence_loss(confidence, fused_logits, labels)
        else:
            L_conf = torch.tensor(0.0, device=fused_logits.device)
        
        # Fusion loss (if weights and reliability available)
        if weights and reliability:
            L_fusion = self.fusion_loss(weights, reliability)
        else:
            L_fusion = torch.tensor(0.0, device=fused_logits.device)
        
        # Consistency loss (if modality logits available)
        if clinical_logits is not None and ecg_logits is not None and pcg_logits is not None:
            L_consist = self.consistency_loss(clinical_logits, ecg_logits, pcg_logits)
        else:
            L_consist = torch.tensor(0.0, device=fused_logits.device)
        
        # Compute total loss
        total_loss = (
            self.lambda_cls * L_cls +
            self.lambda_rel * L_rel +
            self.lambda_conf * L_conf +
            self.lambda_fusion * L_fusion +
            self.lambda_consist * L_consist
        )
        
        if return_components:
            return {
                'total': total_loss,
                'classification': L_cls,
                'reliability': L_rel,
                'confidence': L_conf,
                'fusion': L_fusion,
                'consistency': L_consist
            }
        
        return total_loss


# ============================================================================
# Alternative Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Labels (B,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Labels (B,)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = torch.sum(-true_dist * log_probs, dim=1)
        return loss.mean()


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Testing Composite Loss Function")
    print(f"{'='*80}\n")
    
    # Test parameters
    batch_size = 16
    num_classes = 5
    
    # Create sample outputs
    outputs = {
        'fused_logits': torch.randn(batch_size, num_classes),
        'clinical_logits': torch.randn(batch_size, num_classes),
        'ecg_logits': torch.randn(batch_size, num_classes),
        'pcg_logits': torch.randn(batch_size, num_classes),
        'reliability': {
            'Rc': torch.rand(batch_size, 1),
            'Re': torch.rand(batch_size, 1),
            'Rp': torch.rand(batch_size, 1)
        },
        'confidence': {
            'Cc': torch.rand(batch_size, 1),
            'Ce': torch.rand(batch_size, 1),
            'Cp': torch.rand(batch_size, 1),
            'Cf': torch.rand(batch_size, 1)
        },
        'weights': {
            'Wc': torch.rand(batch_size, 1),
            'We': torch.rand(batch_size, 1),
            'Wp': torch.rand(batch_size, 1)
        }
    }
    
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test 1: Composite Loss
    print("Test 1: Composite Loss")
    print("-" * 80)
    
    composite_loss = CompositeLoss(num_classes=num_classes)
    
    # Compute loss
    loss = composite_loss(outputs, labels)
    print(f"Total loss: {loss.item():.4f}")
    
    # Get loss components
    loss_components = composite_loss(outputs, labels, return_components=True)
    print(f"\nLoss components:")
    for key, value in loss_components.items():
        print(f"  {key:15s}: {value.item():.4f}")
    
    # Test 2: Gradient flow
    print("\nTest 2: Gradient Flow")
    print("-" * 80)
    
    outputs['fused_logits'].requires_grad = True
    loss = composite_loss(outputs, labels)
    loss.backward()
    print(f"Gradient exists: {outputs['fused_logits'].grad is not None}")
    print(f"Gradient shape: {outputs['fused_logits'].grad.shape}")
    
    # Test 3: Focal Loss
    print("\nTest 3: Focal Loss")
    print("-" * 80)
    
    focal_loss = FocalLoss(gamma=2.0)
    logits = torch.randn(batch_size, num_classes)
    loss = focal_loss(logits, labels)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test 4: Label Smoothing Loss
    print("\nTest 4: Label Smoothing Loss")
    print("-" * 80)
    
    ls_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    loss = ls_loss(logits, labels)
    print(f"Label smoothing loss: {loss.item():.4f}")
    
    print(f"\n{'='*80}")
    print("[OK] Composite Loss - Test Complete")
    print(f"{'='*80}\n")
