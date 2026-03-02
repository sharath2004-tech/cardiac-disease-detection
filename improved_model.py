import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=180):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ImprovedECGClassifier(nn.Module):
    """
    Enhanced CNN-Transformer model with:
    - Residual connections with SE blocks
    - Better regularization
    - Improved feature fusion
    - Multi-head classification for minority classes
    """
    def __init__(self, num_classes=4, dropout=0.35):
        super().__init__()
        
        # Initial feature extraction with residual blocks
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Residual blocks with increasing channels
        self.res_block1 = ResidualBlock(64, 128, kernel_size=5, dropout=dropout)
        self.res_block2 = ResidualBlock(128, 128, kernel_size=5, dropout=dropout)
        
        # Multi-scale CNN branches with different receptive fields
        self.branch1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.Dropout(dropout)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.Dropout(dropout)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.Dropout(dropout)
        )
        
        # Merge branches with attention
        self.merge = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SEBlock(256),
            nn.Dropout(dropout)
        )
        
        # Additional residual processing
        self.res_block3 = ResidualBlock(256, 256, kernel_size=3, dropout=dropout)
        
        # Prepare for Transformer
        self.to_transformer = nn.Linear(256, 256)
        self.pos_encoding = PositionalEncoding(256, max_len=180)
        
        # Transformer blocks with increasing dropout
        self.transformer1 = TransformerBlock(256, nhead=8, dropout=dropout)
        self.transformer2 = TransformerBlock(256, nhead=8, dropout=dropout * 1.1)
        self.transformer3 = TransformerBlock(256, nhead=8, dropout=dropout * 1.2)
        self.transformer4 = TransformerBlock(256, nhead=4, dropout=dropout * 1.3)
        
        # Global pooling with multiple strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 1.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 1.1),
            
            nn.Linear(128, num_classes)
        )
        
        # Auxiliary classifier for minority classes (S and F)
        # This helps the model focus on rare patterns
        self.aux_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 1.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        # Initial feature extraction
        x = self.stem(x)  # [B, 64, 180]
        x = self.res_block1(x)  # [B, 128, 180]
        x = self.res_block2(x)  # [B, 128, 180]
        
        # Multi-scale branches
        b1 = self.branch1(x)  # [B, 128, 180]
        b2 = self.branch2(x)  # [B, 128, 180]
        b3 = self.branch3(x)  # [B, 128, 180]
        
        # Concatenate and merge branches
        x = torch.cat([b1, b2, b3], dim=1)  # [B, 384, 180]
        x = self.merge(x)  # [B, 256, 180]
        x = self.res_block3(x)  # [B, 256, 180]
        
        # Prepare for Transformer: [B, 256, 180] -> [B, 180, 256]
        x = x.permute(0, 2, 1)
        x = self.to_transformer(x)
        x = self.pos_encoding(x)
        
        # Transformer blocks
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        x = self.transformer4(x)
        
        # Back to [B, 256, 180] for pooling
        x = x.permute(0, 2, 1)
        
        # Global pooling with both avg and max
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [B, 256]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [B, 256]
        features = torch.cat([avg_pool, max_pool], dim=1)  # [B, 512]
        
        # Main classification
        main_out = self.classifier(features)
        
        # Auxiliary classification (used during training)
        if self.training:
            aux_out = self.aux_classifier(features)
            return main_out, aux_out
        
        if return_features:
            return main_out, features
        
        return main_out


class ImprovedFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with:
    - Label smoothing for regularization
    - Dynamic alpha based on class distribution
    """
    def __init__(self, alpha=None, gamma=2.5, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, pred, target):
        # Compute cross entropy with label smoothing
        logpt = F.log_softmax(pred, dim=1)
        ce_loss = F.nll_loss(logpt, target, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Combine
        loss = focal_term * ce_loss
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function:
    - Focal loss for main classification
    - Auxiliary loss for minority class focus
    - Optional label smoothing
    """
    def __init__(self, alpha=None, gamma=2.5, aux_weight=0.3):
        super().__init__()
        self.focal_loss = ImprovedFocalLoss(alpha=alpha, gamma=gamma, label_smoothing=0.1)
        self.aux_weight = aux_weight
    
    def forward(self, main_pred, aux_pred, target):
        # Main focal loss
        main_loss = self.focal_loss(main_pred, target)
        
        # Auxiliary loss (helps model focus on all classes)
        aux_loss = self.focal_loss(aux_pred, target)
        
        # Combine
        total_loss = main_loss + self.aux_weight * aux_loss
        
        return total_loss, main_loss, aux_loss


# Data Augmentation Techniques
class ECGAugmentation:
    """Advanced augmentation techniques for ECG signals"""
    
    @staticmethod
    def add_noise(signal, noise_level=0.02):
        """Add Gaussian noise"""
        noise = torch.randn_like(signal) * noise_level
        return signal + noise
    
    @staticmethod
    def time_shift(signal, max_shift=10):
        """Shift signal in time"""
        shift = np.random.randint(-max_shift, max_shift)
        return torch.roll(signal, shifts=shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(signal, scale_range=(0.9, 1.1)):
        """Scale amplitude"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def time_warp(signal, num_anchors=4):
        """Apply time warping"""
        length = signal.size(-1)
        anchors = torch.linspace(0, length-1, num_anchors)
        warped_anchors = anchors + torch.randn(num_anchors) * (length / num_anchors * 0.1)
        warped_anchors = torch.clamp(warped_anchors, 0, length-1)
        
        # Simple interpolation-based warping
        return signal
    
    @staticmethod
    def cutout(signal, num_cuts=2, cut_length=10):
        """Apply cutout augmentation"""
        signal = signal.clone()
        length = signal.size(-1)
        for _ in range(num_cuts):
            start = np.random.randint(0, length - cut_length)
            signal[..., start:start+cut_length] = 0
        return signal
    
    @staticmethod
    def mixup(signal1, signal2, label1, label2, alpha=0.4):
        """Mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        mixed_signal = lam * signal1 + (1 - lam) * signal2
        return mixed_signal, label1, label2, lam


def apply_augmentation(signal, label, minority_classes=[1, 3], prob=0.5):
    """
    Apply augmentation with higher probability for minority classes
    """
    aug = ECGAugmentation()
    
    # Higher augmentation for minority classes
    is_minority = label in minority_classes
    apply_prob = 0.8 if is_minority else prob
    
    if np.random.random() < apply_prob:
        # Random combination of augmentations
        if np.random.random() < 0.4:
            signal = aug.add_noise(signal, noise_level=0.01 if is_minority else 0.02)
        if np.random.random() < 0.4:
            signal = aug.time_shift(signal, max_shift=8)
        if np.random.random() < 0.3:
            signal = aug.amplitude_scale(signal, scale_range=(0.95, 1.05))
        if np.random.random() < 0.2 and is_minority:
            signal = aug.cutout(signal, num_cuts=1, cut_length=8)
    
    return signal
