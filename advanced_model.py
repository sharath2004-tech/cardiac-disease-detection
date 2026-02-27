import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class AdvancedECGClassifier(nn.Module):
    """
    Hybrid CNN-Transformer model for ECG classification
    - CNN layers extract local features
    - Transformer layers capture long-range dependencies
    - Multi-scale feature fusion
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Initial Conv Block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-scale CNN branches
        self.branch1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Merge branches
        self.merge = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Prepare for Transformer (Conv1d -> Linear projection)
        self.to_transformer = nn.Linear(256, 256)
        self.pos_encoding = PositionalEncoding(256, max_len=180)
        
        # Transformer blocks
        self.transformer1 = TransformerBlock(256, nhead=8, dropout=0.3)
        self.transformer2 = TransformerBlock(256, nhead=8, dropout=0.4)
        self.transformer3 = TransformerBlock(256, nhead=8, dropout=0.4)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head with attention to minority classes
        self.classifier = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)  # [B, 64, 180]
        
        # Multi-scale branches
        b1 = self.branch1(x)  # [B, 128, 180]
        b2 = self.branch2(x)  # [B, 128, 180]
        b3 = self.branch3(x)  # [B, 128, 180]
        
        # Concatenate branches
        x = torch.cat([b1, b2, b3], dim=1)  # [B, 384, 180]
        x = self.merge(x)  # [B, 256, 180]
        
        # Prepare for Transformer: [B, 256, 180] -> [B, 180, 256]
        x = x.permute(0, 2, 1)
        x = self.to_transformer(x)
        x = self.pos_encoding(x)
        
        # Transformer blocks
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        
        # Back to [B, 256, 180] for pooling
        x = x.permute(0, 2, 1)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [B, 256]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [B, 256]
        x = torch.cat([avg_pool, max_pool], dim=1)  # [B, 512]
        
        # Classification
        x = self.classifier(x)
        return x


class FocalLoss(nn.Module):
    """
    Focal Loss - Better for extreme class imbalance
    Focuses more on hard-to-classify examples
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[target] * focal_loss
        
        return focal_loss.mean()
