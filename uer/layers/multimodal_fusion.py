"""
Multi-Modal Fusion Module for Stage 2 Pretraining

Design based on:
- MM4flow's cross-attention architecture (concatenated query)
- Gating mechanism from Mixture of Experts (MOE)
- Balance loss to prevent modality imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiModalFusion(nn.Module):
    """
    Gating-based Multi-Modal Fusion with Cross-Attention

    Architecture:
    1. Cross-Attention: MM4flow style (concatenated query)
    2. Gating Mechanism: Learn adaptive weights for each modality
    3. Balance Loss: Prevent modality imbalance

    Args:
        hidden_size: Hidden dimension size (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout=0.1):
        super(GatedMultiModalFusion, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # Cross-Attention layers
        self.cross_attn_raw = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_size = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization for cross-attention
        self.layer_norm_raw = nn.LayerNorm(hidden_size)
        self.layer_norm_size = nn.LayerNorm(hidden_size)

        # Gate networks
        self.gate_raw = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        self.gate_size = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )

        # Temperature for gate softmax
        self.gate_temperature = 0.5

    def forward(self, raw_feat, size_feat, return_gate_weights=False):
        """
        Forward pass with cross-attention and gating

        Args:
            raw_feat: [batch, seq_len_raw, hidden] - Raw Packet features from encoder
            size_feat: [batch, seq_len_size, hidden] - Packet Size features from encoder
            return_gate_weights: Whether to return gate weights for balance loss

        Returns:
            raw_fused: [batch, seq_len_raw, hidden] - Fused Raw features
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            gate_weights: (g_raw, g_size) if return_gate_weights=True
        """
        batch_size = raw_feat.size(0)
        seq_len_raw = raw_feat.size(1)
        seq_len_size = size_feat.size(1)

        # 1. Concatenated Query (MM4flow strategy)
        # 让每个token都能attend到两个模态的所有位置
        Q = torch.cat([raw_feat, size_feat], dim=1)  # [batch, seq_len_raw+seq_len_size, hidden]

        # 2. Cross-Attention for Raw modality
        # Query: 包含两个模态, Key/Value: 只来自Raw
        raw_cross, _ = self.cross_attn_raw(
            query=Q,
            key=raw_feat,
            value=raw_feat
        )  # [batch, seq_len_raw+seq_len_size, hidden]

        # 取前seq_len_raw个position (对应Raw Packet)
        raw_cross = raw_cross[:, :seq_len_raw, :]  # [batch, seq_len_raw, hidden]

        # Residual connection + Layer norm
        raw_fused = self.layer_norm_raw(raw_feat + raw_cross)

        # 3. Cross-Attention for Size modality
        # Query: 包含两个模态, Key/Value: 只来自Size
        size_cross, _ = self.cross_attn_size(
            query=Q,
            key=size_feat,
            value=size_feat
        )  # [batch, seq_len_raw+seq_len_size, hidden]

        # 取后seq_len_size个position (对应Packet Size)
        size_cross = size_cross[:, seq_len_raw:, :]  # [batch, seq_len_size, hidden]

        # Residual connection + Layer norm
        size_fused = self.layer_norm_size(size_feat + size_cross)

        # 4. Gating Mechanism (只在[CLS] token上计算gate)
        # [CLS] token is at position 0
        raw_cls = raw_fused[:, 0, :]   # [batch, hidden]
        size_cls = size_fused[:, 0, :]  # [batch, hidden]

        # 计算gate权重
        g_raw = self.gate_raw(raw_cls)    # [batch, 1]
        g_size = self.gate_size(size_cls)  # [batch, 1]

        # Softmax归一化 (确保和为1)
        gates = torch.cat([g_raw, g_size], dim=1)  # [batch, 2]
        gates = F.softmax(gates / self.gate_temperature, dim=1)
        g_raw, g_size = gates[:, 0:1], gates[:, 1:2]  # [batch, 1] each

        # 5. 加权融合 (只在[CLS]位置)
        fused_cls = g_raw * raw_cls + g_size * size_cls  # [batch, hidden]

        # 替换原来的[CLS] token
        raw_fused = raw_fused.clone()
        size_fused = size_fused.clone()
        raw_fused[:, 0, :] = fused_cls
        size_fused[:, 0, :] = fused_cls

        if return_gate_weights:
            return raw_fused, size_fused, (g_raw, g_size)
        else:
            return raw_fused, size_fused


def compute_balance_loss(g_raw, g_size, target_ratio=0.5):
    """
    Balance Loss to prevent modality imbalance

    鼓励两个模态的平均使用率接近target_ratio

    Args:
        g_raw: [batch, 1] - Gate weights for Raw modality
        g_size: [batch, 1] - Gate weights for Size modality
        target_ratio: Target ratio for each modality (default: 0.5)

    Returns:
        loss: Scalar balance loss
    """
    # 计算batch内的平均gate权重
    mean_g_raw = g_raw.mean()
    mean_g_size = g_size.mean()

    # L2 loss: 鼓励接近target_ratio
    loss = (mean_g_raw - target_ratio) ** 2 + (mean_g_size - target_ratio) ** 2

    return loss
