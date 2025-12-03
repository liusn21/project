"""
Multi-Modal Fusion Module for Stage 2 Pretraining (v2)

设计要点：
1. 标准双向 Cross-Attention：raw ↔ size 双向信息交换
2. Gated Fusion：Gate 控制跨模态信息的融合比例
3. Balance Loss：防止 Gate 趋向极端值

参考：
- Gated Multimodal Units (GMU) 的门控思想
- 标准 Cross-Attention 的双向设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.layer_norm import LayerNorm


class GatedMultiModalFusion(nn.Module):
    """
    Gated Bidirectional Cross-Attention Fusion

    Architecture:
    1. Bidirectional Cross-Attention:
       - raw_cross = CrossAttn(Q=raw, K=size, V=size)  # raw 从 size 获取信息
       - size_cross = CrossAttn(Q=size, K=raw, V=raw)  # size 从 raw 获取信息
    2. Gated Fusion:
       - g_raw ∈ [0,1] 控制 raw 吸收多少 size 的信息
       - g_size ∈ [0,1] 控制 size 吸收多少 raw 的信息
    3. Output:
       - raw_fused = LayerNorm(raw + g_raw * raw_cross)
       - size_fused = LayerNorm(size + g_size * size_cross)

    Args:
        hidden_size: Hidden dimension size (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        attention_head_size: Size of each attention head (default: 64)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, hidden_size=768, num_attention_heads=12, attention_head_size=64,
                 dropout=0.1, gate_temperature=0.5):
        super(GatedMultiModalFusion, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size

        # Bidirectional Cross-Attention layers
        # raw2size: raw attend to size (raw 从 size 获取信息)
        self.cross_attn_raw2size = MultiHeadedAttention(
            hidden_size=hidden_size,
            heads_num=num_attention_heads,
            attention_head_size=attention_head_size,
            dropout=dropout,
            has_bias=True,
            with_scale=True
        )
        # size2raw: size attend to raw (size 从 raw 获取信息)
        self.cross_attn_size2raw = MultiHeadedAttention(
            hidden_size=hidden_size,
            heads_num=num_attention_heads,
            attention_head_size=attention_head_size,
            dropout=dropout,
            has_bias=True,
            with_scale=True
        )

        # Layer normalization
        self.layer_norm_raw = LayerNorm(hidden_size)
        self.layer_norm_size = LayerNorm(hidden_size)

        # Gate networks
        # 输入: [CLS] token embedding
        # 输出: scalar gate weight ∈ [0, 1]
        self.gate_raw = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        self.gate_size = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, raw_feat, size_feat, raw_seg=None, size_seg=None, return_gate_weights=False):
        """
        Forward pass with bidirectional cross-attention and gated fusion

        Args:
            raw_feat: [batch, seq_len_raw, hidden] - Raw Packet features from encoder
            size_feat: [batch, seq_len_size, hidden] - Packet Size features from encoder
            raw_seg: [batch, seq_len_raw] - Raw segment indicators (0 for padding, >0 for valid)
            size_seg: [batch, seq_len_size] - Size segment indicators (0 for padding, >0 for valid)
            return_gate_weights: Whether to return gate weights for balance loss

        Returns:
            raw_fused: [batch, seq_len_raw, hidden] - Raw enhanced with size info
            size_fused: [batch, seq_len_size, hidden] - Size enhanced with raw info
            gate_weights: (g_raw, g_size) if return_gate_weights=True
        """
        batch_size = raw_feat.size(0)
        seq_len_raw = raw_feat.size(1)
        seq_len_size = size_feat.size(1)

        # ===== Step 1: Compute Gate Weights (based on original [CLS]) =====
        raw_cls = raw_feat[:, 0, :]   # [batch, hidden]
        size_cls = size_feat[:, 0, :]  # [batch, hidden]

        g_raw = self.gate_raw(raw_cls)    # [batch, 1], range [0, 1]
        g_size = self.gate_size(size_cls)  # [batch, 1], range [0, 1]

        # ===== Step 2: Create Attention Masks =====
        # Mask format: 0.0 for valid positions, -10000.0 for padding
        # Shape: [batch, 1, query_len, key_len]

        # Mask for size K/V (used when raw attends to size)
        if size_seg is not None:
            # size_seg: [batch, seq_len_size], 0=padding, >0=valid
            size_mask = (size_seg > 0).unsqueeze(1).unsqueeze(1)
            # -> [batch, 1, 1, seq_len_size]
            size_mask = size_mask.expand(-1, -1, seq_len_raw, -1)
            # -> [batch, 1, seq_len_raw, seq_len_size]
            size_mask = size_mask.float()
            size_mask = (1.0 - size_mask) * -10000.0
        else:
            size_mask = torch.zeros(batch_size, 1, seq_len_raw, seq_len_size, device=raw_feat.device)

        # Mask for raw K/V (used when size attends to raw)
        if raw_seg is not None:
            raw_mask = (raw_seg > 0).unsqueeze(1).unsqueeze(1)
            # -> [batch, 1, 1, seq_len_raw]
            raw_mask = raw_mask.expand(-1, -1, seq_len_size, -1)
            # -> [batch, 1, seq_len_size, seq_len_raw]
            raw_mask = raw_mask.float()
            raw_mask = (1.0 - raw_mask) * -10000.0
        else:
            raw_mask = torch.zeros(batch_size, 1, seq_len_size, seq_len_raw, device=size_feat.device)

        # ===== Step 3: Bidirectional Cross-Attention =====
        # raw attend to size: raw 从 size 获取信息
        raw_cross, _ = self.cross_attn_raw2size(
            key=size_feat,      # [batch, seq_len_size, hidden]
            value=size_feat,    # [batch, seq_len_size, hidden]
            query=raw_feat,     # [batch, seq_len_raw, hidden]
            mask=size_mask,     # [batch, 1, seq_len_raw, seq_len_size]
            position_bias=None
        )  # -> [batch, seq_len_raw, hidden]

        # size attend to raw: size 从 raw 获取信息
        size_cross, _ = self.cross_attn_size2raw(
            key=raw_feat,       # [batch, seq_len_raw, hidden]
            value=raw_feat,     # [batch, seq_len_raw, hidden]
            query=size_feat,    # [batch, seq_len_size, hidden]
            mask=raw_mask,      # [batch, 1, seq_len_size, seq_len_raw]
            position_bias=None
        )  # -> [batch, seq_len_size, hidden]

        # ===== Step 4: Gated Fusion =====
        # g_raw controls how much raw absorbs from size
        # g_size controls how much size absorbs from raw
        g_raw_expanded = g_raw.unsqueeze(-1)   # [batch, 1, 1]
        g_size_expanded = g_size.unsqueeze(-1)  # [batch, 1, 1]

        # Residual connection with gated cross-modal information
        raw_fused = self.layer_norm_raw(raw_feat + g_raw_expanded * raw_cross)
        size_fused = self.layer_norm_size(size_feat + g_size_expanded * size_cross)

        if return_gate_weights:
            return raw_fused, size_fused, (g_raw, g_size)
        else:
            return raw_fused, size_fused


def compute_balance_loss(g_raw, g_size, target_ratio=0.5):
    """
    Balance Loss to prevent gate weights from going to extremes

    鼓励两个模态都适度地接收跨模态信息，防止：
    - 两个 gate 都趋向 0（完全不交互）
    - 某个 gate 趋向 1 而另一个趋向 0（单向交互）

    Args:
        g_raw: [batch, 1] - Gate weights for raw modality (how much to absorb from size)
        g_size: [batch, 1] - Gate weights for size modality (how much to absorb from raw)
        target_ratio: Target average gate value (default: 0.5)

    Returns:
        loss: Scalar balance loss
    """
    # 计算 batch 内的平均 gate 权重
    mean_g_raw = g_raw.mean()
    mean_g_size = g_size.mean()

    # L2 loss: 鼓励接近 target_ratio
    loss = (mean_g_raw - target_ratio) ** 2 + (mean_g_size - target_ratio) ** 2

    return loss
