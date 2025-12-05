"""
Multi-Modal Fusion Module for Stage 2 Pretraining (v3)

设计要点：
1. 标准双向 Cross-Attention：raw ↔ size 双向信息交换
2. Joint Gated Fusion：Gate 同时看两个模态，用 softmax 约束平衡
3. Entropy-based Balance Loss：最大化熵，鼓励均匀分布

改进 (v3):
- Gate 输入从单模态改为双模态拼接，能感知两个模态的状态
- 使用 softmax 让两个 gate 和为 1，形成竞争约束
- Balance loss 改为熵最大化，鼓励 [0.5, 0.5] 的均匀分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.layer_norm import LayerNorm


class GatedMultiModalFusion(nn.Module):
    """
    Gated Bidirectional Cross-Attention Fusion (v3)

    Architecture:
    1. Bidirectional Cross-Attention:
       - cross_raw = CrossAttn(Q=raw, K=size, V=size)  # raw 从 size 获取信息
       - cross_size = CrossAttn(Q=size, K=raw, V=raw)  # size 从 raw 获取信息
    
    2. Joint Gated Fusion (改进):
       - 输入: concat([raw_cls, size_cls]) 同时看两个模态
       - 输出: softmax([g_raw, g_size])，和为 1，形成竞争
       - g_raw 控制 raw 吸收多少 size 的信息
       - g_size 控制 size 吸收多少 raw 的信息
    
    3. Output:
       - raw_fused = LayerNorm(raw + g_raw * cross_raw)
       - size_fused = LayerNorm(size + g_size * cross_size)
    """

    def __init__(self, hidden_size=768, num_attention_heads=12, attention_head_size=64,
                 dropout=0.1, gate_temperature=1.0):
        super(GatedMultiModalFusion, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.gate_temperature = gate_temperature

        # Bidirectional Cross-Attention layers
        self.cross_attn_raw2size = MultiHeadedAttention(
            hidden_size=hidden_size,
            heads_num=num_attention_heads,
            attention_head_size=attention_head_size,
            dropout=dropout,
            has_bias=True,
            with_scale=True
        )
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

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # 输出 2 个 logits
        )

    def forward(self, raw_feat, size_feat, raw_seg=None, size_seg=None, return_gate_weights=False):
        """
        Forward pass with bidirectional cross-attention and joint gated fusion

        Args:
            raw_feat: [batch, seq_len_raw, hidden] - Raw Packet features
            size_feat: [batch, seq_len_size, hidden] - Packet Size features
            raw_seg: [batch, seq_len_raw] - Raw segment indicators
            size_seg: [batch, seq_len_size] - Size segment indicators
            return_gate_weights: Whether to return gate weights for balance loss

        Returns:
            raw_fused: [batch, seq_len_raw, hidden]
            size_fused: [batch, seq_len_size, hidden]
            gate_weights: (g_raw, g_size, gates) if return_gate_weights=True
                - g_raw: [batch, 1] 
                - g_size: [batch, 1]
                - gates: [batch, 2] softmax 分布，用于计算 entropy loss
        """
        batch_size = raw_feat.size(0)
        seq_len_raw = raw_feat.size(1)
        seq_len_size = size_feat.size(1)

        # ===== Step 1: Extract [CLS] tokens =====
        raw_cls = raw_feat[:, 0, :]    # [batch, hidden]
        size_cls = size_feat[:, 0, :]  # [batch, hidden]

        # ===== Step 2: Joint Gate Computation (v3 改进) =====
        # 拼接两个模态的 CLS，让 gate 能同时感知两者状态
        combined_cls = torch.cat([raw_cls, size_cls], dim=-1)  # [batch, hidden * 2]
        gate_logits = self.gate_net(combined_cls)  # [batch, 2]
        
        # Softmax with temperature: 和为 1，形成竞争约束
        gates = F.softmax(gate_logits / self.gate_temperature, dim=-1)  # [batch, 2]
        
        g_raw = gates[:, 0:1]   # [batch, 1] - raw 从 size 吸收的权重
        g_size = gates[:, 1:2]  # [batch, 1] - size 从 raw 吸收的权重

        # ===== Step 3: Create Attention Masks =====
        if size_seg is not None:
            size_mask = (size_seg > 0).unsqueeze(1).unsqueeze(1)
            size_mask = size_mask.expand(-1, -1, seq_len_raw, -1).float()
            size_mask = (1.0 - size_mask) * -10000.0
        else:
            size_mask = torch.zeros(batch_size, 1, seq_len_raw, seq_len_size, device=raw_feat.device)

        if raw_seg is not None:
            raw_mask = (raw_seg > 0).unsqueeze(1).unsqueeze(1)
            raw_mask = raw_mask.expand(-1, -1, seq_len_size, -1).float()
            raw_mask = (1.0 - raw_mask) * -10000.0
        else:
            raw_mask = torch.zeros(batch_size, 1, seq_len_size, seq_len_raw, device=size_feat.device)

        # ===== Step 4: Bidirectional Cross-Attention =====
        # raw attend to size: raw 从 size 获取信息
        cross_raw, _ = self.cross_attn_raw2size(
            key=size_feat,
            value=size_feat,
            query=raw_feat,
            mask=size_mask,
            position_bias=None
        )  # [batch, seq_len_raw, hidden]

        # size attend to raw: size 从 raw 获取信息
        cross_size, _ = self.cross_attn_size2raw(
            key=raw_feat,
            value=raw_feat,
            query=size_feat,
            mask=raw_mask,
            position_bias=None
        )  # [batch, seq_len_size, hidden]

        # ===== Step 5: Gated Fusion =====
        g_raw_expanded = g_raw.unsqueeze(-1)    # [batch, 1, 1]
        g_size_expanded = g_size.unsqueeze(-1)  # [batch, 1, 1]

        # Residual connection with gated cross-modal information
        raw_fused = self.layer_norm_raw(raw_feat + g_raw_expanded * cross_raw)
        size_fused = self.layer_norm_size(size_feat + g_size_expanded * cross_size)

        if return_gate_weights:
            # 返回 gates 用于计算 entropy-based balance loss
            return raw_fused, size_fused, (g_raw, g_size, gates)
        else:
            return raw_fused, size_fused


def compute_balance_loss(g_raw, g_size, gates=None):
    """
    Entropy-based Balance Loss (v3 改进)
    
    最大化 softmax 分布的熵，鼓励 [0.5, 0.5] 的均匀分布
    
    熵公式: H = -sum(p * log(p))
    均匀分布 [0.5, 0.5] 时熵最大 = log(2) ≈ 0.693
    极端分布 [1, 0] 或 [0, 1] 时熵最小 = 0
    
    Args:
        g_raw: [batch, 1] - 兼容旧接口，但不再使用
        g_size: [batch, 1] - 兼容旧接口，但不再使用
        gates: [batch, 2] - softmax 分布，用于计算熵
        
    Returns:
        loss: Scalar，负熵（最小化 loss = 最大化熵 = 鼓励均匀）
    """
    if gates is not None:
        # 计算每个样本的熵，然后取 batch 平均
        entropy = -(gates * torch.log(gates + 1e-8)).sum(dim=-1)  # [batch]
        mean_entropy = entropy.mean()
        

        max_entropy = torch.log(torch.tensor(2.0, device=gates.device))
        loss = 1.0 - mean_entropy / max_entropy
        
        return loss
    else:
        # 兼容旧接口：如果没有传入 gates，使用 L2 loss
        mean_g_raw = g_raw.mean()
        mean_g_size = g_size.mean()
        loss = (mean_g_raw - 0.5) ** 2 + (mean_g_size - 0.5) ** 2
        return loss
