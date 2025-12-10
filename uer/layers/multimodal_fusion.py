"""
Multi-Modal Fusion Module for Stage 2 Pretraining (ALBEF-style)

Architecture:
- 6层双向Cross-Attention Fusion
- 每层包含: Self-Attention + Cross-Attention (双向) + FFN
- 参考LXMERT的双向设计

改进点 (相比原版本):
1. 去掉Gate机制，采用标准的残差连接
2. 6层深度融合，而非单层
3. 双向Cross-Attention: Raw↔Size
"""

import torch
import torch.nn as nn
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward


class BidirectionalFusionLayer(nn.Module):
    """
    单层双向Cross-Attention Fusion

    结构:
        Raw Branch:  Self-Attn → Cross-Attn(Q=raw, KV=size) → FFN
        Size Branch: Self-Attn → Cross-Attn(Q=size, KV=raw) → FFN

    采用Post-LayerNorm结构 (与原框架一致)
    """

    def __init__(self, hidden_size, heads_num, feedforward_size, hidden_act, dropout):
        super(BidirectionalFusionLayer, self).__init__()

        attention_head_size = hidden_size // heads_num

        # ===== Raw Branch =====
        # Self-Attention
        self.self_attn_raw = MultiHeadedAttention(
            hidden_size=hidden_size,
            heads_num=heads_num,
            attention_head_size=attention_head_size,
            dropout=dropout,
            has_bias=True,
            with_scale=True
        )
        self.layer_norm_raw_1 = LayerNorm(hidden_size)
        self.dropout_raw_1 = nn.Dropout(dropout)

        # Cross-Attention (Q=raw, KV=size)
        self.cross_attn_raw = MultiHeadedAttention(
            hidden_size=hidden_size,
            heads_num=heads_num,
            attention_head_size=attention_head_size,
            dropout=dropout,
            has_bias=True,
            with_scale=True
        )
        self.layer_norm_raw_2 = LayerNorm(hidden_size)
        self.dropout_raw_2 = nn.Dropout(dropout)

        # FFN
        self.ffn_raw = PositionwiseFeedForward(
            hidden_size=hidden_size,
            feedforward_size=feedforward_size,
            hidden_act=hidden_act,
            has_bias=True
        )
        self.layer_norm_raw_3 = LayerNorm(hidden_size)
        self.dropout_raw_3 = nn.Dropout(dropout)

        # ===== Size Branch =====
        # Self-Attention
        self.self_attn_size = MultiHeadedAttention(
            hidden_size=hidden_size,
            heads_num=heads_num,
            attention_head_size=attention_head_size,
            dropout=dropout,
            has_bias=True,
            with_scale=True
        )
        self.layer_norm_size_1 = LayerNorm(hidden_size)
        self.dropout_size_1 = nn.Dropout(dropout)

        # Cross-Attention (Q=size, KV=raw)
        self.cross_attn_size = MultiHeadedAttention(
            hidden_size=hidden_size,
            heads_num=heads_num,
            attention_head_size=attention_head_size,
            dropout=dropout,
            has_bias=True,
            with_scale=True
        )
        self.layer_norm_size_2 = LayerNorm(hidden_size)
        self.dropout_size_2 = nn.Dropout(dropout)

        # FFN
        self.ffn_size = PositionwiseFeedForward(
            hidden_size=hidden_size,
            feedforward_size=feedforward_size,
            hidden_act=hidden_act,
            has_bias=True
        )
        self.layer_norm_size_3 = LayerNorm(hidden_size)
        self.dropout_size_3 = nn.Dropout(dropout)

    def forward(self, raw_feat, size_feat, raw_mask, size_mask, cross_mask_r2s, cross_mask_s2r):
        """
        Args:
            raw_feat: [batch, seq_len_raw, hidden] - Raw特征
            size_feat: [batch, seq_len_size, hidden] - Size特征
            raw_mask: [batch, 1, seq_len_raw, seq_len_raw] - Raw self-attention mask
            size_mask: [batch, 1, seq_len_size, seq_len_size] - Size self-attention mask
            cross_mask_r2s: [batch, 1, seq_len_raw, seq_len_size] - Raw attend to Size的mask
            cross_mask_s2r: [batch, 1, seq_len_size, seq_len_raw] - Size attend to Raw的mask

        Returns:
            raw_out: [batch, seq_len_raw, hidden]
            size_out: [batch, seq_len_size, hidden]
        """
        # ===== Raw Branch =====
        # Self-Attention
        raw_self, _ = self.self_attn_raw(raw_feat, raw_feat, raw_feat, raw_mask, None)
        raw_self = self.dropout_raw_1(raw_self)
        raw_feat = self.layer_norm_raw_1(raw_feat + raw_self)

        # Cross-Attention (Q=raw, KV=size)
        raw_cross, _ = self.cross_attn_raw(size_feat, size_feat, raw_feat, cross_mask_r2s, None)
        raw_cross = self.dropout_raw_2(raw_cross)
        raw_feat = self.layer_norm_raw_2(raw_feat + raw_cross)

        # FFN
        raw_ffn = self.ffn_raw(raw_feat)
        raw_ffn = self.dropout_raw_3(raw_ffn)
        raw_out = self.layer_norm_raw_3(raw_feat + raw_ffn)

        # ===== Size Branch =====
        # Self-Attention
        size_self, _ = self.self_attn_size(size_feat, size_feat, size_feat, size_mask, None)
        size_self = self.dropout_size_1(size_self)
        size_feat = self.layer_norm_size_1(size_feat + size_self)

        # Cross-Attention (Q=size, KV=raw)
        size_cross, _ = self.cross_attn_size(raw_feat, raw_feat, size_feat, cross_mask_s2r, None)
        size_cross = self.dropout_size_2(size_cross)
        size_feat = self.layer_norm_size_2(size_feat + size_cross)

        # FFN
        size_ffn = self.ffn_size(size_feat)
        size_ffn = self.dropout_size_3(size_ffn)
        size_out = self.layer_norm_size_3(size_feat + size_ffn)

        return raw_out, size_out


class MultiModalFusionEncoder(nn.Module):
    """
    多层双向Cross-Attention Fusion Encoder

    包含6层BidirectionalFusionLayer
    """

    def __init__(self, args, num_layers=6):
        super(MultiModalFusionEncoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = args.hidden_size

        # 构建6层Fusion
        self.fusion_layers = nn.ModuleList([
            BidirectionalFusionLayer(
                hidden_size=args.hidden_size,
                heads_num=args.heads_num,
                feedforward_size=args.feedforward_size,
                hidden_act=args.hidden_act,
                dropout=args.dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, raw_feat, size_feat, raw_seg, size_seg):
        """
        Args:
            raw_feat: [batch, seq_len_raw, hidden] - Encoder输出的Raw特征
            size_feat: [batch, seq_len_size, hidden] - Encoder输出的Size特征
            raw_seg: [batch, seq_len_raw] - Raw segment (用于生成mask)
            size_seg: [batch, seq_len_size] - Size segment (用于生成mask)

        Returns:
            raw_fused: [batch, seq_len_raw, hidden] - Fusion后的Raw特征
            size_fused: [batch, seq_len_size, hidden] - Fusion后的Size特征
        """
        batch_size = raw_feat.size(0)
        seq_len_raw = raw_feat.size(1)
        seq_len_size = size_feat.size(1)
        device = raw_feat.device

        # ===== 生成Attention Masks =====
        # Raw self-attention mask: [batch, 1, seq_len_raw, seq_len_raw]
        raw_mask = (raw_seg > 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len_raw]
        raw_mask = raw_mask.expand(-1, -1, seq_len_raw, -1).float()  # [batch, 1, seq_len_raw, seq_len_raw]
        raw_mask = (1.0 - raw_mask) * -10000.0

        # Size self-attention mask: [batch, 1, seq_len_size, seq_len_size]
        size_mask = (size_seg > 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len_size]
        size_mask = size_mask.expand(-1, -1, seq_len_size, -1).float()  # [batch, 1, seq_len_size, seq_len_size]
        size_mask = (1.0 - size_mask) * -10000.0

        # Cross-attention mask: Raw attend to Size
        # [batch, 1, seq_len_raw, seq_len_size]
        cross_mask_r2s = (size_seg > 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len_size]
        cross_mask_r2s = cross_mask_r2s.expand(-1, -1, seq_len_raw, -1).float()
        cross_mask_r2s = (1.0 - cross_mask_r2s) * -10000.0

        # Cross-attention mask: Size attend to Raw
        # [batch, 1, seq_len_size, seq_len_raw]
        cross_mask_s2r = (raw_seg > 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len_raw]
        cross_mask_s2r = cross_mask_s2r.expand(-1, -1, seq_len_size, -1).float()
        cross_mask_s2r = (1.0 - cross_mask_s2r) * -10000.0

        # ===== 逐层Fusion =====
        raw_hidden = raw_feat
        size_hidden = size_feat

        for layer in self.fusion_layers:
            raw_hidden, size_hidden = layer(
                raw_hidden, size_hidden,
                raw_mask, size_mask,
                cross_mask_r2s, cross_mask_s2r
            )

        return raw_hidden, size_hidden
