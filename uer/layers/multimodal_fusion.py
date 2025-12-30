"""
Multi-Modal Fusion Module for Stage 2 Pretraining (ALBEF-style)

Architecture:
- 6层双向Cross-Attention Fusion
- 每层包含: Self-Attention + Cross-Attention (双向) + FFN
- 参考LXMERT的双向设计

改进点 (v2 - 带Logits级Gate):
1. 添加可学习的Cross-Attention Gate机制
2. 双层Gate: 模态级 + Token级
3. Gate作用于attention logits (softmax前)，更有效地抑制噪声模态
4. 可通过参数控制是否启用Gate

Gate设计原理:
- 模态级Gate: 用两个模态的CLS token判断源模态整体可靠性
- Token级Gate: 用query的hidden state判断每个位置是否需要跨模态信息
- 组合方式: gate = g_modality * g_token
- 应用方式: logits' = logits - (1-gate) * 10000 (在softmax前)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward


class CrossAttentionGate(nn.Module):
    """
    Cross-Attention Gate Module (Logits级)

    为Cross-Attention生成logits级gate，在softmax前抑制噪声模态。

    双层设计:
    1. 模态级Gate (Modality-level): 用CLS tokens判断源模态整体可靠性
       - 输入: [CLS_query; CLS_key] 拼接
       - 输出: [B, H, 1, 1] 每个head一个gate值

    2. Token级Gate (Token-level): 用query hidden判断每个位置的需求
       - 输入: query_feat (self-attention后)
       - 输出: [B, 1, L_q, 1] 每个query位置一个gate值

    最终Gate: gate = g_modality * g_token -> [B, H, L_q, 1]
    """

    def __init__(self, hidden_size, heads_num, dropout=0.1):
        super(CrossAttentionGate, self).__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # ===== 模态级Gate: 判断源模态整体可靠性 =====
        # 输入: [CLS_query; CLS_key] -> 输出: 每个head一个gate
        self.modality_gate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, heads_num)
            # 不加sigmoid，后面统一处理
        )

        # ===== Token级Gate: 判断每个query位置是否需要跨模态信息 =====
        # 输入: query_feat -> 输出: 每个位置一个gate
        self.token_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
            # 不加sigmoid，后面统一处理
        )

        # 初始化: 让gate初始值接近1 (保持原有行为)
        self._init_weights()

    def _init_weights(self):
        """
        初始化权重，使gate初始输出接近1

        sigmoid(2.0) ≈ 0.88，保证训练初期模型正常使用跨模态信息
        """
        # 模态级gate的最后一层bias
        nn.init.zeros_(self.modality_gate[-1].weight)
        nn.init.constant_(self.modality_gate[-1].bias, 2.0)

        # Token级gate的最后一层bias
        nn.init.zeros_(self.token_gate[-1].weight)
        nn.init.constant_(self.token_gate[-1].bias, 2.0)

    def forward(self, query_feat, query_cls, key_cls):
        """
        计算Cross-Attention的logits级gate

        Args:
            query_feat: [B, L_q, H] - Query序列特征 (self-attention后)
            query_cls: [B, H] - Query模态的CLS token
            key_cls: [B, H] - Key模态(源模态)的CLS token

        Returns:
            gate: [B, heads_num, L_q, 1] - 用于logits缩放的gate
                  值域[0,1]，gate=1正常attention，gate=0完全抑制
        """
        B, L_q, H = query_feat.shape

        # ===== 模态级Gate =====
        # 拼接两个模态的CLS token
        cls_concat = torch.cat([query_cls, key_cls], dim=-1)  # [B, 2H]
        # 计算每个head的gate值
        g_modality_logits = self.modality_gate(cls_concat)  # [B, heads_num]
        g_modality = torch.sigmoid(g_modality_logits)  # [B, heads_num]
        # 调整形状: [B, heads_num] -> [B, heads_num, 1, 1]
        g_modality = g_modality.view(B, self.heads_num, 1, 1)

        # ===== Token级Gate =====
        # 计算每个query位置的gate值
        g_token_logits = self.token_gate(query_feat)  # [B, L_q, 1]
        g_token = torch.sigmoid(g_token_logits)  # [B, L_q, 1]
        # 调整形状: [B, L_q, 1] -> [B, 1, L_q, 1]
        g_token = g_token.unsqueeze(1)

        # ===== 组合Gate =====
        # 广播乘法: [B, heads_num, 1, 1] * [B, 1, L_q, 1] -> [B, heads_num, L_q, 1]
        gate = g_modality * g_token

        return gate


class BidirectionalFusionLayer(nn.Module):
    """
    单层双向Cross-Attention Fusion (带可选Gate)

    结构:
        Raw Branch:  Self-Attn → Cross-Attn(Q=raw, KV=size) → FFN
        Size Branch: Self-Attn → Cross-Attn(Q=size, KV=raw) → FFN

    Gate机制 (可选):
        - 在Cross-Attention的logits上应用gate
        - gate作用于softmax前，更有效地抑制噪声
        - gate = g_modality * g_token
    """

    def __init__(self, hidden_size, heads_num, feedforward_size, hidden_act, dropout,
                 use_gate=False):
        super(BidirectionalFusionLayer, self).__init__()

        self.use_gate = use_gate
        self.heads_num = heads_num
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

        # ===== Gate Modules (如果启用) =====
        if self.use_gate:
            # Raw接收Size信息的Gate
            self.gate_raw = CrossAttentionGate(hidden_size, heads_num, dropout)
            # Size接收Raw信息的Gate
            self.gate_size = CrossAttentionGate(hidden_size, heads_num, dropout)

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
        # ===== Step 1: Self-Attention (parallel for both branches) =====
        # Raw Self-Attention
        raw_self, _ = self.self_attn_raw(raw_feat, raw_feat, raw_feat, raw_mask, None)
        raw_self = self.dropout_raw_1(raw_self)
        raw_feat_sa = self.layer_norm_raw_1(raw_feat + raw_self)

        # Size Self-Attention
        size_self, _ = self.self_attn_size(size_feat, size_feat, size_feat, size_mask, None)
        size_self = self.dropout_size_1(size_self)
        size_feat_sa = self.layer_norm_size_1(size_feat + size_self)

        # ===== Step 2: Cross-Attention (with optional Gate) =====
        if self.use_gate:
            # 提取CLS tokens用于模态级Gate
            raw_cls = raw_feat_sa[:, 0, :]  # [B, H]
            size_cls = size_feat_sa[:, 0, :]  # [B, H]

            # 计算Gate
            # Raw要从Size获取信息: query=raw, key=size
            gate_raw = self.gate_raw(raw_feat_sa, raw_cls, size_cls)  # [B, heads, L_r, 1]
            # Size要从Raw获取信息: query=size, key=raw
            gate_size = self.gate_size(size_feat_sa, size_cls, raw_cls)  # [B, heads, L_s, 1]

            # Raw Cross-Attention with Gate
            raw_cross, _ = self.cross_attn_raw(
                size_feat_sa, size_feat_sa, raw_feat_sa,
                cross_mask_r2s, None,
                logits_gate=gate_raw
            )

            # Size Cross-Attention with Gate
            size_cross, _ = self.cross_attn_size(
                raw_feat_sa, raw_feat_sa, size_feat_sa,
                cross_mask_s2r, None,
                logits_gate=gate_size
            )
        else:
            # 原始Cross-Attention (无Gate)
            raw_cross, _ = self.cross_attn_raw(size_feat_sa, size_feat_sa, raw_feat_sa, cross_mask_r2s, None)
            size_cross, _ = self.cross_attn_size(raw_feat_sa, raw_feat_sa, size_feat_sa, cross_mask_s2r, None)

        raw_cross = self.dropout_raw_2(raw_cross)
        raw_feat_ca = self.layer_norm_raw_2(raw_feat_sa + raw_cross)

        size_cross = self.dropout_size_2(size_cross)
        size_feat_ca = self.layer_norm_size_2(size_feat_sa + size_cross)

        # ===== Step 3: FFN (parallel for both branches) =====
        # Raw FFN
        raw_ffn = self.ffn_raw(raw_feat_ca)
        raw_ffn = self.dropout_raw_3(raw_ffn)
        raw_out = self.layer_norm_raw_3(raw_feat_ca + raw_ffn)

        # Size FFN
        size_ffn = self.ffn_size(size_feat_ca)
        size_ffn = self.dropout_size_3(size_ffn)
        size_out = self.layer_norm_size_3(size_feat_ca + size_ffn)

        return raw_out, size_out


class MultiModalFusionEncoder(nn.Module):
    """
    多层双向Cross-Attention Fusion Encoder

    包含多层BidirectionalFusionLayer，支持可选的Gate机制
    """

    def __init__(self, args, num_layers=6, use_gate=False):
        super(MultiModalFusionEncoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = args.hidden_size
        self.use_gate = use_gate

        # 构建多层Fusion
        self.fusion_layers = nn.ModuleList([
            BidirectionalFusionLayer(
                hidden_size=args.hidden_size,
                heads_num=args.heads_num,
                feedforward_size=args.feedforward_size,
                hidden_act=args.hidden_act,
                dropout=args.dropout,
                use_gate=use_gate
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
