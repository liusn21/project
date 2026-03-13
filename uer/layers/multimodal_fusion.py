"""
Multi-Modal Fusion Module for Stage 2 Pretraining (ALBEF-style)

Architecture:
- 6层双向Cross-Attention Fusion
- 每层包含: Self-Attention + Cross-Attention (双向) + FFN
- 参考LXMERT的双向设计

Gate mechanism - ITGCA (Information-Theoretic Gated Cross-Attention) — Asymmetric:
- Size←Raw (Raw source, may degrade):
  - Source-side V gating: V_raw *= 0.1 + 0.9 * local_ent_raw
  - Modality gate: flow-level Shannon entropy prior + bilinear CLS correction
  - Token gate: pure learned (SA residual)
- Raw←Size (Size source, stable):
  - Modality gate: pure learned (no statistical prior)
  - Token gate: pure learned (SA residual)
- Hierarchical: g = r_mod * g_token + (1-r_mod) * g_default
- Gate applied AFTER final_linear, per-position [B, L_q, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward


class ITGCrossAttentionGate(nn.Module):
    """
    Information-Theoretic Gated Cross-Attention Gate (ITGCA) — Asymmetric

    Hierarchical gate:
    1. Modality gate (r_mod):
       - If has_modality_prior: r_mod = r_stat + sigmoid(alpha_1) * (r_learned - r_stat)
       - Otherwise: r_mod = r_learned (pure learned)

    2. Token gate (g_token): pure learned from SA residual
       g_token = sigmoid(sum(W_k(q) * W_v(delta)))

    Combined: g = r_mod * g_token + (1 - r_mod) * sigmoid(g_default_logit)
    Output: [B, L_q, 1] (per-position, applied after final_linear)
    """

    def __init__(self, hidden_size, dropout=0.1, has_modality_prior=False):
        super(ITGCrossAttentionGate, self).__init__()

        self.hidden_size = hidden_size
        self.has_modality_prior = has_modality_prior
        bottleneck = hidden_size // 4

        # ===== Modality Gate =====
        # Bilinear: r_learned = sigmoid(c_q^T W c_k + bias)
        self.bilinear_W = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bilinear_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.bilinear_W, gain=0.1)

        # alpha_1: gated residual mixing. sigmoid(-2.0) ≈ 0.12
        # Only created when has_modality_prior=True (Size←Raw direction)
        if self.has_modality_prior:
            self.alpha_modality = nn.Parameter(torch.tensor(-2.0))

        # ===== Token Gate (pure learned, no local_ent prior) =====
        # SA residual pointwise attention: g_token = sigmoid(sum(W_k(q) * W_v(delta)))
        self.W_k = nn.Linear(hidden_size, bottleneck, bias=False)
        self.W_v = nn.Linear(hidden_size, bottleneck, bias=False)

        # ===== Default gate (when modality unreliable) =====
        # sigmoid(0.0) = 0.5
        self.g_default_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, query_feat, sa_delta, encoder_cls_q, encoder_cls_k,
                r_stat):
        """
        Args:
            query_feat: [B, L_q, H] - Query features (post-SA, post-LayerNorm)
            sa_delta: [B, L_q, H] - SA residual (post_SA_normed - pre_SA)
            encoder_cls_q: [B, H] - Query encoder CLS (detached, pre-fusion)
            encoder_cls_k: [B, H] - Source encoder CLS (detached, pre-fusion)
            r_stat: [B] or None - Statistical reliability of source modality
                    (only used when has_modality_prior=True)

        Returns:
            gate: [B, L_q, 1] - Final gate values
            r_mod: [B] - Modality gate values (for CRC loss)
        """
        B, L_q, H = query_feat.shape

        # ===== Modality Gate =====
        # r_learned = sigmoid(c_q^T W c_k + bias)
        r_logit = torch.sum(
            torch.matmul(encoder_cls_q, self.bilinear_W) * encoder_cls_k,
            dim=-1
        ) + self.bilinear_bias.squeeze()  # [B]
        r_learned = torch.sigmoid(r_logit)  # [B]

        if self.has_modality_prior and r_stat is not None:
            # Gated residual: r_mod = r_stat + beta_1 * (r_learned - r_stat)
            beta_1 = torch.sigmoid(self.alpha_modality)
            r_mod = r_stat + beta_1 * (r_learned - r_stat)  # [B]
        else:
            r_mod = r_learned  # [B]

        # ===== Token Gate (pure learned) =====
        # g_token = sigmoid(sum(W_k(query) * W_v(delta), dim=-1))
        q_proj = self.W_k(query_feat)   # [B, L_q, bottleneck]
        d_proj = self.W_v(sa_delta)     # [B, L_q, bottleneck]
        t_logit = (q_proj * d_proj).sum(dim=-1)  # [B, L_q]
        g_token = torch.sigmoid(t_logit)  # [B, L_q]

        # ===== Hierarchical Combination =====
        # g = r_mod * g_token + (1 - r_mod) * g_default
        r_mod_ex = r_mod.unsqueeze(1)  # [B, 1]
        g_def = torch.sigmoid(self.g_default_logit)  # scalar in [0, 1]
        gate = r_mod_ex * g_token + (1 - r_mod_ex) * g_def  # [B, L_q]

        gate = gate.unsqueeze(-1)  # [B, L_q, 1]

        return gate, r_mod


class BidirectionalFusionLayer(nn.Module):
    """
    单层双向Cross-Attention Fusion

    结构:
        Raw Branch:  Self-Attn → Cross-Attn(Q=raw, KV=size) → FFN
        Size Branch: Self-Attn → Cross-Attn(Q=size, KV=raw) → FFN

    Gate: ITGCA (use_itgca=True) provides information-theoretic per-position gating.
    """

    def __init__(self, hidden_size, heads_num, feedforward_size, hidden_act, dropout,
                 use_itgca=False):
        super(BidirectionalFusionLayer, self).__init__()

        self.use_itgca = use_itgca
        self.heads_num = heads_num
        attention_head_size = hidden_size // heads_num

        # ===== Raw Branch =====
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

        self.ffn_raw = PositionwiseFeedForward(
            hidden_size=hidden_size,
            feedforward_size=feedforward_size,
            hidden_act=hidden_act,
            has_bias=True
        )
        self.layer_norm_raw_3 = LayerNorm(hidden_size)
        self.dropout_raw_3 = nn.Dropout(dropout)

        # ===== Size Branch =====
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

        self.ffn_size = PositionwiseFeedForward(
            hidden_size=hidden_size,
            feedforward_size=feedforward_size,
            hidden_act=hidden_act,
            has_bias=True
        )
        self.layer_norm_size_3 = LayerNorm(hidden_size)
        self.dropout_size_3 = nn.Dropout(dropout)

        # ===== Gate Modules (Asymmetric) =====
        if self.use_itgca:
            # Raw←Size: no statistical prior (Size is stable source)
            self.gate_raw = ITGCrossAttentionGate(hidden_size, dropout, has_modality_prior=False)
            # Size←Raw: has modality prior (Raw source may degrade)
            self.gate_size = ITGCrossAttentionGate(hidden_size, dropout, has_modality_prior=True)

    def forward(self, raw_feat, size_feat, raw_mask, size_mask,
                cross_mask_r2s, cross_mask_s2r,
                raw_cls_enc=None, size_cls_enc=None,
                r_stat_raw=None,
                local_ent_raw=None):
        """
        Args:
            raw_feat: [batch, seq_len_raw, hidden]
            size_feat: [batch, seq_len_size, hidden]
            raw_mask: [batch, 1, seq_len_raw, seq_len_raw]
            size_mask: [batch, 1, seq_len_size, seq_len_size]
            cross_mask_r2s: [batch, 1, seq_len_raw, seq_len_size]
            cross_mask_s2r: [batch, 1, seq_len_size, seq_len_raw]
            raw_cls_enc: [batch, hidden] - Raw encoder CLS (detached), for ITGCA
            size_cls_enc: [batch, hidden] - Size encoder CLS (detached), for ITGCA
            r_stat_raw: [batch] - Raw flow-level reliability, for ITGCA (Size←Raw direction only)
            local_ent_raw: [batch, seq_len_raw] - Raw local entropy, for source-side V gating

        Returns:
            raw_out: [batch, seq_len_raw, hidden]
            size_out: [batch, seq_len_size, hidden]
            gate_info: dict or None - Gate statistics for ITGCA losses
        """
        # ===== Step 1: Self-Attention =====
        raw_self, _ = self.self_attn_raw(raw_feat, raw_feat, raw_feat, raw_mask, None)
        raw_self = self.dropout_raw_1(raw_self)
        raw_feat_sa = self.layer_norm_raw_1(raw_feat + raw_self)

        size_self, _ = self.self_attn_size(size_feat, size_feat, size_feat, size_mask, None)
        size_self = self.dropout_size_1(size_self)
        size_feat_sa = self.layer_norm_size_1(size_feat + size_self)

        # ===== Step 2: Cross-Attention =====
        gate_info = None

        if self.use_itgca:
            # SA residual: delta = post_SA_normed - pre_SA
            raw_delta = raw_feat_sa - raw_feat    # [B, L_raw, H]
            size_delta = size_feat_sa - size_feat  # [B, L_size, H]

            # Gate for Raw←Size (raw receives from size): no statistical prior
            gate_raw, r_mod_r2s = self.gate_raw(
                raw_feat_sa, raw_delta, raw_cls_enc, size_cls_enc,
                r_stat=None
            )

            # Gate for Size←Raw (size receives from raw): has modality prior
            gate_size, r_mod_s2r = self.gate_size(
                size_feat_sa, size_delta, size_cls_enc, raw_cls_enc,
                r_stat=r_stat_raw
            )

            # Source-side V gating for Size←Raw: scale Raw values by local entropy
            # raw_V_gated ∈ [0.1, 1.0] * raw_feat_sa (low entropy → attenuated)
            if local_ent_raw is not None:
                raw_V_gated = raw_feat_sa * (0.1 + 0.9 * local_ent_raw.unsqueeze(-1))
            else:
                raw_V_gated = raw_feat_sa

            # Cross-attention with ITGCA gate
            # Raw←Size: Key=size, Value=size, Query=raw
            raw_cross, _ = self.cross_attn_raw(
                size_feat_sa, size_feat_sa, raw_feat_sa,
                cross_mask_r2s, None, logits_gate=gate_raw
            )
            # Size←Raw: Key=raw, Value=raw_V_gated (source-side gated), Query=size
            size_cross, _ = self.cross_attn_size(
                raw_feat_sa, raw_V_gated, size_feat_sa,
                cross_mask_s2r, None, logits_gate=gate_size
            )

            gate_info = {
                'r_mod_r2s': r_mod_r2s,              # [B]
                'r_mod_s2r': r_mod_s2r,              # [B]
                'gate_raw': gate_raw.squeeze(-1),     # [B, L_raw]
                'gate_size': gate_size.squeeze(-1),   # [B, L_size]
            }

        else:
            raw_cross, _ = self.cross_attn_raw(
                size_feat_sa, size_feat_sa, raw_feat_sa, cross_mask_r2s, None
            )
            size_cross, _ = self.cross_attn_size(
                raw_feat_sa, raw_feat_sa, size_feat_sa, cross_mask_s2r, None
            )

        # Residual + LayerNorm
        raw_cross = self.dropout_raw_2(raw_cross)
        raw_feat_ca = self.layer_norm_raw_2(raw_feat_sa + raw_cross)

        size_cross = self.dropout_size_2(size_cross)
        size_feat_ca = self.layer_norm_size_2(size_feat_sa + size_cross)

        # ===== Step 3: FFN =====
        raw_ffn = self.ffn_raw(raw_feat_ca)
        raw_ffn = self.dropout_raw_3(raw_ffn)
        raw_out = self.layer_norm_raw_3(raw_feat_ca + raw_ffn)

        size_ffn = self.ffn_size(size_feat_ca)
        size_ffn = self.dropout_size_3(size_ffn)
        size_out = self.layer_norm_size_3(size_feat_ca + size_ffn)

        return raw_out, size_out, gate_info


class MultiModalFusionEncoder(nn.Module):
    """
    多层双向Cross-Attention Fusion Encoder

    包含多层BidirectionalFusionLayer，支持可选的ITGCA门控机制
    """

    def __init__(self, args, num_layers=6, use_itgca=False):
        super(MultiModalFusionEncoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = args.hidden_size
        self.use_itgca = use_itgca

        self.fusion_layers = nn.ModuleList([
            BidirectionalFusionLayer(
                hidden_size=args.hidden_size,
                heads_num=args.heads_num,
                feedforward_size=args.feedforward_size,
                hidden_act=args.hidden_act,
                dropout=args.dropout,
                use_itgca=use_itgca
            )
            for _ in range(num_layers)
        ])

    def forward(self, raw_feat, size_feat, raw_seg, size_seg,
                raw_cls_enc=None, size_cls_enc=None,
                r_stat_raw=None,
                local_ent_raw=None):
        """
        Args:
            raw_feat: [batch, seq_len_raw, hidden]
            size_feat: [batch, seq_len_size, hidden]
            raw_seg: [batch, seq_len_raw] - for mask generation
            size_seg: [batch, seq_len_size] - for mask generation
            raw_cls_enc: [batch, hidden] - Raw encoder CLS (detached), for ITGCA
            size_cls_enc: [batch, hidden] - Size encoder CLS (detached), for ITGCA
            r_stat_raw: [batch] - Raw flow-level reliability, for ITGCA (Size←Raw only)
            local_ent_raw: [batch, seq_len_raw] - Raw local entropy, for source-side V gating

        Returns:
            raw_fused: [batch, seq_len_raw, hidden]
            size_fused: [batch, seq_len_size, hidden]
            all_gate_info: list of gate_info dicts (one per layer) or None
        """
        batch_size = raw_feat.size(0)
        seq_len_raw = raw_feat.size(1)
        seq_len_size = size_feat.size(1)

        # ===== Generate Attention Masks =====
        raw_mask = (raw_seg > 0).unsqueeze(1).unsqueeze(2)
        raw_mask = raw_mask.expand(-1, -1, seq_len_raw, -1).float()
        raw_mask = (1.0 - raw_mask) * -10000.0

        size_mask = (size_seg > 0).unsqueeze(1).unsqueeze(2)
        size_mask = size_mask.expand(-1, -1, seq_len_size, -1).float()
        size_mask = (1.0 - size_mask) * -10000.0

        cross_mask_r2s = (size_seg > 0).unsqueeze(1).unsqueeze(2)
        cross_mask_r2s = cross_mask_r2s.expand(-1, -1, seq_len_raw, -1).float()
        cross_mask_r2s = (1.0 - cross_mask_r2s) * -10000.0

        cross_mask_s2r = (raw_seg > 0).unsqueeze(1).unsqueeze(2)
        cross_mask_s2r = cross_mask_s2r.expand(-1, -1, seq_len_size, -1).float()
        cross_mask_s2r = (1.0 - cross_mask_s2r) * -10000.0

        # ===== Layer-by-layer Fusion =====
        raw_hidden = raw_feat
        size_hidden = size_feat
        all_gate_info = []

        for layer in self.fusion_layers:
            raw_hidden, size_hidden, gate_info = layer(
                raw_hidden, size_hidden,
                raw_mask, size_mask,
                cross_mask_r2s, cross_mask_s2r,
                raw_cls_enc=raw_cls_enc,
                size_cls_enc=size_cls_enc,
                r_stat_raw=r_stat_raw,
                local_ent_raw=local_ent_raw
            )
            if gate_info is not None:
                all_gate_info.append(gate_info)

        return raw_hidden, size_hidden, all_gate_info if all_gate_info else None
