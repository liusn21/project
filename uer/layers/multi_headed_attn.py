import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale = True):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
            )
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None, logits_gate=None):
        """
        Args:
            key: [batch_size x seq_length_k x hidden_size]
            value: [batch_size x seq_length_k x hidden_size]
            query: [batch_size x seq_length_q x hidden_size]
            mask: [batch_size x 1 x seq_length_q x seq_length_k]
            position_bias: [1 x heads_num x seq_length x seq_length]
            logits_gate: Optional gate for cross-attention control.
                         Shape: [batch_size x heads_num x seq_length_q x 1] or broadcastable.
                         Values in [0, 1]. When gate=0, cross-attention output is suppressed.
                         Applied as: attn_output = attn_output * gate (per-head, per-position)
        Returns:
            output: [batch_size x seq_length_q x hidden_size]
            probs: [batch_size x heads_num x seq_length_q x seq_length_k]
        """
        batch_size, seq_length_q, _ = query.size()
        seq_length_k = key.size(1)
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length_q, self.inner_hidden_size)

        # Linear projections: [B, L, H*d] -> [B, H, L, d]
        query = self.linear_layers[0](query).view(batch_size, seq_length_q, heads_num, per_head_size).transpose(1, 2)
        key = self.linear_layers[1](key).view(batch_size, seq_length_k, heads_num, per_head_size).transpose(1, 2)
        value = self.linear_layers[2](value).view(batch_size, seq_length_k, heads_num, per_head_size).transpose(1, 2)

        # Attention scores: [B, H, L_q, L_k]
        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))

        # Apply padding mask
        scores = scores + mask

        # === OLD: Additive gate on logits ===
        # if logits_gate is not None:
        #     gate_mask = (1.0 - logits_gate) * -10000.0
        #     scores = scores + gate_mask

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        # === NEW: Output-level gate (per-head, per-position) ===
        # gate: [B, H, L_q, 1] × attn_output: [B, H, L_q, d] → [B, H, L_q, d]
        # gate=1: full cross-attention; gate=0: output zeroed, residual preserves self-attention
        attn_output = torch.matmul(probs, value)  # [B, H, L_q, d]
        if logits_gate is not None:
            attn_output = attn_output * logits_gate
        output = unshape(attn_output)

        output = self.final_linear(output)
        return output, probs
