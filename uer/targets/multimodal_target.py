"""
Multi-Modal Target for Stage 2 Pretraining

包含两个预训练任务:
1. CMM (Cross-Modal Matching): 跨模态匹配 (50% 正样本, 50% 负样本)
2. CMMP (Cross-Modal Masked Prediction): 跨模态掩码预测 (Raw → Size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalTarget(nn.Module):
    """
    Multi-Modal Pretraining Target

    Tasks:
    1. CMM: Binary classification (matching or not)
    2. CMMP: Predict masked Size tokens from Raw features

    Args:
        hidden_size: Hidden dimension
        vocab_size_raw: Vocabulary size for Raw Packet
        vocab_size_size: Vocabulary size for Packet Size
    """
    def __init__(self, hidden_size, vocab_size_raw, vocab_size_size):
        super(MultiModalTarget, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size_raw = vocab_size_raw
        self.vocab_size_size = vocab_size_size

        # CMM: Cross-Modal Matching head
        # 使用[CLS] token进行二分类
        self.cmm_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Binary classification
        )

        # CMMP: Cross-Modal Masked Prediction head for Size
        # 预测被mask的Size tokens
        self.cmmp_linear_1 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.cmmp_linear_2 = nn.Linear(hidden_size, vocab_size_size)
        self.activation = nn.GELU()

        # Loss functions
        self.cmm_criterion = nn.BCEWithLogitsLoss()
        self.cmmp_criterion = nn.CrossEntropyLoss()

    def forward(self, raw_fused, size_fused, tgt_cmm, tgt_cmmp_size):
        """
        Compute CMM and CMMP losses

        Args:
            raw_fused: [batch, seq_len_raw, hidden] - Fused Raw features
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            tgt_cmm: [batch] - CMM labels (0 or 1, 1 for matching)
            tgt_cmmp_size: [batch, seq_len_size] - Target Size tokens for CMMP

        Returns:
            cmm_loss: CMM task loss
            cmmp_loss: CMMP task loss
            cmm_correct: Number of correct CMM predictions
            cmmp_correct: Number of correct CMMP predictions
            cmmp_denominator: Number of masked positions in CMMP
        """
        batch_size = raw_fused.size(0)

        # ==================== CMM Task ====================
        # 使用fused [CLS] token (position 0)
        raw_cls = raw_fused[:, 0, :]   # [batch, hidden]
        size_cls = size_fused[:, 0, :]  # [batch, hidden]

        # 计算相似度 (cosine similarity)
        # Normalize features
        raw_cls_norm = F.normalize(raw_cls, p=2, dim=1)
        size_cls_norm = F.normalize(size_cls, p=2, dim=1)

        # Cosine similarity
        cmm_scores = (raw_cls_norm * size_cls_norm).sum(dim=1)  # [batch]

        # Binary Cross Entropy Loss
        tgt_cmm_float = tgt_cmm.float()  # [batch]
        cmm_loss = self.cmm_criterion(cmm_scores, tgt_cmm_float)

        # CMM accuracy
        cmm_pred = (torch.sigmoid(cmm_scores) > 0.5).long()  # [batch]
        cmm_correct = (cmm_pred == tgt_cmm).sum().float()

        # ==================== CMMP Task ====================
        # 预测被mask的Size tokens (从Raw特征)
        # 只计算masked positions的loss

        # size_fused: [batch, seq_len_size, hidden]
        # tgt_cmmp_size: [batch, seq_len_size]

        # Forward through CMMP head
        size_hidden = self.cmmp_linear_1(size_fused)  # [batch, seq_len_size, hidden]
        size_hidden = self.activation(size_hidden)
        size_hidden = self.layer_norm(size_hidden)
        cmmp_logits = self.cmmp_linear_2(size_hidden)  # [batch, seq_len_size, vocab_size_size]

        # 只计算masked positions (tgt > 0)
        # tgt_cmmp_size: 0 for unmasked, token_id for masked
        mask_positions = (tgt_cmmp_size > 0)  # [batch, seq_len_size]

        if mask_positions.sum() > 0:
            # Flatten for loss computation
            cmmp_logits_masked = cmmp_logits[mask_positions]  # [num_masked, vocab_size_size]
            tgt_cmmp_masked = tgt_cmmp_size[mask_positions]    # [num_masked]

            # Cross Entropy Loss
            cmmp_loss = self.cmmp_criterion(cmmp_logits_masked, tgt_cmmp_masked)

            # CMMP accuracy
            cmmp_pred = cmmp_logits_masked.argmax(dim=-1)  # [num_masked]
            cmmp_correct = (cmmp_pred == tgt_cmmp_masked).sum().float()
            cmmp_denominator = mask_positions.sum().float()
        else:
            # No masked positions (shouldn't happen in practice)
            cmmp_loss = torch.tensor(0.0, device=raw_fused.device)
            cmmp_correct = torch.tensor(0.0, device=raw_fused.device)
            cmmp_denominator = torch.tensor(1.0, device=raw_fused.device)

        return cmm_loss, cmmp_loss, cmm_correct, cmmp_correct, cmmp_denominator


def hard_negative_sampling(raw_features, size_features, temperature=0.07):
    """
    Hard Negative Sampling for CMM task

    基于相似度采样负样本: 相似度越高，越容易被选中

    Args:
        raw_features: [batch, hidden] - Raw [CLS] features
        size_features: [batch, hidden] - Size [CLS] features
        temperature: Temperature for softmax (default: 0.07)

    Returns:
        neg_indices: [batch] - Indices of hard negative samples
    """
    batch_size = raw_features.size(0)
    device = raw_features.device

    # Normalize features
    raw_norm = F.normalize(raw_features, p=2, dim=1)  # [batch, hidden]
    size_norm = F.normalize(size_features, p=2, dim=1)  # [batch, hidden]

    # 计算相似度矩阵
    similarities = torch.matmul(raw_norm, size_norm.T)  # [batch, batch]

    # 对于每个样本i，排除自己 (i, i)
    # 设置对角线为-inf
    similarities.fill_diagonal_(-float('inf'))

    # 基于相似度采样 (相似度高的更容易被选中)
    probs = F.softmax(similarities / temperature, dim=1)  # [batch, batch]

    # 采样
    neg_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [batch]

    return neg_indices
