"""
Multi-Modal Target for Stage 2 Pretraining (v3)

包含三个预训练任务:
1. CMM (Cross-Modal Matching): 跨模态匹配，ITM二分类任务
2. CMMP_raw (Cross-Modal Masked Prediction - Raw): 用Raw的fusion信息预测Raw
3. CMMP_size (Cross-Modal Masked Prediction - Size): 用Size的fusion信息预测Size

Architecture v3:
- CMM和CMMP都在Fusion之后计算
- CMM使用Element-wise Product + MLP进行二分类
- CMM负样本构建已向量化优化
- CMMP_raw和CMMP_size分别复用MlmTarget的实现逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.layer_norm import LayerNorm
from uer.utils import *


class MultiModalTarget(nn.Module):
    """
    Multi-Modal Pretraining Target

    Tasks:
    1. CMM: ITM二分类（困难负样本挖掘 + Element-wise Product + MLP）
    2. CMMP_raw: MLM任务（预测masked Raw tokens，复用MlmTarget逻辑）
    3. CMMP_size: MLM任务（预测masked Size tokens，复用MlmTarget逻辑）

    Args:
        args: Configuration arguments
        hidden_size: Hidden dimension (e.g., 768)
        vocab_size_raw: Vocabulary size for Raw Packet
        vocab_size_size: Vocabulary size for Packet Size
    """

    def __init__(self, args, hidden_size, vocab_size_raw, vocab_size_size):
        super(MultiModalTarget, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size_raw = vocab_size_raw
        self.vocab_size_size = vocab_size_size

        # Get activation function from args (consistent with framework)
        self.act = str2act[args.hidden_act]

        # Get factorized embedding parameterization setting
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.emb_size = args.emb_size

        # ===== CMM: ITM Binary Classification Head =====
        # Element-wise product + MLP
        self.itm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 768 -> 384
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size // 2, 1)  # 384 -> 1
        )
        self.itm_criterion = nn.BCEWithLogitsLoss()

        # ===== CMMP_raw: MLM Head for Raw modality =====
        if self.factorized_embedding_parameterization:
            self.cmmp_raw_linear_1 = nn.Linear(hidden_size, self.emb_size)
            self.cmmp_raw_layer_norm = LayerNorm(self.emb_size)
            self.cmmp_raw_linear_2 = nn.Linear(self.emb_size, vocab_size_raw)
        else:
            self.cmmp_raw_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.cmmp_raw_layer_norm = LayerNorm(hidden_size)
            self.cmmp_raw_linear_2 = nn.Linear(hidden_size, vocab_size_raw)

        self.cmmp_raw_softmax = nn.LogSoftmax(dim=-1)
        self.cmmp_raw_criterion = nn.NLLLoss()

        # ===== CMMP_size: MLM Head for Size modality =====
        if self.factorized_embedding_parameterization:
            self.cmmp_size_linear_1 = nn.Linear(hidden_size, self.emb_size)
            self.cmmp_size_layer_norm = LayerNorm(self.emb_size)
            self.cmmp_size_linear_2 = nn.Linear(self.emb_size, vocab_size_size)
        else:
            self.cmmp_size_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.cmmp_size_layer_norm = LayerNorm(hidden_size)
            self.cmmp_size_linear_2 = nn.Linear(hidden_size, vocab_size_size)

        self.cmmp_size_softmax = nn.LogSoftmax(dim=-1)
        self.cmmp_size_criterion = nn.NLLLoss()

    def forward_cmmp_raw(self, raw_fused, tgt_cmmp_raw):
        """
        CMMP_raw (Cross-Modal Masked Prediction for Raw) 任务

        Args:
            raw_fused: [batch, seq_len_raw, hidden] - Fused Raw features
            tgt_cmmp_raw: [batch, seq_len_raw] - Target tokens (0=unmasked, token_id=masked)

        Returns:
            cmmp_raw_loss: CMMP_raw loss
            cmmp_raw_correct: Number of correct predictions
            cmmp_raw_denominator: Number of masked positions
        """
        # Forward through CMMP_raw head
        output_cmmp = self.act(self.cmmp_raw_linear_1(raw_fused))
        output_cmmp = self.cmmp_raw_layer_norm(output_cmmp)

        # Flatten
        if self.factorized_embedding_parameterization:
            output_cmmp = output_cmmp.contiguous().view(-1, self.emb_size)
        else:
            output_cmmp = output_cmmp.contiguous().view(-1, self.hidden_size)

        tgt_cmmp_raw = tgt_cmmp_raw.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        output_cmmp = output_cmmp[tgt_cmmp_raw > 0, :]
        tgt_cmmp_raw = tgt_cmmp_raw[tgt_cmmp_raw > 0]

        # Prediction head
        output_cmmp = self.cmmp_raw_linear_2(output_cmmp)
        output_cmmp = self.cmmp_raw_softmax(output_cmmp)

        # Compute denominator and correct count
        denominator = torch.tensor(output_cmmp.size(0) + 1e-6)
        if output_cmmp.size(0) == 0:
            correct_cmmp = torch.tensor(0.0)
            cmmp_loss = torch.tensor(0.0, device=raw_fused.device)
        else:
            correct_cmmp = torch.sum((output_cmmp.argmax(dim=-1).eq(tgt_cmmp_raw)).float())
            cmmp_loss = self.cmmp_raw_criterion(output_cmmp, tgt_cmmp_raw)

        return cmmp_loss, correct_cmmp, denominator

    def forward_cmmp_size(self, size_fused, tgt_cmmp_size):
        """
        CMMP_size (Cross-Modal Masked Prediction for Size) 任务

        Args:
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            tgt_cmmp_size: [batch, seq_len_size] - Target tokens (0=unmasked, token_id=masked)

        Returns:
            cmmp_size_loss: CMMP_size loss
            cmmp_size_correct: Number of correct predictions
            cmmp_size_denominator: Number of masked positions
        """
        # Forward through CMMP_size head
        output_cmmp = self.act(self.cmmp_size_linear_1(size_fused))
        output_cmmp = self.cmmp_size_layer_norm(output_cmmp)

        # Flatten
        if self.factorized_embedding_parameterization:
            output_cmmp = output_cmmp.contiguous().view(-1, self.emb_size)
        else:
            output_cmmp = output_cmmp.contiguous().view(-1, self.hidden_size)

        tgt_cmmp_size = tgt_cmmp_size.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        output_cmmp = output_cmmp[tgt_cmmp_size > 0, :]
        tgt_cmmp_size = tgt_cmmp_size[tgt_cmmp_size > 0]

        # Prediction head
        output_cmmp = self.cmmp_size_linear_2(output_cmmp)
        output_cmmp = self.cmmp_size_softmax(output_cmmp)

        # Compute denominator and correct count
        denominator = torch.tensor(output_cmmp.size(0) + 1e-6)
        if output_cmmp.size(0) == 0:
            correct_cmmp = torch.tensor(0.0)
            cmmp_loss = torch.tensor(0.0, device=size_fused.device)
        else:
            correct_cmmp = torch.sum((output_cmmp.argmax(dim=-1).eq(tgt_cmmp_size)).float())
            cmmp_loss = self.cmmp_size_criterion(output_cmmp, tgt_cmmp_size)

        return cmmp_loss, correct_cmmp, denominator

    def forward_cmm_itm(self, raw_fused, size_fused, temperature=0.07):
        """
        CMM作为标准ITM二分类任务 (向量化优化版本)

        实现步骤：
        1. 困难负样本挖掘（基于相似度，batch内采样）
        2. 构建50% pos + 50% neg训练样本（向量化）
        3. Element-wise product作为交互特征
        4. MLP进行二分类

        Args:
            raw_fused: [batch, seq_len_raw, hidden] - Fused Raw features
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            temperature: Temperature for hard negative sampling (default: 0.07)

        Returns:
            cmm_loss: ITM binary classification loss
            cmm_correct: Number of correct predictions
        """
        batch_size = raw_fused.size(0)
        device = raw_fused.device


        # Extract fused [CLS] features
        raw_cls = raw_fused[:, 0, :]  # [batch, hidden]
        size_cls = size_fused[:, 0, :]  # [batch, hidden]

        # ===== Step 1: Hard Negative Mining =====
        with torch.no_grad():
            # Normalize features
            raw_norm = F.normalize(raw_cls, p=2, dim=1)
            size_norm = F.normalize(size_cls, p=2, dim=1)

            # Compute similarity matrix
            similarities = torch.matmul(raw_norm, size_norm.T)  # [batch, batch]
            similarities.fill_diagonal_(-float('inf'))  # Exclude self

            # Sample hard negatives (higher similarity = more likely to be selected)
            probs = F.softmax(similarities / temperature, dim=1)
            neg_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [batch]

        # ===== Step 2: Construct Training Samples (Vectorized) =====
        # Random 50% pos + 50% neg
        is_positive = torch.rand(batch_size, device=device) < 0.5  # [batch]

        # Vectorized selection
        # For positive samples: use size_cls[i]
        # For negative samples: use size_cls[neg_indices[i]]
        neg_size_cls = size_cls[neg_indices]  # [batch, hidden]

        # Use where for vectorized selection
        is_positive_expanded = is_positive.unsqueeze(1)  # [batch, 1]
        size_features = torch.where(is_positive_expanded, size_cls, neg_size_cls)  # [batch, hidden]
        raw_features = raw_cls  # [batch, hidden] (always use raw_cls[i])

        # Labels: 1.0 for positive, 0.0 for negative
        labels = is_positive.float()  # [batch]

        # ===== Step 3: Element-wise Product =====
        interaction = raw_features * size_features  # [batch, hidden]

        # ===== Step 4: MLP Classification =====
        logits = self.itm_head(interaction).squeeze(1)  # [batch]

        # Binary cross-entropy loss
        cmm_loss = self.itm_criterion(logits, labels)

        # Accuracy
        pred = (torch.sigmoid(logits) > 0.5).float()
        cmm_correct = (pred == labels).sum()

        return cmm_loss, cmm_correct