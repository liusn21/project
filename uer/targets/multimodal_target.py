"""
Multi-Modal Target for Stage 2 Pretraining

包含两个预训练任务:
1. CMM (Cross-Modal Matching): 跨模态匹配，ITM二分类任务
2. CMMP (Cross-Modal Masked Prediction): 跨模态掩码预测，MLM任务

Architecture v2:
- CMM和CMMP都在Fusion之后计算
- CMM使用Element-wise Product + MLP进行二分类
- CMMP复用MlmTarget的实现逻辑
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
    2. CMMP: MLM任务（预测masked Size tokens，复用MlmTarget逻辑）

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

        # ===== CMMP: MLM Head (复用MlmTarget架构) =====
        if self.factorized_embedding_parameterization:
            self.cmmp_linear_1 = nn.Linear(hidden_size, self.emb_size)
            self.cmmp_layer_norm = LayerNorm(self.emb_size)
            self.cmmp_linear_2 = nn.Linear(self.emb_size, vocab_size_size)
        else:
            self.cmmp_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.cmmp_layer_norm = LayerNorm(hidden_size)
            self.cmmp_linear_2 = nn.Linear(hidden_size, vocab_size_size)

        self.cmmp_softmax = nn.LogSoftmax(dim=-1)
        self.cmmp_criterion = nn.NLLLoss()

    def cmmp(self, size_fused, tgt_cmmp_size):
        """
        CMMP (Cross-Modal Masked Prediction) 任务

        复用MlmTarget.mlm()的实现逻辑，保持与原框架一致

        Args:
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            tgt_cmmp_size: [batch, seq_len_size] - Target tokens (0=unmasked, token_id=masked)

        Returns:
            cmmp_loss: CMMP loss
            cmmp_correct: Number of correct predictions
            cmmp_denominator: Number of masked positions
        """
        # Forward through CMMP head (same as MlmTarget.mlm())
        output_cmmp = self.act(self.cmmp_linear_1(size_fused))
        output_cmmp = self.cmmp_layer_norm(output_cmmp)

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
        output_cmmp = self.cmmp_linear_2(output_cmmp)
        output_cmmp = self.cmmp_softmax(output_cmmp)

        # Compute denominator and correct count
        denominator = torch.tensor(output_cmmp.size(0) + 1e-6)
        if output_cmmp.size(0) == 0:
            correct_cmmp = torch.tensor(0.0)
        else:
            correct_cmmp = torch.sum((output_cmmp.argmax(dim=-1).eq(tgt_cmmp_size)).float())

        # Compute loss
        cmmp_loss = self.cmmp_criterion(output_cmmp, tgt_cmmp_size)

        return cmmp_loss, correct_cmmp, denominator

    def forward_cmmp_only(self, size_fused, tgt_cmmp_size):
        """
        Compute CMMP loss only

        Args:
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            tgt_cmmp_size: [batch, seq_len_size] - Target Size tokens

        Returns:
            cmmp_loss, cmmp_correct, cmmp_denominator
        """
        return self.cmmp(size_fused, tgt_cmmp_size)

    def forward_cmm_itm(self, raw_fused, size_fused, temperature=0.07):
        """
        CMM作为标准ITM二分类任务

        实现步骤：
        1. 困难负样本挖掘（基于相似度，batch内采样）
        2. 构建50% pos + 50% neg训练样本
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

        # ===== Step 2: Construct Training Samples (50% pos + 50% neg) =====
        raw_features = []
        size_features = []
        labels = []

        # Random 50% pos + 50% neg
        random_probs = torch.rand(batch_size, device=device)
        is_positive = random_probs < 0.5

        for i in range(batch_size):
            if is_positive[i]:
                # Positive sample: matched pair
                raw_features.append(raw_cls[i])
                size_features.append(size_cls[i])
                labels.append(1.0)
            else:
                # Negative sample: hard negative
                raw_features.append(raw_cls[i])
                size_features.append(size_cls[neg_indices[i]])
                labels.append(0.0)

        raw_features = torch.stack(raw_features)  # [batch, hidden]
        size_features = torch.stack(size_features)  # [batch, hidden]
        labels = torch.tensor(labels, dtype=torch.float, device=device)  # [batch]

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
