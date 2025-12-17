"""
Multi-Modal Target for Stage 2 Pretraining (ALBEF-style)

包含三个预训练任务:
1. ITC (Image-Text Contrastive): 对比学习对齐Raw和Size的表示空间
2. ITM (Image-Text Matching): 跨模态匹配二分类任务，使用hard negative mining
3. MLM (Masked Language Modeling): 对两个模态都做MLM预测

参考ALBEF的设计:
- ITC在Fusion之前计算，使用Momentum Encoder + Queue扩大负样本
- ITM在Fusion之后计算，使用ITC的相似度做hard negative sampling
- MLM在Fusion之后计算，利用跨模态信息帮助预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.layer_norm import LayerNorm
from uer.utils import *


class MultiModalTarget(nn.Module):
    """
    Multi-Modal Pretraining Target (ALBEF-style)

    Tasks:
    1. ITC: Contrastive learning between Raw and Size CLS tokens
    2. ITM: Binary classification after fusion (with hard negative mining)
    3. MLM_raw: MLM for Raw modality (after fusion)
    4. MLM_size: MLM for Size modality (after fusion)
    """

    def __init__(self, args, hidden_size, vocab_size_raw, vocab_size_size):
        super(MultiModalTarget, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size_raw = vocab_size_raw
        self.vocab_size_size = vocab_size_size

        # Activation function
        self.act = str2act[args.hidden_act]

        # Factorized embedding parameterization setting
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.emb_size = args.emb_size

        # ===== ITC: Contrastive Learning Head =====
        # Project to lower dimension for contrastive learning (optional, can be same as hidden_size)
        self.itc_proj_raw = nn.Linear(hidden_size, hidden_size)
        self.itc_proj_size = nn.Linear(hidden_size, hidden_size)

        # ===== ITM: Binary Classification Head =====
        # Input: concatenation of raw_fused_cls and size_fused_cls [2 * hidden_size]
        self.itm_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, 2)  # Binary classification
        )
        self.itm_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))

        # ===== MLM_raw: MLM Head for Raw modality =====
        if self.factorized_embedding_parameterization:
            self.mlm_raw_linear_1 = nn.Linear(hidden_size, self.emb_size)
            self.mlm_raw_layer_norm = LayerNorm(self.emb_size)
            self.mlm_raw_linear_2 = nn.Linear(self.emb_size, vocab_size_raw)
        else:
            self.mlm_raw_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.mlm_raw_layer_norm = LayerNorm(hidden_size)
            self.mlm_raw_linear_2 = nn.Linear(hidden_size, vocab_size_raw)

        self.mlm_raw_softmax = nn.LogSoftmax(dim=-1)
        self.mlm_raw_criterion = nn.NLLLoss()

        # ===== MLM_size: MLM Head for Size modality =====
        if self.factorized_embedding_parameterization:
            self.mlm_size_linear_1 = nn.Linear(hidden_size, self.emb_size)
            self.mlm_size_layer_norm = LayerNorm(self.emb_size)
            self.mlm_size_linear_2 = nn.Linear(self.emb_size, vocab_size_size)
        else:
            self.mlm_size_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.mlm_size_layer_norm = LayerNorm(hidden_size)
            self.mlm_size_linear_2 = nn.Linear(hidden_size, vocab_size_size)

        self.mlm_size_softmax = nn.LogSoftmax(dim=-1)
        self.mlm_size_criterion = nn.NLLLoss()

    def forward_itc(self, raw_cls, size_cls, raw_cls_m, size_cls_m,
                    raw_queue, size_queue, temperature=0.07):
        """
        ITC (Image-Text Contrastive) Loss

        使用Momentum特征和Queue扩大负样本数量

        Args:
            raw_cls: [batch, hidden] - 当前batch的Raw CLS特征
            size_cls: [batch, hidden] - 当前batch的Size CLS特征
            raw_cls_m: [batch, hidden] - Momentum encoder的Raw CLS特征
            size_cls_m: [batch, hidden] - Momentum encoder的Size CLS特征
            raw_queue: [queue_size, hidden] - Raw特征队列
            size_queue: [queue_size, hidden] - Size特征队列
            temperature: 温度参数

        Returns:
            itc_loss: ITC loss
            sim_r2s: [batch, batch+queue] 相似度矩阵 (用于ITM hard negative mining)
            sim_s2r: [batch, batch+queue] 相似度矩阵
        """
        batch_size = raw_cls.size(0)

        # Project and normalize
        raw_feat = F.normalize(self.itc_proj_raw(raw_cls), dim=-1)  # [batch, hidden]
        size_feat = F.normalize(self.itc_proj_size(size_cls), dim=-1)  # [batch, hidden]

        # Momentum features (already projected and normalized in model)
        raw_feat_m = F.normalize(raw_cls_m, dim=-1)  # [batch, hidden]
        size_feat_m = F.normalize(size_cls_m, dim=-1)  # [batch, hidden]

        # Concatenate with queue
        # raw_feat_all: [batch + queue_size, hidden]
        raw_feat_all = torch.cat([raw_feat_m, raw_queue.clone().detach()], dim=0)
        size_feat_all = torch.cat([size_feat_m, size_queue.clone().detach()], dim=0)

        # Compute similarity
        # sim_r2s[i, j] = similarity between raw_feat[i] and size_feat_all[j]
        sim_r2s = raw_feat @ size_feat_all.T / temperature  # [batch, batch+queue]
        sim_s2r = size_feat @ raw_feat_all.T / temperature  # [batch, batch+queue]

        # Labels: positive pairs are on diagonal (indices 0 to batch-1)
        labels = torch.arange(batch_size, device=raw_cls.device)

        # Contrastive loss (InfoNCE)
        loss_r2s = F.cross_entropy(sim_r2s, labels)
        loss_s2r = F.cross_entropy(sim_s2r, labels)
        itc_loss = (loss_r2s + loss_s2r) / 2

        return itc_loss, sim_r2s, sim_s2r

    def sample_hard_negatives(self, sim_r2s, sim_s2r, batch_size, temperature=0.07):
        """
        Sample hard negative indices based on ITC similarity

        Args:
            sim_r2s: [batch, batch+queue] - Raw到Size的相似度矩阵
            sim_s2r: [batch, batch+queue] - Size到Raw的相似度矩阵
            batch_size: batch size
            temperature: sampling temperature

        Returns:
            neg_size_idx: [batch] - hard negative size indices for each raw
            neg_raw_idx: [batch] - hard negative raw indices for each size
        """
        with torch.no_grad():
            # 只使用batch内的相似度 (取前batch_size列)
            sim_r2s_batch = sim_r2s[:, :batch_size].clone()
            sim_s2r_batch = sim_s2r[:, :batch_size].clone()

            # Mask out positive pairs (diagonal)
            sim_r2s_batch.fill_diagonal_(-float('inf'))
            sim_s2r_batch.fill_diagonal_(-float('inf'))

            # Sample hard negatives based on similarity
            weights_r2s = F.softmax(sim_r2s_batch / temperature, dim=1)
            weights_s2r = F.softmax(sim_s2r_batch / temperature, dim=1)

            # For each raw, sample a hard negative size
            neg_size_idx = torch.multinomial(weights_r2s, 1).squeeze(1)  # [batch]
            # For each size, sample a hard negative raw
            neg_raw_idx = torch.multinomial(weights_s2r, 1).squeeze(1)  # [batch]

        return neg_size_idx, neg_raw_idx

    def forward_itm(self, pos_raw_cls, pos_size_cls, neg1_raw_cls, neg1_size_cls, neg2_raw_cls, neg2_size_cls):
        """
        ITM (Image-Text Matching) Loss - ALBEF style

        所有输入都是经过Fusion后的CLS特征（正样本和负样本都经过了各自的Fusion）

        Args:
            pos_raw_cls: [batch, hidden] - 正样本raw经过Fusion后的CLS
            pos_size_cls: [batch, hidden] - 正样本size经过Fusion后的CLS
            neg1_raw_cls: [batch, hidden] - 负样本1的raw经过Fusion后的CLS (raw_i with size_neg)
            neg1_size_cls: [batch, hidden] - 负样本1的size经过Fusion后的CLS
            neg2_raw_cls: [batch, hidden] - 负样本2的raw经过Fusion后的CLS (raw_neg with size_i)
            neg2_size_cls: [batch, hidden] - 负样本2的size经过Fusion后的CLS

        Returns:
            itm_loss: ITM loss
            itm_acc: ITM accuracy
        """
        batch_size = pos_raw_cls.size(0)
        device = pos_raw_cls.device

        # Concatenation to preserve all information from both modalities
        pos_feat = torch.cat([pos_raw_cls, pos_size_cls], dim=-1)  # [batch, 2*hidden]
        neg1_feat = torch.cat([neg1_raw_cls, neg1_size_cls], dim=-1)  # [batch, 2*hidden]
        neg2_feat = torch.cat([neg2_raw_cls, neg2_size_cls], dim=-1)  # [batch, 2*hidden]

        # Positive samples
        pos_logits = self.itm_head(pos_feat)  # [batch, 2]
        pos_labels = torch.ones(batch_size, dtype=torch.long, device=device)

        # Negative samples 1: (raw_i, size_neg)
        neg1_logits = self.itm_head(neg1_feat)  # [batch, 2]
        neg1_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Negative samples 2: (raw_neg, size_i)
        neg2_logits = self.itm_head(neg2_feat)  # [batch, 2]
        neg2_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Combine all
        all_logits = torch.cat([pos_logits, neg1_logits, neg2_logits], dim=0)  # [3*batch, 2]
        all_labels = torch.cat([pos_labels, neg1_labels, neg2_labels], dim=0)  # [3*batch]

        # ITM Loss (ensure weight is on correct device)
        if self.itm_criterion.weight is not None and self.itm_criterion.weight.device != device:
            self.itm_criterion.weight = self.itm_criterion.weight.to(device)
        itm_loss = self.itm_criterion(all_logits, all_labels)

        # ITM Accuracy
        preds = all_logits.argmax(dim=-1)
        itm_acc = (preds == all_labels).float().mean()

        return itm_loss, itm_acc

    def forward_mlm_raw(self, raw_fused, tgt_mlm_raw):
        """
        MLM for Raw modality (after fusion)

        Args:
            raw_fused: [batch, seq_len_raw, hidden] - Fused Raw features
            tgt_mlm_raw: [batch, seq_len_raw] - MLM targets (0=unmasked, token_id=masked)

        Returns:
            mlm_loss: MLM loss
            mlm_correct: Number of correct predictions
            mlm_denominator: Number of masked positions
        """
        # Forward through MLM head
        output = self.act(self.mlm_raw_linear_1(raw_fused))
        output = self.mlm_raw_layer_norm(output)

        # Flatten
        if self.factorized_embedding_parameterization:
            output = output.contiguous().view(-1, self.emb_size)
        else:
            output = output.contiguous().view(-1, self.hidden_size)

        tgt = tgt_mlm_raw.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        output = output[tgt > 0, :]
        tgt = tgt[tgt > 0]

        # Prediction
        output = self.mlm_raw_linear_2(output)
        output = self.mlm_raw_softmax(output)

        # Compute metrics
        denominator = torch.tensor(output.size(0) + 1e-6)
        if output.size(0) == 0:
            correct = torch.tensor(0.0)
            loss = torch.tensor(0.0, device=raw_fused.device)
        else:
            correct = torch.sum((output.argmax(dim=-1).eq(tgt)).float())
            loss = self.mlm_raw_criterion(output, tgt)

        return loss, correct, denominator

    def forward_mlm_size(self, size_fused, tgt_mlm_size):
        """
        MLM for Size modality (after fusion)

        Args:
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            tgt_mlm_size: [batch, seq_len_size] - MLM targets (0=unmasked, token_id=masked)

        Returns:
            mlm_loss: MLM loss
            mlm_correct: Number of correct predictions
            mlm_denominator: Number of masked positions
        """
        # Forward through MLM head
        output = self.act(self.mlm_size_linear_1(size_fused))
        output = self.mlm_size_layer_norm(output)

        # Flatten
        if self.factorized_embedding_parameterization:
            output = output.contiguous().view(-1, self.emb_size)
        else:
            output = output.contiguous().view(-1, self.hidden_size)

        tgt = tgt_mlm_size.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        output = output[tgt > 0, :]
        tgt = tgt[tgt > 0]

        # Prediction
        output = self.mlm_size_linear_2(output)
        output = self.mlm_size_softmax(output)

        # Compute metrics
        denominator = torch.tensor(output.size(0) + 1e-6)
        if output.size(0) == 0:
            correct = torch.tensor(0.0)
            loss = torch.tensor(0.0, device=size_fused.device)
        else:
            correct = torch.sum((output.argmax(dim=-1).eq(tgt)).float())
            loss = self.mlm_size_criterion(output, tgt)

        return loss, correct, denominator
