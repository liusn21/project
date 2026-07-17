"""
Multi-Modal Target for Stage 2 Pretraining (ALBEF-style) with Temporal Information

Three pretraining tasks:
1. ITC (Image-Text Contrastive): contrastive alignment of Raw and Size representations
2. ITM (Image-Text Matching): cross-modal binary matching with hard negative mining
3. Masked Reconstruction: reconstruct masked Size and IAT tokens from fused features

Design:
- Raw modality is NOT masked (provides context for reconstruction)
- Size + IAT are synchronously masked and reconstructed
- Temporal reconstruction uses soft-label KL divergence loss (same as Stage 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.layer_norm import LayerNorm
from uer.utils import *


def soft_label_kl_loss(logits, target_idx, num_classes, sigma=10, num_special_tokens=5):
    """
    Compute KL divergence loss with soft Gaussian labels.

    Copied from mlm_target.py for consistency.

    Args:
        logits: [N, num_classes] - raw model outputs (before softmax)
        target_idx: [N] - ground truth token indices
        num_classes: int - vocabulary size
        sigma: float - standard deviation of Gaussian
        num_special_tokens: int - number of special tokens at the beginning of vocab
                            (PAD, UNK, CLS, SEP, MASK) to exclude from soft labels

    Returns:
        loss: scalar - KL divergence loss
    """
    device = logits.device

    x = torch.arange(num_classes, device=device, dtype=torch.float32)
    target_soft = torch.exp(
        -(x.unsqueeze(0) - target_idx.unsqueeze(1).float()) ** 2 / (2 * sigma ** 2)
    )
    # Zero out special token positions (PAD, UNK, CLS, SEP, MASK) before normalization
    # to prevent the model from being trained to predict special tokens
    if num_special_tokens > 0:
        target_soft[:, :num_special_tokens] = 0.0
    target_soft = target_soft / target_soft.sum(dim=1, keepdim=True)

    pred_log_prob = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(target_soft * pred_log_prob, dim=-1).mean()

    return loss


class MultiModalTarget(nn.Module):
    """
    Multi-Modal Pretraining Target (ALBEF-style) with Masked Reconstruction

    Tasks:
    1. ITC: Contrastive learning between Raw and Size CLS tokens
    2. ITM: Binary classification after fusion (with hard negative mining)
    3. Masked Reconstruction: Reconstruct masked Size + IAT tokens using fused features

    Note: MLM_raw is removed as Raw modality is no longer masked
    """

    def __init__(self, args, hidden_size, vocab_size_raw, vocab_size_size, vocab_size_temporal=None):
        super(MultiModalTarget, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size_raw = vocab_size_raw
        self.vocab_size_size = vocab_size_size
        self.vocab_size_temporal = vocab_size_temporal if vocab_size_temporal else vocab_size_size

        # Activation function
        self.act = str2act[args.hidden_act]

        # Factorized embedding parameterization setting
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.emb_size = args.emb_size

        # Temporal soft-label sigma (same as Stage 1)
        self.temporal_sigma = getattr(args, 'temporal_sigma', 10)

        # ===== ITC: Contrastive Learning Head =====
        self.itc_proj_raw = nn.Linear(hidden_size, hidden_size)
        self.itc_proj_size = nn.Linear(hidden_size, hidden_size)

        # ===== ITM: Binary Classification Head =====
        self.itm_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, 2)
        )
        self.itm_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))

        # ===== Masked Reconstruction: Size Head =====
        if self.factorized_embedding_parameterization:
            self.recon_size_linear_1 = nn.Linear(hidden_size, self.emb_size)
            self.recon_size_layer_norm = LayerNorm(self.emb_size)
            self.recon_size_linear_2 = nn.Linear(self.emb_size, vocab_size_size)
        else:
            self.recon_size_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.recon_size_layer_norm = LayerNorm(hidden_size)
            self.recon_size_linear_2 = nn.Linear(hidden_size, vocab_size_size)

        self.recon_size_softmax = nn.LogSoftmax(dim=-1)
        self.recon_size_criterion = nn.NLLLoss()

        # ===== Masked Reconstruction: Temporal Head =====
        # Uses soft-label KL divergence loss (no NLLLoss criterion needed)
        if self.factorized_embedding_parameterization:
            self.recon_temporal_linear_1 = nn.Linear(hidden_size, self.emb_size)
            self.recon_temporal_layer_norm = LayerNorm(self.emb_size)
            self.recon_temporal_linear_2 = nn.Linear(self.emb_size, self.vocab_size_temporal)
        else:
            self.recon_temporal_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.recon_temporal_layer_norm = LayerNorm(hidden_size)
            self.recon_temporal_linear_2 = nn.Linear(hidden_size, self.vocab_size_temporal)

    def forward_itc(self, raw_cls, size_cls, raw_cls_m, size_cls_m,
                    raw_queue, size_queue, temperature=0.07, alpha=0.0):
        """
        ITC (Image-Text Contrastive) Loss

        Uses momentum features plus a feature queue to enlarge the negative pool.

        Args:
            raw_cls: [batch, hidden] - current batch Raw CLS features
            size_cls: [batch, hidden] - current batch Size CLS features
            raw_cls_m: [batch, hidden] - momentum encoder Raw CLS features
            size_cls_m: [batch, hidden] - momentum encoder Size CLS features
            raw_queue: [queue_size, hidden] - Raw feature queue
            size_queue: [queue_size, hidden] - Size feature queue
            temperature: contrastive temperature
            alpha: ALBEF momentum-distillation weight in [0, 1]
                   (0 = pure hard-label InfoNCE; >0 mixes in momentum soft targets)

        Returns:
            itc_loss: ITC loss
            sim_r2s: [batch, batch+queue] similarity matrix (reused by ITM hard-negative mining)
            sim_s2r: [batch, batch+queue] similarity matrix
        """
        batch_size = raw_cls.size(0)

        # Project and normalize student (online) features
        raw_feat = F.normalize(self.itc_proj_raw(raw_cls), dim=-1)   # [batch, hidden]
        size_feat = F.normalize(self.itc_proj_size(size_cls), dim=-1)  # [batch, hidden]

        # Momentum (teacher) features — already projected & normalized in the model
        raw_feat_m = F.normalize(raw_cls_m, dim=-1)   # [batch, hidden]
        size_feat_m = F.normalize(size_cls_m, dim=-1)  # [batch, hidden]

        # Append the feature queue to enlarge the negative pool
        raw_feat_all = torch.cat([raw_feat_m, raw_queue.clone().detach()], dim=0)    # [batch+queue, hidden]
        size_feat_all = torch.cat([size_feat_m, size_queue.clone().detach()], dim=0)  # [batch+queue, hidden]

        # Student similarities (with grad)
        sim_r2s = raw_feat @ size_feat_all.T / temperature  # [batch, batch+queue]
        sim_s2r = size_feat @ raw_feat_all.T / temperature  # [batch, batch+queue]

        # ===== Soft targets: ALBEF momentum distillation =====
        # target = alpha * softmax(teacher_sim) + (1 - alpha) * one_hot_positive.
        # The positive for row i is the teacher feature of sample i, on the diagonal
        # of the first `batch` columns. With alpha=0 the target is a pure one-hot, so
        # the loss below reduces EXACTLY to the previous InfoNCE (cross-entropy with
        # arange labels) — this keeps the change backward-compatible / ablatable.
        with torch.no_grad():
            sim_targets = torch.zeros_like(sim_r2s)
            sim_targets.fill_diagonal_(1.0)
            if alpha > 0:
                sim_r2s_m = raw_feat_m @ size_feat_all.T / temperature
                sim_s2r_m = size_feat_m @ raw_feat_all.T / temperature
                sim_r2s_targets = alpha * F.softmax(sim_r2s_m, dim=1) + (1.0 - alpha) * sim_targets
                sim_s2r_targets = alpha * F.softmax(sim_s2r_m, dim=1) + (1.0 - alpha) * sim_targets
            else:
                sim_r2s_targets = sim_targets
                sim_s2r_targets = sim_targets

        # Soft-label cross-entropy (== InfoNCE when alpha=0)
        loss_r2s = -torch.sum(F.log_softmax(sim_r2s, dim=1) * sim_r2s_targets, dim=1).mean()
        loss_s2r = -torch.sum(F.log_softmax(sim_s2r, dim=1) * sim_s2r_targets, dim=1).mean()
        itc_loss = (loss_r2s + loss_s2r) / 2

        return itc_loss, sim_r2s, sim_s2r

    def sample_hard_negatives(self, sim_r2s, sim_s2r, batch_size):
        """
        Sample hard negative indices based on ITC similarity

        Args:
            sim_r2s: [batch, batch+queue] - Raw-to-Size similarity matrix (already scaled by temperature)
            sim_s2r: [batch, batch+queue] - Size-to-Raw similarity matrix (already scaled by temperature)
            batch_size: batch size

        Returns:
            neg_size_idx: [batch] - hard negative size indices for each raw
            neg_raw_idx: [batch] - hard negative raw indices for each size
        """
        if batch_size < 2:
            raise ValueError(
                "ITM hard-negative mining requires a local batch with at least "
                "2 samples. Check that the multimodal dataloader drops single-sample tail batches."
            )

        with torch.no_grad():
            # Use only the in-batch similarities (the first batch_size columns).
            sim_r2s_batch = sim_r2s[:, :batch_size].clone()
            sim_s2r_batch = sim_s2r[:, :batch_size].clone()

            # Mask out positive pairs (diagonal)
            sim_r2s_batch.fill_diagonal_(-float('inf'))
            sim_s2r_batch.fill_diagonal_(-float('inf'))

            # Sample hard negatives based on similarity
            # sim_r2s and sim_s2r are already divided by temperature in forward_itc
            weights_r2s = F.softmax(sim_r2s_batch, dim=1)
            weights_s2r = F.softmax(sim_s2r_batch, dim=1)

            # For each raw, sample a hard negative size
            neg_size_idx = torch.multinomial(weights_r2s, 1).squeeze(1)  # [batch]
            # For each size, sample a hard negative raw
            neg_raw_idx = torch.multinomial(weights_s2r, 1).squeeze(1)  # [batch]

        return neg_size_idx, neg_raw_idx

    def forward_itm(self, pos_raw_cls, pos_size_cls, neg1_raw_cls, neg1_size_cls, neg2_raw_cls, neg2_size_cls):
        """
        ITM (Image-Text Matching) Loss - ALBEF style

        All inputs are post-fusion CLS features.

        Args:
            pos_raw_cls: [batch, hidden] - positive raw post-fusion CLS
            pos_size_cls: [batch, hidden] - positive size post-fusion CLS
            neg1_raw_cls: [batch, hidden] - negative-type-1 raw post-fusion CLS (raw_i with size_neg)
            neg1_size_cls: [batch, hidden] - negative-type-1 size post-fusion CLS
            neg2_raw_cls: [batch, hidden] - negative-type-2 raw post-fusion CLS (raw_neg with size_i)
            neg2_size_cls: [batch, hidden] - negative-type-2 size post-fusion CLS

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

    def forward_masked_reconstruction(self, size_fused, tgt_mlm_size, tgt_mlm_temporal):
        """
        Masked Reconstruction for Size and Temporal (IAT) tokens

        Uses fused features (which incorporate Raw context) to reconstruct
        masked Size and IAT tokens.

        Args:
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            tgt_mlm_size: [batch, seq_len_size] - Size targets (0=unmasked, token_id=masked)
            tgt_mlm_temporal: [batch, seq_len_size] - Temporal targets (0=unmasked, token_id=masked)

        Returns:
            dict: Dictionary containing:
                - size_loss: Size reconstruction loss (CrossEntropy)
                - size_correct: Number of correct size predictions
                - size_denom: Number of masked size positions
                - temporal_loss: Temporal reconstruction loss (soft-label KL)
                - temporal_correct_exact: Number of exact temporal matches
                - temporal_correct_range: Number of predictions within ±sigma range
                - temporal_denom: Number of masked temporal positions
        """
        device = size_fused.device

        # ===== Size Reconstruction =====
        size_output = self.act(self.recon_size_linear_1(size_fused))
        size_output = self.recon_size_layer_norm(size_output)

        if self.factorized_embedding_parameterization:
            size_output = size_output.contiguous().view(-1, self.emb_size)
        else:
            size_output = size_output.contiguous().view(-1, self.hidden_size)

        tgt_size = tgt_mlm_size.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        size_output_masked = size_output[tgt_size > 0, :]
        tgt_size_masked = tgt_size[tgt_size > 0]

        # Size prediction
        size_logits = self.recon_size_linear_2(size_output_masked)
        size_log_probs = self.recon_size_softmax(size_logits)

        # Size metrics
        size_denom = torch.tensor(size_output_masked.size(0) + 1e-6)
        if size_output_masked.size(0) == 0:
            size_correct = torch.tensor(0.0)
            size_loss = torch.tensor(0.0, device=device)
        else:
            size_correct = torch.sum((size_log_probs.argmax(dim=-1).eq(tgt_size_masked)).float())
            size_loss = self.recon_size_criterion(size_log_probs, tgt_size_masked)

        # ===== Temporal Reconstruction =====
        temporal_output = self.act(self.recon_temporal_linear_1(size_fused))
        temporal_output = self.recon_temporal_layer_norm(temporal_output)

        if self.factorized_embedding_parameterization:
            temporal_output = temporal_output.contiguous().view(-1, self.emb_size)
        else:
            temporal_output = temporal_output.contiguous().view(-1, self.hidden_size)

        tgt_temporal = tgt_mlm_temporal.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        temporal_output_masked = temporal_output[tgt_temporal > 0, :]
        tgt_temporal_masked = tgt_temporal[tgt_temporal > 0]

        # Temporal metrics
        temporal_denom = torch.tensor(temporal_output_masked.size(0) + 1e-6)
        if temporal_output_masked.size(0) == 0:
            temporal_correct_exact = torch.tensor(0.0)
            temporal_correct_range = torch.tensor(0.0)
            temporal_loss = torch.tensor(0.0, device=device)
        else:
            # Get logits (before softmax)
            temporal_logits = self.recon_temporal_linear_2(temporal_output_masked)

            # Soft-label KL divergence loss
            temporal_loss = soft_label_kl_loss(
                temporal_logits, tgt_temporal_masked,
                self.vocab_size_temporal, sigma=self.temporal_sigma
            )

            # Compute accuracy metrics
            pred_idx = temporal_logits.argmax(dim=-1)

            # Exact match accuracy
            temporal_correct_exact = torch.sum((pred_idx == tgt_temporal_masked).float())

            # Range accuracy: prediction within ±sigma of target
            temporal_correct_range = torch.sum(
                ((pred_idx - tgt_temporal_masked).abs() <= self.temporal_sigma).float()
            )

        return {
            'size_loss': size_loss,
            'size_correct': size_correct,
            'size_denom': size_denom,
            'temporal_loss': temporal_loss,
            'temporal_correct_exact': temporal_correct_exact,
            'temporal_correct_range': temporal_correct_range,
            'temporal_denom': temporal_denom,
        }
