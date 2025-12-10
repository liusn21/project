"""
Multi-Modal Model for Stage 2 Pretraining (ALBEF-style)

Architecture:
    - encoder_raw: Raw Packet encoder (from Stage 1)
    - encoder_size: Packet Size encoder (from Stage 1)
    - encoder_raw_m: Momentum Raw encoder (for ITC)
    - encoder_size_m: Momentum Size encoder (for ITC)
    - fusion: MultiModalFusionEncoder (6-layer bidirectional cross-attention)
    - target: MultiModalTarget (ITC + ITM + MLM)

Features:
    - Momentum distillation with EMA update
    - Feature queues for contrastive learning
    - Full parameter training (no freezing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from uer.utils.constants import PAD_ID


class MultiModalModel(nn.Module):
    """
    Multi-Modal Pretraining Model (ALBEF-style)

    Integrates two encoders with momentum distillation,
    fusion module, and multi-modal pretraining objectives.
    """

    def __init__(self, args, embedding_raw, encoder_raw, embedding_size, encoder_size,
                 fusion, target, queue_size=4096, momentum=0.995):
        super(MultiModalModel, self).__init__()

        self.hidden_size = args.hidden_size
        self.queue_size = queue_size
        self.momentum = momentum
        self.itc_temp = getattr(args, 'itc_temperature', 0.07)
        self.itm_temp = getattr(args, 'itm_temperature', 0.07)

        # ===== Main Encoders =====
        self.embedding_raw = embedding_raw
        self.encoder_raw = encoder_raw
        self.embedding_size = embedding_size
        self.encoder_size = encoder_size

        # ===== Momentum Encoders (deep copy) =====
        self.embedding_raw_m = copy.deepcopy(embedding_raw)
        self.encoder_raw_m = copy.deepcopy(encoder_raw)
        self.embedding_size_m = copy.deepcopy(embedding_size)
        self.encoder_size_m = copy.deepcopy(encoder_size)

        # Disable gradient for momentum encoders
        for param in self.embedding_raw_m.parameters():
            param.requires_grad = False
        for param in self.encoder_raw_m.parameters():
            param.requires_grad = False
        for param in self.embedding_size_m.parameters():
            param.requires_grad = False
        for param in self.encoder_size_m.parameters():
            param.requires_grad = False

        # ===== Fusion Module =====
        self.fusion = fusion

        # ===== Target Module =====
        self.target = target

        # ===== Feature Queues =====
        # Register as buffers so they are saved with the model but not trained
        self.register_buffer("raw_queue", torch.randn(queue_size, self.hidden_size))
        self.register_buffer("size_queue", torch.randn(queue_size, self.hidden_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Normalize queues
        self.raw_queue = F.normalize(self.raw_queue, dim=-1)
        self.size_queue = F.normalize(self.size_queue, dim=-1)

    @torch.no_grad()
    def _momentum_update(self):
        """
        Update momentum encoders using EMA
        """
        # Update Raw encoder
        for param, param_m in zip(self.embedding_raw.parameters(),
                                   self.embedding_raw_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)
        for param, param_m in zip(self.encoder_raw.parameters(),
                                   self.encoder_raw_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

        # Update Size encoder
        for param, param_m in zip(self.embedding_size.parameters(),
                                   self.embedding_size_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)
        for param, param_m in zip(self.encoder_size.parameters(),
                                   self.encoder_size_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, raw_feat, size_feat):
        """
        Update feature queues with new features

        Args:
            raw_feat: [batch, hidden] - Raw CLS features (normalized)
            size_feat: [batch, hidden] - Size CLS features (normalized)
        """
        batch_size = raw_feat.size(0)

        ptr = int(self.queue_ptr)

        # If queue is not full, or batch fits exactly
        if ptr + batch_size <= self.queue_size:
            self.raw_queue[ptr:ptr + batch_size] = raw_feat
            self.size_queue[ptr:ptr + batch_size] = size_feat
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.raw_queue[ptr:] = raw_feat[:remaining]
            self.size_queue[ptr:] = size_feat[:remaining]
            self.raw_queue[:batch_size - remaining] = raw_feat[remaining:]
            self.size_queue[:batch_size - remaining] = size_feat[remaining:]
            ptr = batch_size - remaining

        self.queue_ptr[0] = ptr

    def forward(self, raw_src, raw_packet_ids, raw_directions, size_src,
                tgt_mlm_raw, tgt_mlm_size):
        """
        Forward pass for training

        Args:
            raw_src: [batch, seq_len_raw] - Raw Packet token IDs (masked)
            raw_packet_ids: [batch, seq_len_raw] - Packet indices
            raw_directions: [batch, seq_len_raw] - Direction indices
            size_src: [batch, seq_len_size] - Size token IDs (masked)
            tgt_mlm_raw: [batch, seq_len_raw] - MLM targets for Raw
            tgt_mlm_size: [batch, seq_len_size] - MLM targets for Size

        Returns:
            loss_dict: Dictionary containing all losses and metrics
        """
        batch_size = raw_src.size(0)

        # ===== Step 1: Encode with Main Encoders =====
        raw_emb = self.embedding_raw(raw_src, raw_packet_ids, raw_directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)  # [batch, seq_len_raw, hidden]

        size_emb = self.embedding_size(size_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)  # [batch, seq_len_size, hidden]

        # Extract CLS tokens (before fusion, for ITC)
        raw_cls = raw_output[:, 0, :]  # [batch, hidden]
        size_cls = size_output[:, 0, :]  # [batch, hidden]

        # ===== Step 2: Encode with Momentum Encoders (no gradient) =====
        with torch.no_grad():
            self._momentum_update()

            raw_emb_m = self.embedding_raw_m(raw_src, raw_packet_ids, raw_directions)
            raw_output_m = self.encoder_raw_m(raw_emb_m, raw_seg)

            size_emb_m = self.embedding_size_m(size_src)
            size_output_m = self.encoder_size_m(size_emb_m, size_seg)

            # Extract CLS tokens and project
            raw_cls_m = raw_output_m[:, 0, :]  # [batch, hidden]
            size_cls_m = size_output_m[:, 0, :]  # [batch, hidden]

            # Project momentum features (using main model's projection)
            raw_cls_m_proj = F.normalize(self.target.itc_proj_raw(raw_cls_m), dim=-1)
            size_cls_m_proj = F.normalize(self.target.itc_proj_size(size_cls_m), dim=-1)

        # ===== Step 3: ITC Loss =====
        itc_loss, sim_r2s, sim_s2r = self.target.forward_itc(
            raw_cls, size_cls,
            raw_cls_m_proj, size_cls_m_proj,
            self.raw_queue, self.size_queue,
            temperature=self.itc_temp
        )

        # ===== Step 4: Update Queues =====
        with torch.no_grad():
            self._dequeue_and_enqueue(raw_cls_m_proj, size_cls_m_proj)

        # ===== Step 5: Fusion =====
        raw_fused, size_fused = self.fusion(raw_output, size_output, raw_seg, size_seg)

        # Extract fused CLS tokens
        raw_fused_cls = raw_fused[:, 0, :]  # [batch, hidden]
        size_fused_cls = size_fused[:, 0, :]  # [batch, hidden]

        # ===== Step 6: ITM Loss =====
        itm_loss, itm_acc = self.target.forward_itm(
            raw_fused_cls, size_fused_cls,
            sim_r2s, sim_s2r,
            raw_fused_cls, size_fused_cls,  # Use same features for negative sampling
            temperature=self.itm_temp
        )

        # ===== Step 7: MLM Losses =====
        mlm_raw_loss, mlm_raw_correct, mlm_raw_denom = self.target.forward_mlm_raw(
            raw_fused, tgt_mlm_raw
        )
        mlm_size_loss, mlm_size_correct, mlm_size_denom = self.target.forward_mlm_size(
            size_fused, tgt_mlm_size
        )

        # ===== Return all losses and metrics =====
        return {
            'itc_loss': itc_loss,
            'itm_loss': itm_loss,
            'itm_acc': itm_acc,
            'mlm_raw_loss': mlm_raw_loss,
            'mlm_raw_correct': mlm_raw_correct,
            'mlm_raw_denom': mlm_raw_denom,
            'mlm_size_loss': mlm_size_loss,
            'mlm_size_correct': mlm_size_correct,
            'mlm_size_denom': mlm_size_denom,
        }

    def forward_inference(self, raw_src, raw_packet_ids, raw_directions, size_src):
        """
        Forward pass for inference (no training objectives)

        Returns fused features for downstream tasks.
        """
        # Encode
        raw_emb = self.embedding_raw(raw_src, raw_packet_ids, raw_directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)

        size_emb = self.embedding_size(size_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)

        # Fusion
        raw_fused, size_fused = self.fusion(raw_output, size_output, raw_seg, size_seg)

        return raw_fused, size_fused
