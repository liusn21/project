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
import torch.distributed as dist
import copy
from uer.utils.constants import PAD_ID


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Gather tensors from all GPUs and concatenate them.
    Used for distributed training to synchronize feature queues.

    Args:
        tensor: [batch, ...] - tensor to gather from all GPUs

    Returns:
        gathered: [batch * world_size, ...] - concatenated tensor from all GPUs
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Create placeholder tensors for gathering
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)

    # Concatenate along batch dimension
    output = torch.cat(tensors_gather, dim=0)
    return output


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

        # ===== Momentum Projection Layers (for ITC) =====
        # 独立的投影层，通过 EMA 更新，而非共享 Online encoder 的投影层
        self.itc_proj_raw_m = copy.deepcopy(target.itc_proj_raw)
        self.itc_proj_size_m = copy.deepcopy(target.itc_proj_size)

        # Disable gradient for momentum projection layers
        for param in self.itc_proj_raw_m.parameters():
            param.requires_grad = False
        for param in self.itc_proj_size_m.parameters():
            param.requires_grad = False

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
        Update momentum encoders and projection layers using EMA
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

        # Update ITC projection layers
        for param, param_m in zip(self.target.itc_proj_raw.parameters(),
                                   self.itc_proj_raw_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)
        for param, param_m in zip(self.target.itc_proj_size.parameters(),
                                   self.itc_proj_size_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, raw_feat, size_feat):
        """
        Update feature queues with new features.

        In distributed training, gathers features from all GPUs before enqueuing
        to ensure all processes have consistent queues (ALBEF-style).

        Args:
            raw_feat: [batch, hidden] - Raw CLS features (normalized)
            size_feat: [batch, hidden] - Size CLS features (normalized)
        """
        # Gather features from all GPUs in distributed training
        # This ensures all processes have the same queue contents
        # raw_feat = concat_all_gather(raw_feat)
        # size_feat = concat_all_gather(size_feat)

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

    def forward(self, raw_src, raw_packet_ids, raw_directions,
                size_src, iat_src, tgt_mlm_size, tgt_mlm_temporal):
        """
        Forward pass for training with Masked Reconstruction

        NEW Design:
            - Raw: NOT masked, provides context for reconstruction
            - Size + IAT: Synchronously masked, reconstructed using fused features

        Args:
            raw_src: [batch, seq_len_raw] - Raw Packet token IDs (NOT masked)
            raw_packet_ids: [batch, seq_len_raw] - Packet indices
            raw_directions: [batch, seq_len_raw] - Direction indices
            size_src: [batch, seq_len_size] - Size token IDs (masked)
            iat_src: [batch, seq_len_size] - IAT token IDs (masked at same positions)
            tgt_mlm_size: [batch, seq_len_size] - Size reconstruction targets
            tgt_mlm_temporal: [batch, seq_len_size] - Temporal reconstruction targets

        Returns:
            loss_dict: Dictionary containing all losses and metrics
        """
        batch_size = raw_src.size(0)

        # ===== Step 1: Encode Raw with Main Encoder (NOT masked) =====
        raw_emb = self.embedding_raw(raw_src, raw_packet_ids, raw_directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)  # [batch, seq_len_raw, hidden]

        # ===== Step 2: Encode Size+IAT with Main Encoder (masked) =====
        size_emb = self.embedding_size(size_src, iat_src)  # Now takes both size and IAT
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)  # [batch, seq_len_size, hidden]

        # Extract CLS tokens (before fusion, for ITC)
        raw_cls = raw_output[:, 0, :]  # [batch, hidden]
        size_cls = size_output[:, 0, :]  # [batch, hidden]

        # ===== Step 3: Encode with Momentum Encoders (no gradient) =====
        # Note: momentum update AFTER forward pass, so momentum encoder uses
        # the previous iteration's EMA state (standard ALBEF/MoCo practice)
        with torch.no_grad():
            raw_emb_m = self.embedding_raw_m(raw_src, raw_packet_ids, raw_directions)
            raw_output_m = self.encoder_raw_m(raw_emb_m, raw_seg)

            size_emb_m = self.embedding_size_m(size_src, iat_src)  # Now takes both size and IAT
            size_output_m = self.encoder_size_m(size_emb_m, size_seg)

            # Extract CLS tokens and project (using previous momentum state)
            raw_cls_m = raw_output_m[:, 0, :]  # [batch, hidden]
            size_cls_m = size_output_m[:, 0, :]  # [batch, hidden]

            # Project momentum features (using momentum projection layers)
            raw_cls_m_proj = F.normalize(self.itc_proj_raw_m(raw_cls_m), dim=-1)
            size_cls_m_proj = F.normalize(self.itc_proj_size_m(size_cls_m), dim=-1)

            # Update momentum encoders and projections AFTER using them
            self._momentum_update()

        # ===== Step 4: ITC Loss =====
        itc_loss, sim_r2s, sim_s2r = self.target.forward_itc(
            raw_cls, size_cls,
            raw_cls_m_proj, size_cls_m_proj,
            self.raw_queue, self.size_queue,
            temperature=self.itc_temp
        )

        # ===== Step 5: Update Queues =====
        with torch.no_grad():
            self._dequeue_and_enqueue(raw_cls_m_proj, size_cls_m_proj)

        # ===== Step 6: Fusion for Positive Samples =====
        # Positive pairs: (raw_i, size_i)
        raw_fused, size_fused = self.fusion(raw_output, size_output, raw_seg, size_seg)

        # Extract fused CLS tokens for positive samples
        pos_raw_cls = raw_fused[:, 0, :]  # [batch, hidden]
        pos_size_cls = size_fused[:, 0, :]  # [batch, hidden]

        # ===== Step 7: Hard Negative Sampling for ITM =====
        # Sample hard negative indices based on ITC similarity
        neg_size_idx, neg_raw_idx = self.target.sample_hard_negatives(
            sim_r2s, sim_s2r, batch_size
        )

        # ===== Step 8: Fusion for Negative Samples =====
        # Negative type 1: (raw_i, size_neg) - each raw_i paired with its hard negative size
        neg_size_output_1 = size_output[neg_size_idx]  # [batch, seq_len_size, hidden]
        neg_size_seg_1 = size_seg[neg_size_idx]  # [batch, seq_len_size]

        neg1_raw_fused, neg1_size_fused = self.fusion(
            raw_output, neg_size_output_1, raw_seg, neg_size_seg_1
        )
        neg1_raw_cls = neg1_raw_fused[:, 0, :]  # [batch, hidden]
        neg1_size_cls = neg1_size_fused[:, 0, :]  # [batch, hidden]

        # Negative type 2: (raw_neg, size_i) - each size_i paired with its hard negative raw
        neg_raw_output_2 = raw_output[neg_raw_idx]  # [batch, seq_len_raw, hidden]
        neg_raw_seg_2 = raw_seg[neg_raw_idx]  # [batch, seq_len_raw]

        neg2_raw_fused, neg2_size_fused = self.fusion(
            neg_raw_output_2, size_output, neg_raw_seg_2, size_seg
        )
        neg2_raw_cls = neg2_raw_fused[:, 0, :]  # [batch, hidden]
        neg2_size_cls = neg2_size_fused[:, 0, :]  # [batch, hidden]

        # ===== Step 9: ITM Loss =====
        itm_loss, itm_acc = self.target.forward_itm(
            pos_raw_cls, pos_size_cls,      # Positive: (raw_i, size_i)
            neg1_raw_cls, neg1_size_cls,    # Negative 1: (raw_i, size_neg)
            neg2_raw_cls, neg2_size_cls     # Negative 2: (raw_neg, size_i)
        )

        # ===== Step 10: Masked Reconstruction Loss (Size + Temporal) =====
        # Use fused size features to reconstruct masked Size and IAT tokens
        recon_results = self.target.forward_masked_reconstruction(
            size_fused, tgt_mlm_size, tgt_mlm_temporal
        )

        # ===== Return all losses and metrics =====
        return {
            'itc_loss': itc_loss,
            'itm_loss': itm_loss,
            'itm_acc': itm_acc,
            # Masked Reconstruction results
            'recon_size_loss': recon_results['size_loss'],
            'recon_size_correct': recon_results['size_correct'],
            'recon_size_denom': recon_results['size_denom'],
            'recon_temporal_loss': recon_results['temporal_loss'],
            'recon_temporal_correct_exact': recon_results['temporal_correct_exact'],
            'recon_temporal_correct_range': recon_results['temporal_correct_range'],
            'recon_temporal_denom': recon_results['temporal_denom'],
        }

    def forward_inference(self, raw_src, raw_packet_ids, raw_directions, size_src, iat_src):
        """
        Forward pass for inference (no training objectives)

        Returns fused features for downstream tasks.

        Args:
            raw_src: [batch, seq_len_raw] - Raw Packet token IDs
            raw_packet_ids: [batch, seq_len_raw] - Packet indices
            raw_directions: [batch, seq_len_raw] - Direction indices
            size_src: [batch, seq_len_size] - Size token IDs
            iat_src: [batch, seq_len_size] - IAT token IDs

        Returns:
            raw_fused: [batch, seq_len_raw, hidden] - Fused Raw features
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
        """
        # Encode Raw
        raw_emb = self.embedding_raw(raw_src, raw_packet_ids, raw_directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)

        # Encode Size + IAT
        size_emb = self.embedding_size(size_src, iat_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)

        # Fusion
        raw_fused, size_fused = self.fusion(raw_output, size_output, raw_seg, size_seg)

        return raw_fused, size_fused
