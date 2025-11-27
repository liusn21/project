"""
Multi-Modal Model for Stage 2 Pretraining

Architecture:
    - encoder_raw: Pretrained Raw Packet encoder
    - encoder_size: Pretrained Packet Size encoder
    - fusion: GatedMultiModalFusion (cross-attention + gating)
    - target: MultiModalTarget (CMM + CMMP tasks)

Two-phase training:
    - Phase 1: Freeze encoders, train fusion only
    - Phase 2: Joint training with differential learning rates
"""

import torch
import torch.nn as nn
from uer.utils.constants import PAD_ID


class MultiModalModel(nn.Module):
    """
    Multi-Modal Pretraining Model

    Integrates two pretrained encoders (Raw Packet + Packet Size)
    with fusion module and multi-modal pretraining objectives.
    """

    def __init__(self, args, embedding_raw, encoder_raw, embedding_size, encoder_size, fusion, target):
        super(MultiModalModel, self).__init__()

        # Raw Packet modality
        self.embedding_raw = embedding_raw
        self.encoder_raw = encoder_raw

        # Packet Size modality
        self.embedding_size = embedding_size
        self.encoder_size = encoder_size

        # Fusion module
        self.fusion = fusion

        # Target module
        self.target = target

        # Phase control (for two-phase training)
        self.freeze_encoders = args.freeze_encoders if hasattr(args, 'freeze_encoders') else False

        # Apply freezing if needed
        if self.freeze_encoders:
            self._freeze_encoders()

    def _freeze_encoders(self):
        """Freeze encoder parameters (Phase 1)"""
        print("Freezing encoder parameters (Phase 1)...")
        for param in self.embedding_raw.parameters():
            param.requires_grad = False
        for param in self.encoder_raw.parameters():
            param.requires_grad = False
        for param in self.embedding_size.parameters():
            param.requires_grad = False
        for param in self.encoder_size.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze encoder parameters (Phase 2)"""
        print("Unfreezing encoder parameters (Phase 2)...")
        for param in self.embedding_raw.parameters():
            param.requires_grad = True
        for param in self.encoder_raw.parameters():
            param.requires_grad = True
        for param in self.embedding_size.parameters():
            param.requires_grad = True
        for param in self.encoder_size.parameters():
            param.requires_grad = True
        self.freeze_encoders = False

    def forward(self, raw_src, raw_packet_ids, raw_directions, size_src):
        """
        Forward pass for inference (no training objectives)

        NOTE: During training, MultiModalTrainer manually forwards each component
        to implement hard negative sampling for CMM task. This forward() is used
        for inference/evaluation only.

        Args:
            raw_src: [batch, seq_len_raw] - Raw Packet token IDs
            raw_packet_ids: [batch, seq_len_raw] - Packet indices
            raw_directions: [batch, seq_len_raw] - Direction indices
            size_src: [batch, seq_len_size] - Size token IDs

        Returns:
            raw_fused: [batch, seq_len_raw, hidden] - Fused Raw features
            size_fused: [batch, seq_len_size, hidden] - Fused Size features
            (g_raw, g_size): Gate weights
        """

        # ===== Encode Raw Packet =====
        raw_emb = self.embedding_raw(raw_src, raw_packet_ids, raw_directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)  # [batch, seq_len_raw, hidden]

        # ===== Encode Packet Size =====
        size_emb = self.embedding_size(size_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)  # [batch, seq_len_size, hidden]

        # ===== Fusion =====
        raw_fused, size_fused, (g_raw, g_size) = self.fusion(
            raw_output, size_output, return_gate_weights=True
        )

        return raw_fused, size_fused, (g_raw, g_size)
