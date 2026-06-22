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
    - ITGCA (Information-Theoretic Gated Cross-Attention) support
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
from uer.utils.constants import PAD_ID


@torch.no_grad()
def concat_all_gather(tensor):
    """Gather tensors from all GPUs and concatenate them."""
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output


# ===== ITGCA Statistical Prior Computation =====

@torch.no_grad()
def compute_flow_reliability_raw(raw_src, pad_id=0, vocab_size=None):
    """
    Compute flow-level reliability for raw modality (GPU-vectorized).
    r_stat = 1 - H(flow) / H_max

    Encrypted payload has near-uniform byte distribution → high entropy → low reliability.
    Plaintext has repeated patterns (HTTP headers, HTML) → low entropy → high reliability.

    Uses scatter_add for GPU-side bincount — no CPU transfer or Python loops.

    Args:
        raw_src: [B, L] - Raw token IDs
        pad_id: PAD token ID to exclude
        vocab_size: int or None - Vocabulary size (avoids GPU-CPU sync for max())

    Returns:
        reliability: [B] - Flow-level reliability in [0, 1]
    """
    B, L = raw_src.shape
    device = raw_src.device
    V = vocab_size if vocab_size is not None else int(raw_src.max().item()) + 1

    # Non-pad mask and token count per sample
    non_pad = (raw_src != pad_id)                            # [B, L]
    n = non_pad.float().sum(dim=1)                           # [B]

    # Bincount via scatter_add: count occurrences of each token per sample
    # PAD positions contribute 0.0 (non_pad is False), so counts are unaffected
    counts = torch.zeros(B, V, device=device)
    counts.scatter_add_(1, raw_src, non_pad.float())         # [B, V]

    # Shannon entropy: H = -Σ p * log(p)
    n_safe = n.clamp(min=1)
    probs = counts / n_safe.unsqueeze(1)                     # [B, V]
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1) # [B]

    # Normalized reliability: 1 - H / H_max
    # H_max = log(min(n, V)): when n > V, entropy can't exceed log(V),
    # so use log(V) to avoid artificial floor on r_stat.
    # n <= 1: single token or empty → fully predictable → reliability = 1.0
    H_max = torch.log(n_safe.clamp(max=float(V)))            # [B]
    valid = H_max > 0
    reliability = torch.where(
        valid,
        (1.0 - entropy / H_max).clamp(0.0, 1.0),
        torch.ones_like(entropy)
    )

    return reliability


@torch.no_grad()
def compute_local_entropy(tokens, window_size=16, pad_id=0):
    """
    Vectorized GPU computation of local sliding-window Shannon entropy.
    t_stat[j] = 1 - H(window_j) / log(n_j)

    H_max is per-position log(n) where n = number of non-pad tokens in the window,
    so that all-unique tokens always give reliability ≈ 0 regardless of window size.

    Uses pairwise equality within windows — no Python loops, runs entirely on GPU.

    Args:
        tokens: [B, L] - Token IDs (torch tensor)
        window_size: Sliding window size (default 16)
        pad_id: PAD token ID to exclude

    Returns:
        local_reliability: [B, L] - Per-position reliability in [0, 1]
    """
    B, L = tokens.shape
    half_w = window_size // 2
    actual_w = 2 * half_w + 1

    # Pad with pad_id so boundary positions get correct variable-size windows
    padded = F.pad(tokens, (half_w, half_w), value=pad_id)  # [B, L + 2*half_w]
    windows = padded.unfold(1, actual_w, 1)                 # [B, L, actual_w]

    # Non-pad mask within each window
    non_pad = (windows != pad_id)                            # [B, L, actual_w]
    n = non_pad.float().sum(dim=-1)                          # [B, L]

    # Pairwise equality within each window (both positions must be non-pad)
    eq = (windows.unsqueeze(-1) == windows.unsqueeze(-2))    # [B, L, W, W]
    eq = eq & non_pad.unsqueeze(-1) & non_pad.unsqueeze(-2)
    counts = eq.sum(dim=-1).float()                          # [B, L, W]

    # H = log(n) - (1/n) * sum_i log(count_i)  over non-pad positions
    log_counts = torch.log(counts.clamp(min=1))              # [B, L, W]
    sum_log_counts = (log_counts * non_pad.float()).sum(dim=-1)  # [B, L]
    n_safe = n.clamp(min=1)
    H = torch.log(n_safe) - sum_log_counts / n_safe         # [B, L]

    # Per-position H_max = log(n): all-unique → H=log(n) → reliability=0
    H_max = torch.log(n_safe)                                # [B, L]

    # n <= 1: H=0, H_max=0 → reliability=1 (single token = fully predictable)
    # n >= 2: reliability = 1 - H / log(n), clamped to [0, 1]
    valid = H_max > 0
    result = torch.where(valid, (1.0 - H / H_max).clamp(0.0, 1.0),
                         torch.ones_like(H))

    # Pad positions in original input → 0
    result[tokens == pad_id] = 0.0

    return result


# ===== ITGCA Auxiliary Losses =====


class MultiModalModel(nn.Module):
    """
    Multi-Modal Pretraining Model (ALBEF-style) with optional ITGCA.
    """

    def __init__(self, args, embedding_raw, encoder_raw, embedding_size, encoder_size,
                 fusion, target, queue_size=4096, momentum=0.995):
        super(MultiModalModel, self).__init__()

        self.hidden_size = args.hidden_size
        self.queue_size = queue_size
        self.momentum = momentum
        self.itc_temp = getattr(args, 'itc_temperature', 0.07)

        # ITGCA parameters
        self.use_itgca = getattr(args, 'use_itgca', False)
        if self.use_itgca:
            self.itgca_window_size = getattr(args, 'itgca_window_size', 16)
            self.vocab_size_raw = embedding_raw.token_embedding.num_embeddings
            # Component-level ablation flags
            self.ablate_r_stat = getattr(args, 'ablate_r_stat', False)
            self.ablate_g_token = getattr(args, 'ablate_g_token', False)
            self.ablate_source_bias = getattr(args, 'ablate_source_bias', False)

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
        self.itc_proj_raw_m = copy.deepcopy(target.itc_proj_raw)
        self.itc_proj_size_m = copy.deepcopy(target.itc_proj_size)

        for param in self.itc_proj_raw_m.parameters():
            param.requires_grad = False
        for param in self.itc_proj_size_m.parameters():
            param.requires_grad = False

        # ===== Feature Queues =====
        self.register_buffer("raw_queue", torch.randn(queue_size, self.hidden_size))
        self.register_buffer("size_queue", torch.randn(queue_size, self.hidden_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.raw_queue = F.normalize(self.raw_queue, dim=-1)
        self.size_queue = F.normalize(self.size_queue, dim=-1)

    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoders and projection layers using EMA."""
        for param, param_m in zip(self.embedding_raw.parameters(),
                                   self.embedding_raw_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)
        for param, param_m in zip(self.encoder_raw.parameters(),
                                   self.encoder_raw_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

        for param, param_m in zip(self.embedding_size.parameters(),
                                   self.embedding_size_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)
        for param, param_m in zip(self.encoder_size.parameters(),
                                   self.encoder_size_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

        for param, param_m in zip(self.target.itc_proj_raw.parameters(),
                                   self.itc_proj_raw_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)
        for param, param_m in zip(self.target.itc_proj_size.parameters(),
                                   self.itc_proj_size_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def init_momentum_encoders(self):
        """
        Hard-copy every online module into its momentum counterpart.

        Must be called ONCE after (a) Stage-1 weights are loaded into the online
        encoders and (b) the trainer's blanket re-initialization of all
        non-pretrained parameters. The ITC projection heads (itc_proj_*_m) are
        deep-copied at construction time, but their online versions are
        re-initialized later in the trainer; without this re-sync the momentum
        (teacher) projections would start from a different random init than the
        online (student) ones, corrupting the ITC contrastive target for the
        first thousands of steps. Encoders/embeddings are already identical
        (loaded into both online and momentum), so copying them is a harmless
        no-op that also future-proofs against any further desync.
        """
        module_pairs = [
            (self.embedding_raw, self.embedding_raw_m),
            (self.encoder_raw, self.encoder_raw_m),
            (self.embedding_size, self.embedding_size_m),
            (self.encoder_size, self.encoder_size_m),
            (self.target.itc_proj_raw, self.itc_proj_raw_m),
            (self.target.itc_proj_size, self.itc_proj_size_m),
        ]
        for online, momentum in module_pairs:
            for p, p_m in zip(online.parameters(), momentum.parameters()):
                p_m.data.copy_(p.data)
                p_m.requires_grad = False

    @torch.no_grad()
    def _dequeue_and_enqueue(self, raw_feat, size_feat):
        """Update feature queues with new features."""
        batch_size = raw_feat.size(0)
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.raw_queue[ptr:ptr + batch_size] = raw_feat
            self.size_queue[ptr:ptr + batch_size] = size_feat
            ptr = (ptr + batch_size) % self.queue_size
        else:
            remaining = self.queue_size - ptr
            self.raw_queue[ptr:] = raw_feat[:remaining]
            self.size_queue[ptr:] = size_feat[:remaining]
            self.raw_queue[:batch_size - remaining] = raw_feat[remaining:]
            self.size_queue[:batch_size - remaining] = size_feat[remaining:]
            ptr = batch_size - remaining

        self.queue_ptr[0] = ptr

    def _compute_itgca_signals(self, raw_src, raw_cls, size_cls):
        """
        Compute ITGCA statistical priors and encoder CLS signals (asymmetric).

        Only computes priors for the Raw modality (Size is stable, no prior needed).
        Local entropy is computed on-the-fly using vectorized GPU ops (< 1ms/batch).

        Args:
            raw_src: [B, L_raw] - Raw token IDs
            raw_cls: [B, H] - Raw encoder CLS (will be detached)
            size_cls: [B, H] - Size encoder CLS (will be detached)

        Returns:
            itgca_kwargs: dict of keyword arguments for fusion forward
        """
        # Skip r_stat / local_ent computation when their downstream consumers
        # are ablated — saves a few ms per batch and ensures the gate's
        # `r_stat is None` branch fires for "w/o r_stat" runs.
        if self.ablate_r_stat:
            r_stat_raw = None
        else:
            r_stat_raw = compute_flow_reliability_raw(raw_src, vocab_size=self.vocab_size_raw)
        if self.ablate_source_bias:
            local_ent_raw = None
        else:
            local_ent_raw = compute_local_entropy(raw_src, self.itgca_window_size)

        raw_cls_enc = raw_cls.detach()
        size_cls_enc = size_cls.detach()

        return {
            'raw_cls_enc': raw_cls_enc,
            'size_cls_enc': size_cls_enc,
            'r_stat_raw': r_stat_raw,
            'local_ent_raw': local_ent_raw,
        }

    def forward(self, raw_src, raw_packet_ids, raw_directions,
                size_src_clean, iat_src_clean,
                size_src_masked, iat_src_masked,
                tgt_mlm_size, tgt_mlm_temporal, itc_alpha=0.0, update_momentum=True):
        """
        Forward pass for training with Masked Reconstruction (ALBEF-style).

        Args:
            raw_src: [batch, seq_len_raw] - Raw Packet token IDs (NOT masked)
            raw_packet_ids: [batch, seq_len_raw] - Packet indices
            raw_directions: [batch, seq_len_raw] - Direction indices
            size_src_clean: [batch, seq_len_size] - Size token IDs (clean, for ITC/ITM)
            iat_src_clean: [batch, seq_len_size] - IAT token IDs (clean, for ITC/ITM)
            size_src_masked: [batch, seq_len_size] - Size token IDs (masked, for reconstruction)
            iat_src_masked: [batch, seq_len_size] - IAT token IDs (masked, for reconstruction)
            tgt_mlm_size: [batch, seq_len_size] - Size reconstruction targets
            tgt_mlm_temporal: [batch, seq_len_size] - Temporal reconstruction targets

        Returns:
            loss_dict: Dictionary containing all losses and metrics
        """
        batch_size = raw_src.size(0)

        # ===== Step 1: Encode Raw =====
        raw_emb = self.embedding_raw(raw_src, raw_packet_ids, raw_directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)

        # ===== Step 2: Encode CLEAN Size+IAT =====
        size_emb_clean = self.embedding_size(size_src_clean, iat_src_clean)
        size_seg = (size_src_clean != PAD_ID).long()
        size_output_clean = self.encoder_size(size_emb_clean, size_seg)

        # Extract CLS tokens (before fusion, for ITC)
        raw_cls = raw_output[:, 0, :]
        size_cls = size_output_clean[:, 0, :]

        # ===== ITGCA: Compute statistical priors =====
        if self.use_itgca:
            itgca_kwargs = self._compute_itgca_signals(
                raw_src, raw_cls, size_cls
            )
        else:
            itgca_kwargs = {}

        # ===== Step 3: Momentum Encoders =====
        with torch.no_grad():
            raw_emb_m = self.embedding_raw_m(raw_src, raw_packet_ids, raw_directions)
            raw_output_m = self.encoder_raw_m(raw_emb_m, raw_seg)

            size_emb_m = self.embedding_size_m(size_src_clean, iat_src_clean)
            size_output_m = self.encoder_size_m(size_emb_m, size_seg)

            raw_cls_m = raw_output_m[:, 0, :]
            size_cls_m = size_output_m[:, 0, :]

            raw_cls_m_proj = F.normalize(self.itc_proj_raw_m(raw_cls_m), dim=-1)
            size_cls_m_proj = F.normalize(self.itc_proj_size_m(size_cls_m), dim=-1)

            # EMA update once per OPTIMIZER step: only the last micro-batch of an
            # accumulation group passes update_momentum=True. This keeps the teacher
            # constant within a group and makes the EMA rate independent of
            # accumulation_steps (with accum=1, update_momentum is always True).
            if update_momentum:
                self._momentum_update()

        # ===== Step 4: ITC Loss =====
        itc_loss, sim_r2s, sim_s2r = self.target.forward_itc(
            raw_cls, size_cls,
            raw_cls_m_proj, size_cls_m_proj,
            self.raw_queue, self.size_queue,
            temperature=self.itc_temp, alpha=itc_alpha
        )

        # ===== Step 5: Update Queues =====
        with torch.no_grad():
            self._dequeue_and_enqueue(raw_cls_m_proj, size_cls_m_proj)

        # ===== Step 6: Fusion for Positive Samples =====
        raw_fused, size_fused, gate_info_pos = self.fusion(
            raw_output, size_output_clean, raw_seg, size_seg,
            **itgca_kwargs
        )

        pos_raw_cls = raw_fused[:, 0, :]
        pos_size_cls = size_fused[:, 0, :]

        # ===== Step 7: Hard Negative Sampling =====
        neg_size_idx, neg_raw_idx = self.target.sample_hard_negatives(
            sim_r2s, sim_s2r, batch_size
        )

        # ===== Step 8: Fusion for Negative Samples =====
        # Negative type 1: (raw_i, size_neg)
        neg_size_output_1 = size_output_clean[neg_size_idx]
        neg_size_seg_1 = size_seg[neg_size_idx]

        if self.use_itgca:
            itgca_kwargs_neg1 = {
                'raw_cls_enc': itgca_kwargs['raw_cls_enc'],
                'size_cls_enc': itgca_kwargs['size_cls_enc'][neg_size_idx],
                'r_stat_raw': itgca_kwargs['r_stat_raw'],
                'local_ent_raw': itgca_kwargs['local_ent_raw'],
            }
        else:
            itgca_kwargs_neg1 = {}

        neg1_raw_fused, neg1_size_fused, _ = self.fusion(
            raw_output, neg_size_output_1, raw_seg, neg_size_seg_1,
            **itgca_kwargs_neg1
        )
        neg1_raw_cls = neg1_raw_fused[:, 0, :]
        neg1_size_cls = neg1_size_fused[:, 0, :]

        # Negative type 2: (raw_neg, size_i)
        neg_raw_output_2 = raw_output[neg_raw_idx]
        neg_raw_seg_2 = raw_seg[neg_raw_idx]

        if self.use_itgca:
            # r_stat_raw / local_ent_raw may be None when their components are ablated
            r_stat_neg2 = itgca_kwargs['r_stat_raw']
            if r_stat_neg2 is not None:
                r_stat_neg2 = r_stat_neg2[neg_raw_idx]
            local_ent_neg2 = itgca_kwargs['local_ent_raw']
            if local_ent_neg2 is not None:
                local_ent_neg2 = local_ent_neg2[neg_raw_idx]
            itgca_kwargs_neg2 = {
                'raw_cls_enc': itgca_kwargs['raw_cls_enc'][neg_raw_idx],
                'size_cls_enc': itgca_kwargs['size_cls_enc'],
                'r_stat_raw': r_stat_neg2,
                'local_ent_raw': local_ent_neg2,
            }
        else:
            itgca_kwargs_neg2 = {}

        neg2_raw_fused, neg2_size_fused, _ = self.fusion(
            neg_raw_output_2, size_output_clean, neg_raw_seg_2, size_seg,
            **itgca_kwargs_neg2
        )
        neg2_raw_cls = neg2_raw_fused[:, 0, :]
        neg2_size_cls = neg2_size_fused[:, 0, :]

        # ===== Step 9: ITM Loss =====
        itm_loss, itm_acc = self.target.forward_itm(
            pos_raw_cls, pos_size_cls,
            neg1_raw_cls, neg1_size_cls,
            neg2_raw_cls, neg2_size_cls
        )

        # ===== Step 10: Encode MASKED Size+IAT =====
        size_emb_masked = self.embedding_size(size_src_masked, iat_src_masked)
        size_seg_masked = (size_src_masked != PAD_ID).long()
        size_output_masked = self.encoder_size(size_emb_masked, size_seg_masked)

        # ===== Step 11: Fusion with Masked Size features =====
        if self.use_itgca:
            # Use clean encoder CLS and clean r_stat/local_ent for masked fusion
            itgca_kwargs_masked = dict(itgca_kwargs)
        else:
            itgca_kwargs_masked = {}

        _, size_fused_masked, _ = self.fusion(
            raw_output, size_output_masked, raw_seg, size_seg_masked,
            **itgca_kwargs_masked
        )

        # ===== Step 12: Masked Reconstruction Loss =====
        recon_results = self.target.forward_masked_reconstruction(
            size_fused_masked, tgt_mlm_size, tgt_mlm_temporal
        )

        # ===== Return all losses =====
        loss_dict = {
            'itc_loss': itc_loss,
            'itm_loss': itm_loss,
            'itm_acc': itm_acc,
            'recon_size_loss': recon_results['size_loss'],
            'recon_size_correct': recon_results['size_correct'],
            'recon_size_denom': recon_results['size_denom'],
            'recon_temporal_loss': recon_results['temporal_loss'],
            'recon_temporal_correct_exact': recon_results['temporal_correct_exact'],
            'recon_temporal_correct_range': recon_results['temporal_correct_range'],
            'recon_temporal_denom': recon_results['temporal_denom'],
        }

        return loss_dict

    def forward_inference(self, raw_src, raw_packet_ids, raw_directions, size_src, iat_src):
        """
        Forward pass for inference (no training objectives).

        Returns fused features for downstream tasks.
        """
        # Encode Raw
        raw_emb = self.embedding_raw(raw_src, raw_packet_ids, raw_directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)

        # Encode Size + IAT
        size_emb = self.embedding_size(size_src, iat_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)

        # ITGCA signals for inference
        if self.use_itgca:
            raw_cls = raw_output[:, 0, :]
            size_cls = size_output[:, 0, :]
            itgca_kwargs = self._compute_itgca_signals(
                raw_src, raw_cls, size_cls
            )
        else:
            itgca_kwargs = {}

        # Fusion
        raw_fused, size_fused, _ = self.fusion(
            raw_output, size_output, raw_seg, size_seg,
            **itgca_kwargs
        )

        return raw_fused, size_fused
