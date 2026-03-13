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
def compute_flow_reliability_raw(raw_src, pad_id=0):
    """
    Compute flow-level reliability for raw modality.
    r_stat = 1 - H(flow) / H_max

    Encrypted payload has near-uniform bigram distribution → high entropy → low reliability.
    Plaintext has repeated patterns (HTTP headers, HTML) → low entropy → high reliability.

    Args:
        raw_src: [B, L] - Raw token IDs
        pad_id: PAD token ID to exclude

    Returns:
        reliability: [B] - Flow-level reliability in [0, 1]
    """
    B, L = raw_src.shape
    device = raw_src.device
    tokens_cpu = raw_src.cpu().numpy()
    result = np.zeros(B, dtype=np.float32)

    for b in range(B):
        tokens = tokens_cpu[b]
        tokens = tokens[tokens != pad_id]
        n = len(tokens)
        if n <= 1:
            result[b] = 1.0
            continue
        _, counts = np.unique(tokens, return_counts=True)
        probs = counts.astype(np.float32) / n
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        H_max = np.log(n)
        result[b] = max(0.0, min(1.0, 1.0 - entropy / H_max))

    return torch.from_numpy(result).to(device)


def compute_local_entropy_np(tokens_np, window_size=16, pad_id=0):
    """
    Compute local sliding-window Shannon entropy for each position (pure numpy).

    Can be called from DataLoader workers (no torch dependency).
    t_stat[j] = 1 - H(window_j) / log(window_size)

    Args:
        tokens_np: numpy array [L] (single sample) or [B, L] (batch)
        window_size: Sliding window size (default 16)
        pad_id: PAD token ID to exclude

    Returns:
        numpy array of same shape as input, values in [0, 1]
    """
    if tokens_np.ndim == 1:
        tokens_np = tokens_np[np.newaxis, :]  # [1, L]
        squeeze = True
    else:
        squeeze = False

    B, L = tokens_np.shape
    half_w = window_size // 2
    H_max = math.log(window_size) if window_size > 1 else 1.0

    result = np.zeros((B, L), dtype=np.float32)

    for b in range(B):
        for j in range(L):
            if tokens_np[b, j] == pad_id:
                continue
            start = max(0, j - half_w)
            end = min(L, j + half_w + 1)
            window = tokens_np[b, start:end]
            window = window[window != pad_id]
            n = len(window)
            if n <= 1:
                result[b, j] = 1.0
                continue
            _, counts = np.unique(window, return_counts=True)
            probs = counts.astype(np.float32) / n
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            result[b, j] = max(0.0, min(1.0, 1.0 - entropy / H_max))

    return result[0] if squeeze else result


@torch.no_grad()
def compute_local_entropy(tokens, window_size=16, pad_id=0):
    """
    Compute local sliding-window Shannon entropy (torch wrapper).

    Prefer using precomputed local_ent from DataLoader to avoid GPU idle time.
    This function is kept for inference / fine-tuning where precomputation is not set up.

    Args:
        tokens: [B, L] - Token IDs (torch tensor)
        window_size: Sliding window size (default 16)
        pad_id: PAD token ID to exclude

    Returns:
        local_reliability: [B, L] - Per-position reliability in [0, 1]
    """
    device = tokens.device
    result = compute_local_entropy_np(tokens.cpu().numpy(), window_size, pad_id)
    return torch.from_numpy(result).to(device)


# ===== ITGCA Auxiliary Losses =====

def compute_crc_loss(r_mod, r_stat, margin=0.1, k_ratio=0.25):
    """
    Contrastive Reliability Calibration (CRC) ranking loss.

    Ensures the modality gate preserves the ordering of statistical reliability:
    if r_stat(A) > r_stat(B), then r_mod(A) should be > r_mod(B) + margin.

    Only requires ordering preservation, not value matching (unlike MSE).
    Uses top-k vs bottom-k pairs for efficiency.

    Args:
        r_mod: [B] - Learned modality gate values
        r_stat: [B] - Statistical reliability values
        margin: Ranking margin
        k_ratio: Fraction of batch for top-k and bottom-k

    Returns:
        crc_loss: scalar
    """
    B = r_stat.shape[0]
    k = max(1, int(B * k_ratio))

    if B < 2 or k < 1:
        return torch.tensor(0.0, device=r_mod.device)

    _, sorted_idx = r_stat.sort(descending=True)
    top_idx = sorted_idx[:k]
    bottom_idx = sorted_idx[-k:]

    r_mod_top = r_mod[top_idx]        # [k]
    r_mod_bottom = r_mod[bottom_idx]  # [k]

    # Pairwise margin ranking: max(0, margin - (top - bottom))
    diff = r_mod_top.unsqueeze(1) - r_mod_bottom.unsqueeze(0)  # [k, k]
    loss = F.relu(margin - diff)

    return loss.mean()


def compute_entropy_reg(gate_values_list):
    """
    Entropy regularization to prevent gate collapse (all-open or all-closed).

    Encourages batch-mean gate value toward 0.5 (maximum binary entropy).

    Args:
        gate_values_list: list of [B, L] gate tensors

    Returns:
        neg_entropy: scalar (minimize to maximize entropy)
    """
    if not gate_values_list:
        return torch.tensor(0.0)

    total_sum = 0.0
    total_count = 0
    for g in gate_values_list:
        total_sum = total_sum + g.sum()
        total_count += g.numel()

    gate_mean = total_sum / total_count

    eps = 1e-7
    gate_mean = gate_mean.clamp(eps, 1 - eps)
    entropy = -(gate_mean * gate_mean.log() + (1 - gate_mean) * (1 - gate_mean).log())

    return -entropy  # Negative entropy as loss


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
            self.crc_margin = getattr(args, 'crc_margin', 0.1)
            self.lambda_crc = getattr(args, 'lambda_crc', 0.1)
            self.lambda_ent = getattr(args, 'lambda_ent', 0.01)

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

    def _compute_itgca_signals(self, raw_src, raw_cls, size_cls,
                               local_ent_raw=None):
        """
        Compute ITGCA statistical priors and encoder CLS signals (asymmetric).

        Only computes priors for the Raw modality (Size is stable, no prior needed).

        Args:
            raw_src: [B, L_raw] - Raw token IDs
            raw_cls: [B, H] - Raw encoder CLS (will be detached)
            size_cls: [B, H] - Size encoder CLS (will be detached)
            local_ent_raw: [B, L_raw] or None - Precomputed local entropy (from DataLoader)

        Returns:
            itgca_kwargs: dict of keyword arguments for fusion forward
        """
        r_stat_raw = compute_flow_reliability_raw(raw_src)

        # Use precomputed local entropy if available, otherwise compute on-the-fly
        if local_ent_raw is None:
            local_ent_raw = compute_local_entropy(raw_src, self.itgca_window_size)

        raw_cls_enc = raw_cls.detach()
        size_cls_enc = size_cls.detach()

        return {
            'raw_cls_enc': raw_cls_enc,
            'size_cls_enc': size_cls_enc,
            'r_stat_raw': r_stat_raw,
            'local_ent_raw': local_ent_raw,
        }

    def _compute_itgca_losses(self, gate_info_list, r_stat_raw):
        """
        Compute ITGCA auxiliary losses from gate info (asymmetric).

        CRC loss only for Size←Raw direction (where r_stat_raw is the prior).
        Raw←Size has no statistical prior, so no CRC needed.
        Entropy regularization applies to all gates (both directions).

        Args:
            gate_info_list: list of gate_info dicts (one per fusion layer)
            r_stat_raw: [B] - Raw statistical reliability

        Returns:
            crc_loss: scalar - CRC ranking loss
            ent_loss: scalar - Entropy regularization loss
        """
        crc_loss = torch.tensor(0.0, device=r_stat_raw.device)
        all_gates = []

        for gi in gate_info_list:
            # CRC only for Size←Raw direction (source is raw, has statistical prior)
            crc_loss = crc_loss + compute_crc_loss(gi['r_mod_s2r'], r_stat_raw, self.crc_margin)

            all_gates.append(gi['gate_raw'])
            all_gates.append(gi['gate_size'])

        crc_loss = crc_loss / len(gate_info_list)
        ent_loss = compute_entropy_reg(all_gates)

        return crc_loss, ent_loss

    def forward(self, raw_src, raw_packet_ids, raw_directions,
                size_src_clean, iat_src_clean,
                size_src_masked, iat_src_masked,
                tgt_mlm_size, tgt_mlm_temporal,
                local_ent_raw=None):
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
            local_ent_raw: [batch, seq_len_raw] or None - Precomputed local entropy from DataLoader

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
                raw_src, raw_cls, size_cls,
                local_ent_raw=local_ent_raw
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
            itgca_kwargs_neg2 = {
                'raw_cls_enc': itgca_kwargs['raw_cls_enc'][neg_raw_idx],
                'size_cls_enc': itgca_kwargs['size_cls_enc'],
                'r_stat_raw': itgca_kwargs['r_stat_raw'][neg_raw_idx],
                'local_ent_raw': itgca_kwargs['local_ent_raw'][neg_raw_idx],
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

        # ===== ITGCA Auxiliary Losses =====
        if self.use_itgca and gate_info_pos is not None:
            crc_loss, ent_loss = self._compute_itgca_losses(
                gate_info_pos,
                itgca_kwargs['r_stat_raw']
            )
        else:
            crc_loss = torch.tensor(0.0, device=raw_src.device)
            ent_loss = torch.tensor(0.0, device=raw_src.device)

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
            'crc_loss': crc_loss,
            'ent_loss': ent_loss,
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
