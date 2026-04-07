"""
Robustness Evaluation against Adaptive Traffic Shaping Adversaries

Applies additive perturbations to behavior modality (packet size / IAT)
while keeping content modality (raw payload) unchanged.
Measures accuracy/F1 degradation across perturbation intensities.

Threat model (CertTA / NetMasquerade):
  - Adversary can only INCREASE packet sizes (padding) and INCREASE IAT (delay)
  - Non-negative constraint: δ_l ≥ 0, δ_t ≥ 0
  - Content (encrypted payload) is untouched

Usage:
    # Size padding attack on Stage2 model
    python fine-tuning/robustness_eval.py \
        --model_path models/classifier_stage2.bin \
        --model_type stage2 \
        --test_path datasets/test.pkl \
        --label2id_path datasets/label2id.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --config_path models/bert/base_config.json \
        --attack_type size \
        --intensities "0,100,300,500,1000,1500" \
        --output_path results/robustness_size.csv \
        --use_itgca --seed 42

    # Timing delay attack on Stage1-size model
    python fine-tuning/robustness_eval.py \
        --model_path models/classifier_stage1_size.bin \
        --model_type stage1_size \
        --test_path datasets/test.pkl \
        --label2id_path datasets/label2id.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --config_path models/bert/base_config.json \
        --attack_type timing \
        --intensities "0,10,50,200,1000,5000" \
        --output_path results/robustness_timing.csv \
        --seed 42
"""

import os
import sys
sys.path.append(os.getcwd())

import argparse
import csv
import pickle
import math
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

from run_classifier_stage2 import (
    Stage2Classifier, load_dataset, pre_tensorize, batch_loader
)
from run_classifier_stage1 import Stage1Classifier
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import model_opts


# ============================================================
# Token encoding constants (verified from vocab files)
# ============================================================
SPECIAL_OFFSET = 5       # [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
SIZE_MID = 1500          # size_value = size * direction + 1500
SIZE_TOKEN_MAX = 3005    # 3000 + offset
SIZE_TOKEN_MIN = 5       # 0 + offset
IAT_TOKEN_MAX = 1004     # 999 + offset
IAT_TOKEN_MIN = 5        # 0 + offset
IAT_BINS = 1000
CLS_ID = 2
SEP_ID = 3
PAD_ID = 0
SPECIAL_PKT_ID = 8       # packet_id for CLS/SEP/PAD
NEUTRAL_DIR = 1           # direction for special tokens


# ============================================================
# IAT encode / decode helpers
# ============================================================

def decode_iat_token(token_id):
    """Decode IAT token ID → seconds."""
    iat_bin = token_id - SPECIAL_OFFSET
    normalized = max(0.001, min(0.999, iat_bin / IAT_BINS))
    log_iat = math.log(normalized / (1.0 - normalized))
    return 10.0 ** log_iat


def encode_iat_seconds(iat_seconds):
    """Encode IAT seconds → token ID."""
    log_iat = math.log10(max(1e-10, iat_seconds))
    normalized = 1.0 / (1.0 + math.exp(-log_iat))
    iat_bin = int(normalized * IAT_BINS)
    iat_bin = max(0, min(IAT_BINS - 1, iat_bin))
    return iat_bin + SPECIAL_OFFSET


def compute_size_range(size_src):
    """Compute (min_size, max_size) in bytes from size tokens."""
    real = size_src[size_src >= SPECIAL_OFFSET] - SPECIAL_OFFSET
    sizes_bytes = torch.abs(real.float() - SIZE_MID)
    return int(sizes_bytes.min().item()), int(sizes_bytes.max().item())


def compute_dataset_stats(size_src, iat_src):
    """Compute dataset-level statistics for auto-deriving perturbation parameters.

    Returns:
        dict with keys: size_max, size_median, iat_median_ms, iat_mean_ms
    """
    # Size stats (bytes)
    real_size = size_src[size_src >= SPECIAL_OFFSET] - SPECIAL_OFFSET
    sizes_bytes = torch.abs(real_size.float() - SIZE_MID)

    # IAT stats (decode to milliseconds)
    real_iat = iat_src[iat_src >= SPECIAL_OFFSET]
    iat_ms_list = []
    for t in real_iat.tolist():
        iat_ms_list.append(decode_iat_token(t) * 1000.0)  # seconds → ms
    iat_ms = np.array(iat_ms_list)

    return {
        'size_max': int(sizes_bytes.max().item()),
        'size_median': int(sizes_bytes.median().item()),
        'iat_median_ms': float(np.median(iat_ms)),
        'iat_mean_ms': float(np.mean(iat_ms)),
    }


def make_rate_mask(seq_len, n_real, n1, rate_front, rate_rear, rng):
    """Generate per-position boolean mask with two-region rates.

    Returns BoolTensor [seq_len] — True where perturbation should apply.
    Only real-packet positions (first n_real non-special tokens) are candidates.
    """
    mask = np.zeros(seq_len, dtype=bool)
    for j in range(n_real):
        rate = rate_front if j < n1 else rate_rear
        if rng.random() < rate:
            mask[j] = True
    return mask


# ============================================================
# Perturbation functions
# ============================================================

def perturb_size(size_src, delta_max, rng, rate_front=1.0, rate_rear=1.0,
                 n1=8, skip_mask=None):
    """
    Additive size padding with two-region rate control.

    For each non-special, non-skipped token (with probability = region rate):
      - Uplink  (size_value > 1500): token += δ
      - Downlink (size_value < 1500): token -= δ

    Args:
        size_src: LongTensor [N, L] — size token IDs (modified in-place)
        delta_max: int — max padding in bytes (0 = no perturbation)
        rng: numpy RandomState
        rate_front: perturbation probability for first n1 packets
        rate_rear: perturbation probability for remaining packets
        n1: front region size (default 8)
        skip_mask: BoolTensor [N, L] or None — True = skip (e.g., dummy positions)
    """
    if delta_max <= 0:
        return

    N, L = size_src.shape
    for i in range(N):
        seq = size_src[i]
        real_positions = (seq >= SPECIAL_OFFSET).nonzero(as_tuple=True)[0]
        for pkt_idx, pos in enumerate(real_positions):
            pos = pos.item()
            if skip_mask is not None and skip_mask[i, pos]:
                continue
            rate = rate_front if pkt_idx < n1 else rate_rear
            if rng.random() >= rate:
                continue
            size_val = seq[pos].item() - SPECIAL_OFFSET
            size_bytes = abs(size_val - SIZE_MID)
            # Skip small packets (payload < 64B) to keep raw-size consistency
            if size_bytes < 64:
                continue
            delta = rng.randint(0, delta_max + 1)
            if size_val > SIZE_MID:  # uplink
                size_src[i, pos] = min(seq[pos].item() + delta, SIZE_TOKEN_MAX)
            elif size_val < SIZE_MID:  # downlink
                size_src[i, pos] = max(seq[pos].item() - delta, SIZE_TOKEN_MIN)


def perturb_timing(iat_src, delta_ms, rng, rate_front=1.0, rate_rear=1.0,
                   n1=8, skip_mask=None):
    """
    Additive timing delay with two-region rate control.

    For each non-special, non-skipped token (with probability = region rate):
      Decode IAT → add Exp(delta_ms) delay → re-encode.

    Args:
        iat_src: LongTensor [N, L] — IAT token IDs (modified in-place)
        delta_ms: float — mean delay in ms (0 = no perturbation)
        rng: numpy RandomState
        rate_front: perturbation probability for first n1 packets
        rate_rear: perturbation probability for remaining packets
        n1: front region size (default 8)
        skip_mask: BoolTensor [N, L] or None — True = skip (e.g., dummy positions)
    """
    if delta_ms <= 0:
        return

    delta_seconds = delta_ms / 1000.0
    N, L = iat_src.shape
    for i in range(N):
        seq = iat_src[i]
        real_positions = (seq >= SPECIAL_OFFSET).nonzero(as_tuple=True)[0]
        for pkt_idx, pos in enumerate(real_positions):
            pos = pos.item()
            if skip_mask is not None and skip_mask[i, pos]:
                continue
            rate = rate_front if pkt_idx < n1 else rate_rear
            if rng.random() >= rate:
                continue
            iat_sec = decode_iat_token(seq[pos].item())
            delay = rng.exponential(scale=delta_seconds)
            iat_src[i, pos] = encode_iat_seconds(iat_sec + delay)


def perturb_size_fixed(size_src, fixed_size, skip_mask=None):
    """
    Tamaraw-style: pad ALL non-special packets to a fixed size.
    Completely eliminates size distribution information.

    Args:
        size_src: LongTensor [N, L] — size token IDs (modified in-place)
        fixed_size: int — target size in bytes (e.g., 750 or 1500)
        skip_mask: BoolTensor [N, L] or None — True = skip (dummy positions)
    """
    N, L = size_src.shape
    for i in range(N):
        seq = size_src[i]
        real_positions = (seq >= SPECIAL_OFFSET).nonzero(as_tuple=True)[0]
        for pos in real_positions:
            pos = pos.item()
            if skip_mask is not None and skip_mask[i, pos]:
                continue
            size_val = seq[pos].item() - SPECIAL_OFFSET
            # Preserve direction, replace size with fixed_size
            if size_val > SIZE_MID:  # uplink
                size_src[i, pos] = min(fixed_size + SIZE_MID + SPECIAL_OFFSET, SIZE_TOKEN_MAX)
            elif size_val < SIZE_MID:  # downlink
                size_src[i, pos] = max(SIZE_MID - fixed_size + SPECIAL_OFFSET, SIZE_TOKEN_MIN)


def perturb_timing_fixed(iat_src, fixed_interval_ms, skip_mask=None):
    """
    Tamaraw-style: replace ALL IATs with a fixed interval.
    Completely eliminates timing pattern information.

    Args:
        iat_src: LongTensor [N, L] — IAT token IDs (modified in-place)
        fixed_interval_ms: float — fixed interval in ms (e.g., 10.0)
        skip_mask: BoolTensor [N, L] or None — True = skip (dummy positions)
    """
    fixed_token = encode_iat_seconds(fixed_interval_ms / 1000.0)
    N, L = iat_src.shape
    for i in range(N):
        seq = iat_src[i]
        real_positions = (seq >= SPECIAL_OFFSET).nonzero(as_tuple=True)[0]
        for pos in real_positions:
            pos = pos.item()
            if skip_mask is not None and skip_mask[i, pos]:
                continue
            iat_src[i, pos] = fixed_token


def perturb_dummy_insertion(tensors, rate_front, rate_rear, n1, rng,
                            dummy_fill='random', size_range=(40, 1400)):
    """
    Insert dummy packets into flows with two-region rate control.

    Front region (first n1 packets): modifies raw_src + size_src + iat_src
      - Dummy bytes inserted into raw_src, pushing last real packet out
      - packet_ids and directions updated accordingly
    Rear region (remaining packets): modifies only size_src + iat_src

    Dummy packet properties:
      - Size: Uniform(size_range[0], size_range[1])
      - Direction: random uplink/downlink
      - IAT: splits original interval by r ~ Uniform(0.2, 0.8)
      - Payload: random bytes (dummy_fill='random') or 0x00 (dummy_fill='zero')

    Args:
        tensors: dict of tensors (modified in-place)
        rate_front / rate_rear: insertion probability per region
        n1: front region size (= max_raw_packets, default 8)
        rng: numpy RandomState
        dummy_fill: 'random' or 'zero'
        size_range: (min_bytes, max_bytes) for dummy size

    Returns:
        dummy_mask: BoolTensor [N, L_size] — True at dummy positions
    """
    N, L_size = tensors['size_src'].shape
    _, L_raw = tensors['raw_src'].shape
    dummy_mask = torch.zeros(N, L_size, dtype=torch.bool)

    for i in range(N):
        size_seq = tensors['size_src'][i]
        iat_seq = tensors['iat_src'][i]

        # Find real packet positions in size_src
        real_pos = (size_seq >= SPECIAL_OFFSET).nonzero(as_tuple=True)[0].tolist()
        n_real = len(real_pos)
        if n_real == 0:
            continue

        # Collect original tokens
        real_sizes = [size_seq[p].item() for p in real_pos]
        real_iats = [iat_seq[p].item() for p in real_pos]

        # Decide insertion points
        insert_after = set()
        for j in range(n_real):
            rate = rate_front if j < n1 else rate_rear
            if rng.random() < rate:
                insert_after.add(j)

        if not insert_after:
            continue

        # ----- Build new size/iat sequences with dummies -----
        new_sizes = []
        new_iats = []
        is_dummy_list = []
        front_dummy_sizes_bytes = []  # track dummy sizes (bytes) for raw_src consistency
        pending_split = None  # (original_next_iat_token, split_ratio)

        for j in range(n_real):
            # Apply pending IAT split from previous dummy insertion
            pkt_iat = real_iats[j]
            if pending_split is not None:
                orig_token, r = pending_split
                orig_sec = decode_iat_token(orig_token)
                pkt_iat = encode_iat_seconds(orig_sec * (1.0 - r))
                pending_split = None

            new_sizes.append(real_sizes[j])
            new_iats.append(pkt_iat)
            is_dummy_list.append(False)

            if j in insert_after:
                # Dummy size: clamp to [1, 1500] bytes
                dummy_bytes = min(1500, rng.randint(size_range[0], size_range[1] + 1))
                dummy_dir = rng.choice([-1, 1])  # -1=downlink, +1=uplink
                dummy_size_val = int(dummy_bytes * dummy_dir + SIZE_MID)
                dummy_size_token = max(SIZE_TOKEN_MIN,
                                       min(SIZE_TOKEN_MAX, dummy_size_val + SPECIAL_OFFSET))

                # Dummy IAT: split gap to next packet
                r = rng.uniform(0.2, 0.8)
                if j + 1 < n_real:
                    next_iat_sec = decode_iat_token(real_iats[j + 1])
                    dummy_iat_token = encode_iat_seconds(next_iat_sec * r)
                    pending_split = (real_iats[j + 1], r)
                else:
                    dummy_iat_token = encode_iat_seconds(0.01)  # 10ms default

                new_sizes.append(dummy_size_token)
                new_iats.append(dummy_iat_token)
                is_dummy_list.append(True)
                if j < n1:
                    front_dummy_sizes_bytes.append(dummy_bytes)

        # Rebuild size_src / iat_src (keep original special-token structure)
        # Find special token positions (CLS, SEP) in original
        special_positions = (size_seq < SPECIAL_OFFSET).nonzero(as_tuple=True)[0].tolist()
        # non-PAD specials (CLS=2, SEP=3)
        pre_specials = [p for p in special_positions if p < real_pos[0] and size_seq[p].item() > 0]
        post_specials = [p for p in special_positions if p > real_pos[-1] and size_seq[p].item() > 0]

        new_size_seq = []
        new_iat_seq = []
        new_dummy_flags = []

        # Add pre-specials (CLS)
        for p in pre_specials:
            new_size_seq.append(size_seq[p].item())
            new_iat_seq.append(iat_seq[p].item())
            new_dummy_flags.append(False)

        # Add real + dummy packets (truncate to fit)
        max_packets = L_size - len(pre_specials) - len(post_specials)
        for k in range(min(len(new_sizes), max_packets)):
            new_size_seq.append(new_sizes[k])
            new_iat_seq.append(new_iats[k])
            new_dummy_flags.append(is_dummy_list[k])

        # Add post-specials (SEP)
        for p in post_specials:
            new_size_seq.append(size_seq[p].item())
            new_iat_seq.append(iat_seq[p].item())
            new_dummy_flags.append(False)

        # Pad
        while len(new_size_seq) < L_size:
            new_size_seq.append(PAD_ID)
            new_iat_seq.append(PAD_ID)
            new_dummy_flags.append(False)

        # Truncate and write back
        tensors['size_src'][i] = torch.tensor(new_size_seq[:L_size], dtype=size_seq.dtype)
        tensors['iat_src'][i] = torch.tensor(new_iat_seq[:L_size], dtype=iat_seq.dtype)
        dummy_mask[i, :len(new_dummy_flags[:L_size])] = torch.tensor(
            new_dummy_flags[:L_size], dtype=torch.bool)

        # ----- Front-region: also modify raw_src -----
        front_inserts = sorted([j for j in insert_after if j < min(n1, n_real)])
        if not front_inserts:
            continue

        raw_seq = tensors['raw_src'][i]
        pkt_ids = tensors['packet_ids'][i]
        dir_seq = tensors['directions'][i]

        # Parse raw_src into per-packet byte lists
        raw_packets = []  # list of (bytes_list, direction)
        for pkt_id in range(8):
            pkt_mask = (pkt_ids == pkt_id)
            if pkt_mask.sum() == 0:
                break
            pkt_bytes = raw_seq[pkt_mask].tolist()
            pkt_dir = dir_seq[pkt_mask][0].item()
            raw_packets.append((pkt_bytes, pkt_dir))

        # Insert dummy packets into raw_src
        new_raw_packets = []
        front_dummy_counter = 0
        for j, (pkt_bytes, pkt_dir) in enumerate(raw_packets):
            new_raw_packets.append((pkt_bytes, pkt_dir))
            if j in front_inserts:
                # raw byte count = min(64, dummy_size) for modality consistency
                d_size = front_dummy_sizes_bytes[front_dummy_counter] if front_dummy_counter < len(front_dummy_sizes_bytes) else 64
                n_bytes = min(64, max(1, d_size))
                if dummy_fill == 'random':
                    d_bytes = (rng.randint(0, 256, size=n_bytes) + SPECIAL_OFFSET).tolist()
                else:
                    d_bytes = [SPECIAL_OFFSET] * n_bytes  # byte 0x00
                d_dir = 0 if rng.random() < 0.5 else 2  # downlink or uplink
                new_raw_packets.append((d_bytes, d_dir))
                front_dummy_counter += 1

        # Take first 8 packets
        new_raw_packets = new_raw_packets[:8]

        # Rebuild raw_src
        new_raw = [CLS_ID]
        new_pids = [SPECIAL_PKT_ID]
        new_dirs = [NEUTRAL_DIR]

        for pkt_idx, (pkt_bytes, pkt_dir) in enumerate(new_raw_packets):
            for bt in pkt_bytes:
                new_raw.append(bt)
                new_pids.append(pkt_idx)
                new_dirs.append(pkt_dir)

        new_raw.append(SEP_ID)
        new_pids.append(SPECIAL_PKT_ID)
        new_dirs.append(NEUTRAL_DIR)

        # Pad and truncate
        while len(new_raw) < L_raw:
            new_raw.append(PAD_ID)
            new_pids.append(PAD_ID)
            new_dirs.append(NEUTRAL_DIR)

        tensors['raw_src'][i] = torch.tensor(new_raw[:L_raw], dtype=raw_seq.dtype)
        tensors['packet_ids'][i] = torch.tensor(new_pids[:L_raw], dtype=pkt_ids.dtype)
        tensors['directions'][i] = torch.tensor(new_dirs[:L_raw], dtype=dir_seq.dtype)

    return dummy_mask


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, tensors, device, batch_size=64):
    """Run inference, return accuracy and macro-F1."""
    model.eval()
    y_true, y_pred = [], []

    for raw_src, packet_ids, directions, size_src, iat_src, tgt in \
            batch_loader(batch_size, tensors):
        raw_src = raw_src.to(device, non_blocking=True)
        packet_ids = packet_ids.to(device, non_blocking=True)
        directions = directions.to(device, non_blocking=True)
        size_src = size_src.to(device, non_blocking=True)
        iat_src = iat_src.to(device, non_blocking=True)

        output = model(raw_src, packet_ids, directions, size_src, iat_src)
        if isinstance(output, tuple):
            # Stage1: (None, logits); Stage2+SCL: (logits, feat)
            logits = output[1] if output[0] is None else output[0]
        else:
            logits = output
        pred = torch.argmax(logits, dim=-1)

        y_true.extend(tgt.tolist())
        y_pred.extend(pred.cpu().tolist())

    acc = sum(p == g for p, g in zip(y_pred, y_true)) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, macro_f1


# ============================================================
# Model loading
# ============================================================

def load_model(args, vocab_raw, vocab_size, vocab_temporal, labels_num, device):
    """Load Stage1 or Stage2 classifier from checkpoint."""
    state_dict = torch.load(args.model_path, map_location='cpu')

    if args.model_type == 'stage2':
        model = Stage2Classifier(
            args, len(vocab_raw), len(vocab_size), len(vocab_temporal), labels_num
        )
    elif args.model_type == 'stage1_size':
        args.modality = 'size'
        model = Stage1Classifier(
            args, len(vocab_raw), len(vocab_size), len(vocab_temporal), labels_num
        )
    elif args.model_type == 'stage1_raw':
        args.modality = 'raw'
        model = Stage1Classifier(
            args, len(vocab_raw), len(vocab_size), len(vocab_temporal), labels_num
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Robustness Evaluation against Traffic Shaping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["stage1_size", "stage1_raw", "stage2"],
                        help="Model type: stage1_size | stage1_raw | stage2")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--label2id_path", type=str, required=True)
    parser.add_argument("--vocab_path_raw", type=str, required=True)
    parser.add_argument("--vocab_path_size", type=str, required=True)
    parser.add_argument("--vocab_path_temporal", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json")
    parser.add_argument("--output_path", type=str, default="results/robustness.csv")

    # Attack config
    parser.add_argument("--attack_type", type=str, default="size",
                        choices=["size", "timing", "dummy", "all"],
                        help="size: padding; timing: delay; dummy: packet insertion; "
                             "all: dummy + size + timing combined")
    parser.add_argument("--intensities", type=str, default="0,0.1,0.2,0.5,1.0",
                        help="Comma-separated rate values (perturbation probability per packet).")
    parser.add_argument("--delta_size_max", type=int, default=None,
                        help="Max padding bytes per packet. Default: dataset max packet size")
    parser.add_argument("--delta_timing_ms", type=float, default=None,
                        help="Mean delay ms per packet. Default: dataset median IAT")
    parser.add_argument("--n1", type=int, default=8,
                        help="Front region size — first n1 packets affect both modalities")
    parser.add_argument("--dummy_fill", type=str, default="random",
                        choices=["random", "zero"],
                        help="Dummy packet payload: random bytes or all-zero")
    parser.add_argument("--pad_mode", type=str, default="random",
                        choices=["random", "fixed"],
                        help="random: additive Uniform[0,delta_max]; "
                             "fixed: Tamaraw-style pad to fixed size (all packets same size)")
    parser.add_argument("--fixed_size", type=int, default=None,
                        help="Target size in bytes for fixed padding mode. Default: dataset max")
    parser.add_argument("--timing_mode", type=str, default="delay",
                        choices=["delay", "fixed_interval"],
                        help="delay: additive Exp(delta_ms); "
                             "fixed_interval: Tamaraw-style replace with fixed interval")
    parser.add_argument("--fixed_interval_ms", type=float, default=None,
                        help="Fixed interval in ms for fixed_interval mode. Default: dataset median IAT")

    # Model architecture (must match checkpoint)
    model_opts(parser)
    parser.add_argument("--num_fusion_layers", type=int, default=6)
    parser.add_argument("--use_itgca", action="store_true")
    parser.add_argument("--itgca_window_size", type=int, default=16)
    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

    # Eval config
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)

    # Compat placeholders
    parser.add_argument("--simple_classifier", action="store_true")
    parser.add_argument("--use_attn_pooling", action="store_true")
    parser.add_argument("--is_moe", action="store_true")
    parser.add_argument("--vocab_size", type=int, required=False)
    parser.add_argument("--moebert_expert_dim", type=int, default=3072)
    parser.add_argument("--moebert_expert_num", type=int, required=False)
    parser.add_argument("--moebert_route_method", default="hash-random")
    parser.add_argument("--moebert_route_hash_list", default=None)
    parser.add_argument("--moebert_load_balance", type=float, default=0.0)

    args = parser.parse_args()

    if args.config_path:
        args = load_hyperparam(args)
    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)
    args.num_dropouts = 1
    args.use_scl = False
    args.dropout = getattr(args, 'dropout', 0.1)
    set_seed(args.seed)

    # Parse intensities
    intensities = [float(x) for x in args.intensities.split(',')]

    # ---- Device ----
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Vocabularies ----
    vocab_raw = Vocab()
    vocab_raw.load(args.vocab_path_raw)
    vocab_size_vocab = Vocab()
    vocab_size_vocab.load(args.vocab_path_size)
    vocab_temporal = Vocab()
    vocab_temporal.load(args.vocab_path_temporal)
    print(f"Vocab — Raw: {len(vocab_raw)}, Size: {len(vocab_size_vocab)}, "
          f"Temporal: {len(vocab_temporal)}")

    # ---- Labels ----
    with open(args.label2id_path, 'rb') as f:
        label2id = pickle.load(f)
    id2label = {v: k for k, v in label2id.items()}
    labels_num = len(label2id)
    print(f"Labels: {labels_num}")

    # ---- Model ----
    print(f"Loading model: {args.model_path} (type={args.model_type})")
    model = load_model(args, vocab_raw, vocab_size_vocab, vocab_temporal,
                       labels_num, device)

    # ---- Dataset ----
    print(f"Loading test set: {args.test_path}")
    dataset = load_dataset(args.test_path)
    print(f"  Samples: {len(dataset)}")

    # Pre-tensorize (base tensors, not yet perturbed)
    base_tensors = pre_tensorize(dataset)

    # Compute dataset statistics for auto-deriving perturbation parameters
    stats = compute_dataset_stats(base_tensors['size_src'], base_tensors['iat_src'])
    size_range = compute_size_range(base_tensors['size_src'])
    print(f"  Dataset stats: size_max={stats['size_max']}B, "
          f"size_median={stats['size_median']}B, "
          f"iat_median={stats['iat_median_ms']:.1f}ms, "
          f"iat_mean={stats['iat_mean_ms']:.1f}ms")

    # Auto-derive parameters from dataset if not specified
    if args.delta_size_max is None:
        args.delta_size_max = stats['size_max']
        print(f"  Auto delta_size_max: {args.delta_size_max}B (dataset max)")
    if args.delta_timing_ms is None:
        args.delta_timing_ms = stats['iat_median_ms']
        print(f"  Auto delta_timing_ms: {args.delta_timing_ms:.1f}ms (dataset median)")
    if args.fixed_size is None:
        args.fixed_size = stats['size_max']
        print(f"  Auto fixed_size: {args.fixed_size}B (dataset max)")
    if args.fixed_interval_ms is None:
        args.fixed_interval_ms = stats['iat_median_ms']
        print(f"  Auto fixed_interval_ms: {args.fixed_interval_ms:.1f}ms (dataset median)")

    # ---- Run evaluation at each rate ----
    results = []
    attack = args.attack_type
    print(f"\nAttack: {attack} | Rates: {intensities}")
    print(f"  pad_mode: {args.pad_mode}, timing_mode: {args.timing_mode}")
    if args.pad_mode == 'random':
        print(f"  delta_size_max: {args.delta_size_max}B")
    else:
        print(f"  fixed_size: {args.fixed_size}B")
    if args.timing_mode == 'delay':
        print(f"  delta_timing_ms: {args.delta_timing_ms:.1f}ms")
    else:
        print(f"  fixed_interval_ms: {args.fixed_interval_ms:.1f}ms")
    if attack in ('dummy', 'all'):
        print(f"  dummy_fill: {args.dummy_fill}, n1: {args.n1}")
    print("=" * 70)
    print(f"{'Rate':>8} {'Rate_R':>8} {'Accuracy':>10} {'Macro-F1':>10} {'Acc Drop':>10}")
    print("-" * 70)

    baseline_acc = None

    for rate in intensities:
        rate_front = rate
        rate_rear = rate
        rng = np.random.RandomState(args.seed)

        # Create perturbed tensors (copy base)
        tensors = {k: v.clone() for k, v in base_tensors.items()}

        # Step 1: Dummy insertion (changes sequence structure)
        dummy_mask = None
        if attack in ('dummy', 'all'):
            dummy_mask = perturb_dummy_insertion(
                tensors, rate_front, rate_rear, args.n1, rng,
                dummy_fill=args.dummy_fill, size_range=size_range
            )

        # Step 2: Size perturbation (skip dummy positions)
        if attack in ('size', 'all'):
            if args.pad_mode == 'fixed':
                perturb_size_fixed(tensors['size_src'], args.fixed_size,
                                   skip_mask=dummy_mask)
            else:
                perturb_size(tensors['size_src'], args.delta_size_max, rng,
                             rate_front, rate_rear, args.n1, skip_mask=dummy_mask)

        # Step 3: Timing perturbation (skip dummy positions)
        if attack in ('timing', 'all'):
            if args.timing_mode == 'fixed_interval':
                perturb_timing_fixed(tensors['iat_src'], args.fixed_interval_ms,
                                     skip_mask=dummy_mask)
            else:
                perturb_timing(tensors['iat_src'], args.delta_timing_ms, rng,
                               rate_front, rate_rear, args.n1, skip_mask=dummy_mask)

        acc, macro_f1 = evaluate(model, tensors, device, args.batch_size)

        if baseline_acc is None:
            baseline_acc = acc
        acc_drop = baseline_acc - acc

        results.append({
            'attack_type': attack,
            'rate_front': rate_front,
            'rate_rear': rate_rear,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'acc_drop': acc_drop,
        })

        print(f"{rate_front:>8.2f} {rate_rear:>8.2f} {acc:>10.4f} "
              f"{macro_f1:>10.4f} {acc_drop:>+10.4f}")

    print("=" * 70)

    # ---- Save CSV ----
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'attack_type', 'rate_front', 'rate_rear', 'accuracy', 'macro_f1', 'acc_drop'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
