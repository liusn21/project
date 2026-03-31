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


# ============================================================
# Perturbation functions
# ============================================================

def perturb_size(size_src, delta_max, rng):
    """
    Additive size padding (non-negative, per-token random).

    For each non-special token:
      - Uplink  (size_value > 1500): token += δ  (pad → larger)
      - Downlink (size_value < 1500): token -= δ  (pad → smaller token value,
                                                    because size_value = 1500 - size)

    Args:
        size_src: LongTensor [N, L] — original size token IDs
        delta_max: int — max padding in bytes (0 = no perturbation)
        rng: numpy RandomState for reproducibility

    Returns:
        perturbed: LongTensor [N, L] — perturbed size token IDs
    """
    if delta_max <= 0:
        return size_src.clone()

    perturbed = size_src.clone()
    # Mask: non-special tokens only (token_id >= SPECIAL_OFFSET)
    mask = perturbed >= SPECIAL_OFFSET  # [N, L]

    # Random δ per token: Uniform(0, delta_max), integers
    deltas = torch.from_numpy(
        rng.randint(0, delta_max + 1, size=perturbed.shape)
    ).to(perturbed.dtype)

    # Decode direction from size_value
    size_values = perturbed - SPECIAL_OFFSET  # [N, L]
    is_uplink = (size_values > SIZE_MID) & mask
    is_downlink = (size_values < SIZE_MID) & mask

    # Uplink: increase token (pad payload)
    perturbed[is_uplink] = torch.clamp(
        perturbed[is_uplink] + deltas[is_uplink], max=SIZE_TOKEN_MAX
    )
    # Downlink: decrease token (same padding logic, direction-encoded)
    perturbed[is_downlink] = torch.clamp(
        perturbed[is_downlink] - deltas[is_downlink], min=SIZE_TOKEN_MIN
    )

    return perturbed


def perturb_timing(iat_src, delta_ms, rng):
    """
    Additive timing delay (non-negative, per-token random).

    Decodes IAT token → actual time → adds delay → re-encodes.
    IAT encoding: token = int(sigmoid(log10(iat)) * 1000)

    Args:
        iat_src: LongTensor [N, L] — original IAT token IDs
        delta_ms: float — mean delay in milliseconds (Exponential distribution)
        rng: numpy RandomState for reproducibility

    Returns:
        perturbed: LongTensor [N, L] — perturbed IAT token IDs
    """
    if delta_ms <= 0:
        return iat_src.clone()

    perturbed = iat_src.clone()
    mask = perturbed >= SPECIAL_OFFSET  # non-special tokens

    # Extract tokens that need perturbation
    idx = mask.nonzero(as_tuple=True)
    tokens = perturbed[idx].float()

    # Decode: token_id → iat_bin → actual IAT (seconds)
    iat_bin = tokens - SPECIAL_OFFSET  # [0, 999]
    normalized = iat_bin / IAT_BINS
    # Clamp to avoid log(0) or log(inf) in logit
    normalized = normalized.clamp(0.001, 0.999)
    log_iat = torch.log(normalized / (1.0 - normalized))  # logit = log10(iat)
    iat_seconds = torch.pow(10.0, log_iat)

    # Add random delay: δ_t ~ Exponential(mean = delta_ms / 1000)
    delta_seconds = delta_ms / 1000.0
    delays = torch.from_numpy(
        rng.exponential(scale=delta_seconds, size=(len(iat_seconds),))
    ).float()
    new_iat = iat_seconds + delays

    # Re-encode: actual IAT → token
    new_log_iat = torch.log10(new_iat.clamp(min=1e-10))
    new_normalized = torch.sigmoid(new_log_iat)
    new_bin = (new_normalized * IAT_BINS).long()
    new_bin = new_bin.clamp(0, IAT_BINS - 1)
    new_token = new_bin + SPECIAL_OFFSET

    perturbed[idx] = new_token.to(perturbed.dtype)
    return perturbed


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
                        choices=["size", "timing", "both"],
                        help="size: packet padding; timing: delay injection; both: combined")
    parser.add_argument("--intensities", type=str, default="0,100,300,500,1000,1500",
                        help="Comma-separated perturbation intensities. "
                             "For size: bytes (0-1500). For timing: milliseconds.")

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

    # ---- Run evaluation at each intensity ----
    results = []
    print(f"\nAttack: {args.attack_type} | Intensities: {intensities}")
    print("=" * 65)
    print(f"{'Intensity':>12} {'Accuracy':>10} {'Macro-F1':>10} {'Acc Drop':>10}")
    print("-" * 65)

    baseline_acc = None

    for intensity in intensities:
        rng = np.random.RandomState(args.seed)

        # Create perturbed tensors (copy base, modify behavior only)
        tensors = {k: v.clone() for k, v in base_tensors.items()}

        if args.attack_type in ('size', 'both'):
            delta_size = int(intensity) if args.attack_type == 'size' else int(intensity)
            tensors['size_src'] = perturb_size(
                tensors['size_src'], delta_size, rng
            )

        if args.attack_type in ('timing', 'both'):
            delta_timing = float(intensity) if args.attack_type == 'timing' else float(intensity)
            tensors['iat_src'] = perturb_timing(
                tensors['iat_src'], delta_timing, rng
            )

        # Verify content unchanged
        assert torch.equal(tensors['raw_src'], base_tensors['raw_src']), \
            "BUG: raw_src was modified!"

        acc, macro_f1 = evaluate(model, tensors, device, args.batch_size)

        if baseline_acc is None:
            baseline_acc = acc
        acc_drop = baseline_acc - acc

        results.append({
            'attack_type': args.attack_type,
            'intensity': intensity,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'acc_drop': acc_drop,
        })

        print(f"{intensity:>12.0f} {acc:>10.4f} {macro_f1:>10.4f} {acc_drop:>+10.4f}")

    print("=" * 65)

    # ---- Save CSV ----
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'attack_type', 'intensity', 'accuracy', 'macro_f1', 'acc_drop'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
