"""
Multi-Modal Classifier — Inference under Behavior-Modality Adversarial Attacks

Loads a fine-tuned Stage2Classifier checkpoint and evaluates on a test set under
a user-configured sweep of clean + (any subset of) jitter / padding / combined
attack settings.

Three independent attack strategies (any combination, any number of strengths):
  - jitter:   Δt' = Δt · (1 + ε), ε ~ U(0, α), clamp ≥ 1μs    (only iat_src)
  - padding:  |s'| = min(|s| + p, 1500), p ~ U(0, P) bytes    (only size_src)
  - combined: jitter (α) AND padding (P) on the same sample   (both)

Both perturbations are non-negative (delays only, padding only) — matching a
realistic adversary that cannot send packets earlier or shrink payloads.

Perturbation is applied at the *physical* level (seconds, bytes) by inverting
the tokenization, applying noise, and re-tokenizing. Special tokens
([PAD]/[UNK]/[CLS]/[SEP]/[MASK]) are skipped.

Usage examples:
    # Three combined tiers (mild / moderate / severe) — original 4-row sweep
    python fine-tuning/run_inference_with_attack.py \
        --load_model_path ... --test_path ... --label2id_path ... \
        --combined_pairs 0.05:32,0.15:96,0.3:256 \
        --output_csv results/combined.csv

    # Decoupled: only jitter, three strengths
    python fine-tuning/run_inference_with_attack.py \
        ... --jitter_alphas 0.05,0.15,0.3

    # Mix everything (clean + 3 jitter + 3 padding + 3 combined = 10 rows)
    python fine-tuning/run_inference_with_attack.py \
        ... --jitter_alphas 0.05,0.15,0.3 \
            --padding_strengths 32,96,256 \
            --combined_pairs 0.05:32,0.15:96,0.3:256

    # Clean only (empty sweep) — all attack flags omitted
    python fine-tuning/run_inference_with_attack.py ...

    # Stage1 single-modality (e.g., behavior-only). Use --stage 1 + --modality.
    python fine-tuning/run_inference_with_attack.py \
        --stage 1 --modality size \
        --load_model_path .../stage1_size.bin \
        ... --combined_pairs 0.05:32,0.15:96,0.3:256
"""

import os
import sys
sys.path.append(os.getcwd())

import argparse
import copy
import csv
import math
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from run_classifier_stage1 import Stage1Classifier
from run_classifier_stage2 import (
    Stage2Classifier, load_dataset, pre_tensorize, batch_loader
)
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam, apply_modality_configs
from uer.utils.seed import set_seed
from uer.opts import model_opts


# Vocab offset: first 5 tokens are [PAD] [UNK] [CLS] [SEP] [MASK]
VOCAB_OFFSET = 5


# ============================================================
# Perturbation
# ============================================================

def perturb_iat(iat_src, alpha, rng):
    """In-place multiplicative delay on IAT tokens (non-negative only).

    Δt' = Δt · (1 + ε),  ε ~ U(0, α),  clamp Δt' ≥ 1μs.
    Adversary can delay packets but not send them earlier.

    Forward tokenization (data_generation/multimodal_data_gen.py):
        log10_iat  = log10(iat_seconds)
        normalized = sigmoid(log10_iat)              # sigmoid uses e
        bin        = int(normalized * 1000)          # clipped to [0, 999]

    To invert we need TWO different log bases:
        log10_iat = ln(normalized / (1 - normalized))   # natural log inverts sigmoid
        iat_sec   = 10 ** log10_iat                     # base-10 inverts log10
    """
    for i, vid in enumerate(iat_src):
        if vid < VOCAB_OFFSET:
            continue
        bin_ = vid - VOCAB_OFFSET  # 0..999
        # bin=0 and bin=999 hit logit boundaries (∞); skip to avoid blowup.
        if bin_ <= 0 or bin_ >= 999:
            continue
        normalized = bin_ / 1000.0
        log10_iat = math.log(normalized / (1.0 - normalized))   # natural log → undoes sigmoid
        iat_seconds = 10.0 ** log10_iat                         # 10^x       → undoes log10

        eps = rng.uniform(0.0, alpha)                           # delays only
        iat_seconds = max(iat_seconds * (1.0 + eps), 1e-6)      # clamp ≥ 1μs

        log10_iat_new = math.log10(iat_seconds)
        new_normalized = 1.0 / (1.0 + math.exp(-log10_iat_new))
        new_bin = int(new_normalized * 1000)
        new_bin = max(0, min(999, new_bin))
        iat_src[i] = new_bin + VOCAB_OFFSET


def perturb_size(size_src, P, rng):
    """In-place additive non-negative padding on size tokens.

    |s'| = min(|s| + p, 1500),  p ~ U(0, P).  Direction unchanged.
    Skip empty payload (size = 0), per FRONT semantics.

    Inversion of `size_token = size * direction + 1500`:
        raw = vid - 5  (range 0..3000)
        signed = raw - 1500  (range -1500..1500)
        direction = sign(signed),  size = |signed|
    """
    for i, vid in enumerate(size_src):
        if vid < VOCAB_OFFSET:
            continue
        raw = vid - VOCAB_OFFSET  # 0..3000
        signed = raw - 1500  # -1500..1500
        if signed == 0:
            continue
        direction = 1 if signed > 0 else -1
        magnitude = abs(signed)

        p = rng.randint(0, P)
        new_mag = min(magnitude + p, 1500)

        new_raw = new_mag * direction + 1500
        size_src[i] = new_raw + VOCAB_OFFSET


def apply_attack(dataset, mode, alpha, P, seed):
    """Return a deep-copied dataset with the requested perturbation applied.

    mode ∈ {'jitter', 'padding', 'combined'}:
        jitter   uses alpha only (P ignored)
        padding  uses P only (alpha ignored)
        combined uses both
    """
    out = copy.deepcopy(dataset)
    # Independent sub-streams so jitter ↔ padding draws don't interleave.
    rng_iat = random.Random(seed)
    rng_size = random.Random(seed + 1)
    for sample in out:
        if mode in ('jitter', 'combined'):
            sample['iat_src'] = list(sample['iat_src'])
            perturb_iat(sample['iat_src'], alpha, rng_iat)
        if mode in ('padding', 'combined'):
            sample['size_src'] = list(sample['size_src'])
            perturb_size(sample['size_src'], P, rng_size)
    return out


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def evaluate(args, model, eval_tensors):
    model.eval()
    y_true, y_pred = [], []
    for raw_src, packet_ids, directions, size_src, iat_src, tgt in \
            batch_loader(args.batch_size, eval_tensors):
        raw_src = raw_src.to(args.device, non_blocking=True)
        packet_ids = packet_ids.to(args.device, non_blocking=True)
        directions = directions.to(args.device, non_blocking=True)
        size_src = size_src.to(args.device, non_blocking=True)
        iat_src = iat_src.to(args.device, non_blocking=True)
        tgt = tgt.to(args.device, non_blocking=True)

        output = model(raw_src, packet_ids, directions, size_src, iat_src)
        if isinstance(output, tuple):
            logits = output[1] if output[0] is None else output[0]
        else:
            logits = output

        pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        y_true.extend(tgt.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    correct = sum(1 for p, g in zip(y_pred, y_true) if p == g)
    ac = correct / max(len(y_true), 1)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return ac, macro_f1


# ============================================================
# Args (mirrors run_inference.py / visualize_gate.py)
# ============================================================

def build_args():
    parser = argparse.ArgumentParser(
        description="Stage1 / Stage2 inference under combined behavior-modality attacks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage selection
    parser.add_argument("--stage", type=int, choices=[1, 2], default=2,
                        help="1: single-modality (Stage1Classifier, requires --modality). "
                             "2: full multimodal (Stage2Classifier).")
    parser.add_argument("--modality", choices=["raw", "size", "both"], default="size",
                        help="[Stage 1 only] Single modality "
                             "('size' = behavior-only [size + IAT], 'raw' = content-only).")

    # Paths
    parser.add_argument("--load_model_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--label2id_path", type=str, required=True)
    parser.add_argument("--vocab_path_raw", type=str, default="./vocab_raw.txt")
    parser.add_argument("--vocab_path_size", type=str, default="./vocab_size.txt")
    parser.add_argument("--vocab_path_temporal", type=str, default="./vocab_temporal.txt")
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json")
    parser.add_argument("--config_path_raw", type=str, default=None)
    parser.add_argument("--config_path_size", type=str, default=None)

    # Model architecture (must match training)
    model_opts(parser)
    parser.add_argument("--num_fusion_layers", type=int, default=6)
    parser.add_argument("--use_itgca", action="store_true")
    parser.add_argument("--itgca_window_size", type=int, default=16)
    parser.add_argument("--ablate_r_stat", action="store_true")
    parser.add_argument("--ablate_g_token", action="store_true")
    parser.add_argument("--ablate_source_bias", action="store_true")
    parser.add_argument("--simple_classifier", action="store_true")
    parser.add_argument("--use_attn_pooling", action="store_true")
    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

    # Inference
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int)

    # Attack — three independent strategies, all optional. Empty = clean only.
    parser.add_argument("--jitter_alphas", type=str, default="",
                        help="Comma-separated list of α values for IAT-only "
                             "jitter (e.g., '0.05,0.15,0.3'). Empty = skip.")
    parser.add_argument("--padding_strengths", type=str, default="",
                        help="Comma-separated list of P values (bytes) for "
                             "size-only padding (e.g., '32,96,256'). Empty = skip.")
    parser.add_argument("--combined_pairs", type=str, default="",
                        help="Comma-separated 'α:P' pairs for combined "
                             "jitter+padding (e.g., '0.05:32,0.15:96,0.3:256'). "
                             "Empty = skip.")
    parser.add_argument("--attack_seed", type=int, default=1234,
                        help="RNG seed for perturbations (decoupled from --seed)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional CSV output path for the metrics table.")

    # MoE compat (unused, mirrors run_inference.py)
    parser.add_argument("--is_moe", action="store_true")
    parser.add_argument("--vocab_size", type=int, required=False)
    parser.add_argument("--moebert_expert_dim", type=int, default=3072)
    parser.add_argument("--moebert_expert_num", type=int, required=False)
    parser.add_argument("--moebert_route_method", default="hash-random", type=str)
    parser.add_argument("--moebert_route_hash_list", default=None, type=str)
    parser.add_argument("--moebert_load_balance", type=float, default=0.0)

    args = parser.parse_args()

    if args.config_path:
        args = load_hyperparam(args)
    args = apply_modality_configs(args)

    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)
    args.num_dropouts = 1
    args.use_scl = False
    args.dropout = getattr(args, 'dropout', 0.1)

    return args


def setup_device(args):
    if len(args.gpu_ranks) >= 1:
        assert torch.cuda.is_available(), "No available GPUs."
        gpu_id = args.gpu_ranks[0]
        args.device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id}")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        print("Using default GPU: cuda:0")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")


def load_checkpoint_strict(model, ckpt_path, args):
    """Replicate the safe-skip logic from run_inference.py.

    Works for both Stage2 (full ITGCA flags) and Stage1 (only scl_projection
    skip applies).
    """
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    unexpected = sorted(ckpt_keys - model_keys)
    missing = sorted(model_keys - ckpt_keys)

    safe_to_skip = {k for k in unexpected if 'scl_projection' in k}
    if getattr(args, 'ablate_g_token', False):
        safe_to_skip |= {k for k in unexpected
                         if any(f in k for f in ['.W_k.', '.W_v.', '.token_gate_bias'])}
    if getattr(args, 'ablate_r_stat', False):
        safe_to_skip |= {k for k in unexpected
                         if any(f in k for f in ['.alpha_modality',
                                                 '.gate_size.stat_scale',
                                                 '.gate_size.stat_shift'])}
    if getattr(args, 'ablate_source_bias', False):
        safe_to_skip |= {k for k in unexpected
                         if any(f in k for f in ['.local_stat_scale', '.local_stat_shift'])}

    unsafe_unexpected = sorted(set(unexpected) - safe_to_skip)
    if unsafe_unexpected or missing:
        print(f"\n[ERROR] checkpoint/model key mismatch.")
        if unsafe_unexpected:
            print(f"  Unexpected ({len(unsafe_unexpected)}): {unsafe_unexpected[:6]}...")
        if missing:
            print(f"  Missing    ({len(missing)}): {missing[:6]}...")
        print("  Re-check architecture flags (--stage / --use_itgca / --modality / "
              "--simple_classifier / --num_fusion_layers / --ablate_*) against "
              "this checkpoint.")
        sys.exit(1)

    filtered = {k: v for k, v in state_dict.items() if k not in safe_to_skip}
    model.load_state_dict(filtered, strict=True)
    print(f"  Loaded checkpoint (skipped {len(safe_to_skip)} training-only keys).")


def build_model(args, vocab_raw, vocab_size, vocab_temporal):
    """Build either Stage1Classifier (single-modality) or Stage2Classifier."""
    if args.stage == 1:
        print(f"Building Stage1 model (modality={args.modality})...")
        model = Stage1Classifier(
            args, len(vocab_raw), len(vocab_size), len(vocab_temporal),
            args.labels_num,
        )
    else:
        print("Building Stage2 model...")
        model = Stage2Classifier(
            args, len(vocab_raw), len(vocab_size), len(vocab_temporal),
            args.labels_num,
        )
    print(f"Loading checkpoint: {args.load_model_path}")
    load_checkpoint_strict(model, args.load_model_path, args)
    return model


# ============================================================
# Main
# ============================================================

def build_settings(args):
    """Parse CLI strings into a list of (name, mode, alpha, P) tuples.

    Always starts with the clean baseline. Order: clean → jitter* → padding* →
    combined*, in the order given on the command line.
    """
    settings = [('clean', 'none', None, None)]

    if args.jitter_alphas.strip():
        for tok in args.jitter_alphas.split(','):
            tok = tok.strip()
            if not tok:
                continue
            a = float(tok)
            settings.append((f'jitter_a{a:g}', 'jitter', a, None))

    if args.padding_strengths.strip():
        for tok in args.padding_strengths.split(','):
            tok = tok.strip()
            if not tok:
                continue
            P = int(tok)
            settings.append((f'padding_P{P}', 'padding', None, P))

    if args.combined_pairs.strip():
        for tok in args.combined_pairs.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if ':' not in tok:
                raise ValueError(f"--combined_pairs entry '{tok}' must be 'α:P'")
            a_str, P_str = tok.split(':')
            a = float(a_str)
            P = int(P_str)
            settings.append((f'combined_a{a:g}_P{P}', 'combined', a, P))

    return settings


def main():
    args = build_args()
    set_seed(args.seed)

    # Vocab
    print("Loading vocabularies...")
    vocab_raw = Vocab(); vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab(); vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab(); vocab_temporal.load(args.vocab_path_temporal)

    # Labels
    label2id = load_dataset(args.label2id_path)
    args.labels_num = len(label2id)
    print(f"  Classes: {args.labels_num}")

    # Test data
    print(f"Loading test set: {args.test_path}")
    test_data = load_dataset(args.test_path)
    print(f"  Samples: {len(test_data)}")

    # Model (Stage1 single-modality or Stage2 multimodal)
    model = build_model(args, vocab_raw, vocab_size, vocab_temporal)
    setup_device(args)
    model = model.to(args.device)

    # Build sweep
    settings = build_settings(args)
    print(f"\nRunning attack sweep ({len(settings)} settings)...")
    results = []
    clean_ac, clean_f1 = None, None
    for name, mode, alpha, P in settings:
        if mode == 'none':
            data = test_data
            print(f"  [{name}] no perturbation")
        else:
            tag = f"α={alpha}" if mode == 'jitter' else (
                  f"P={P}" if mode == 'padding' else f"α={alpha}, P={P}")
            print(f"  [{name}] mode={mode}, {tag} — perturbing...")
            data = apply_attack(test_data, mode, alpha, P, args.attack_seed)
        tensors = pre_tensorize(data)
        ac, f1 = evaluate(args, model, tensors)
        if name == 'clean':
            clean_ac, clean_f1 = ac, f1
        d_ac = ac - clean_ac if clean_ac is not None and name != 'clean' else None
        d_f1 = f1 - clean_f1 if clean_f1 is not None and name != 'clean' else None
        results.append({
            'setting': name, 'mode': mode, 'alpha': alpha, 'P': P,
            'AC': ac, 'F1': f1, 'dAC': d_ac, 'dF1': d_f1,
        })

    # Print table — width adapts to longest setting name
    name_w = max(8, max(len(r['setting']) for r in results))
    total_w = name_w + 2 + 8 + 1 + 8 + 1 + 9 + 1 + 9
    print("\n" + "=" * total_w)
    stage_tag = (f"Stage1 (modality={args.modality})"
                 if args.stage == 1 else "Stage2 (multimodal)")
    print(f"  Model: {stage_tag}")
    print(f"{'setting':<{name_w}s}  {'AC':>8s} {'F1':>8s} {'ΔAC':>9s} {'ΔF1':>9s}")
    print("-" * total_w)
    for r in results:
        d_ac_s = f"{r['dAC']:+.4f}" if r['dAC'] is not None else "    -    "
        d_f1_s = f"{r['dF1']:+.4f}" if r['dF1'] is not None else "    -    "
        print(f"{r['setting']:<{name_w}s}  {r['AC']:>8.4f} {r['F1']:>8.4f} "
              f"{d_ac_s:>9s} {d_f1_s:>9s}")
    print("=" * total_w)

    # CSV
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['setting', 'mode', 'alpha', 'P',
                        'AC', 'F1_macro', 'dAC', 'dF1'])
            for r in results:
                w.writerow([
                    r['setting'], r['mode'],
                    r['alpha'] if r['alpha'] is not None else '',
                    r['P'] if r['P'] is not None else '',
                    f"{r['AC']:.6f}", f"{r['F1']:.6f}",
                    f"{r['dAC']:.6f}" if r['dAC'] is not None else '',
                    f"{r['dF1']:.6f}" if r['dF1'] is not None else '',
                ])
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
