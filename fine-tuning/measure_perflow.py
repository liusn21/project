"""measure_perflow.py — Per-flow inference benchmark for MM-TrafficBERT (Stage 2).

Loads a fine-tuned Stage2Classifier .bin checkpoint + a pickled test set, runs
the model one flow at a time (batch_size=1) and logs per flow:

    flow_id, true_label, pred_label, correct,
    max_confidence, pred_entropy, latency_ms

Plus prints an aggregate summary including the "easy-flow waste" table.

USAGE (run from project/ root):

    cd /Users/liusn25/Downloads/project

    python fine-tuning/measure_perflow.py \\
        --load_model_path     models/classifier_stage2.bin \\
        --test_path           datasets/processed/test.pkl \\
        --label2id_path       datasets/processed/label2id.pkl \\
        --vocab_path_raw      models/vocab_raw.txt \\
        --vocab_path_size     models/vocab_size.txt \\
        --vocab_path_temporal models/vocab_temporal.txt \\
        --config_path         models/bert/base_config.json \\
        --use_itgca \\
        --output              results/<dataset>_perflow.csv

The ITGCA / ablation / fusion-layer flags MUST match the fine-tuned checkpoint
(same rule as run_inference.py). If the loader complains about missing /
unexpected keys, set --use_itgca / --ablate_r_stat / --ablate_g_token /
--ablate_source_bias accordingly.

Default output: results/<load_model_path basename>_perflow.csv  (override --output).

NOTE: MM-TrafficBERT uses fixed-shape padded inputs across raw + size + IAT, so
per-flow latency variance reflects system jitter, not data heterogeneity. The
motivating heterogeneity lives in the (confidence, correctness) pairs,
summarised in the easy-flow waste table.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Allow `from run_classifier_stage2 import ...` and `from uer...`
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _PROJECT_ROOT, os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from run_classifier_stage2 import Stage2Classifier, load_dataset, pre_tensorize  # noqa: E402
from uer.utils.vocab import Vocab  # noqa: E402
from uer.utils.config import load_hyperparam, apply_modality_configs  # noqa: E402
from uer.utils.seed import set_seed  # noqa: E402
from uer.opts import model_opts  # noqa: E402


def _cuda_sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_logits(out):
    """Stage2Classifier returns logits directly; tolerate (logits, scl_feat)
    or (None, logits) variants for safety."""
    if isinstance(out, tuple):
        return out[1] if out[0] is None else out[0]
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="MM-TrafficBERT Stage 2 per-flow inference benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Paths ----
    parser.add_argument("--load_model_path", required=True,
                        help="Path to fine-tuned Stage 2 .bin checkpoint")
    parser.add_argument("--test_path", required=True,
                        help="Path to test set pickle file (list[dict])")
    parser.add_argument("--label2id_path", required=True,
                        help="Path to label2id mapping (pickle)")
    parser.add_argument("--vocab_path_raw", default="models/vocab_raw.txt")
    parser.add_argument("--vocab_path_size", default="models/vocab_size.txt")
    parser.add_argument("--vocab_path_temporal", default="models/vocab_temporal.txt")
    parser.add_argument("--config_path", default="models/bert/base_config.json")
    parser.add_argument("--config_path_raw", default=None,
                        help="Optional per-modality config for the Raw encoder")
    parser.add_argument("--config_path_size", default=None,
                        help="Optional per-modality config for the Size encoder")

    # ---- Model architecture (must match training) ----
    model_opts(parser)

    parser.add_argument("--num_fusion_layers", type=int, default=6)
    parser.add_argument("--use_itgca", action="store_true",
                        help="Must match the fine-tuned checkpoint")
    parser.add_argument("--itgca_window_size", type=int, default=16)
    parser.add_argument("--ablate_r_stat", action="store_true")
    parser.add_argument("--ablate_g_token", action="store_true")
    parser.add_argument("--ablate_source_bias", action="store_true")
    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

    # ---- Measurement options ----
    parser.add_argument("--output", default=None,
                        help="Output CSV. Default: results/<ckpt basename>_perflow.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only measure first N flows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ranks", default=[], nargs="+", type=int,
                        help="GPU ranks to use. Empty = auto (cuda:0 or CPU)")

    args = parser.parse_args()

    # Pull emb_size, hidden_size, layers_num, heads_num, feedforward_size, dropout, etc.
    if args.config_path:
        args = load_hyperparam(args)
    args = apply_modality_configs(args)

    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)
    args.dropout = getattr(args, "dropout", 0.1)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.load_model_path))[0]
        args.output = os.path.join("results", f"{base}_perflow.csv")

    return args


def setup_device(args):
    if len(args.gpu_ranks) >= 1:
        assert torch.cuda.is_available(), "No available GPUs."
        gpu_id = args.gpu_ranks[0]
        args.device = torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")


def _filter_ablated_keys(state_dict, model_keys, args):
    """Same logic as run_inference.py: drop checkpoint keys that the model
    intentionally omits when ablation flags are set."""
    unexpected = sorted(set(state_dict.keys()) - set(model_keys))
    safe_to_skip = set()
    if args.ablate_g_token:
        safe_to_skip |= {k for k in unexpected
                         if any(f in k for f in [".W_k.", ".W_v.", ".token_gate_bias"])}
    if args.ablate_r_stat:
        safe_to_skip |= {k for k in unexpected
                         if any(f in k for f in [".alpha_modality",
                                                 ".gate_size.stat_scale",
                                                 ".gate_size.stat_shift"])}
    if args.ablate_source_bias:
        safe_to_skip |= {k for k in unexpected
                         if any(f in k for f in [".local_stat_scale", ".local_stat_shift"])}
    return {k: v for k, v in state_dict.items() if k not in safe_to_skip}, safe_to_skip


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_device(args)
    device = args.device

    print(f"[device]   {device}")
    print(f"[paths]    load_model = {args.load_model_path}")
    print(f"           test       = {args.test_path}")
    print(f"           label2id   = {args.label2id_path}")
    print(f"           output     = {args.output}")

    # ---- Vocabularies ----
    vocab_raw = Vocab(); vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab(); vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab(); vocab_temporal.load(args.vocab_path_temporal)
    print(f"[vocab]    raw={len(vocab_raw)}  size={len(vocab_size)}  "
          f"temporal={len(vocab_temporal)}")

    # ---- Labels ----
    label2id = load_dataset(args.label2id_path)
    id2label = {v: k for k, v in label2id.items()}
    args.labels_num = len(label2id)
    print(f"[labels]   {args.labels_num} classes")

    # ---- Test set ----
    test_data = load_dataset(args.test_path)
    n_total = len(test_data)
    n = n_total if args.limit is None else min(args.limit, n_total)
    print(f"[data]     samples={n_total}; measuring {n}")
    test_tensors = pre_tensorize(test_data)

    # ---- Model ----
    print(f"[model]    Stage2Classifier  "
          f"hidden={args.hidden_size} layers={args.layers_num} heads={args.heads_num} "
          f"fusion={args.num_fusion_layers} use_itgca={args.use_itgca}")
    model = Stage2Classifier(args,
                             len(vocab_raw), len(vocab_size), len(vocab_temporal),
                             args.labels_num)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f} M")

    # ---- Checkpoint ----
    print(f"[ckpt]     {args.load_model_path}")
    state_dict = _safe_torch_load(args.load_model_path, "cpu")
    # Unwrap common wrappers
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Strip DataParallel prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model_keys = set(model.state_dict().keys())
    filtered, skipped = _filter_ablated_keys(state_dict, model_keys, args)
    if skipped:
        print(f"  Skipping {len(skipped)} ablated keys.")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  [warn] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys (first 5): {unexpected[:5]}")
    model.to(device).eval()

    # ---- Warmup ----
    raw_w  = test_tensors["raw_src"][0:1].to(device)
    pid_w  = test_tensors["packet_ids"][0:1].to(device)
    dir_w  = test_tensors["directions"][0:1].to(device)
    size_w = test_tensors["size_src"][0:1].to(device)
    iat_w  = test_tensors["iat_src"][0:1].to(device)
    print(f"[warmup]   {args.warmup} iterations")
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(raw_w, pid_w, dir_w, size_w, iat_w)
    _cuda_sync(device)

    is_cuda = str(device).startswith("cuda")
    starter = torch.cuda.Event(enable_timing=True) if is_cuda else None
    ender = torch.cuda.Event(enable_timing=True) if is_cuda else None

    # ---- Per-flow measurement ----
    print("[measure]  per-flow forward (bs=1)")
    rows = []
    with torch.inference_mode():
        for i in range(n):
            raw_src    = test_tensors["raw_src"][i:i+1].to(device, non_blocking=True)
            packet_ids = test_tensors["packet_ids"][i:i+1].to(device, non_blocking=True)
            directions = test_tensors["directions"][i:i+1].to(device, non_blocking=True)
            size_src   = test_tensors["size_src"][i:i+1].to(device, non_blocking=True)
            iat_src    = test_tensors["iat_src"][i:i+1].to(device, non_blocking=True)
            true_label = int(test_tensors["label"][i].item())

            if is_cuda:
                starter.record()
                out = model(raw_src, packet_ids, directions, size_src, iat_src)
                ender.record()
                torch.cuda.synchronize(device)
                lat_ms = float(starter.elapsed_time(ender))
            else:
                t0 = time.perf_counter()
                out = model(raw_src, packet_ids, directions, size_src, iat_src)
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0

            logits = _extract_logits(out)
            probs = F.softmax(logits, dim=-1)
            max_conf = float(probs.max(dim=-1).values.item())
            ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).item())
            pred = int(logits.argmax(dim=-1).item())

            rows.append({
                "flow_id": i,
                "true_label": true_label,
                "pred_label": pred,
                "correct": int(pred == true_label),
                "max_confidence": max_conf,
                "pred_entropy": ent,
                "latency_ms": lat_ms,
            })

            if (i + 1) % 500 == 0 or (i + 1) == n:
                lats = [r["latency_ms"] for r in rows]
                accs = [r["correct"] for r in rows]
                print(f"  [{i+1}/{n}] mean lat={np.mean(lats):.3f} ms  "
                      f"acc={np.mean(accs):.4f}")

    # ---- Write CSV ----
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[write]    {len(rows)} rows -> {args.output}")

    # ---- Aggregate ----
    lats = np.array([r["latency_ms"] for r in rows], dtype=np.float64)
    confs = np.array([r["max_confidence"] for r in rows], dtype=np.float64)
    correct = np.array([r["correct"] for r in rows], dtype=np.float64)

    print("\n=== Aggregate ===")
    print(f"Accuracy:               {correct.mean():.4f}  "
          f"({int(correct.sum())}/{len(rows)})")
    print(f"Latency (ms):           mean={lats.mean():.3f}  std={lats.std():.3f}")
    print(f"  P50 / P95 / P99:      {np.percentile(lats,50):.3f} / "
          f"{np.percentile(lats,95):.3f} / {np.percentile(lats,99):.3f}")
    print(f"Confidence:             mean={confs.mean():.4f}  "
          f"median={np.median(confs):.4f}")

    print("\n=== Easy-flow waste (confidently correct → could be early-exited / cheaper-path) ===")
    print("  threshold | %flows | %total-latency")
    for tau in (0.70, 0.80, 0.90, 0.95, 0.99):
        mask = (confs >= tau) & (correct == 1)
        pct_flows = mask.mean() * 100
        pct_lat = lats[mask].sum() / lats.sum() * 100 if lats.sum() > 0 else 0
        print(f"  >= {tau:.2f}    | {pct_flows:5.1f}% | {pct_lat:5.1f}%")


if __name__ == "__main__":
    main()
