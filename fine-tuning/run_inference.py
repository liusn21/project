"""
Multi-Modal Classifier — Inference & Evaluation (Stage 1 & Stage 2)

Loads a fine-tuned Stage1Classifier or Stage2Classifier .bin checkpoint and
evaluates on a test set. Prints overall metrics, per-class metrics, and
optionally lists misclassified samples with confidence scores for error analysis.

Usage:
    # Stage 1 inference
    python fine-tuning/run_inference.py --stage 1 \
        --load_model_path models/classifier_stage1.bin \
        --test_path datasets/processed/test.pkl \
        --label2id_path datasets/processed/label2id.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --config_path models/bert/base_config.json \
        --modality both

    # Stage 2 inference
    python fine-tuning/run_inference.py --stage 2 \
        --load_model_path models/classifier_stage2.bin \
        --test_path datasets/processed/test.pkl \
        --label2id_path datasets/processed/label2id.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --config_path models/bert/base_config.json

    # Detailed: also show misclassified samples & save to CSV
    python fine-tuning/run_inference.py --stage 2 \
        --load_model_path models/classifier_stage2.bin \
        --test_path datasets/processed/test.pkl \
        --label2id_path datasets/processed/label2id.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --config_path models/bert/base_config.json \
        --detailed --top_k 30 --output_path results/errors.csv
"""

import os
import sys
sys.path.append(os.getcwd())

import argparse
import csv
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report
)

from run_classifier_stage1 import Stage1Classifier
from run_classifier_stage2 import (
    Stage2Classifier, load_dataset, pre_tensorize, batch_loader
)
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam, apply_modality_configs
from uer.utils.seed import set_seed
from uer.opts import model_opts


def get_stage2_classifier(old=False):
    """Return the Stage 2 classifier implementation matching the checkpoint."""
    if not old:
        return Stage2Classifier

    try:
        from run_classifier_stage2_old import Stage2Classifier as OldStage2Classifier
    except ImportError as exc:
        raise ImportError(
            "--old requires fine-tuning/run_classifier_stage2_old.py"
        ) from exc
    return OldStage2Classifier


def build_args():
    parser = argparse.ArgumentParser(
        description="Stage1/Stage2 Classifier Inference & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---- Stage selection ----
    parser.add_argument("--stage", type=int, choices=[1, 2], default=2,
                        help="Classifier stage: 1 (no fusion) or 2 (with fusion)")

    # ---- Path options ----
    parser.add_argument("--load_model_path", type=str, required=True,
                        help="Path to fine-tuned .bin model checkpoint")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test set pickle file")
    parser.add_argument("--label2id_path", type=str, required=True,
                        help="Path to label2id mapping (pickle)")
    parser.add_argument("--vocab_path_raw", type=str, default="./test/vocab_raw.txt",
                        help="Path to raw modality vocabulary")
    parser.add_argument("--vocab_path_size", type=str, default="./test/vocab_size.txt",
                        help="Path to size modality vocabulary")
    parser.add_argument("--vocab_path_temporal", type=str, default="./test/vocab_temporal.txt",
                        help="Path to temporal (IAT) modality vocabulary")
    parser.add_argument("--config_path", default="models/bert/base_config.json",
                        type=str, help="Path to model config JSON")
    parser.add_argument("--config_path_raw", type=str, default=None,
                        help="[Stage 2] Optional per-modality config for the Raw encoder; "
                             "layers_num-only override, shared dims must match --config_path.")
    parser.add_argument("--config_path_size", type=str, default=None,
                        help="[Stage 2] Optional per-modality config for the Size encoder.")

    # ---- Model architecture (must match training) ----
    model_opts(parser)

    # Stage 1 options
    parser.add_argument("--modality", choices=["raw", "size", "both"], default="both",
                        help="[Stage 1] Modality mode (must match training)")

    # Stage 2 options
    parser.add_argument("--num_fusion_layers", type=int, default=6,
                        help="[Stage 2] Number of fusion layers (must match training)")
    parser.add_argument("--use_itgca", action="store_true",
                        help="[Stage 2] Enable ITGCA (must match training config)")
    parser.add_argument("--itgca_window_size", type=int, default=16,
                        help="[Stage 2] ITGCA sliding window size")
    parser.add_argument(
        "--old",
        action="store_true",
        help=(
            "[Stage 2] Use the legacy classifier implementation from "
            "run_classifier_stage2_old.py"
        ),
    )
    parser.add_argument(
        "--use_attn_pooling",
        action="store_true",
        help="[Stage 2 --old] Enable the legacy attention-pooling modules",
    )
    parser.add_argument(
        "--use_scl",
        action="store_true",
        help="[Stage 2 --old] Enable the legacy SCL projection head",
    )

    # ITGCA component-level ablation flags — MUST match the pretrained checkpoint config.
    parser.add_argument("--ablate_r_stat", action="store_true",
                        help="[Stage 2] Disable r_stat prior. Must match the pretrained checkpoint.")
    parser.add_argument("--ablate_g_token", action="store_true",
                        help="[Stage 2] Disable token-level gate. Must match the pretrained checkpoint.")
    parser.add_argument("--ablate_source_bias", action="store_true",
                        help="[Stage 2] Disable source-side attention bias. Must match the pretrained checkpoint.")

    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

    # ---- Inference options ----
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Inference batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int,
                        help="GPU ranks to use. Empty = auto (cuda:0 or CPU)")

    # ---- Output options ----
    parser.add_argument("--detailed", action="store_true",
                        help="Enable detailed mode: list misclassified samples")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of misclassified samples to display (0 = all)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Save detailed results (misclassified samples) to CSV "
                             "(only used with --detailed)")

    args = parser.parse_args()
    legacy_cli_options = (args.old, args.use_attn_pooling, args.use_scl)

    # Load hyperparameters from config JSON
    if args.config_path:
        args = load_hyperparam(args)
    args = apply_modality_configs(args)
    args.old, args.use_attn_pooling, args.use_scl = legacy_cli_options

    if args.old and args.stage != 2:
        parser.error("--old is only valid with --stage 2")
    if (args.use_attn_pooling or args.use_scl) and not args.old:
        parser.error("--use_attn_pooling and --use_scl require --old")

    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)
    args.dropout = getattr(args, 'dropout', 0.1)
    return args


def setup_device(args):
    """Setup GPU/CPU device."""
    ranks_num = len(args.gpu_ranks)
    if ranks_num >= 1:
        assert torch.cuda.is_available(), "No available GPUs."
        gpu_id = args.gpu_ranks[0]
        args.device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id}")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        print("Using default GPU: cuda:0")
    else:
        args.device = torch.device("cpu")
        print("Using CPU mode")


def evaluate_detailed(args, model, eval_tensors, id2label):
    """
    Run inference and collect per-sample predictions with confidence.

    Returns:
        results: dict with keys:
            y_true, y_pred: lists of integer labels
            confidences: list of float (softmax prob of predicted class)
            confusion: [num_classes, num_classes] tensor
    """
    model.eval()

    y_true = []
    y_pred = []
    confidences = []
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    with torch.no_grad():
        for raw_src, packet_ids, directions, size_src, iat_src, tgt in \
                batch_loader(args.batch_size, eval_tensors):
            raw_src = raw_src.to(args.device, non_blocking=True)
            packet_ids = packet_ids.to(args.device, non_blocking=True)
            directions = directions.to(args.device, non_blocking=True)
            size_src = size_src.to(args.device, non_blocking=True)
            iat_src = iat_src.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)

            output = model(raw_src, packet_ids, directions, size_src, iat_src)
            # Stage 1 returns (None, logits); Stage 2 returns logits or (logits, scl_feat)
            if isinstance(output, tuple):
                logits = output[1] if output[0] is None else output[0]
            else:
                logits = output

            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

            for p, g, c in zip(pred.cpu().tolist(), tgt.cpu().tolist(),
                               conf.cpu().tolist()):
                confusion[g, p] += 1
                y_true.append(g)
                y_pred.append(p)
                confidences.append(c)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'confidences': confidences,
        'confusion': confusion,
    }


def print_overall_metrics(y_true, y_pred):
    """Print overall classification metrics."""
    correct = sum(1 for p, g in zip(y_pred, y_true) if p == g)
    total = len(y_true)
    acc = correct / total

    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    micro_p = precision_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_p = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    micro_r = recall_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 55)
    print("  Overall Metrics")
    print("=" * 55)
    print(f"  Accuracy:  {acc:.4f} ({correct}/{total})")
    print(f"  Precision  - Macro: {macro_p:.4f}  Micro: {micro_p:.4f}  Weighted: {weighted_p:.4f}")
    print(f"  Recall     - Macro: {macro_r:.4f}  Micro: {micro_r:.4f}  Weighted: {weighted_r:.4f}")
    print(f"  F1         - Macro: {macro_f1:.4f}  Micro: {micro_f1:.4f}  Weighted: {weighted_f1:.4f}")


def print_per_class_metrics(confusion, id2label):
    """Print per-class precision, recall, F1, and support, sorted by F1."""
    num_classes = confusion.shape[0]
    eps = 1e-9

    rows = []
    for i in range(num_classes):
        tp = confusion[i, i].item()
        col_sum = confusion[:, i].sum().item()  # predicted as i
        row_sum = confusion[i, :].sum().item()  # actual i

        if row_sum == 0:
            continue  # skip classes not present in test set

        p = tp / (col_sum + eps)
        r = tp / (row_sum + eps)
        f1 = 2 * p * r / (p + r + eps)
        label_name = id2label.get(i, str(i))
        rows.append((label_name, p, r, f1, int(row_sum)))

    # Sort by F1 ascending so worst classes are at top
    rows.sort(key=lambda x: x[3])

    print("\n" + "=" * 75)
    print("  Per-Class Metrics (sorted by F1, ascending)")
    print("=" * 75)
    print(f"  {'Class':<30s} {'Precision':>9s} {'Recall':>9s} {'F1':>9s} {'Support':>9s}")
    print("  " + "-" * 68)
    for label_name, p, r, f1, support in rows:
        name_display = label_name[:28] if len(label_name) > 28 else label_name
        print(f"  {name_display:<30s} {p:>9.4f} {r:>9.4f} {f1:>9.4f} {support:>9d}")

    print("  " + "-" * 68)
    print(f"  Total classes with samples: {len(rows)}")


def print_misclassified_samples(y_true, y_pred, confidences, id2label, top_k):
    """Print misclassified samples sorted by confidence (low → high)."""
    errors = []
    for idx, (gt, pred, conf) in enumerate(zip(y_true, y_pred, confidences)):
        if gt != pred:
            errors.append((idx, gt, pred, conf))

    if not errors:
        print("\n  No misclassified samples!")
        return errors

    # Sort by confidence ascending — least confident errors first
    errors.sort(key=lambda x: x[3])

    total_errors = len(errors)
    display = errors if (top_k <= 0 or top_k >= total_errors) else errors[:top_k]

    print(f"\n{'=' * 85}")
    print(f"  Misclassified Samples ({len(display)}/{total_errors} shown, sorted by confidence)")
    print(f"{'=' * 85}")
    print(f"  {'#':<5s} {'Index':<8s} {'True Label':<25s} {'Predicted Label':<25s} {'Conf':>8s}")
    print("  " + "-" * 73)

    for rank, (idx, gt, pred, conf) in enumerate(display, 1):
        gt_name = id2label.get(gt, str(gt))
        pred_name = id2label.get(pred, str(pred))
        gt_display = gt_name[:23] if len(gt_name) > 23 else gt_name
        pred_display = pred_name[:23] if len(pred_name) > 23 else pred_name
        print(f"  {rank:<5d} {idx:<8d} {gt_display:<25s} {pred_display:<25s} {conf:>8.4f}")

    if top_k > 0 and top_k < total_errors:
        print(f"\n  ... {total_errors - top_k} more errors not shown. "
              f"Use --top_k 0 to show all, or --output_path to save to CSV.")

    return errors


def save_errors_to_csv(errors, id2label, output_path):
    """Save all misclassified samples to a CSV file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index', 'true_label_id', 'true_label_name',
                         'pred_label_id', 'pred_label_name', 'confidence'])
        for idx, gt, pred, conf in errors:
            writer.writerow([
                idx,
                gt, id2label.get(gt, str(gt)),
                pred, id2label.get(pred, str(pred)),
                f"{conf:.6f}"
            ])

    print(f"\n  Saved {len(errors)} misclassified samples to: {output_path}")


def main():
    args = build_args()
    set_seed(args.seed)

    print(f"Stage: {args.stage}")

    # ---- Load vocabularies ----
    print("Loading vocabularies...")
    vocab_raw = Vocab()
    vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab()
    vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab()
    vocab_temporal.load(args.vocab_path_temporal)
    print(f"  Raw: {len(vocab_raw)}, Size: {len(vocab_size)}, Temporal: {len(vocab_temporal)}")

    # ---- Load label mapping ----
    print("Loading label mapping...")
    label2id = load_dataset(args.label2id_path)
    id2label = {v: k for k, v in label2id.items()}
    args.labels_num = len(label2id)
    print(f"  Number of classes: {args.labels_num}")

    # ---- Load test data ----
    print("Loading test data...")
    test_data = load_dataset(args.test_path)
    print(f"  Test samples: {len(test_data)}")
    test_tensors = pre_tensorize(test_data)

    # ---- Build model ----
    print("Building model...")
    if args.stage == 1:
        print(f"  Modality: {args.modality}")
        model = Stage1Classifier(
            args, len(vocab_raw), len(vocab_size), len(vocab_temporal), args.labels_num
        )
    else:
        stage2_classifier = get_stage2_classifier(args.old)
        if args.old:
            print(
                "  Legacy classifier: "
                f"attention_pooling={args.use_attn_pooling}, "
                f"SCL={args.use_scl}"
            )
        model = stage2_classifier(
            args, len(vocab_raw), len(vocab_size), len(vocab_temporal), args.labels_num
        )

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {args.load_model_path}")
    state_dict = torch.load(args.load_model_path, map_location='cpu')

    # Compare model keys vs checkpoint keys to detect mismatches.
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    common = model_keys & ckpt_keys
    unexpected = sorted(ckpt_keys - model_keys)   # in checkpoint, not in model
    missing = sorted(model_keys - ckpt_keys)       # in model, not in checkpoint

    print(f"  Model keys: {len(model_keys)}, Checkpoint keys: {len(ckpt_keys)}, "
          f"Common: {len(common)}")

    # Classify unexpected keys: which are safe to skip?
    safe_to_skip = set()

    # Inference-time ablation: checkpoint has components the model intentionally omits.
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

    if safe_to_skip:
        print(f"  Skipping {len(safe_to_skip)} training-only / ablated keys:")
        for k in sorted(safe_to_skip)[:8]:
            print(f"    - {k}")
        if len(safe_to_skip) > 8:
            print(f"    ... +{len(safe_to_skip)-8} more")

    # Diagnose unresolvable mismatches with actionable hints
    has_error = False

    if unsafe_unexpected:
        has_error = True
        print(f"\n  ERROR: {len(unsafe_unexpected)} unexpected keys in checkpoint "
              f"(no matching model parameter):")
        for k in unsafe_unexpected:
            print(f"    - {k}")
        if any('gate_raw' in k or 'gate_size' in k or 'local_stat_' in k
               for k in unsafe_unexpected):
            print(f"\n  HINT: Checkpoint contains ITGCA gate weights. "
                  f"Add --use_itgca to match.")

    if missing:
        has_error = True
        print(f"\n  ERROR: {len(missing)} missing keys (model expects, checkpoint lacks).")
        print(f"  These would be RANDOMLY INITIALIZED — catastrophic for inference!")
        for k in missing:
            print(f"    - {k}")
        if any('gate_raw' in k or 'gate_size' in k or 'local_stat_' in k
               for k in missing):
            print(f"\n  HINT: Model defines ITGCA parameters not in checkpoint.")
            print(f"        If checkpoint was trained WITHOUT ITGCA → remove --use_itgca.")
            print(f"        If checkpoint was trained WITH ablation → add matching "
                  f"--ablate_r_stat / --ablate_g_token / --ablate_source_bias.")
        if any('classifier' in k for k in missing):
            print(f"\n  HINT: Classifier shape mismatch. Check --labels_num.")

    if has_error:
        print(f"\n  Current inference flags:")
        print(f"    --use_itgca={getattr(args, 'use_itgca', False)}")
        print(f"    --ablate_r_stat={getattr(args, 'ablate_r_stat', False)}")
        print(f"    --ablate_g_token={getattr(args, 'ablate_g_token', False)}")
        print(f"    --ablate_source_bias={getattr(args, 'ablate_source_bias', False)}")
        print(f"    --num_fusion_layers={getattr(args, 'num_fusion_layers', 6)}")
        print(f"    --old={getattr(args, 'old', False)}")
        print(f"    --use_attn_pooling={getattr(args, 'use_attn_pooling', False)}")
        print(f"    --use_scl={getattr(args, 'use_scl', False)}")
        sys.exit(1)

    # Filter and load with strict=True
    filtered = {k: v for k, v in state_dict.items() if k not in safe_to_skip}
    model.load_state_dict(filtered, strict=True)
    print(f"  Loaded successfully (strict=True, skipped {len(safe_to_skip)}).")

    # ---- Setup device ----
    setup_device(args)
    model = model.to(args.device)

    # ---- Run inference ----
    print("\nRunning inference...")
    results = evaluate_detailed(args, model, test_tensors, id2label)

    # ---- Print metrics ----
    print_overall_metrics(results['y_true'], results['y_pred'])
    print_per_class_metrics(results['confusion'], id2label)

    # ---- Detailed mode ----
    if args.detailed:
        errors = print_misclassified_samples(
            results['y_true'], results['y_pred'], results['confidences'],
            id2label, args.top_k
        )
        if args.output_path and errors:
            save_errors_to_csv(errors, id2label, args.output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
