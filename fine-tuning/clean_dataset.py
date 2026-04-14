"""
Clean dataset by removing high-confidence misclassified samples (likely mislabeled).

Three mutually exclusive modes:
    --top_n N          Remove top N errors (by confidence, high -> low)
    --top_pct P        Remove top P% of errors (by confidence, high -> low)
    --conf_threshold T Remove all errors with confidence > T

Usage:
    # Remove top 50 highest-confidence errors
    python fine-tuning/clean_dataset.py \
        --input datasets/processed/test.pkl \
        --errors_csv results/errors.csv \
        --output datasets/processed/test_cleaned.pkl \
        --top_n 50

    # Remove top 30% highest-confidence errors
    python fine-tuning/clean_dataset.py \
        --input datasets/processed/test.pkl \
        --errors_csv results/errors.csv \
        --output datasets/processed/test_cleaned.pkl \
        --top_pct 0.3

    # Remove all errors with confidence > 0.9
    python fine-tuning/clean_dataset.py \
        --input datasets/processed/test.pkl \
        --errors_csv results/errors.csv \
        --output datasets/processed/test_cleaned.pkl \
        --conf_threshold 0.9
"""

import argparse
import csv
import pickle
import random
import sys


def load_errors(errors_csv_path):
    """Load misclassified samples from inference CSV.

    Expected columns: sample_index, true_label_id, true_label_name,
                      pred_label_id, pred_label_name, confidence
    """
    errors = []
    with open(errors_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            errors.append({
                'sample_index': int(row['sample_index']),
                'true_label': row['true_label_name'],
                'pred_label': row['pred_label_name'],
                'confidence': float(row['confidence']),
            })
    return errors


def select_indices_to_remove(errors, args):
    """Determine which sample indices to remove based on the selected mode.

    All modes sort by confidence descending (high confidence = likely mislabeled).
    """
    # Sort by confidence high -> low
    errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)

    if args.top_n is not None:
        n = min(args.top_n, len(errors_sorted))
        selected = errors_sorted[:n]
        print(f"Mode: top_n = {args.top_n}")
        print(f"  Selecting top {n} errors by confidence")

    elif args.top_pct is not None:
        n = max(1, int(len(errors_sorted) * args.top_pct))
        n = min(n, len(errors_sorted))
        selected = errors_sorted[:n]
        print(f"Mode: top_pct = {args.top_pct:.1%}")
        print(f"  Selecting top {n}/{len(errors_sorted)} errors "
              f"({n / len(errors_sorted):.1%})")

    elif args.conf_threshold is not None:
        selected = [e for e in errors_sorted if e['confidence'] > args.conf_threshold]
        print(f"Mode: conf_threshold > {args.conf_threshold}")
        print(f"  Selecting {len(selected)}/{len(errors_sorted)} errors "
              f"with confidence > {args.conf_threshold}")

    else:
        print("ERROR: Must specify one of --top_n, --top_pct, --conf_threshold",
              file=sys.stderr)
        sys.exit(1)

    if not selected:
        print("  No samples to remove.")
        return set()

    # Print confidence range of selected samples
    confs = [e['confidence'] for e in selected]
    print(f"  Confidence range: [{min(confs):.4f}, {max(confs):.4f}]")

    return {e['sample_index'] for e in selected}


def clean_pkl(input_path, output_path, remove_indices):
    """Remove specified samples from pkl file by index."""
    with open(input_path, 'rb') as f:
        dataset = pickle.load(f)

    total = len(dataset)

    # Validate indices
    out_of_range = {i for i in remove_indices if i >= total}
    if out_of_range:
        print(f"  WARNING: {len(out_of_range)} indices out of range "
              f"(max={total - 1}), skipped: {sorted(out_of_range)[:5]}...")

    cleaned = [sample for i, sample in enumerate(dataset) if i not in remove_indices]

    with open(output_path, 'wb') as f:
        pickle.dump(cleaned, f)

    removed = total - len(cleaned)
    print(f"\nResult:")
    print(f"  Input:   {total} samples")
    print(f"  Removed: {removed} samples")
    print(f"  Output:  {len(cleaned)} samples")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove high-confidence misclassified samples from pkl dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input pkl file (e.g. test.pkl)")
    parser.add_argument("--errors_csv", type=str, required=True,
                        help="Misclassified samples CSV from run_inference.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output cleaned pkl file")

    # Three mutually exclusive modes
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--top_n", type=int, default=None,
                      help="Remove top N errors (highest confidence first)")
    mode.add_argument("--top_pct", type=float, default=None,
                      help="Remove top X%% of errors, e.g. 0.3 = 30%% "
                           "(highest confidence first)")
    mode.add_argument("--conf_threshold", type=float, default=None,
                      help="Remove all errors with confidence > this value")

    # Correct-sample removal mode
    parser.add_argument("--remove_correct", action="store_true",
                        help="Remove ALL misclassified samples + randomly remove a portion "
                             "of correct samples. Only supports --top_n / --top_pct.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --remove_correct sampling")

    args = parser.parse_args()

    # Validate
    if args.remove_correct and args.conf_threshold is not None:
        print("ERROR: --conf_threshold is not supported with --remove_correct "
              "(no confidence info for correct samples). Use --top_n or --top_pct.",
              file=sys.stderr)
        sys.exit(1)
    if args.top_pct is not None and not (0.0 < args.top_pct <= 1.0):
        print("ERROR: --top_pct must be in (0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.conf_threshold is not None and not (0.0 < args.conf_threshold < 1.0):
        print("ERROR: --conf_threshold must be in (0, 1)", file=sys.stderr)
        sys.exit(1)

    # Load errors
    print(f"Loading errors from: {args.errors_csv}")
    errors = load_errors(args.errors_csv)
    print(f"  Total misclassified samples: {len(errors)}")

    if args.remove_correct:
        # Mode: keep ALL errors + randomly keep a portion of correct samples
        random.seed(args.seed)
        error_indices = {e['sample_index'] for e in errors}

        with open(args.input, 'rb') as f:
            total = len(pickle.load(f))
        correct_indices = sorted(set(range(total)) - error_indices)
        print(f"  Total samples: {total}, Correct: {len(correct_indices)}")

        if args.top_n is not None:
            n_keep = min(args.top_n, len(correct_indices))
            print(f"Mode: remove_correct + top_n = {args.top_n}")
        else:  # top_pct
            n_keep = max(1, int(len(correct_indices) * args.top_pct))
            n_keep = min(n_keep, len(correct_indices))
            print(f"Mode: remove_correct + top_pct = {args.top_pct:.1%}")

        kept_correct = set(random.sample(correct_indices, n_keep))
        keep_indices = error_indices | kept_correct
        remove_indices = set(range(total)) - keep_indices
        print(f"  Correct samples kept:    {n_keep}/{len(correct_indices)}")
        print(f"  Error samples kept:      {len(error_indices)} (all)")
        print(f"  Total kept:              {len(keep_indices)}")
        print(f"  Total removed:           {len(remove_indices)}")

        print(f"\nCleaning: {args.input}")
        clean_pkl(args.input, args.output, remove_indices)
        return

    if not errors:
        print("No errors found. Nothing to remove.")
        return

    # Select which to remove
    remove_indices = select_indices_to_remove(errors, args)

    if not remove_indices:
        return

    # Clean pkl
    print(f"\nCleaning: {args.input}")
    clean_pkl(args.input, args.output, remove_indices)


if __name__ == "__main__":
    main()
