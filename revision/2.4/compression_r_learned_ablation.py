#!/usr/bin/env python3
"""Run only the missing no-r_learned comparison on existing compression flows.

The compression audit and all full-model/baseline predictions are reused from
an existing ``flow_results.csv``.  Only positive-exposure rows are tokenized,
and only a separately fine-tuned ``--ablate_r_learned`` classifier is run.

The main result is the per-bin Macro-F1 difference required by the paper:

    Macro-F1(full ITGCA) - Macro-F1(no r_learned)

The original result file is never modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compression_checkpoint_inference as common  # noqa: E402


SCRIPT_VERSION = "1.2.0"
DEFAULT_BOOTSTRAP_REPETITIONS = 2000

BIN_DEFINITIONS = (
    ("e_gt_0_le_0_25", "0 < e_i <= 0.25", lambda value: 0.0 < value <= 0.25),
    (
        "e_gt_0_25_le_0_50",
        "0.25 < e_i <= 0.50",
        lambda value: 0.25 < value <= 0.50,
    ),
    ("e_gt_0_50_le_1", "0.50 < e_i <= 1.00", lambda value: 0.50 < value <= 1.0),
)

FLOW_FIELDS = [
    "dataset",
    "evaluation_scope",
    "label",
    "relative_pcap",
    "e_i",
    "e_bin",
    "existing_itgca_pred",
    "existing_itgca_correct",
    "no_r_learned_pred",
    "no_r_learned_correct",
    "full_correct_no_r_learned_wrong",
    "no_r_learned_correct_full_wrong",
    "inference_status",
    "error",
]

SUMMARY_FIELDS = [
    "dataset",
    "evaluation_scope",
    "group",
    "group_display",
    "flow_count",
    "evaluated_flows",
    "failed_flows",
    "labels_present",
    "existing_itgca_macro_f1",
    "no_r_learned_macro_f1",
    "delta_macro_f1_pp_full_minus_no_r_learned",
    "delta_macro_f1_pp_ci_lower",
    "delta_macro_f1_pp_ci_upper",
    "full_correct_no_r_learned_wrong_count",
    "no_r_learned_correct_full_wrong_count",
    "bootstrap_repetitions",
]


def _finite_float(value: Any, field: str, row_number: int, path: Path) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: row {row_number} has invalid {field}={value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"{path}: row {row_number} has non-finite {field}={value!r}"
        )
    return parsed


def _binary(value: Any, field: str, row_number: int, path: Path) -> bool:
    parsed = _finite_float(value, field, row_number, path)
    if parsed not in (0.0, 1.0):
        raise ValueError(
            f"{path}: row {row_number} has non-binary {field}={value!r}"
        )
    return bool(parsed)


def _canonical_bin(e_i: float) -> str:
    for name, _, predicate in BIN_DEFINITIONS:
        if predicate(e_i):
            return name
    raise ValueError(f"positive compression exposure is outside (0, 1]: {e_i}")


def _resolve_pcap(dataset_dir: Path, relative_pcap: str) -> Path:
    root = dataset_dir.resolve()
    path = (root / relative_pcap).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"relative_pcap escapes --dataset-dir: {relative_pcap!r}"
        ) from exc
    return path


def _read_existing_results(
    path: Path,
    dataset_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read only successful positive-exposure rows from the completed run."""

    common._maximize_csv_field_size_limit()
    records: List[Dict[str, Any]] = []
    datasets = set()
    scopes = set()
    seen_flows = set()
    successful_source_rows = 0
    skipped_zero_exposure = 0
    skipped_failed_source = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"existing result CSV has no header: {path}")
        required = {
            "label",
            "relative_pcap",
            "e_i",
            "itgca_pred",
            "itgca_correct",
            "inference_status",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "existing result CSV is missing required columns: "
                + ", ".join(missing)
            )

        for row_number, row in enumerate(reader, start=2):
            if str(row.get("inference_status", "")).strip().lower() != "ok":
                skipped_failed_source += 1
                continue
            successful_source_rows += 1

            e_i = _finite_float(row.get("e_i"), "e_i", row_number, path)
            if not 0.0 <= e_i <= 1.0:
                raise ValueError(f"{path}: row {row_number} has e_i outside [0, 1]")
            if e_i == 0.0:
                skipped_zero_exposure += 1
                continue

            label = str(row.get("label", "")).strip()
            relative_pcap = str(row.get("relative_pcap", "")).strip()
            full_prediction = str(row.get("itgca_pred", "")).strip()
            if not label or not relative_pcap or not full_prediction:
                raise ValueError(
                    f"{path}: row {row_number} has an empty label, "
                    "relative_pcap, or itgca_pred"
                )

            identity = (label, relative_pcap)
            if identity in seen_flows:
                raise ValueError(
                    f"{path}: duplicate positive-exposure flow {identity!r}"
                )
            seen_flows.add(identity)

            full_correct = _binary(
                row.get("itgca_correct"), "itgca_correct", row_number, path
            )
            if full_correct != (full_prediction == label):
                raise ValueError(
                    f"{path}: row {row_number} has inconsistent itgca_pred and "
                    "itgca_correct"
                )

            dataset = str(row.get("dataset", "")).strip()
            scope = str(row.get("evaluation_scope", "")).strip()
            if dataset:
                datasets.add(dataset)
            if scope:
                scopes.add(scope)

            records.append(
                {
                    "dataset": dataset,
                    "evaluation_scope": scope,
                    "label": label,
                    "relative_pcap": relative_pcap,
                    "pcap_path": _resolve_pcap(dataset_dir, relative_pcap),
                    "e_i": e_i,
                    "e_bin": _canonical_bin(e_i),
                    "existing_itgca_pred": full_prediction,
                    "existing_itgca_correct": full_correct,
                    "inference_status": "pending",
                    "error": "",
                }
            )

    if not records:
        raise ValueError(
            "existing result CSV contains no successful flow with e_i > 0"
        )
    if len(datasets) > 1:
        raise ValueError(f"existing result CSV contains multiple datasets: {datasets}")
    if len(scopes) > 1:
        raise ValueError(
            f"existing result CSV contains multiple evaluation scopes: {scopes}"
        )

    metadata = {
        "dataset_from_source": next(iter(datasets), path.parent.name),
        "evaluation_scope": next(iter(scopes), ""),
        "successful_source_rows": successful_source_rows,
        "selected_positive_exposure_rows": len(records),
        "skipped_zero_exposure_rows": skipped_zero_exposure,
        "skipped_failed_source_rows": skipped_failed_source,
    }
    return records, metadata


def _macro_f1(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> float:
    """Macro-F1 over the ground-truth labels present in this exposure bin."""

    if len(true_labels) != len(predicted_labels) or not true_labels:
        raise ValueError("Macro-F1 received empty or misaligned predictions")
    labels = sorted(set(true_labels))
    scores = []
    for label in labels:
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_positive = sum(
            truth != label and prediction == label
            for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_negative = sum(
            truth == label and prediction != label
            for truth, prediction in zip(true_labels, predicted_labels)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "labels_present": 0,
            "existing_itgca_macro_f1": None,
            "no_r_learned_macro_f1": None,
            "delta_macro_f1_pp_full_minus_no_r_learned": None,
            "full_correct_no_r_learned_wrong_count": 0,
            "no_r_learned_correct_full_wrong_count": 0,
        }

    true_labels = [str(row["label"]) for row in rows]
    full_predictions = [str(row["existing_itgca_pred"]) for row in rows]
    no_r_predictions = [str(row["no_r_learned_pred"]) for row in rows]
    full_macro_f1 = _macro_f1(true_labels, full_predictions)
    no_r_macro_f1 = _macro_f1(true_labels, no_r_predictions)
    return {
        "labels_present": len(set(true_labels)),
        "existing_itgca_macro_f1": full_macro_f1,
        "no_r_learned_macro_f1": no_r_macro_f1,
        "delta_macro_f1_pp_full_minus_no_r_learned": 100.0
        * (full_macro_f1 - no_r_macro_f1),
        "full_correct_no_r_learned_wrong_count": sum(
            bool(row["existing_itgca_correct"])
            and not bool(row["no_r_learned_correct"])
            for row in rows
        ),
        "no_r_learned_correct_full_wrong_count": sum(
            bool(row["no_r_learned_correct"])
            and not bool(row["existing_itgca_correct"])
            for row in rows
        ),
    }


def _stratified_sample(
    rows: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> List[Mapping[str, Any]]:
    by_label: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)

    sampled: List[Mapping[str, Any]] = []
    for label in sorted(by_label):
        label_rows = by_label[label]
        indices = rng.integers(0, len(label_rows), size=len(label_rows))
        sampled.extend(label_rows[int(index)] for index in indices)
    return sampled


def _bootstrap_delta_interval(
    rows: Sequence[Mapping[str, Any]],
    repetitions: int,
    rng: np.random.Generator,
) -> Tuple[Optional[float], Optional[float]]:
    if not rows:
        return None, None
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        estimates[index] = float(
            _metrics(_stratified_sample(rows, rng))[
                "delta_macro_f1_pp_full_minus_no_r_learned"
            ]
        )
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def _paired_rows(
    records: Sequence[Mapping[str, Any]],
    store: common.FeatureStore,
    predictions: np.ndarray,
    id2label: Mapping[int, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if not bool(store.success[index]):
            rows.append(
                {
                    "dataset": record["dataset"],
                    "evaluation_scope": record["evaluation_scope"],
                    "label": record["label"],
                    "relative_pcap": record["relative_pcap"],
                    "e_i": record["e_i"],
                    "e_bin": record["e_bin"],
                    "existing_itgca_pred": record["existing_itgca_pred"],
                    "existing_itgca_correct": int(
                        record["existing_itgca_correct"]
                    ),
                    "inference_status": record["inference_status"],
                    "error": record["error"],
                }
            )
            continue

        prediction_id = int(predictions[index])
        no_r_prediction = id2label[prediction_id]
        no_r_correct = no_r_prediction == record["label"]
        full_correct = bool(record["existing_itgca_correct"])
        rows.append(
            {
                "dataset": record["dataset"],
                "evaluation_scope": record["evaluation_scope"],
                "label": record["label"],
                "relative_pcap": record["relative_pcap"],
                "e_i": record["e_i"],
                "e_bin": record["e_bin"],
                "existing_itgca_pred": record["existing_itgca_pred"],
                "existing_itgca_correct": int(full_correct),
                "no_r_learned_pred": no_r_prediction,
                "no_r_learned_correct": int(no_r_correct),
                "full_correct_no_r_learned_wrong": int(
                    full_correct and not no_r_correct
                ),
                "no_r_learned_correct_full_wrong": int(
                    no_r_correct and not full_correct
                ),
                "inference_status": "ok",
                "error": "",
            }
        )
    return rows


def _summary_rows(
    paired: Sequence[Mapping[str, Any]],
    dataset: str,
    scope: str,
    repetitions: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for group, display, predicate in BIN_DEFINITIONS:
        selected = [row for row in paired if predicate(float(row["e_i"]))]
        evaluated = [row for row in selected if row["inference_status"] == "ok"]
        metrics = _metrics(evaluated)
        lower, upper = _bootstrap_delta_interval(evaluated, repetitions, rng)
        output: Dict[str, Any] = {
            "dataset": dataset,
            "evaluation_scope": scope,
            "group": group,
            "group_display": display,
            "flow_count": len(selected),
            "evaluated_flows": len(evaluated),
            "failed_flows": len(selected) - len(evaluated),
            "delta_macro_f1_pp_ci_lower": lower,
            "delta_macro_f1_pp_ci_upper": upper,
            "bootstrap_repetitions": repetitions,
        }
        output.update(metrics)
        rows.append(output)

    if sum(int(row["flow_count"]) for row in rows) != len(paired):
        raise RuntimeError("the three positive-exposure bins do not cover all rows")
    return rows


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only a fine-tuned no-r_learned classifier and calculate the "
            "missing paired Macro-F1 column from an existing flow_results.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("existing_flow_results", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--no-r-learned-checkpoint", type=Path, required=True)
    parser.add_argument("--label2id-path", type=Path, required=True)
    parser.add_argument(
        "--raw-config",
        type=Path,
        default=common.PROJECT_ROOT / "models/bert/base_config.json",
    )
    parser.add_argument(
        "--behavior-config",
        type=Path,
        default=common.PROJECT_ROOT / "models/bert/base_behavior_config.json",
    )
    parser.add_argument("--itgca-window-size", type=int, default=16)
    parser.add_argument("--seq-length-raw", type=int, default=512)
    parser.add_argument("--seq-length-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=common.DEFAULT_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=common.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bootstrap-repetitions",
        type=_positive_integer,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--use-attn-pooling", action="store_true")
    parser.add_argument("--use-scl", action="store_true")
    parser.add_argument(
        "--vocab-path-raw",
        type=Path,
        default=common.PROJECT_ROOT / "models/bert/vocab_raw.txt",
    )
    parser.add_argument(
        "--vocab-path-size",
        type=Path,
        default=common.PROJECT_ROOT / "models/bert/vocab_size.txt",
    )
    parser.add_argument(
        "--vocab-path-temporal",
        type=Path,
        default=common.PROJECT_ROOT / "models/bert/vocab_temporal.txt",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    args = parser.parse_args(argv)
    if (args.use_attn_pooling or args.use_scl) and not args.old:
        parser.error("--use-attn-pooling and --use-scl require --old")
    return args


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "existing_flow_results",
        "dataset_dir",
        "no_r_learned_checkpoint",
        "label2id_path",
        "raw_config",
        "behavior_config",
        "vocab_path_raw",
        "vocab_path_size",
        "vocab_path_temporal",
    ):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())


def _validate_paths(args: argparse.Namespace) -> None:
    required_files = {
        "existing flow results": args.existing_flow_results,
        "fine-tuned no-r_learned checkpoint": args.no_r_learned_checkpoint,
        "label2id": args.label2id_path,
        "raw config": args.raw_config,
        "behavior config": args.behavior_config,
        "raw vocabulary": args.vocab_path_raw,
        "size vocabulary": args.vocab_path_size,
        "temporal vocabulary": args.vocab_path_temporal,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.is_file()
    ]
    if missing:
        raise ValueError("required files not found:\n  " + "\n  ".join(missing))
    if not args.dataset_dir.is_dir():
        raise ValueError(f"dataset directory not found: {args.dataset_dir}")
    for name in (
        "itgca_window_size",
        "seq_length_raw",
        "seq_length_size",
        "batch_size",
        "workers",
        "chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _validate_labels(
    records: Sequence[Mapping[str, Any]],
    label2id: Mapping[str, int],
) -> None:
    unknown_true = sorted({row["label"] for row in records} - set(label2id))
    unknown_full_predictions = sorted(
        {row["existing_itgca_pred"] for row in records} - set(label2id)
    )
    if unknown_true:
        raise ValueError(
            "source rows contain labels absent from label2id: "
            + ", ".join(map(repr, unknown_true))
        )
    if unknown_full_predictions:
        raise ValueError(
            "source ITGCA predictions contain labels absent from label2id: "
            + ", ".join(map(repr, unknown_full_predictions))
        )


def _build_no_r_learned_model(
    checkpoint_path: Path,
    raw_config_path: Path,
    behavior_config_path: Path,
    itgca_window_size: int,
    vocab_lengths: Tuple[int, int, int],
    labels_num: int,
    seq_length_raw: int,
    seq_length_size: int,
    old: bool,
    use_attn_pooling: bool,
    use_scl: bool,
) -> Any:
    """Build the strict ablated classifier without modifying the old runner."""

    state = common._safe_torch_load(checkpoint_path)
    if not any(str(key).startswith("classifier.") for key in state):
        raise ValueError(
            "--no-r-learned-checkpoint has no classifier head. Pass the "
            "fine-tuned --ablate_r_learned checkpoint, not the Stage 2 "
            "pre-training checkpoint."
        )

    fusion_depth = common._infer_stack_depth(state, "fusion.fusion_layers.")
    if fusion_depth <= 0:
        raise ValueError(
            "no-r_learned checkpoint has no fusion.fusion_layers parameters"
        )

    forbidden_components = {
        "r_learned bilinear gate": any(
            ".gate_raw.bilinear_" in key or ".gate_size.bilinear_" in key
            for key in state
        ),
        "r_stat/r_learned mixing coefficient": any(
            ".alpha_modality" in key for key in state
        ),
    }
    present_forbidden = [
        name for name, present in forbidden_components.items() if present
    ]
    if present_forbidden:
        raise ValueError(
            "--no-r-learned-checkpoint still contains "
            + ", ".join(present_forbidden)
            + ". Fine-tune a checkpoint built with --ablate_r_learned."
        )

    required_components = {
        "r_stat calibration path": (
            any(".gate_size.stat_scale" in key for key in state)
            and any(".gate_size.stat_shift" in key for key in state)
        ),
        "Raw<-Size token gate": (
            any(".gate_raw.W_k.weight" in key for key in state)
            and any(".gate_raw.W_v.weight" in key for key in state)
        ),
        "Size<-Raw token gate": (
            any(".gate_size.W_k.weight" in key for key in state)
            and any(".gate_size.W_v.weight" in key for key in state)
        ),
        "source-side entropy bias": (
            any(".local_stat_scale" in key for key in state)
            and any(".local_stat_shift" in key for key in state)
        ),
    }
    missing_components = [
        name for name, present in required_components.items() if not present
    ]
    if missing_components:
        raise ValueError(
            "checkpoint is not the isolated no-r_learned ITGCA ablation; missing: "
            + ", ".join(missing_components)
        )

    args = common._new_model_args(
        raw_config_path, seq_length_raw, seq_length_size
    )
    args.config_path_raw = str(raw_config_path)
    args.config_path_size = str(behavior_config_path)
    args = common.apply_modality_configs(args)
    args.num_fusion_layers = fusion_depth
    args.use_itgca = True
    args.use_mlp_gate = False
    args.itgca_window_size = itgca_window_size
    args.ablate_r_stat = False
    args.ablate_r_learned = True
    args.ablate_g_token = False
    args.ablate_source_bias = False
    args.use_msd = False
    args.msd_num = 1
    if old:
        args.use_attn_pooling = use_attn_pooling
        args.use_scl = use_scl

    common._validate_config_depth(
        state,
        "encoder_raw.transformer.",
        int(args.layers_num_raw),
        "no-r_learned raw encoder",
    )
    common._validate_config_depth(
        state,
        "encoder_size.transformer.",
        int(args.layers_num_size),
        "no-r_learned behavior encoder",
    )

    classifier = common._get_stage2_classifier(old)
    model = classifier(args, *vocab_lengths, labels_num)
    checkpoint_name = (
        "legacy no-r_learned ITGCA" if old else "no-r_learned ITGCA"
    )
    common._strict_load(model, state, checkpoint_name)
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _resolve_paths(args)
    _validate_paths(args)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.existing_flow_results.parent / "r_learned_ablation"
    )
    output_paths = {
        "flows": output_dir / "r_learned_ablation_flows.csv",
        "summary": output_dir / "r_learned_ablation_summary.csv",
        "provenance": output_dir / "r_learned_ablation_provenance.json",
    }
    existing_outputs = [path for path in output_paths.values() if path.exists()]
    if existing_outputs and not args.overwrite:
        raise ValueError(
            "output file(s) already exist; pass --overwrite to replace them: "
            + ", ".join(map(str, existing_outputs))
        )

    label2id_value = common._load_pickle(args.label2id_path)
    if not isinstance(label2id_value, dict) or not label2id_value:
        raise ValueError("label2id must be a non-empty dictionary")
    label2id = {
        str(name): int(label_id) for name, label_id in label2id_value.items()
    }
    if set(label2id.values()) != set(range(len(label2id))):
        raise ValueError("label2id IDs must be unique and contiguous from 0")
    id2label = {label_id: name for name, label_id in label2id.items()}

    records, source_metadata = _read_existing_results(
        args.existing_flow_results, args.dataset_dir
    )
    _validate_labels(records, label2id)
    dataset = (
        str(args.dataset_name).strip()
        if args.dataset_name is not None
        else source_metadata["dataset_from_source"]
    )
    if not dataset:
        raise ValueError("--dataset-name must not be empty")
    scope = source_metadata["evaluation_scope"]
    for record in records:
        record["dataset"] = dataset
        record["evaluation_scope"] = scope

    if not args.quiet:
        print(f"[version] {SCRIPT_VERSION}")
        print(
            f"[reuse] dataset={dataset} positive_exposure_rows={len(records)}"
        )
        print(
            "[reuse] audit, baselines, full ITGCA, utility, and gate statistics "
            "will not be regenerated"
        )

    store, vocab_lengths = common._prepare_features(
        records,
        label2id,
        (args.vocab_path_raw, args.vocab_path_size, args.vocab_path_temporal),
        args.seq_length_raw,
        args.seq_length_size,
        args.workers,
        args.chunk_size,
        args.quiet,
    )
    successful_indices = np.flatnonzero(store.success)
    if not len(successful_indices):
        raise ValueError("none of the selected positive-exposure flows were readable")

    device = common._resolve_device(args.device)
    if not args.quiet:
        print(
            f"[features] ready={len(successful_indices)} "
            f"failed={len(records) - len(successful_indices)}"
        )
        print(f"[device] {device}")

    model = _build_no_r_learned_model(
        args.no_r_learned_checkpoint,
        args.raw_config,
        args.behavior_config,
        args.itgca_window_size,
        vocab_lengths,
        len(label2id),
        args.seq_length_raw,
        args.seq_length_size,
        args.old,
        args.use_attn_pooling,
        args.use_scl,
    )
    logits = common._run_model_logits(
        "no-r_learned",
        model,
        "itgca",
        store,
        successful_indices,
        args.batch_size,
        device,
        vocab_lengths[0],
        len(label2id),
        args.quiet,
    )
    common._release_model(model, device)
    del model

    predictions = np.full(len(records), -1, dtype=np.int64)
    predictions[successful_indices] = np.argmax(
        logits[successful_indices], axis=1
    ).astype(np.int64)
    paired = _paired_rows(records, store, predictions, id2label)
    summaries = _summary_rows(
        paired,
        dataset,
        scope,
        args.bootstrap_repetitions,
        args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    common._atomic_write_csv(output_paths["flows"], paired, FLOW_FIELDS)
    common._atomic_write_csv(output_paths["summary"], summaries, SUMMARY_FIELDS)
    common._atomic_write_json(
        output_paths["provenance"],
        {
            "script": str(Path(__file__).resolve()),
            "script_version": SCRIPT_VERSION,
            "existing_flow_results": str(args.existing_flow_results),
            "dataset_dir": str(args.dataset_dir),
            "dataset_name": dataset,
            "no_r_learned_checkpoint": str(args.no_r_learned_checkpoint),
            "label2id_path": str(args.label2id_path),
            "raw_config": str(args.raw_config),
            "behavior_config": str(args.behavior_config),
            "selected_rows": len(records),
            "evaluated_rows": int(len(successful_indices)),
            "failed_rows": int(len(records) - len(successful_indices)),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.seed,
            "source": source_metadata,
            "reused_outputs": [
                "compression exposure e_i and bins",
                "full ITGCA predictions",
            ],
            "new_inference": "fine-tuned no-r_learned classifier only",
            "metric": (
                "100 * (Macro-F1(full ITGCA) - "
                "Macro-F1(no r_learned)); Macro-F1 averages over ground-truth "
                "labels present in each exposure bin"
            ),
        },
    )

    if not args.quiet:
        for path in output_paths.values():
            print(f"[write] {path}")
        print(
            f"[done] evaluated={len(successful_indices)} "
            f"failed={len(records) - len(successful_indices)}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
