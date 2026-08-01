#!/usr/bin/env python3
"""Export per-flow conditional content-utility inputs for Revision Sec. 2.3.

The script pairs two Stage-1 classifiers that never use ``r_stat``:

* concat: ``Stage1Classifier(modality="both")`` estimates p(y | C, B);
* behavior: ``Stage1Classifier(modality="size")`` estimates p(y | B).

Each model is temperature-calibrated on the same validation split.  The script
then evaluates both checkpoints on the same ordered test pickle and writes:

    content_utility
      = log p_concat(y_true | C, B) - log p_behavior(y_true | B)

Only the ten requested per-flow fields are written.  Correlations, AUROC,
quantile bins, bootstrap intervals, and aggregate summaries deliberately belong
in a later analysis script.

Example:

    python revision/2.3/content_utility_inference.py \
        --concat-checkpoint /path/to/concat.bin \
        --behavior-checkpoint /path/to/behavior.bin \
        --validation-path /path/to/val.pkl \
        --test-path /path/to/test.pkl \
        --label2id-path /path/to/label2id.pkl \
        --raw-config models/bert/base_config.json \
        --behavior-config models/bert/behavior_6_config.json \
        --output-path /path/to/flow_results.csv

The concat classifier uses ``--raw-config`` for its raw encoder and
``--behavior-config`` for its behavior encoder.  The behavior-only classifier
reuses that same ``--behavior-config``.

The ``sample_index`` column is the zero-based position in ``test.pkl``.  Both
models are run from that one in-memory test object, so row alignment cannot
depend on filenames or on separately generated prediction files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


SCRIPT_VERSION = "1.2.0"
DEFAULT_SEQ_LENGTH_RAW = 512
DEFAULT_SEQ_LENGTH_SIZE = 256
DEFAULT_BATCH_SIZE = 64
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0
TEMPERATURE_GRID_SIZE = 121

OUTPUT_FIELDS = [
    "sample_index",
    "true_label",
    "r_stat",
    "concat_pred",
    "concat_correct",
    "concat_p_true",
    "behavior_pred",
    "behavior_correct",
    "behavior_p_true",
    "content_utility",
]

REQUIRED_SAMPLE_FIELDS = (
    "raw_src",
    "packet_ids",
    "directions",
    "size_src",
    "iat_src",
    "label",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_DIR = PROJECT_ROOT / "fine-tuning"
for _path in (str(FINETUNE_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Reuse the exact current classifier and entropy-prior implementations.
from run_classifier_stage1 import Stage1Classifier  # noqa: E402
from uer.models.multimodal_model import compute_flow_reliability_raw  # noqa: E402
from uer.opts import model_opts  # noqa: E402
from uer.utils.config import apply_modality_configs  # noqa: E402
from uer.utils.vocab import Vocab  # noqa: E402


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_vocab_size(path: Path) -> int:
    vocab = Vocab()
    vocab.load(str(path), is_quiet=True)
    if len(vocab) <= 0:
        raise ValueError(f"vocabulary is empty: {path}")
    return len(vocab)


def _new_model_args(
    modality: str,
    seq_length_raw: int,
    seq_length_size: int,
    raw_config_path: Path,
    behavior_config_path: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    model_opts(parser)
    args = parser.parse_args([])

    if modality == "both":
        main_config_path = raw_config_path
    elif modality == "size":
        main_config_path = behavior_config_path
    else:
        raise ValueError(f"unsupported Stage-1 modality: {modality}")

    with main_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(
            f"model config must contain a JSON object: {main_config_path}"
        )

    vars(args).update(config)
    args.modality = modality
    args.max_seq_length = max(seq_length_raw, seq_length_size)
    args.dropout = getattr(args, "dropout", 0.1)
    args.config_path_raw = str(raw_config_path) if modality == "both" else None
    args.config_path_size = str(behavior_config_path)
    return apply_modality_configs(args)


def _safe_torch_load(path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"checkpoint is not a non-empty state_dict: {path}")

    state = dict(checkpoint)
    for prefix in ("module.", "_orig_mod."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {
                str(key)[len(prefix):]: value
                for key, value in state.items()
            }

    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"checkpoint contains non-tensor state entries: {path}")
    return state


def _infer_stack_depth(
    state: Mapping[str, torch.Tensor],
    prefix: str,
) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    indices = {
        int(match.group(1))
        for key in state
        for match in [pattern.match(str(key))]
        if match is not None
    }
    return max(indices) + 1 if indices else 0


def _validate_stack_depth(
    state: Mapping[str, torch.Tensor],
    prefix: str,
    configured_depth: int,
    description: str,
) -> None:
    observed_depth = _infer_stack_depth(state, prefix)
    if observed_depth <= 0:
        raise ValueError(
            f"{description} parameters were not found in the checkpoint"
        )
    if observed_depth != configured_depth:
        raise ValueError(
            f"{description} depth mismatch: "
            f"config={configured_depth}, checkpoint={observed_depth}"
        )


def _build_stage1_model(
    modality: str,
    checkpoint_path: Path,
    raw_config_path: Path,
    behavior_config_path: Path,
    vocab_sizes: Tuple[int, int, int],
    labels_num: int,
    seq_length_raw: int,
    seq_length_size: int,
) -> torch.nn.Module:
    state = _safe_torch_load(checkpoint_path)
    args = _new_model_args(
        modality,
        seq_length_raw,
        seq_length_size,
        raw_config_path,
        behavior_config_path,
    )
    base_depth = int(args.layers_num)
    raw_depth = int(getattr(args, "layers_num_raw", base_depth) or base_depth)
    size_depth = int(getattr(args, "layers_num_size", base_depth) or base_depth)

    if modality == "both":
        _validate_stack_depth(
            state,
            "encoder_raw.transformer.",
            raw_depth,
            "concat raw encoder",
        )
        _validate_stack_depth(
            state,
            "encoder_size.transformer.",
            size_depth,
            "concat behavior encoder",
        )
        checkpoint_description = "concat Stage-1"
    elif modality == "size":
        _validate_stack_depth(
            state,
            "encoder_size.transformer.",
            size_depth,
            "behavior-only encoder",
        )
        checkpoint_description = "behavior-only Stage-1"
    else:
        raise ValueError(f"unsupported Stage-1 modality: {modality}")

    model = Stage1Classifier(args, *vocab_sizes, labels_num)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"{checkpoint_description} checkpoint does not match "
            f"the supplied configuration. Inference was stopped rather than "
            f"using missing or randomly initialized parameters.\n{exc}"
        ) from exc
    return model


def _validate_label_mapping(label2id: Any) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not isinstance(label2id, dict) or not label2id:
        raise ValueError("label2id must be a non-empty dictionary")

    normalized = {str(name): int(label_id) for name, label_id in label2id.items()}
    expected_ids = set(range(len(normalized)))
    if set(normalized.values()) != expected_ids:
        raise ValueError("label2id IDs must be unique and contiguous from 0")
    return normalized, {label_id: name for name, label_id in normalized.items()}


def _tensorize_dataset(
    dataset: Any,
    split_name: str,
    labels_num: int,
) -> Dict[str, torch.Tensor]:
    if not isinstance(dataset, (list, tuple)) or not dataset:
        raise ValueError(f"{split_name} dataset must be a non-empty list or tuple")

    for index, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            raise ValueError(
                f"{split_name}[{index}] is not a dictionary"
            )
        missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(
                f"{split_name}[{index}] is missing fields: {', '.join(missing)}"
            )

        try:
            label = int(sample["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{split_name}[{index}] has an invalid label: {sample['label']!r}"
            ) from exc
        if not 0 <= label < labels_num:
            raise ValueError(
                f"{split_name}[{index}] label {label} is outside "
                f"[0, {labels_num - 1}]"
            )

    tensors: Dict[str, torch.Tensor] = {}
    for field in REQUIRED_SAMPLE_FIELDS[:-1]:
        try:
            values = np.stack(
                [np.asarray(sample[field], dtype=np.int64) for sample in dataset]
            )
        except ValueError as exc:
            raise ValueError(
                f"{split_name} field {field!r} has inconsistent sample shapes"
            ) from exc
        tensors[field] = torch.from_numpy(values)

    tensors["label"] = torch.tensor(
        [int(sample["label"]) for sample in dataset],
        dtype=torch.long,
    )
    return tensors


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is unavailable: {requested}")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"visible device count is {torch.cuda.device_count()}"
            )
    return device


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, tuple):
        candidates = [
            value
            for value in output
            if torch.is_tensor(value) and value.ndim == 2
        ]
        if len(candidates) != 1:
            raise ValueError(
                "could not identify a unique [batch, classes] logits tensor "
                "in the classifier output"
            )
        logits = candidates[0]
    else:
        raise ValueError(
            f"unsupported classifier output type: {type(output).__name__}"
        )

    if logits.ndim != 2:
        raise ValueError(
            f"classifier logits must have rank 2, got shape {tuple(logits.shape)}"
        )
    return logits


def _to_device(
    tensor: torch.Tensor,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    return tensor[start:end].to(device=device, dtype=torch.long, non_blocking=True)


def _collect_logits(
    model: torch.nn.Module,
    modality: str,
    tensors: Mapping[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    labels_num: int,
    raw_vocab_size: int,
    collect_r_stat: bool,
    quiet: bool,
    split_name: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    sample_count = int(tensors["label"].shape[0])
    model.to(device).eval()

    logits_parts = []
    r_stat_parts = []
    total_batches = (sample_count + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_number, start in enumerate(
            range(0, sample_count, batch_size),
            start=1,
        ):
            end = min(sample_count, start + batch_size)

            if modality == "both":
                raw_src = _to_device(tensors["raw_src"], start, end, device)
                packet_ids = _to_device(tensors["packet_ids"], start, end, device)
                directions = _to_device(tensors["directions"], start, end, device)
                size_src = _to_device(tensors["size_src"], start, end, device)
                iat_src = _to_device(tensors["iat_src"], start, end, device)
                output = model(
                    raw_src,
                    packet_ids,
                    directions,
                    size_src,
                    iat_src,
                )

                if collect_r_stat:
                    r_stat = compute_flow_reliability_raw(
                        raw_src,
                        vocab_size=raw_vocab_size,
                    )
                    r_stat_parts.append(
                        r_stat.detach().cpu().numpy().astype(np.float64)
                    )
            elif modality == "size":
                size_src = _to_device(tensors["size_src"], start, end, device)
                iat_src = _to_device(tensors["iat_src"], start, end, device)
                output = model(None, None, None, size_src, iat_src)
            else:
                raise ValueError(f"unsupported inference modality: {modality}")

            logits = _extract_logits(output)
            if logits.shape[1] != labels_num:
                raise ValueError(
                    f"{modality} logits have {logits.shape[1]} classes, "
                    f"but label2id contains {labels_num}"
                )
            logits_parts.append(
                logits.detach().cpu().numpy().astype(np.float64)
            )

            if not quiet and (
                batch_number == total_batches or batch_number % 50 == 0
            ):
                print(
                    f"[{modality}:{split_name}] "
                    f"{batch_number}/{total_batches} batches"
                )

    all_logits = np.concatenate(logits_parts, axis=0)
    if all_logits.shape != (sample_count, labels_num):
        raise RuntimeError(
            f"{modality} inference returned shape {all_logits.shape}, "
            f"expected {(sample_count, labels_num)}"
        )

    all_r_stat: Optional[np.ndarray] = None
    if collect_r_stat:
        all_r_stat = np.concatenate(r_stat_parts, axis=0)
        if all_r_stat.shape != (sample_count,):
            raise RuntimeError(
                f"r_stat returned shape {all_r_stat.shape}, "
                f"expected {(sample_count,)}"
            )
    return all_logits, all_r_stat


def _release_model(model: torch.nn.Module, device: torch.device) -> None:
    model.to(torch.device("cpu"))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _log_softmax_numpy(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite: {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    row_max = np.max(scaled, axis=1, keepdims=True)
    shifted = scaled - row_max
    log_normalizer = row_max + np.log(
        np.exp(shifted).sum(axis=1, keepdims=True)
    )
    return scaled - log_normalizer


def _mean_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    log_probs = _log_softmax_numpy(logits, temperature)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def _fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float, bool]:
    """Fit one scalar temperature by bounded validation-set NLL minimization."""

    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("temperature fitting received misaligned logits and labels")
    if not np.isfinite(logits).all():
        raise ValueError("temperature fitting received non-finite logits")

    log_min = math.log(TEMPERATURE_MIN)
    log_max = math.log(TEMPERATURE_MAX)
    grid = np.linspace(log_min, log_max, TEMPERATURE_GRID_SIZE)
    # Explicitly include T=1 so calibration can never be forced to worsen NLL.
    grid = np.unique(np.concatenate([grid, np.asarray([0.0])]))
    objectives = np.asarray(
        [_mean_nll(logits, labels, math.exp(value)) for value in grid]
    )
    best_index = int(np.argmin(objectives))
    best_log_temperature = float(grid[best_index])
    best_nll = float(objectives[best_index])

    # Refine an interior grid optimum with a deterministic golden-section search.
    if 0 < best_index < len(grid) - 1:
        left = float(grid[best_index - 1])
        right = float(grid[best_index + 1])
        golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = right - golden_ratio * (right - left)
        x2 = left + golden_ratio * (right - left)
        f1 = _mean_nll(logits, labels, math.exp(x1))
        f2 = _mean_nll(logits, labels, math.exp(x2))

        for _ in range(60):
            if f1 <= f2:
                right = x2
                x2 = x1
                f2 = f1
                x1 = right - golden_ratio * (right - left)
                f1 = _mean_nll(logits, labels, math.exp(x1))
            else:
                left = x1
                x1 = x2
                f1 = f2
                x2 = left + golden_ratio * (right - left)
                f2 = _mean_nll(logits, labels, math.exp(x2))

        refined_log_temperature = (left + right) / 2.0
        refined_temperature = math.exp(refined_log_temperature)
        refined_nll = _mean_nll(logits, labels, refined_temperature)
        if refined_nll < best_nll:
            best_log_temperature = refined_log_temperature
            best_nll = refined_nll

    temperature = math.exp(best_log_temperature)
    nll_before = _mean_nll(logits, labels, 1.0)
    at_boundary = (
        math.isclose(temperature, TEMPERATURE_MIN, rel_tol=1e-6)
        or math.isclose(temperature, TEMPERATURE_MAX, rel_tol=1e-6)
    )
    return temperature, nll_before, best_nll, at_boundary


def _calibrated_outputs(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> Dict[str, np.ndarray]:
    log_probs = _log_softmax_numpy(logits, temperature)
    row_indices = np.arange(len(labels))
    log_p_true = log_probs[row_indices, labels]
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    return {
        "pred": predictions,
        "correct": predictions == labels,
        "log_p_true": log_p_true,
        "p_true": np.exp(log_p_true),
    }


def _write_flow_results(
    output_path: Path,
    labels: np.ndarray,
    id2label: Mapping[int, str],
    r_stat: np.ndarray,
    concat: Mapping[str, np.ndarray],
    behavior: Mapping[str, np.ndarray],
) -> None:
    content_utility = concat["log_p_true"] - behavior["log_p_true"]
    sample_count = len(labels)

    aligned_arrays = {
        "r_stat": r_stat,
        "concat_pred": concat["pred"],
        "concat_correct": concat["correct"],
        "concat_p_true": concat["p_true"],
        "behavior_pred": behavior["pred"],
        "behavior_correct": behavior["correct"],
        "behavior_p_true": behavior["p_true"],
        "content_utility": content_utility,
    }
    for name, values in aligned_arrays.items():
        if len(values) != sample_count:
            raise RuntimeError(
                f"output field {name} has {len(values)} rows; "
                f"expected {sample_count}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for index in range(sample_count):
            true_id = int(labels[index])
            concat_pred_id = int(concat["pred"][index])
            behavior_pred_id = int(behavior["pred"][index])
            writer.writerow(
                {
                    "sample_index": index,
                    "true_label": id2label[true_id],
                    "r_stat": f"{float(r_stat[index]):.12g}",
                    "concat_pred": id2label[concat_pred_id],
                    "concat_correct": int(bool(concat["correct"][index])),
                    "concat_p_true": f"{float(concat['p_true'][index]):.12g}",
                    "behavior_pred": id2label[behavior_pred_id],
                    "behavior_correct": int(bool(behavior["correct"][index])),
                    "behavior_p_true": f"{float(behavior['p_true'][index]):.12g}",
                    "content_utility": f"{float(content_utility[index]):.12g}",
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export paired concat/behavior predictions and conditional "
            "content utility for Revision Sec. 2.3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--concat-checkpoint", type=Path, required=True)
    parser.add_argument("--behavior-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--label2id-path", type=Path, required=True)
    parser.add_argument(
        "--raw-config",
        type=Path,
        default=PROJECT_ROOT / "models/bert/base_config.json",
        help=(
            "raw encoder config for the concat classifier; it also supplies "
            "the concat model's shared dimensions and hyperparameters"
        ),
    )
    parser.add_argument(
        "--behavior-config",
        type=Path,
        default=PROJECT_ROOT / "models/bert/behavior_6_config.json",
        help=(
            "shared behavior-encoder config for both the concat classifier "
            "and the behavior-only classifier"
        ),
    )
    parser.add_argument(
        "--vocab-path-raw",
        type=Path,
        default=PROJECT_ROOT / "models/bert/vocab_raw.txt",
    )
    parser.add_argument(
        "--vocab-path-size",
        type=Path,
        default=PROJECT_ROOT / "models/bert/vocab_size.txt",
    )
    parser.add_argument(
        "--vocab-path-temporal",
        type=Path,
        default=PROJECT_ROOT / "models/bert/vocab_temporal.txt",
    )
    parser.add_argument("--seq-length-raw", type=int, default=DEFAULT_SEQ_LENGTH_RAW)
    parser.add_argument("--seq-length-size", type=int, default=DEFAULT_SEQ_LENGTH_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit device such as cuda:1",
    )
    parser.add_argument(
        "--skip-temperature-scaling",
        action="store_true",
        help="use temperature 1.0 for both models (not recommended for log-p comparison)",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output CSV",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _resolve_and_validate_paths(args: argparse.Namespace) -> None:
    path_fields = (
        "concat_checkpoint",
        "behavior_checkpoint",
        "validation_path",
        "test_path",
        "label2id_path",
        "raw_config",
        "behavior_config",
        "vocab_path_raw",
        "vocab_path_size",
        "vocab_path_temporal",
    )
    for field in path_fields:
        path = getattr(args, field).expanduser().resolve()
        setattr(args, field, path)
        if not path.is_file():
            raise ValueError(
                f"required file does not exist ({field}): {path}"
            )

    args.output_path = args.output_path.expanduser().resolve()
    if args.output_path.exists() and not args.overwrite:
        raise ValueError(
            f"output already exists: {args.output_path}\n"
            "Pass --overwrite only if replacing it is intentional."
        )
    if args.validation_path == args.test_path:
        raise ValueError(
            "validation and test paths are identical; temperature calibration "
            "must not use the evaluated test split"
        )

    for field in ("seq_length_raw", "seq_length_size", "batch_size"):
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _resolve_and_validate_paths(args)
    device = _resolve_device(args.device)

    label2id, id2label = _validate_label_mapping(
        _load_pickle(args.label2id_path)
    )
    labels_num = len(label2id)
    vocab_sizes = (
        _load_vocab_size(args.vocab_path_raw),
        _load_vocab_size(args.vocab_path_size),
        _load_vocab_size(args.vocab_path_temporal),
    )

    if not args.quiet:
        print(f"content-utility inference v{SCRIPT_VERSION}")
        print(f"device: {device}")
        print(f"labels: {labels_num}")
        print(
            "vocab sizes: "
            f"raw={vocab_sizes[0]}, size={vocab_sizes[1]}, "
            f"temporal={vocab_sizes[2]}"
        )

    validation_dataset = _load_pickle(args.validation_path)
    test_dataset = _load_pickle(args.test_path)
    validation_tensors = _tensorize_dataset(
        validation_dataset,
        "validation",
        labels_num,
    )
    test_tensors = _tensorize_dataset(test_dataset, "test", labels_num)
    del validation_dataset, test_dataset
    validation_labels = validation_tensors["label"].numpy()
    test_labels = test_tensors["label"].numpy()

    if not args.quiet:
        print(
            f"samples: validation={len(validation_labels)}, "
            f"test={len(test_labels)}"
        )

    concat_model = _build_stage1_model(
        "both",
        args.concat_checkpoint,
        args.raw_config,
        args.behavior_config,
        vocab_sizes,
        labels_num,
        args.seq_length_raw,
        args.seq_length_size,
    )
    concat_validation_logits, _ = _collect_logits(
        concat_model,
        "both",
        validation_tensors,
        args.batch_size,
        device,
        labels_num,
        vocab_sizes[0],
        collect_r_stat=False,
        quiet=args.quiet,
        split_name="validation",
    )
    concat_test_logits, test_r_stat = _collect_logits(
        concat_model,
        "both",
        test_tensors,
        args.batch_size,
        device,
        labels_num,
        vocab_sizes[0],
        collect_r_stat=True,
        quiet=args.quiet,
        split_name="test",
    )
    _release_model(concat_model, device)
    del concat_model

    behavior_model = _build_stage1_model(
        "size",
        args.behavior_checkpoint,
        args.raw_config,
        args.behavior_config,
        vocab_sizes,
        labels_num,
        args.seq_length_raw,
        args.seq_length_size,
    )
    behavior_validation_logits, _ = _collect_logits(
        behavior_model,
        "size",
        validation_tensors,
        args.batch_size,
        device,
        labels_num,
        vocab_sizes[0],
        collect_r_stat=False,
        quiet=args.quiet,
        split_name="validation",
    )
    behavior_test_logits, _ = _collect_logits(
        behavior_model,
        "size",
        test_tensors,
        args.batch_size,
        device,
        labels_num,
        vocab_sizes[0],
        collect_r_stat=False,
        quiet=args.quiet,
        split_name="test",
    )
    _release_model(behavior_model, device)
    del behavior_model

    if args.skip_temperature_scaling:
        concat_temperature = 1.0
        behavior_temperature = 1.0
        concat_nll_before = concat_nll_after = _mean_nll(
            concat_validation_logits,
            validation_labels,
            concat_temperature,
        )
        behavior_nll_before = behavior_nll_after = _mean_nll(
            behavior_validation_logits,
            validation_labels,
            behavior_temperature,
        )
        concat_at_boundary = behavior_at_boundary = False
    else:
        (
            concat_temperature,
            concat_nll_before,
            concat_nll_after,
            concat_at_boundary,
        ) = _fit_temperature(concat_validation_logits, validation_labels)
        (
            behavior_temperature,
            behavior_nll_before,
            behavior_nll_after,
            behavior_at_boundary,
        ) = _fit_temperature(behavior_validation_logits, validation_labels)

    print(
        "concat temperature: "
        f"{concat_temperature:.8g} "
        f"(validation NLL {concat_nll_before:.8g} -> {concat_nll_after:.8g})"
    )
    print(
        "behavior temperature: "
        f"{behavior_temperature:.8g} "
        f"(validation NLL {behavior_nll_before:.8g} -> {behavior_nll_after:.8g})"
    )
    if concat_at_boundary or behavior_at_boundary:
        print(
            "WARNING: at least one fitted temperature reached the search "
            f"boundary [{TEMPERATURE_MIN}, {TEMPERATURE_MAX}].",
            file=sys.stderr,
        )

    concat_outputs = _calibrated_outputs(
        concat_test_logits,
        test_labels,
        concat_temperature,
    )
    behavior_outputs = _calibrated_outputs(
        behavior_test_logits,
        test_labels,
        behavior_temperature,
    )
    if test_r_stat is None:
        raise RuntimeError("internal error: test r_stat was not collected")

    _write_flow_results(
        args.output_path,
        test_labels,
        id2label,
        test_r_stat,
        concat_outputs,
        behavior_outputs,
    )
    if not args.quiet:
        content_utility = (
            concat_outputs["log_p_true"] - behavior_outputs["log_p_true"]
        )
        rescue_count = int(
            np.sum(concat_outputs["correct"] & ~behavior_outputs["correct"])
        )
        harm_count = int(
            np.sum(behavior_outputs["correct"] & ~concat_outputs["correct"])
        )
        print(f"wrote {len(test_labels)} rows: {args.output_path}")
        print(
            "sanity check only: "
            f"mean content_utility={content_utility.mean():.8g}, "
            f"rescue={rescue_count}, harm={harm_count}"
        )
        print("No aggregate analysis files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
