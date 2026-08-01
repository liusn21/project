#!/usr/bin/env python3
"""Evaluate conditional content utility and ITGCA correction on audited flows.

The input is ``flow_details.csv`` produced by ``compression_audit.py``.  The
script selects valid, confirmed-compressed rows, recreates their model inputs
with the exact fine-tuning PCAP pipeline, and runs four aligned predictors:

* concat Stage 1: p(y | content, behavior), without ``r_stat``;
* behavior Stage 1: p(y | behavior), without ``r_stat``;
* the full ITGCA model;
* a same-checkpoint stat-only intervention with beta approximately zero.

All four predictors are independently temperature-scaled on the validation
split unless ``--skip-temperature-scaling`` is supplied.  The task-aligned
conditional content utility is

    u_i = log p_concat(y_i) - log p_behavior(y_i).

Four result files are written:

* ``flow_results.csv``: aligned predictions and flow-level gate diagnostics;
* ``summary.csv``: aggregate results for all compressed flows and e_i groups;
* ``gate_layers.csv``: one row per successfully evaluated flow and fusion layer;
* ``calibration.json``: validation temperatures and NLL diagnostics.

The audit normally covers the complete fine-tuning PCAP directory.  Therefore
the default evaluation scope is explicitly diagnostic rather than held-out.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score


SCRIPT_VERSION = "2.0.2"
BYTES_PER_PACKET = 64
MAX_RAW_PACKETS = 8
MAX_SIZE_PACKETS = 256
DEFAULT_SEQ_LENGTH_RAW = 512
DEFAULT_SEQ_LENGTH_SIZE = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHUNK_SIZE = 16
DEFAULT_WORKERS = max(1, min(32, (os.cpu_count() or 1) - 1))
DEFAULT_UTILITY_THRESHOLD = 0.01
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0
TEMPERATURE_GRID_SIZE = 121
STAT_ONLY_ALPHA = -30.0

def _discover_project_layout(script_path: Path) -> Tuple[Path, Path]:
    """Find the repository root without depending on the launch directory.

    The normal layout is ``PROJECT/revision/2.4/this_file.py`` plus
    ``PROJECT/fine-tuning/multimodal_data_utils.py``.  Searching ancestors is
    more robust than a fixed ``parents[N]`` assumption when the revision folder
    is copied or symlinked on a remote machine.
    """

    checked: List[Path] = []
    for ancestor in script_path.resolve().parents:
        for directory_name in ("fine-tuning", "fine_tuning"):
            finetune_dir = ancestor / directory_name
            module_path = finetune_dir / "multimodal_data_utils.py"
            checked.append(module_path)
            if module_path.is_file() and (ancestor / "uer").is_dir():
                return ancestor, finetune_dir
    raise RuntimeError(
        "could not locate the MM-TrafficBERT project root from "
        f"{script_path.resolve()}. Expected both uer/ and "
        "fine-tuning/multimodal_data_utils.py (or fine_tuning/...). "
        "Checked: "
        + ", ".join(str(path) for path in checked)
    )


PROJECT_ROOT, FINETUNE_DIR = _discover_project_layout(Path(__file__))
for _path in (str(FINETUNE_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# These imports intentionally reuse the current fine-tuning implementation.
from multimodal_data_utils import (  # noqa: E402
    _init_worker as _finetune_init_worker,
    _process_pcap_worker as _finetune_process_pcap_worker,
    build_all_caches,
    create_tokenizer,
)
from run_classifier_stage1 import Stage1Classifier  # noqa: E402
from run_classifier_stage2 import Stage2Classifier  # noqa: E402
from uer.models.multimodal_model import compute_flow_reliability_raw  # noqa: E402
from uer.opts import model_opts  # noqa: E402
from uer.utils.config import apply_modality_configs  # noqa: E402


FLOW_OUTPUT_FIELDS = [
    "dataset",
    "compression_level",
    "evaluation_scope",
    "label",
    "relative_pcap",
    "e_i",
    "e_bin",
    "model_entropy_bits",
    "audit_model_r_stat",
    "inference_r_stat",
    "concat_pred",
    "concat_correct",
    "concat_p_true",
    "concat_log_p_true",
    "behavior_pred",
    "behavior_correct",
    "behavior_p_true",
    "behavior_log_p_true",
    "itgca_pred",
    "itgca_correct",
    "itgca_p_true",
    "itgca_log_p_true",
    "stat_only_pred",
    "stat_only_correct",
    "stat_only_p_true",
    "stat_only_log_p_true",
    "content_utility",
    "full_gain",
    "correction_gain",
    "utility_threshold",
    "utility_group",
    "content_helpful",
    "r_calibrated",
    "r_learned",
    "r_mod",
    "delta_learned",
    "delta_mod",
    "gate_up",
    "content_opportunity",
    "itgca_rescue",
    "utility_recovered",
    "gate_supported_recovery",
    "inference_status",
    "error",
]

GATE_LAYER_FIELDS = [
    "dataset",
    "compression_level",
    "evaluation_scope",
    "label",
    "relative_pcap",
    "e_i",
    "e_bin",
    "layer_index",
    "beta",
    "r_stat",
    "r_calibrated",
    "r_learned",
    "r_mod",
    "delta_learned",
    "delta_mod",
    "content_utility",
    "utility_group",
]


SUMMARY_FIELDS = [
    "dataset",
    "compression_level",
    "evaluation_scope",
    "group",
    "group_display",
    "flow_count",
    "evaluated_flows",
    "failed_flows",
    "mean_model_entropy_bits",
    "mean_r_stat",
    "concat_accuracy",
    "behavior_accuracy",
    "itgca_accuracy",
    "stat_only_accuracy",
    "concat_macro_f1",
    "behavior_macro_f1",
    "itgca_macro_f1",
    "stat_only_macro_f1",
    "concat_minus_behavior_accuracy",
    "itgca_minus_behavior_accuracy",
    "itgca_minus_stat_only_accuracy",
    "mean_content_utility",
    "median_content_utility",
    "content_helpful_count",
    "content_helpful_rate",
    "mean_beta",
    "mean_r_calibrated",
    "mean_r_learned",
    "mean_r_mod",
    "mean_delta_learned",
    "mean_delta_mod",
    "helpful_gate_up_count",
    "helpful_gate_up_rate",
    "helpful_median_delta_learned",
    "helpful_median_delta_mod",
    "helpful_mean_full_gain",
    "helpful_mean_correction_gain",
    "helpful_stat_only_improvement_count",
    "helpful_stat_only_improvement_rate",
    "utility_recovery_count",
    "utility_recovery_rate",
    "gate_supported_recovery_count",
    "gate_supported_recovery_rate",
    "content_opportunity_count",
    "itgca_rescue_count",
    "itgca_rescue_rate",
    "concat_checkpoint",
    "behavior_checkpoint",
    "itgca_checkpoint",
]


def _maximize_csv_field_size_limit() -> None:
    """Raise csv's parser limit without overflowing platforms with a C long."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = math.nan) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _read_audit_metadata(flow_path: Path) -> Dict[str, Any]:
    metadata_path = flow_path.parent / "audit_metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, ValueError):
        return {}


def _resolve_pcap_path(
    row: Mapping[str, str],
    flow_path: Path,
    dataset_dir: Optional[Path],
) -> Path:
    candidates: List[Path] = []

    raw_path = str(row.get("pcap_path", "")).strip()
    if raw_path:
        pcap_path = Path(raw_path).expanduser()
        if not pcap_path.is_absolute():
            pcap_path = flow_path.parent / pcap_path
        candidates.append(pcap_path)

    relative = str(row.get("relative_pcap", "")).strip()
    if relative and dataset_dir is not None:
        candidates.append(dataset_dir / relative)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return candidates[-1].resolve() if candidates else Path("")


def _read_confirmed_rows(
    flow_path: Path,
    dataset_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """Stream the large audit CSV and retain only compact required fields."""

    _maximize_csv_field_size_limit()
    records: List[Dict[str, Any]] = []
    with flow_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"audit CSV has no header: {flow_path}")

        required = {"status", "confirmed_compressed", "label", "relative_pcap", "e_i"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"audit CSV is missing required columns: {', '.join(missing)}"
            )

        for csv_row_number, row in enumerate(reader, start=2):
            if str(row.get("status", "")).strip().lower() != "valid":
                continue
            if not _is_true(row.get("confirmed_compressed")):
                continue

            e_i = _as_float(row.get("e_i"))
            if not math.isfinite(e_i) or e_i < 0.0 or e_i > 1.0:
                raise ValueError(
                    f"invalid e_i={row.get('e_i')!r} at CSV row {csv_row_number}"
                )

            records.append({
                "dataset": str(row.get("dataset", "")).strip(),
                "label": str(row.get("label", "")).strip(),
                "relative_pcap": str(row.get("relative_pcap", "")).strip(),
                "pcap_path": _resolve_pcap_path(row, flow_path, dataset_dir),
                "e_i": e_i,
                "e_bin": str(row.get("e_bin", "")).strip(),
                "model_entropy_bits": _as_float(row.get("model_entropy_bits")),
                "audit_model_r_stat": _as_float(row.get("model_r_stat")),
                "inference_status": "pending",
                "error": "",
            })

    if not records:
        raise ValueError(
            "no rows satisfy status=valid and confirmed_compressed=1"
        )

    datasets = {record["dataset"] for record in records if record["dataset"]}
    if len(datasets) > 1:
        raise ValueError(
            "flow_details.csv contains more than one dataset: "
            + ", ".join(sorted(datasets))
        )
    return records


def _preprocess_worker(task: Tuple[int, str, str, int]) -> Tuple[int, Any, str]:
    index, pcap_path, label_name, label_id = task
    try:
        sample = _finetune_process_pcap_worker((pcap_path, label_name))
        if sample is None:
            return index, None, "fine-tuning PCAP extraction returned no payload flow"
        sample["label"] = label_id
        return index, sample, ""
    except Exception as exc:  # preserve unexpected worker failures in the result CSV
        return index, None, f"{type(exc).__name__}: {exc}"


class FeatureStore:
    """Compact fixed-shape feature arrays aligned to audit-row indices."""

    def __init__(self, count: int, seq_length_raw: int, seq_length_size: int):
        self.raw_src = np.empty((count, seq_length_raw), dtype=np.int32)
        self.packet_ids = np.empty((count, seq_length_raw), dtype=np.uint8)
        self.directions = np.empty((count, seq_length_raw), dtype=np.uint8)
        self.size_src = np.empty((count, seq_length_size), dtype=np.int32)
        self.iat_src = np.empty((count, seq_length_size), dtype=np.int32)
        self.labels = np.full(count, -1, dtype=np.int64)
        self.success = np.zeros(count, dtype=bool)

    def put(self, index: int, sample: Mapping[str, Any]) -> None:
        self.raw_src[index] = sample["raw_src"]
        self.packet_ids[index] = sample["packet_ids"]
        self.directions[index] = sample["directions"]
        self.size_src[index] = sample["size_src"]
        self.iat_src[index] = sample["iat_src"]
        self.labels[index] = int(sample["label"])
        self.success[index] = True


REQUIRED_SAMPLE_FIELDS = (
    "raw_src",
    "packet_ids",
    "directions",
    "size_src",
    "iat_src",
    "label",
)


def _feature_store_from_pickle(
    path: Path,
    split_name: str,
    labels_num: int,
    seq_length_raw: int,
    seq_length_size: int,
) -> FeatureStore:
    """Load one processed split into the same storage used by audited PCAPs."""

    dataset = _load_pickle(path)
    if not isinstance(dataset, (list, tuple)) or not dataset:
        raise ValueError(
            f"{split_name} dataset must be a non-empty list or tuple: {path}"
        )

    store = FeatureStore(len(dataset), seq_length_raw, seq_length_size)
    expected_shapes = {
        "raw_src": (seq_length_raw,),
        "packet_ids": (seq_length_raw,),
        "directions": (seq_length_raw,),
        "size_src": (seq_length_size,),
        "iat_src": (seq_length_size,),
    }
    for index, sample in enumerate(dataset):
        if not isinstance(sample, Mapping):
            raise ValueError(f"{split_name}[{index}] is not a mapping")
        missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(
                f"{split_name}[{index}] is missing fields: {', '.join(missing)}"
            )

        normalized: Dict[str, Any] = {}
        for field, expected_shape in expected_shapes.items():
            values = np.asarray(sample[field])
            if values.shape != expected_shape:
                raise ValueError(
                    f"{split_name}[{index}].{field} has shape {values.shape}; "
                    f"expected {expected_shape}"
                )
            normalized[field] = values

        try:
            label = int(sample["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{split_name}[{index}] has invalid label {sample['label']!r}"
            ) from exc
        if not 0 <= label < labels_num:
            raise ValueError(
                f"{split_name}[{index}] label {label} is outside "
                f"[0, {labels_num - 1}]"
            )
        normalized["label"] = label
        store.put(index, normalized)

    return store


def _prepare_features(
    records: List[Dict[str, Any]],
    label2id: Mapping[str, int],
    vocab_paths: Tuple[Path, Path, Path],
    seq_length_raw: int,
    seq_length_size: int,
    workers: int,
    chunk_size: int,
    quiet: bool,
) -> Tuple[FeatureStore, Tuple[int, int, int]]:
    tokenizer_raw = create_tokenizer(str(vocab_paths[0]))
    tokenizer_size = create_tokenizer(str(vocab_paths[1]))
    tokenizer_temporal = create_tokenizer(str(vocab_paths[2]))
    raw_cache, size_cache, iat_cache = build_all_caches(
        tokenizer_raw, tokenizer_size, tokenizer_temporal
    )

    vocab_raw = tokenizer_raw.vocab
    vocab_size = tokenizer_size.vocab
    vocab_temporal = tokenizer_temporal.vocab
    vocab_lengths = (len(vocab_raw), len(vocab_size), len(vocab_temporal))

    extractor_params = {
        "bytes_per_packet": BYTES_PER_PACKET,
        "max_raw_packets": MAX_RAW_PACKETS,
        "max_size_packets": MAX_SIZE_PACKETS,
    }
    init_args = (
        extractor_params,
        raw_cache,
        size_cache,
        iat_cache,
        vocab_raw,
        vocab_size,
        vocab_temporal,
        seq_length_raw,
        seq_length_size,
    )

    store = FeatureStore(len(records), seq_length_raw, seq_length_size)
    tasks: List[Tuple[int, str, str, int]] = []
    for index, record in enumerate(records):
        label_name = record["label"]
        if label_name not in label2id:
            record["inference_status"] = "label_not_in_mapping"
            record["error"] = f"label {label_name!r} is absent from label2id"
            continue

        pcap_path = record["pcap_path"]
        if not pcap_path or not Path(pcap_path).is_file():
            record["inference_status"] = "pcap_missing"
            record["error"] = f"PCAP not found: {pcap_path}"
            continue
        tasks.append((index, str(pcap_path), label_name, int(label2id[label_name])))

    if not quiet:
        print(
            f"[preprocess] selected={len(records)} readable={len(tasks)} "
            f"workers={workers}"
        )

    def consume(results: Iterable[Tuple[int, Any, str]]) -> None:
        completed = 0
        for index, sample, error in results:
            completed += 1
            if sample is None:
                records[index]["inference_status"] = "preprocess_failed"
                records[index]["error"] = error
            else:
                store.put(index, sample)
                records[index]["inference_status"] = "ready"
            if not quiet and (completed % 500 == 0 or completed == len(tasks)):
                print(f"[preprocess] {completed}/{len(tasks)}")

    if workers > 1 and tasks:
        with mp.Pool(
            processes=workers,
            initializer=_finetune_init_worker,
            initargs=init_args,
        ) as pool:
            consume(pool.imap_unordered(_preprocess_worker, tasks, chunksize=chunk_size))
    else:
        _finetune_init_worker(*init_args)
        consume(_preprocess_worker(task) for task in tasks)

    return store, vocab_lengths


def _new_model_args(config_path: Path, seq_length_raw: int, seq_length_size: int):
    parser = argparse.ArgumentParser(add_help=False)
    model_opts(parser)
    args = parser.parse_args([])

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"model config must contain a JSON object: {config_path}")
    vars(args).update(config)

    args.max_seq_length = max(seq_length_raw, seq_length_size)
    args.dropout = getattr(args, "dropout", 0.1)
    return args


def _get_stage2_classifier(old: bool):
    """Return the Stage 2 classifier implementation matching the checkpoint."""
    if not old:
        return Stage2Classifier

    try:
        from run_classifier_stage2_old import (  # noqa: E402
            Stage2Classifier as OldStage2Classifier,
        )
    except ImportError as exc:
        raise ValueError(
            "--old requires fine-tuning/run_classifier_stage2_old.py"
        ) from exc
    return OldStage2Classifier


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
            state = {str(key)[len(prefix):]: value for key, value in state.items()}
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"checkpoint contains non-tensor state entries: {path}")
    return state


def _infer_stack_depth(state: Mapping[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    indices = {
        int(match.group(1))
        for key in state
        for match in [pattern.match(key)]
        if match is not None
    }
    return max(indices) + 1 if indices else 0


def _strict_load(model: torch.nn.Module, state: Mapping[str, torch.Tensor], name: str) -> None:
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} checkpoint does not match its model configuration. "
            f"Inference was stopped instead of using random/missing parameters.\n{exc}"
        ) from exc


def _validate_config_depth(
    state: Mapping[str, torch.Tensor],
    prefix: str,
    configured: int,
    description: str,
) -> None:
    observed = _infer_stack_depth(state, prefix)
    if observed and observed != configured:
        raise ValueError(
            f"{description} depth mismatch: config={configured}, checkpoint={observed}"
        )


def _build_stage1_model(
    modality: str,
    checkpoint_path: Path,
    raw_config_path: Path,
    behavior_config_path: Path,
    vocab_lengths: Tuple[int, int, int],
    labels_num: int,
    seq_length_raw: int,
    seq_length_size: int,
) -> torch.nn.Module:
    state = _safe_torch_load(checkpoint_path)
    if modality == "both":
        main_config_path = raw_config_path
    elif modality == "size":
        main_config_path = behavior_config_path
    else:
        raise ValueError(f"unsupported Stage-1 modality: {modality}")

    args = _new_model_args(main_config_path, seq_length_raw, seq_length_size)
    args.modality = modality
    args.config_path_raw = str(raw_config_path) if modality == "both" else None
    args.config_path_size = str(behavior_config_path)
    args = apply_modality_configs(args)

    base_depth = int(args.layers_num)
    raw_depth = int(getattr(args, "layers_num_raw", base_depth) or base_depth)
    size_depth = int(getattr(args, "layers_num_size", base_depth) or base_depth)
    if modality == "both":
        _validate_config_depth(
            state,
            "encoder_raw.transformer.",
            raw_depth,
            "concat raw encoder",
        )
        _validate_config_depth(
            state,
            "encoder_size.transformer.",
            size_depth,
            "concat behavior encoder",
        )
        description = "concat Stage-1"
    else:
        _validate_config_depth(
            state,
            "encoder_size.transformer.",
            size_depth,
            "behavior-only encoder",
        )
        description = "behavior-only Stage-1"

    model = Stage1Classifier(args, *vocab_lengths, labels_num)
    _strict_load(model, state, description)
    return model


def _build_itgca_model(
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
) -> torch.nn.Module:
    state = _safe_torch_load(checkpoint_path)
    # The full model deliberately reuses the exact two configs supplied to the
    # concat/behavior baselines.  This leaves no independent ITGCA config that
    # could silently drift from the auxiliary utility models.
    args = _new_model_args(raw_config_path, seq_length_raw, seq_length_size)
    args.config_path_raw = str(raw_config_path)
    args.config_path_size = str(behavior_config_path)
    args = apply_modality_configs(args)

    fusion_depth = _infer_stack_depth(state, "fusion.fusion_layers.")
    if fusion_depth <= 0:
        raise ValueError("ITGCA checkpoint has no fusion.fusion_layers parameters")

    has_itgca = any(".gate_size.bilinear_W" in key for key in state)
    has_r_stat = any(".gate_size.alpha_modality" in key for key in state)
    has_token_gate = any(".gate_size.W_k.weight" in key for key in state)
    has_source_bias = any(".local_stat_scale" in key for key in state)
    missing_components = [
        name
        for name, present in (
            ("ITGCA modality gate", has_itgca),
            ("r_stat correction path", has_r_stat),
            ("token gate", has_token_gate),
            ("source-side entropy bias", has_source_bias),
        )
        if not present
    ]
    if missing_components:
        raise ValueError(
            "--itgca-checkpoint is not a full ITGCA checkpoint; missing: "
            + ", ".join(missing_components)
        )

    args.num_fusion_layers = fusion_depth
    args.use_itgca = True
    args.itgca_window_size = itgca_window_size
    args.ablate_r_stat = False
    args.ablate_g_token = False
    args.ablate_source_bias = False
    args.use_msd = False  # MSD changes training-time sampling only, not eval parameters.
    args.msd_num = 1
    if old:
        args.use_attn_pooling = use_attn_pooling
        args.use_scl = use_scl

    _validate_config_depth(
        state,
        "encoder_raw.transformer.",
        int(args.layers_num_raw),
        "ITGCA raw encoder",
    )
    _validate_config_depth(
        state,
        "encoder_size.transformer.",
        int(args.layers_num_size),
        "ITGCA size encoder",
    )

    stage2_classifier = _get_stage2_classifier(old)
    model = stage2_classifier(args, *vocab_lengths, labels_num)
    checkpoint_name = "legacy full ITGCA" if old else "full ITGCA"
    _strict_load(model, state, checkpoint_name)
    return model


def _to_device(array: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array[indices])
    return torch.from_numpy(contiguous).to(device=device, dtype=torch.long)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[1] if output[0] is None else output[0]
    return output


class GateCollector:
    """Collect the full ITGCA Size<-Raw modality gate without changing forward."""

    def __init__(self, model: Stage2Classifier, flow_count: int):
        self.layers = list(model.fusion.fusion_layers)
        self.layer_count = len(self.layers)
        self.r_stat = np.full(flow_count, np.nan, dtype=np.float64)
        self.r_calibrated = np.full(
            (flow_count, self.layer_count), np.nan, dtype=np.float64
        )
        self.r_learned = np.full_like(self.r_calibrated, np.nan)
        self.r_mod = np.full_like(self.r_calibrated, np.nan)
        self.betas = np.empty(self.layer_count, dtype=np.float64)
        self._current_indices: Optional[np.ndarray] = None
        self._current_r_stat: Optional[torch.Tensor] = None
        self._current_layer_values: Optional[List[Any]] = None
        self._validated_layers = np.zeros(self.layer_count, dtype=bool)
        self._handles: List[Any] = []

        for layer_index, layer in enumerate(self.layers):
            gate_module = layer.gate_size
            self.betas[layer_index] = float(
                torch.sigmoid(gate_module.alpha_modality.detach()).cpu().item()
            )
            self._handles.append(
                gate_module.register_forward_hook(self._make_hook(layer_index))
            )

    def begin_batch(self, indices: np.ndarray, r_stat: torch.Tensor) -> None:
        self._current_indices = np.asarray(indices, dtype=np.int64)
        self._current_r_stat = r_stat.detach()
        self._current_layer_values = [None] * self.layer_count

    def end_batch(self) -> None:
        if (
            self._current_indices is None
            or self._current_r_stat is None
            or self._current_layer_values is None
        ):
            raise RuntimeError("cannot finish an inactive ITGCA diagnostic batch")
        if any(value is None for value in self._current_layer_values):
            raise RuntimeError("one or more ITGCA gate hooks did not run")

        indices = self._current_indices
        calibrated = torch.stack(
            [value[0] for value in self._current_layer_values], dim=1
        )
        learned = torch.stack(
            [value[1] for value in self._current_layer_values], dim=1
        )
        r_mod = torch.stack(
            [value[2] for value in self._current_layer_values], dim=1
        )
        self.r_stat[indices] = self._current_r_stat.cpu().numpy()
        self.r_calibrated[indices] = calibrated.cpu().numpy()
        self.r_learned[indices] = learned.cpu().numpy()
        self.r_mod[indices] = r_mod.cpu().numpy()

        self.abort_batch()

    def abort_batch(self) -> None:
        self._current_indices = None
        self._current_r_stat = None
        self._current_layer_values = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.layers.clear()

    def _make_hook(self, layer_index: int):
        def hook(module, inputs, output):
            if self._current_indices is None or self._current_r_stat is None:
                raise RuntimeError("ITGCA gate hook fired outside an inference batch")
            if len(inputs) < 4:
                raise RuntimeError("unexpected ITGCA gate input signature")

            encoder_cls_q = inputs[2]
            encoder_cls_k = inputs[3]
            r_stat = self._current_r_stat.to(
                device=encoder_cls_q.device, dtype=encoder_cls_q.dtype
            )
            r_logit = torch.sum(
                torch.matmul(encoder_cls_q, module.bilinear_W) * encoder_cls_k,
                dim=-1,
            ) + module.bilinear_bias.squeeze()
            r_learned = torch.sigmoid(r_logit)
            r_calibrated = torch.sigmoid(module.stat_scale * r_stat + module.stat_shift)
            beta = torch.sigmoid(module.alpha_modality)
            expected_r_mod = r_calibrated + beta * (r_learned - r_calibrated)
            r_mod = output[1]

            # One first-batch assertion per layer catches formula drift without
            # forcing a GPU synchronization for every layer of every batch.
            if not self._validated_layers[layer_index]:
                if not torch.allclose(expected_r_mod, r_mod, atol=1e-5, rtol=1e-5):
                    raise RuntimeError(
                        f"ITGCA layer {layer_index} r_mod does not match the current gate formula"
                    )
                self._validated_layers[layer_index] = True

            if self._current_layer_values is None:
                raise RuntimeError("missing active storage for ITGCA gate diagnostics")
            self._current_layer_values[layer_index] = (
                r_calibrated.detach(),
                r_learned.detach(),
                r_mod.detach(),
            )

        return hook


def _run_model_logits(
    name: str,
    model: torch.nn.Module,
    modality: str,
    store: FeatureStore,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    raw_vocab_size: int,
    labels_num: int,
    quiet: bool,
    gate_collector: Optional[GateCollector] = None,
) -> np.ndarray:
    logits_result = np.full(
        (len(store.success), labels_num),
        np.nan,
        dtype=np.float64,
    )
    model.to(device).eval()
    total_batches = (len(indices) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(indices), batch_size), start=1):
            batch_indices = indices[start:start + batch_size]

            if modality == "raw":
                raw_src = _to_device(store.raw_src, batch_indices, device)
                packet_ids = _to_device(store.packet_ids, batch_indices, device)
                directions = _to_device(store.directions, batch_indices, device)
                output = model(raw_src, packet_ids, directions, None, None)
            elif modality == "both":
                raw_src = _to_device(store.raw_src, batch_indices, device)
                packet_ids = _to_device(store.packet_ids, batch_indices, device)
                directions = _to_device(store.directions, batch_indices, device)
                size_src = _to_device(store.size_src, batch_indices, device)
                iat_src = _to_device(store.iat_src, batch_indices, device)
                output = model(
                    raw_src,
                    packet_ids,
                    directions,
                    size_src,
                    iat_src,
                )
            elif modality == "size":
                size_src = _to_device(store.size_src, batch_indices, device)
                iat_src = _to_device(store.iat_src, batch_indices, device)
                output = model(None, None, None, size_src, iat_src)
            elif modality == "itgca":
                raw_src = _to_device(store.raw_src, batch_indices, device)
                packet_ids = _to_device(store.packet_ids, batch_indices, device)
                directions = _to_device(store.directions, batch_indices, device)
                size_src = _to_device(store.size_src, batch_indices, device)
                iat_src = _to_device(store.iat_src, batch_indices, device)
                if gate_collector is not None:
                    r_stat = compute_flow_reliability_raw(
                        raw_src, vocab_size=raw_vocab_size
                    )
                    gate_collector.begin_batch(batch_indices, r_stat)
                    try:
                        output = model(
                            raw_src, packet_ids, directions, size_src, iat_src
                        )
                    except Exception:
                        gate_collector.abort_batch()
                        raise
                    else:
                        gate_collector.end_batch()
                else:
                    output = model(
                        raw_src, packet_ids, directions, size_src, iat_src
                    )
            else:
                raise ValueError(f"unsupported modality: {modality}")

            logits = _extract_logits(output)
            if logits.ndim != 2 or logits.shape[1] != labels_num:
                raise ValueError(
                    f"{name} logits have shape {tuple(logits.shape)}; "
                    f"expected [batch, {labels_num}]"
                )
            logits_result[batch_indices] = (
                logits.detach().cpu().numpy().astype(np.float64)
            )

            if not quiet and (
                batch_number % 50 == 0 or batch_number == total_batches
            ):
                print(f"[inference:{name}] {batch_number}/{total_batches} batches")

    if not np.isfinite(logits_result[indices]).all():
        raise RuntimeError(f"{name} produced missing or non-finite logits")
    return logits_result


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
    """Fit one scalar temperature by bounded validation NLL minimization."""

    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("temperature fitting received misaligned logits and labels")
    if not np.isfinite(logits).all():
        raise ValueError("temperature fitting received non-finite logits")

    log_min = math.log(TEMPERATURE_MIN)
    log_max = math.log(TEMPERATURE_MAX)
    grid = np.linspace(log_min, log_max, TEMPERATURE_GRID_SIZE)
    grid = np.unique(np.concatenate([grid, np.asarray([0.0])]))
    objectives = np.asarray(
        [_mean_nll(logits, labels, math.exp(value)) for value in grid]
    )
    best_index = int(np.argmin(objectives))
    best_log_temperature = float(grid[best_index])
    best_nll = float(objectives[best_index])

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
    evaluated: np.ndarray,
    temperature: float,
) -> Dict[str, np.ndarray]:
    count = len(labels)
    outputs = {
        "pred": np.full(count, -1, dtype=np.int64),
        "correct": np.zeros(count, dtype=bool),
        "log_p_true": np.full(count, np.nan, dtype=np.float64),
        "p_true": np.full(count, np.nan, dtype=np.float64),
    }
    selected_logits = logits[evaluated]
    selected_labels = labels[evaluated]
    log_probs = _log_softmax_numpy(selected_logits, temperature)
    predictions = np.argmax(selected_logits, axis=1).astype(np.int64)
    log_p_true = log_probs[
        np.arange(len(selected_labels)),
        selected_labels,
    ]
    outputs["pred"][evaluated] = predictions
    outputs["correct"][evaluated] = predictions == selected_labels
    outputs["log_p_true"][evaluated] = log_p_true
    outputs["p_true"][evaluated] = np.exp(log_p_true)
    return outputs


@contextmanager
def _stat_only_intervention(model: torch.nn.Module):
    """Temporarily force beta~0 without changing any checkpoint weights on disk."""

    parameters = [
        layer.gate_size.alpha_modality
        for layer in model.fusion.fusion_layers
    ]
    originals = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter in parameters:
                parameter.fill_(STAT_ONLY_ALPHA)
        yield
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)


def _release_model(model: torch.nn.Module, device: torch.device) -> None:
    # Moving the model back to CPU releases its device allocations even before
    # the caller drops its final Python reference.
    model.to(torch.device("cpu"))
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _rowwise_nanmean(matrix: np.ndarray) -> np.ndarray:
    """Mean over finite layer values without warnings for failed-flow rows."""

    finite = np.isfinite(matrix)
    counts = finite.sum(axis=1)
    totals = np.where(finite, matrix, 0.0).sum(axis=1)
    result = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    result[valid] = totals[valid] / counts[valid]
    return result


def _nanmean(values: np.ndarray) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _nanmedian(values: np.ndarray) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _accuracy(correct: np.ndarray) -> Optional[float]:
    return float(np.mean(correct)) if len(correct) else None


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if not len(y_true):
        return None
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _group_definitions() -> List[Tuple[str, str, Any]]:
    return [
        ("all_confirmed", "All confirmed compressed flows", lambda e: True),
        ("e_eq_0", "e_i = 0", lambda e: e == 0.0),
        ("e_gt_0", "e_i > 0", lambda e: e > 0.0),
        ("e_gt_0_le_0_25", "0 < e_i <= 0.25", lambda e: 0.0 < e <= 0.25),
        (
            "e_gt_0_25_le_0_50",
            "0.25 < e_i <= 0.50",
            lambda e: 0.25 < e <= 0.50,
        ),
        ("e_gt_0_50_le_1", "0.50 < e_i <= 1.00", lambda e: 0.50 < e <= 1.0),
    ]


def _build_summary_rows(
    records: Sequence[Mapping[str, Any]],
    store: FeatureStore,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    gate: Mapping[str, np.ndarray],
    content_utility: np.ndarray,
    full_gain: np.ndarray,
    correction_gain: np.ndarray,
    content_helpful: np.ndarray,
    utility_recovered: np.ndarray,
    gate_supported_recovery: np.ndarray,
    content_opportunity: np.ndarray,
    itgca_rescue: np.ndarray,
    dataset: str,
    compression_level: Optional[int],
    evaluation_scope: str,
    mean_beta: float,
    checkpoint_paths: Mapping[str, Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    all_indices = np.arange(len(records), dtype=np.int64)

    for group_name, group_display, predicate in _group_definitions():
        selected = np.asarray(
            [index for index, record in enumerate(records) if predicate(record["e_i"])],
            dtype=np.int64,
        )
        evaluated = selected[store.success[selected]] if len(selected) else selected
        labels = store.labels[evaluated]

        concat_correct = predictions["concat"]["correct"][evaluated]
        behavior_correct = predictions["behavior"]["correct"][evaluated]
        itgca_correct = predictions["itgca"]["correct"][evaluated]
        stat_only_correct = predictions["stat_only"]["correct"][evaluated]
        concat_accuracy = _accuracy(concat_correct)
        behavior_accuracy = _accuracy(behavior_correct)
        itgca_accuracy = _accuracy(itgca_correct)
        stat_only_accuracy = _accuracy(stat_only_correct)

        opportunity = content_opportunity[evaluated]
        rescue = itgca_rescue[evaluated]
        opportunity_count = int(opportunity.sum())
        rescue_count = int(rescue.sum())
        helpful = content_helpful[evaluated]
        helpful_count = int(helpful.sum())
        helpful_indices = evaluated[helpful]
        helpful_gate_up = gate["delta_learned"][helpful_indices] > 0.0
        helpful_gate_up_count = int(helpful_gate_up.sum())
        recovered = utility_recovered[evaluated]
        recovered_count = int(recovered.sum())
        gate_supported = gate_supported_recovery[evaluated]
        gate_supported_count = int(gate_supported.sum())
        stat_only_improved = correction_gain[helpful_indices] > 0.0
        stat_only_improved_count = int(stat_only_improved.sum())

        model_entropy = np.asarray(
            [records[index]["model_entropy_bits"] for index in evaluated],
            dtype=np.float64,
        )

        rows.append({
            "dataset": dataset,
            "compression_level": compression_level,
            "evaluation_scope": evaluation_scope,
            "group": group_name,
            "group_display": group_display,
            "flow_count": int(len(selected)),
            "evaluated_flows": int(len(evaluated)),
            "failed_flows": int(len(selected) - len(evaluated)),
            "mean_model_entropy_bits": _nanmean(model_entropy),
            "mean_r_stat": _nanmean(gate["r_stat"][evaluated]),
            "concat_accuracy": concat_accuracy,
            "behavior_accuracy": behavior_accuracy,
            "itgca_accuracy": itgca_accuracy,
            "stat_only_accuracy": stat_only_accuracy,
            "concat_macro_f1": _macro_f1(
                labels, predictions["concat"]["pred"][evaluated]
            ),
            "behavior_macro_f1": _macro_f1(
                labels, predictions["behavior"]["pred"][evaluated]
            ),
            "itgca_macro_f1": _macro_f1(labels, predictions["itgca"]["pred"][evaluated]),
            "stat_only_macro_f1": _macro_f1(
                labels, predictions["stat_only"]["pred"][evaluated]
            ),
            "concat_minus_behavior_accuracy": (
                concat_accuracy - behavior_accuracy
                if concat_accuracy is not None and behavior_accuracy is not None
                else None
            ),
            "itgca_minus_behavior_accuracy": (
                itgca_accuracy - behavior_accuracy
                if itgca_accuracy is not None and behavior_accuracy is not None
                else None
            ),
            "itgca_minus_stat_only_accuracy": (
                itgca_accuracy - stat_only_accuracy
                if itgca_accuracy is not None and stat_only_accuracy is not None
                else None
            ),
            "mean_content_utility": _nanmean(content_utility[evaluated]),
            "median_content_utility": _nanmedian(content_utility[evaluated]),
            "content_helpful_count": helpful_count,
            "content_helpful_rate": (
                helpful_count / len(evaluated) if len(evaluated) else None
            ),
            "mean_beta": mean_beta,
            "mean_r_calibrated": _nanmean(gate["r_calibrated"][evaluated]),
            "mean_r_learned": _nanmean(gate["r_learned"][evaluated]),
            "mean_r_mod": _nanmean(gate["r_mod"][evaluated]),
            "mean_delta_learned": _nanmean(gate["delta_learned"][evaluated]),
            "mean_delta_mod": _nanmean(gate["delta_mod"][evaluated]),
            "helpful_gate_up_count": helpful_gate_up_count,
            "helpful_gate_up_rate": (
                helpful_gate_up_count / helpful_count if helpful_count else None
            ),
            "helpful_median_delta_learned": _nanmedian(
                gate["delta_learned"][helpful_indices]
            ),
            "helpful_median_delta_mod": _nanmedian(
                gate["delta_mod"][helpful_indices]
            ),
            "helpful_mean_full_gain": _nanmean(full_gain[helpful_indices]),
            "helpful_mean_correction_gain": _nanmean(
                correction_gain[helpful_indices]
            ),
            "helpful_stat_only_improvement_count": stat_only_improved_count,
            "helpful_stat_only_improvement_rate": (
                stat_only_improved_count / helpful_count if helpful_count else None
            ),
            "utility_recovery_count": recovered_count,
            "utility_recovery_rate": (
                recovered_count / helpful_count if helpful_count else None
            ),
            "gate_supported_recovery_count": gate_supported_count,
            "gate_supported_recovery_rate": (
                gate_supported_count / helpful_count if helpful_count else None
            ),
            "content_opportunity_count": opportunity_count,
            "itgca_rescue_count": rescue_count,
            "itgca_rescue_rate": (
                rescue_count / opportunity_count if opportunity_count else None
            ),
            "concat_checkpoint": checkpoint_paths["concat"].name,
            "behavior_checkpoint": checkpoint_paths["behavior"].name,
            "itgca_checkpoint": checkpoint_paths["itgca"].name,
        })

    # Guard against accidental changes to the mutually exclusive positive bins.
    positive_bins = rows[3:]
    positive_count = sum(int(row["flow_count"]) for row in positive_bins)
    if positive_count != int(rows[2]["flow_count"]):
        raise RuntimeError("positive e_i bins do not sum to the e_i > 0 group")
    if int(rows[1]["flow_count"]) + int(rows[2]["flow_count"]) != len(all_indices):
        raise RuntimeError("e_i = 0 and e_i > 0 groups do not cover all selected flows")
    return rows


def _atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _flow_output_rows(
    records: Sequence[Mapping[str, Any]],
    store: FeatureStore,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    gate: Mapping[str, np.ndarray],
    content_utility: np.ndarray,
    full_gain: np.ndarray,
    correction_gain: np.ndarray,
    utility_group: np.ndarray,
    content_helpful: np.ndarray,
    utility_threshold: float,
    content_opportunity: np.ndarray,
    itgca_rescue: np.ndarray,
    utility_recovered: np.ndarray,
    gate_supported_recovery: np.ndarray,
    id2label: Mapping[int, str],
    dataset: str,
    compression_level: Optional[int],
    evaluation_scope: str,
) -> Iterable[Dict[str, Any]]:
    for index, record in enumerate(records):
        success = bool(store.success[index])

        def prediction_label(model_name: str) -> Optional[str]:
            if not success:
                return None
            prediction_id = int(predictions[model_name]["pred"][index])
            return id2label.get(prediction_id, str(prediction_id))

        yield {
            "dataset": dataset,
            "compression_level": compression_level,
            "evaluation_scope": evaluation_scope,
            "label": record["label"],
            "relative_pcap": record["relative_pcap"],
            "e_i": record["e_i"],
            "e_bin": record["e_bin"],
            "model_entropy_bits": record["model_entropy_bits"],
            "audit_model_r_stat": record["audit_model_r_stat"],
            "inference_r_stat": gate["r_stat"][index] if success else None,
            "concat_pred": prediction_label("concat"),
            "concat_correct": (
                int(predictions["concat"]["correct"][index]) if success else None
            ),
            "concat_p_true": (
                predictions["concat"]["p_true"][index] if success else None
            ),
            "concat_log_p_true": (
                predictions["concat"]["log_p_true"][index] if success else None
            ),
            "behavior_pred": prediction_label("behavior"),
            "behavior_correct": (
                int(predictions["behavior"]["correct"][index]) if success else None
            ),
            "behavior_p_true": (
                predictions["behavior"]["p_true"][index] if success else None
            ),
            "behavior_log_p_true": (
                predictions["behavior"]["log_p_true"][index] if success else None
            ),
            "itgca_pred": prediction_label("itgca"),
            "itgca_correct": int(predictions["itgca"]["correct"][index]) if success else None,
            "itgca_p_true": predictions["itgca"]["p_true"][index] if success else None,
            "itgca_log_p_true": (
                predictions["itgca"]["log_p_true"][index] if success else None
            ),
            "stat_only_pred": prediction_label("stat_only"),
            "stat_only_correct": (
                int(predictions["stat_only"]["correct"][index]) if success else None
            ),
            "stat_only_p_true": (
                predictions["stat_only"]["p_true"][index] if success else None
            ),
            "stat_only_log_p_true": (
                predictions["stat_only"]["log_p_true"][index] if success else None
            ),
            "content_utility": content_utility[index] if success else None,
            "full_gain": full_gain[index] if success else None,
            "correction_gain": correction_gain[index] if success else None,
            "utility_threshold": utility_threshold,
            "utility_group": utility_group[index] if success else None,
            "content_helpful": int(content_helpful[index]) if success else None,
            "r_calibrated": gate["r_calibrated"][index] if success else None,
            "r_learned": gate["r_learned"][index] if success else None,
            "r_mod": gate["r_mod"][index] if success else None,
            "delta_learned": gate["delta_learned"][index] if success else None,
            "delta_mod": gate["delta_mod"][index] if success else None,
            "gate_up": (
                int(gate["delta_learned"][index] > 0.0) if success else None
            ),
            "content_opportunity": int(content_opportunity[index]) if success else None,
            "itgca_rescue": (
                int(itgca_rescue[index])
                if success and content_opportunity[index]
                else None
            ),
            "utility_recovered": (
                int(utility_recovered[index]) if success else None
            ),
            "gate_supported_recovery": (
                int(gate_supported_recovery[index]) if success else None
            ),
            "inference_status": "ok" if success else record["inference_status"],
            "error": "" if success else record["error"],
        }


def _gate_layer_rows(
    records: Sequence[Mapping[str, Any]],
    store: FeatureStore,
    collector: GateCollector,
    content_utility: np.ndarray,
    utility_group: np.ndarray,
    dataset: str,
    compression_level: Optional[int],
    evaluation_scope: str,
) -> Iterable[Dict[str, Any]]:
    for index, record in enumerate(records):
        if not store.success[index]:
            continue
        for layer_index in range(collector.layer_count):
            r_calibrated = float(collector.r_calibrated[index, layer_index])
            r_learned = float(collector.r_learned[index, layer_index])
            r_mod = float(collector.r_mod[index, layer_index])
            yield {
                "dataset": dataset,
                "compression_level": compression_level,
                "evaluation_scope": evaluation_scope,
                "label": record["label"],
                "relative_pcap": record["relative_pcap"],
                "e_i": record["e_i"],
                "e_bin": record["e_bin"],
                "layer_index": layer_index + 1,
                "beta": float(collector.betas[layer_index]),
                "r_stat": float(collector.r_stat[index]),
                "r_calibrated": r_calibrated,
                "r_learned": r_learned,
                "r_mod": r_mod,
                "delta_learned": r_learned - r_calibrated,
                "delta_mod": r_mod - r_calibrated,
                "content_utility": float(content_utility[index]),
                "utility_group": str(utility_group[index]),
            }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {device_arg}")
    return device


def _calibration_record(
    name: str,
    validation_logits: Optional[np.ndarray],
    validation_labels: Optional[np.ndarray],
    skip_temperature_scaling: bool,
    quiet: bool,
) -> Dict[str, Any]:
    if skip_temperature_scaling:
        record = {
            "temperature": 1.0,
            "validation_nll_before": None,
            "validation_nll_after": None,
            "temperature_at_search_boundary": False,
        }
    else:
        if validation_logits is None or validation_labels is None:
            raise RuntimeError(f"missing validation predictions for {name}")
        temperature, nll_before, nll_after, at_boundary = _fit_temperature(
            validation_logits,
            validation_labels,
        )
        record = {
            "temperature": temperature,
            "validation_nll_before": nll_before,
            "validation_nll_after": nll_after,
            "temperature_at_search_boundary": at_boundary,
        }

    if not quiet:
        if record["validation_nll_before"] is None:
            print(f"[temperature:{name}] T=1 (scaling skipped)")
        else:
            print(
                f"[temperature:{name}] T={record['temperature']:.8g} "
                f"NLL={record['validation_nll_before']:.8g}"
                f"->{record['validation_nll_after']:.8g}"
            )
        if record["temperature_at_search_boundary"]:
            print(
                f"WARNING: {name} temperature reached search boundary "
                f"[{TEMPERATURE_MIN}, {TEMPERATURE_MAX}]",
                file=sys.stderr,
            )
    return record


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure concat-vs-behavior content utility and ITGCA reliability "
            "correction on confirmed compressed flows."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("flow_details", type=Path, help="compression audit flow_details.csv")
    parser.add_argument("--concat-checkpoint", type=Path, required=True)
    parser.add_argument("--behavior-checkpoint", type=Path, required=True)
    parser.add_argument("--itgca-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=None,
        help=(
            "processed validation.pkl used to fit four scalar temperatures; "
            "required unless --skip-temperature-scaling is used"
        ),
    )
    parser.add_argument("--label2id-path", type=Path, required=True)

    parser.add_argument(
        "--raw-config",
        type=Path,
        default=PROJECT_ROOT / "models/bert/base_config.json",
        help="raw encoder and shared architecture config for concat/full models",
    )
    parser.add_argument(
        "--behavior-config",
        type=Path,
        default=PROJECT_ROOT / "models/bert/behavior_6_config.json",
        help="behavior encoder config shared by concat/behavior/full models",
    )
    parser.add_argument("--itgca-window-size", type=int, default=16)
    parser.add_argument(
        "--old",
        action="store_true",
        help=(
            "use the legacy classifier implementation from "
            "run_classifier_stage2_old.py for the full checkpoint"
        ),
    )
    parser.add_argument(
        "--use-attn-pooling",
        "--use_attn_pooling",
        dest="use_attn_pooling",
        action="store_true",
        help="with --old, enable the legacy attention-pooling modules",
    )
    parser.add_argument(
        "--use-scl",
        "--use_scl",
        dest="use_scl",
        action="store_true",
        help="with --old, enable the legacy SCL projection head",
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
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="fallback root used with relative_pcap when audit pcap_path is stale",
    )
    parser.add_argument("--seq-length-raw", type=int, default=DEFAULT_SEQ_LENGTH_RAW)
    parser.add_argument("--seq-length-size", type=int, default=DEFAULT_SEQ_LENGTH_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--utility-threshold",
        type=float,
        default=DEFAULT_UTILITY_THRESHOLD,
        help=(
            "epsilon for helpful (u_i>epsilon), near-zero "
            "(|u_i|<=epsilon), and harmful (u_i<-epsilon) groups"
        ),
    )
    parser.add_argument(
        "--skip-temperature-scaling",
        action="store_true",
        help="use T=1 for all predictors; not recommended for log-score comparisons",
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=("audit_directory_diagnostic", "heldout_test", "clean_unseen"),
        default="audit_directory_diagnostic",
        help="provenance label written to both result files",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing result files in the output directory",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    args = parser.parse_args(argv)
    if (args.use_attn_pooling or args.use_scl) and not args.old:
        parser.error("--use-attn-pooling and --use-scl require --old")
    if args.validation_path is None and not args.skip_temperature_scaling:
        parser.error(
            "--validation-path is required unless --skip-temperature-scaling is used"
        )
    return args


def _validate_paths(args: argparse.Namespace) -> None:
    required_files = {
        "flow_details": args.flow_details,
        "concat checkpoint": args.concat_checkpoint,
        "behavior checkpoint": args.behavior_checkpoint,
        "ITGCA checkpoint": args.itgca_checkpoint,
        "label2id": args.label2id_path,
        "raw config": args.raw_config,
        "behavior config": args.behavior_config,
        "raw vocabulary": args.vocab_path_raw,
        "size vocabulary": args.vocab_path_size,
        "temporal vocabulary": args.vocab_path_temporal,
    }
    if args.validation_path is not None:
        required_files["validation split"] = args.validation_path
    missing = [f"{name}: {path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise ValueError("required files not found:\n  " + "\n  ".join(missing))
    if args.dataset_dir is not None and not args.dataset_dir.is_dir():
        raise ValueError(f"dataset directory not found: {args.dataset_dir}")
    for name in ("seq_length_raw", "seq_length_size", "batch_size", "workers", "chunk_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.itgca_window_size <= 0:
        raise ValueError("--itgca-window-size must be positive")
    if not math.isfinite(args.utility_threshold) or args.utility_threshold < 0.0:
        raise ValueError("--utility-threshold must be finite and non-negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.flow_details = args.flow_details.expanduser().resolve()
    args.concat_checkpoint = args.concat_checkpoint.expanduser().resolve()
    args.behavior_checkpoint = args.behavior_checkpoint.expanduser().resolve()
    args.itgca_checkpoint = args.itgca_checkpoint.expanduser().resolve()
    args.validation_path = (
        args.validation_path.expanduser().resolve()
        if args.validation_path is not None
        else None
    )
    args.label2id_path = args.label2id_path.expanduser().resolve()
    args.raw_config = args.raw_config.expanduser().resolve()
    args.behavior_config = args.behavior_config.expanduser().resolve()
    args.vocab_path_raw = args.vocab_path_raw.expanduser().resolve()
    args.vocab_path_size = args.vocab_path_size.expanduser().resolve()
    args.vocab_path_temporal = args.vocab_path_temporal.expanduser().resolve()
    args.dataset_dir = (
        args.dataset_dir.expanduser().resolve() if args.dataset_dir else None
    )

    metadata = _read_audit_metadata(args.flow_details)
    if args.dataset_dir is None:
        metadata_dataset = metadata.get("dataset_path")
        if metadata_dataset and Path(metadata_dataset).expanduser().is_dir():
            args.dataset_dir = Path(metadata_dataset).expanduser().resolve()

    _validate_paths(args)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.flow_details.parent / "checkpoint_inference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "flows": output_dir / "flow_results.csv",
        "summary": output_dir / "summary.csv",
        "layers": output_dir / "gate_layers.csv",
        "calibration": output_dir / "calibration.json",
    }
    existing_outputs = [path for path in output_paths.values() if path.exists()]
    if existing_outputs and not args.overwrite:
        raise ValueError(
            "output file(s) already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    label2id = _load_pickle(args.label2id_path)
    if not isinstance(label2id, dict) or not label2id:
        raise ValueError("label2id must be a non-empty dictionary")
    label2id = {str(name): int(label_id) for name, label_id in label2id.items()}
    expected_ids = set(range(len(label2id)))
    if set(label2id.values()) != expected_ids:
        raise ValueError("label2id IDs must be unique and contiguous from 0")
    id2label = {label_id: name for name, label_id in label2id.items()}

    records = _read_confirmed_rows(args.flow_details, args.dataset_dir)
    dataset = next(
        (record["dataset"] for record in records if record["dataset"]),
        args.flow_details.parent.name,
    )
    compression_level = _as_int(
        metadata.get("compression_scope", {}).get("selected_level")
        if isinstance(metadata.get("compression_scope"), dict)
        else None
    )

    if not args.quiet:
        print(f"[version] {SCRIPT_VERSION}")
        print(f"[audit] dataset={dataset} confirmed_compressed={len(records)}")
        print(f"[scope] {args.evaluation_scope}")
        if args.evaluation_scope == "audit_directory_diagnostic":
            print(
                "WARNING: audit_directory_diagnostic may include training flows; "
                "do not present its accuracy as held-out performance.",
                file=sys.stderr,
            )

    store, vocab_lengths = _prepare_features(
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
        raise ValueError("none of the selected audit flows could be preprocessed")
    validation_store = (
        _feature_store_from_pickle(
            args.validation_path,
            "validation",
            len(label2id),
            args.seq_length_raw,
            args.seq_length_size,
        )
        if args.validation_path is not None
        else None
    )
    validation_indices = (
        np.flatnonzero(validation_store.success)
        if validation_store is not None
        else np.empty(0, dtype=np.int64)
    )
    validation_labels = (
        validation_store.labels
        if validation_store is not None
        else None
    )

    device = _resolve_device(args.device)
    if not args.quiet:
        print(
            f"[features] ready={len(successful_indices)} "
            f"failed={len(records) - len(successful_indices)}"
        )
        print(f"[device] {device}")

    audit_logits: Dict[str, np.ndarray] = {}
    validation_logits: Dict[str, np.ndarray] = {}
    calibration: Dict[str, Dict[str, Any]] = {}

    concat_model = _build_stage1_model(
        "both",
        args.concat_checkpoint,
        args.raw_config,
        args.behavior_config,
        vocab_lengths,
        len(label2id),
        args.seq_length_raw,
        args.seq_length_size,
    )
    if validation_store is not None:
        validation_logits["concat"] = _run_model_logits(
            "concat-validation",
            concat_model,
            "both",
            validation_store,
            validation_indices,
            args.batch_size,
            device,
            vocab_lengths[0],
            len(label2id),
            args.quiet,
        )
    audit_logits["concat"] = _run_model_logits(
        "concat",
        concat_model,
        "both",
        store,
        successful_indices,
        args.batch_size,
        device,
        vocab_lengths[0],
        len(label2id),
        args.quiet,
    )
    _release_model(concat_model, device)
    del concat_model
    calibration["concat"] = _calibration_record(
        "concat",
        (
            validation_logits["concat"][validation_indices]
            if validation_store is not None
            else None
        ),
        (
            validation_labels[validation_indices]
            if validation_labels is not None
            else None
        ),
        args.skip_temperature_scaling,
        args.quiet,
    )

    behavior_model = _build_stage1_model(
        "size",
        args.behavior_checkpoint,
        args.raw_config,
        args.behavior_config,
        vocab_lengths,
        len(label2id),
        args.seq_length_raw,
        args.seq_length_size,
    )
    if validation_store is not None:
        validation_logits["behavior"] = _run_model_logits(
            "behavior-validation",
            behavior_model,
            "size",
            validation_store,
            validation_indices,
            args.batch_size,
            device,
            vocab_lengths[0],
            len(label2id),
            args.quiet,
        )
    audit_logits["behavior"] = _run_model_logits(
        "behavior",
        behavior_model,
        "size",
        store,
        successful_indices,
        args.batch_size,
        device,
        vocab_lengths[0],
        len(label2id),
        args.quiet,
    )
    _release_model(behavior_model, device)
    del behavior_model
    calibration["behavior"] = _calibration_record(
        "behavior",
        (
            validation_logits["behavior"][validation_indices]
            if validation_store is not None
            else None
        ),
        (
            validation_labels[validation_indices]
            if validation_labels is not None
            else None
        ),
        args.skip_temperature_scaling,
        args.quiet,
    )

    itgca_model = _build_itgca_model(
        args.itgca_checkpoint,
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
    if validation_store is not None:
        validation_logits["itgca"] = _run_model_logits(
            "itgca-validation",
            itgca_model,
            "itgca",
            validation_store,
            validation_indices,
            args.batch_size,
            device,
            vocab_lengths[0],
            len(label2id),
            args.quiet,
        )
    collector = GateCollector(itgca_model, len(records))
    try:
        audit_logits["itgca"] = _run_model_logits(
            "itgca",
            itgca_model,
            "itgca",
            store,
            successful_indices,
            args.batch_size,
            device,
            vocab_lengths[0],
            len(label2id),
            args.quiet,
            gate_collector=collector,
        )
    finally:
        collector.close()
    calibration["itgca"] = _calibration_record(
        "itgca",
        (
            validation_logits["itgca"][validation_indices]
            if validation_store is not None
            else None
        ),
        (
            validation_labels[validation_indices]
            if validation_labels is not None
            else None
        ),
        args.skip_temperature_scaling,
        args.quiet,
    )

    with _stat_only_intervention(itgca_model):
        if validation_store is not None:
            validation_logits["stat_only"] = _run_model_logits(
                "stat-only-validation",
                itgca_model,
                "itgca",
                validation_store,
                validation_indices,
                args.batch_size,
                device,
                vocab_lengths[0],
                len(label2id),
                args.quiet,
            )
        audit_logits["stat_only"] = _run_model_logits(
            "stat-only",
            itgca_model,
            "itgca",
            store,
            successful_indices,
            args.batch_size,
            device,
            vocab_lengths[0],
            len(label2id),
            args.quiet,
        )
    calibration["stat_only"] = _calibration_record(
        "stat_only",
        (
            validation_logits["stat_only"][validation_indices]
            if validation_store is not None
            else None
        ),
        (
            validation_labels[validation_indices]
            if validation_labels is not None
            else None
        ),
        args.skip_temperature_scaling,
        args.quiet,
    )
    _release_model(itgca_model, device)
    del itgca_model

    predictions = {
        name: _calibrated_outputs(
            logits,
            store.labels,
            successful_indices,
            float(calibration[name]["temperature"]),
        )
        for name, logits in audit_logits.items()
    }

    for name, matrix in (
        ("r_calibrated", collector.r_calibrated),
        ("r_learned", collector.r_learned),
        ("r_mod", collector.r_mod),
    ):
        if not np.isfinite(matrix[successful_indices]).all():
            raise RuntimeError(f"missing {name} values for one or more inferred flows")

    gate = {
        "r_stat": collector.r_stat,
        "r_calibrated": _rowwise_nanmean(collector.r_calibrated),
        "r_learned": _rowwise_nanmean(collector.r_learned),
        "r_mod": _rowwise_nanmean(collector.r_mod),
    }
    gate["delta_learned"] = gate["r_learned"] - gate["r_calibrated"]
    gate["delta_mod"] = gate["r_mod"] - gate["r_calibrated"]

    audit_r_stat = np.asarray(
        [record["audit_model_r_stat"] for record in records], dtype=np.float64
    )
    comparable = store.success & np.isfinite(audit_r_stat) & np.isfinite(gate["r_stat"])
    if comparable.any():
        max_r_stat_difference = float(
            np.max(np.abs(audit_r_stat[comparable] - gate["r_stat"][comparable]))
        )
        if max_r_stat_difference > 1e-5:
            raise ValueError(
                "audit/inference r_stat mismatch "
                f"(max absolute difference={max_r_stat_difference:.6g}). "
                "Check the raw vocabulary and window configuration."
            )
        if not args.quiet:
            print(f"[validate] max |audit r_stat - inference r_stat|={max_r_stat_difference:.3g}")

    content_utility = np.full(len(records), np.nan, dtype=np.float64)
    full_gain = np.full(len(records), np.nan, dtype=np.float64)
    correction_gain = np.full(len(records), np.nan, dtype=np.float64)
    content_utility[successful_indices] = (
        predictions["concat"]["log_p_true"][successful_indices]
        - predictions["behavior"]["log_p_true"][successful_indices]
    )
    full_gain[successful_indices] = (
        predictions["itgca"]["log_p_true"][successful_indices]
        - predictions["behavior"]["log_p_true"][successful_indices]
    )
    correction_gain[successful_indices] = (
        predictions["itgca"]["log_p_true"][successful_indices]
        - predictions["stat_only"]["log_p_true"][successful_indices]
    )
    utility_group = np.full(len(records), "", dtype=object)
    utility_group[successful_indices] = np.select(
        [
            content_utility[successful_indices] > args.utility_threshold,
            content_utility[successful_indices] < -args.utility_threshold,
        ],
        ["helpful", "harmful"],
        default="near_zero",
    )
    content_helpful = store.success & (
        content_utility > args.utility_threshold
    )
    content_opportunity = (
        store.success
        & predictions["concat"]["correct"]
        & ~predictions["behavior"]["correct"]
    )
    itgca_rescue = content_opportunity & predictions["itgca"]["correct"]
    utility_recovered = content_helpful & (full_gain > 0.0)
    gate_supported_recovery = (
        utility_recovered & (gate["delta_learned"] > 0.0)
    )

    checkpoint_paths = {
        "concat": args.concat_checkpoint,
        "behavior": args.behavior_checkpoint,
        "itgca": args.itgca_checkpoint,
    }
    summary_rows = _build_summary_rows(
        records,
        store,
        predictions,
        gate,
        content_utility,
        full_gain,
        correction_gain,
        content_helpful,
        utility_recovered,
        gate_supported_recovery,
        content_opportunity,
        itgca_rescue,
        dataset,
        compression_level,
        args.evaluation_scope,
        float(collector.betas.mean()),
        checkpoint_paths,
    )

    _atomic_write_csv(
        output_paths["flows"],
        _flow_output_rows(
            records,
            store,
            predictions,
            gate,
            content_utility,
            full_gain,
            correction_gain,
            utility_group,
            content_helpful,
            args.utility_threshold,
            content_opportunity,
            itgca_rescue,
            utility_recovered,
            gate_supported_recovery,
            id2label,
            dataset,
            compression_level,
            args.evaluation_scope,
        ),
        FLOW_OUTPUT_FIELDS,
    )
    _atomic_write_csv(output_paths["summary"], summary_rows, SUMMARY_FIELDS)
    _atomic_write_csv(
        output_paths["layers"],
        _gate_layer_rows(
            records,
            store,
            collector,
            content_utility,
            utility_group,
            dataset,
            compression_level,
            args.evaluation_scope,
        ),
        GATE_LAYER_FIELDS,
    )
    _atomic_write_json(
        output_paths["calibration"],
        {
            "script": str(Path(__file__).resolve()),
            "script_version": SCRIPT_VERSION,
            "validation_path": (
                str(args.validation_path) if args.validation_path is not None else None
            ),
            "configs": {
                "raw": str(args.raw_config),
                "behavior": str(args.behavior_config),
                "policy": (
                    "the same two configs are used for concat, behavior-only, "
                    "and full ITGCA checkpoint construction"
                ),
            },
            "inferred_fusion_layers": collector.layer_count,
            "temperature_scaling_skipped": args.skip_temperature_scaling,
            "utility_threshold": args.utility_threshold,
            "stat_only_alpha": STAT_ONLY_ALPHA,
            "models": calibration,
        },
    )

    for path in output_paths.values():
        print(f"[write] {path}")
    print(
        f"[done] selected={len(records)} evaluated={len(successful_indices)} "
        f"failed={len(records) - len(successful_indices)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
