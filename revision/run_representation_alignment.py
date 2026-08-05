#!/usr/bin/env python3
"""Extract representations for a Stage-1 motivation and CMC-effect analysis.

This is the end-to-end driver for the Challenge-1 motivating figure.  It starts
from the artifacts already produced by MM-TrafficBERT:

* a Stage-1 content checkpoint;
* a Stage-1 behavior checkpoint;
* a Stage-2 *pre-training* checkpoint; and
* a processed fine-tuning pickle (for example ``test.pkl``).

No training or model update is performed.  The script loads only the online
content/behavior encoders and the two CMC projection heads; it does not execute
fusion, momentum encoders, queues, reconstruction heads, or a classifier.

For the Stage-1 motivation, the script compares only the within-modality
geometry produced by the two independently pretrained encoders.  Separately,
it loads the Stage-2 online encoders and learned CMC projections to produce a
before/after alignment-effect diagnostic.  Both analyses process the exact
same deterministic sample of flows.

Outputs
-------
The output directory contains:

``before_alignment.npz`` / ``after_alignment.npz``
    Projected representations under keys ``content`` and ``behavior``.  Raw
    pre-fusion CLS representations are also retained as ``content_cls`` and
    ``behavior_cls`` so alternative plots do not require another model pass.

``representation_geometry_motivation.png`` / ``.pdf``
    The Stage-1-only motivation figure: content geometry, behavior geometry,
    and their normalized disagreement for the same flows.

``representation_alignment_similarity.png`` / ``.pdf``
    A separate two-panel CMC-effect diagnostic.  The left panel uses Stage-1
    encoder CLS features, while the right panel uses learned Stage-2 CMC
    projection features.  Diagonal cells represent paired views of one flow.

Example
-------
Run from the project root::

    python3 revision/run_representation_alignment.py \\
        --stage1-raw-checkpoint /path/to/raw_encoder.bin \\
        --stage1-behavior-checkpoint /path/to/size_encoder.bin \\
        --stage2-checkpoint /path/to/mm_trafficbert.bin \\
        --data-path /path/to/test.pkl \\
        --config-path-raw models/bert/base_config.json \\
        --config-path-behavior models/bert/behavior_6_config.json \\
        --device cuda:0

The processed pickle is expected to be the list-of-dictionaries written by
``fine-tuning/multimodal_data_utils.py``.  Each record must contain
``raw_src``, ``packet_ids``, ``directions``, ``size_src``, and ``iat_src``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FINE_TUNING_DIR = PROJECT_ROOT / "fine-tuning"
for import_path in (PROJECT_ROOT, FINE_TUNING_DIR):
    path_string = str(import_path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from uer.encoders import str2encoder  # noqa: E402
from uer.layers import PacketSizeEmbedding, RawPacketEmbedding  # noqa: E402
from uer.opts import model_opts  # noqa: E402
from uer.utils.config import apply_modality_configs, load_hyperparam  # noqa: E402
from uer.utils.constants import PAD_ID  # noqa: E402
from uer.utils.vocab import Vocab  # noqa: E402


REQUIRED_RECORD_KEYS = (
    "raw_src",
    "packet_ids",
    "directions",
    "size_src",
    "iat_src",
)


class AlignmentEncoder(nn.Module):
    """Only the online unimodal encoders and CMC projection heads."""

    def __init__(
        self,
        args: Namespace,
        vocab_size_raw: int,
        vocab_size_behavior: int,
        vocab_size_temporal: int,
    ) -> None:
        super().__init__()
        base_layers = int(args.layers_num)
        raw_layers = int(getattr(args, "layers_num_raw", base_layers) or base_layers)
        behavior_layers = int(
            getattr(args, "layers_num_size", base_layers) or base_layers
        )

        self.embedding_raw = RawPacketEmbedding(args, vocab_size_raw)
        args.layers_num = raw_layers
        self.encoder_raw = str2encoder[args.encoder](args)

        self.embedding_behavior = PacketSizeEmbedding(
            args,
            vocab_size_behavior,
            vocab_size_temporal,
        )
        args.layers_num = behavior_layers
        self.encoder_behavior = str2encoder[args.encoder](args)
        args.layers_num = base_layers

        self.proj_raw = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_behavior = nn.Linear(args.hidden_size, args.hidden_size)

    def initialize_cmc_projections(self, std: float = 0.02) -> None:
        """Match the blanket Stage-2 initialization used by uer.trainer."""
        for projection in (self.proj_raw, self.proj_behavior):
            for parameter in projection.parameters():
                parameter.data.normal_(0.0, std)

    def forward(
        self,
        raw_src: torch.Tensor,
        packet_ids: torch.Tensor,
        directions: torch.Tensor,
        size_src: torch.Tensor,
        iat_src: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_seg = (raw_src != PAD_ID).long()
        raw_embedding = self.embedding_raw(raw_src, packet_ids, directions)
        raw_output = self.encoder_raw(raw_embedding, raw_seg)

        behavior_seg = (size_src != PAD_ID).long()
        behavior_embedding = self.embedding_behavior(size_src, iat_src)
        behavior_output = self.encoder_behavior(behavior_embedding, behavior_seg)

        raw_cls = raw_output[:, 0, :]
        behavior_cls = behavior_output[:, 0, :]
        raw_projected = F.normalize(self.proj_raw(raw_cls), dim=-1)
        behavior_projected = F.normalize(
            self.proj_behavior(behavior_cls),
            dim=-1,
        )
        return raw_cls, behavior_cls, raw_projected, behavior_projected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    artifact_group = parser.add_argument_group("existing model and data artifacts")
    artifact_group.add_argument(
        "--stage1-raw-checkpoint",
        type=Path,
        required=True,
        help="Stage-1 raw/content pre-training checkpoint.",
    )
    artifact_group.add_argument(
        "--stage1-behavior-checkpoint",
        type=Path,
        required=True,
        help="Stage-1 size/IAT behavior pre-training checkpoint.",
    )
    artifact_group.add_argument(
        "--stage2-checkpoint",
        type=Path,
        required=True,
        help=(
            "Stage-2 pre-training checkpoint containing target.itc_proj_*; "
            "a fine-tuned classifier checkpoint is insufficient."
        ),
    )
    artifact_group.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Processed train/val/test pickle from multimodal_data_utils.py.",
    )

    config_group = parser.add_argument_group("model configuration")
    config_group.add_argument(
        "--config-path-raw",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "base_config.json",
    )
    config_group.add_argument(
        "--config-path-behavior",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "behavior_6_config.json",
    )
    config_group.add_argument(
        "--vocab-path-raw",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "vocab_raw.txt",
    )
    config_group.add_argument(
        "--vocab-path-behavior",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "vocab_size.txt",
    )
    config_group.add_argument(
        "--vocab-path-temporal",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "vocab_temporal.txt",
    )
    config_group.add_argument("--seq-length-raw", type=int, default=512)
    config_group.add_argument("--seq-length-behavior", type=int, default=256)
    model_opts(parser)

    extraction_group = parser.add_argument_group("embedding extraction")
    extraction_group.add_argument(
        "--max-flows",
        type=int,
        default=2000,
        help="Deterministically sample at most this many flows; 0 means all.",
    )
    extraction_group.add_argument("--batch-size", type=int, default=64)
    extraction_group.add_argument(
        "--sample-seed",
        type=int,
        default=20260803,
        help="Seed for selecting flows from the processed dataset.",
    )
    extraction_group.add_argument(
        "--model-init-seed",
        type=int,
        default=7,
        help="Seed for deterministic model construction.",
    )
    extraction_group.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cuda:0 or cpu; auto prefers CUDA.",
    )

    output_group = parser.add_argument_group("intermediate and final outputs")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "representation_alignment",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing NPZ/figure outputs.",
    )
    output_group.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Skip model loading when both output NPZ files already exist.",
    )
    output_group.add_argument(
        "--skip-plot",
        action="store_true",
        help="Only generate the two NPZ intermediate files.",
    )

    plot_group = parser.add_argument_group("motivation and diagnostic figures")
    plot_group.add_argument("--plot-sample-size", type=int, default=48)
    plot_group.add_argument("--plot-seed", type=int, default=20260803)
    plot_group.add_argument(
        "--geometry-colormap",
        default="RdBu_r",
        help="Colormap shared by the two Stage-1 geometry panels.",
    )
    plot_group.add_argument(
        "--difference-colormap",
        default="magma",
        help="Colormap for normalized Stage-1 geometry disagreement.",
    )
    plot_group.add_argument(
        "--colormap",
        default="magma",
        help="Colormap for the separate CMC-effect diagnostic.",
    )
    plot_group.add_argument(
        "--similarity-vmin",
        type=float,
        default=None,
        help="Optional shared lower cosine-similarity limit.",
    )
    plot_group.add_argument(
        "--similarity-vmax",
        type=float,
        default=None,
        help="Optional shared upper cosine-similarity limit.",
    )
    plot_group.add_argument("--font-scale", type=float, default=1.0)
    plot_group.add_argument("--fig-width", type=float, default=7.1)
    plot_group.add_argument("--fig-height", type=float, default=2.75)
    plot_group.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    required_paths = (
        args.stage1_raw_checkpoint,
        args.stage1_behavior_checkpoint,
        args.stage2_checkpoint,
        args.data_path,
        args.config_path_raw,
        args.config_path_behavior,
        args.vocab_path_raw,
        args.vocab_path_behavior,
        args.vocab_path_temporal,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        parser.error("required files do not exist: " + ", ".join(missing))
    if args.max_flows < 0:
        parser.error("--max-flows must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.seq_length_raw <= 0 or args.seq_length_behavior <= 0:
        parser.error("sequence lengths must be positive")
    if args.plot_sample_size < 0 or 0 < args.plot_sample_size < 3:
        parser.error("--plot-sample-size must be 0 or at least 3")
    if args.similarity_vmin is not None and not -1.0 <= args.similarity_vmin <= 1.0:
        parser.error("--similarity-vmin must lie in [-1, 1]")
    if args.similarity_vmax is not None and not -1.0 <= args.similarity_vmax <= 1.0:
        parser.error("--similarity-vmax must lie in [-1, 1]")
    if (
        args.similarity_vmin is not None
        and args.similarity_vmax is not None
        and args.similarity_vmin >= args.similarity_vmax
    ):
        parser.error("--similarity-vmin must be smaller than --similarity-vmax")
    if args.font_scale <= 0.0:
        parser.error("--font-scale must be positive")
    if args.fig_width <= 0.0 or args.fig_height <= 0.0:
        parser.error("figure dimensions must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.reuse_embeddings and args.overwrite:
        parser.error("--reuse-embeddings and --overwrite are mutually exclusive")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def load_vocab_size(path: Path) -> int:
    vocab = Vocab()
    vocab.load(str(path))
    return len(vocab)


def build_model_args(args: argparse.Namespace) -> Namespace:
    model_args = Namespace(**vars(args))
    model_args.config_path = str(args.config_path_raw)
    model_args.config_path_raw = str(args.config_path_raw)
    model_args.config_path_size = str(args.config_path_behavior)
    model_args = load_hyperparam(model_args)
    model_args = apply_modality_configs(model_args)
    model_args.max_seq_length = max(
        args.seq_length_raw,
        args.seq_length_behavior,
        int(getattr(model_args, "max_seq_length", 0)),
    )
    return model_args


def safe_torch_load(path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError(f"checkpoint is not a non-empty state dict: {path}")

    state = dict(checkpoint)
    for prefix in ("module.", "_orig_mod."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix):]: value for key, value in state.items()}
    non_tensor = [str(key) for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensor:
        raise ValueError(
            f"checkpoint contains non-tensor state entries: {path}; "
            f"first keys: {non_tensor[:5]}"
        )
    return state


def prefixed_state(
    state: Mapping[str, torch.Tensor],
    prefix: str,
    checkpoint: Path,
) -> Dict[str, torch.Tensor]:
    selected = {
        str(key)[len(prefix):]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not selected:
        examples = sorted(str(key) for key in state)[:8]
        raise ValueError(
            f"{checkpoint}: no parameters with prefix '{prefix}'. "
            f"First checkpoint keys: {examples}"
        )
    return selected


def strict_load_module(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    prefix: str,
    checkpoint: Path,
    description: str,
) -> None:
    selected = prefixed_state(state, prefix, checkpoint)
    try:
        module.load_state_dict(selected, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"{description} does not match its checkpoint/configuration: "
            f"{checkpoint}\n{exc}"
        ) from exc


def build_before_model(
    model_args: Namespace,
    vocab_sizes: Tuple[int, int, int],
    raw_checkpoint: Path,
    behavior_checkpoint: Path,
    init_seed: int,
) -> AlignmentEncoder:
    set_deterministic_seed(init_seed)
    model = AlignmentEncoder(model_args, *vocab_sizes)
    raw_state = safe_torch_load(raw_checkpoint)
    behavior_state = safe_torch_load(behavior_checkpoint)

    strict_load_module(
        model.embedding_raw,
        raw_state,
        "embedding.",
        raw_checkpoint,
        "Stage-1 content embedding",
    )
    strict_load_module(
        model.encoder_raw,
        raw_state,
        "encoder.",
        raw_checkpoint,
        "Stage-1 content encoder",
    )
    strict_load_module(
        model.embedding_behavior,
        behavior_state,
        "embedding.",
        behavior_checkpoint,
        "Stage-1 behavior embedding",
    )
    strict_load_module(
        model.encoder_behavior,
        behavior_state,
        "encoder.",
        behavior_checkpoint,
        "Stage-1 behavior encoder",
    )
    model.initialize_cmc_projections()
    return model


def build_after_model(
    model_args: Namespace,
    vocab_sizes: Tuple[int, int, int],
    stage2_checkpoint: Path,
    init_seed: int,
) -> AlignmentEncoder:
    set_deterministic_seed(init_seed)
    model = AlignmentEncoder(model_args, *vocab_sizes)
    state = safe_torch_load(stage2_checkpoint)

    checkpoint_mapping = (
        (model.embedding_raw, "embedding_raw.", "Stage-2 content embedding"),
        (model.encoder_raw, "encoder_raw.", "Stage-2 content encoder"),
        (model.embedding_behavior, "embedding_size.", "Stage-2 behavior embedding"),
        (model.encoder_behavior, "encoder_size.", "Stage-2 behavior encoder"),
        (model.proj_raw, "target.itc_proj_raw.", "Stage-2 content CMC projection"),
        (
            model.proj_behavior,
            "target.itc_proj_size.",
            "Stage-2 behavior CMC projection",
        ),
    )
    for module, prefix, description in checkpoint_mapping:
        strict_load_module(
            module,
            state,
            prefix,
            stage2_checkpoint,
            description,
        )
    return model


def load_processed_dataset(path: Path) -> Sequence[Mapping[str, object]]:
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, (list, tuple)) or not dataset:
        raise ValueError(f"processed dataset must be a non-empty list/tuple: {path}")
    for index, record in enumerate(dataset[:10]):
        if not isinstance(record, Mapping):
            raise ValueError(f"dataset record {index} is not a dictionary-like mapping")
        missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
        if missing:
            raise ValueError(f"dataset record {index} is missing keys: {missing}")
    return dataset


def select_dataset_indices(total: int, maximum: int, seed: int) -> np.ndarray:
    count = total if maximum == 0 else min(total, maximum)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False)).astype(np.int64)


def validate_selected_records(
    dataset: Sequence[Mapping[str, object]],
    indices: np.ndarray,
    seq_length_raw: int,
    seq_length_behavior: int,
) -> None:
    expected_lengths = {
        "raw_src": seq_length_raw,
        "packet_ids": seq_length_raw,
        "directions": seq_length_raw,
        "size_src": seq_length_behavior,
        "iat_src": seq_length_behavior,
    }
    for source_index in indices:
        record = dataset[int(source_index)]
        missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
        if missing:
            raise ValueError(f"dataset record {source_index} is missing keys: {missing}")
        for key, expected in expected_lengths.items():
            actual = len(record[key])  # type: ignore[arg-type]
            if actual != expected:
                raise ValueError(
                    f"dataset record {source_index}: {key} length={actual}, "
                    f"expected {expected}"
                )


def iter_batches(
    dataset: Sequence[Mapping[str, object]],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Iterable[Tuple[np.ndarray, Tuple[torch.Tensor, ...]]]:
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        records = [dataset[int(index)] for index in batch_indices]
        tensors = tuple(
            torch.as_tensor(
                np.asarray([record[key] for record in records], dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            for key in REQUIRED_RECORD_KEYS
        )
        yield batch_indices, tensors


def extract_embeddings(
    model: AlignmentEncoder,
    dataset: Sequence[Mapping[str, object]],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    state_name: str,
) -> Dict[str, np.ndarray]:
    model = model.to(device)
    model.eval()
    content_cls_parts = []
    behavior_cls_parts = []
    content_parts = []
    behavior_parts = []
    batch_count = (indices.size + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_number, (_, tensors) in enumerate(
            iter_batches(dataset, indices, batch_size, device),
            start=1,
        ):
            raw_cls, behavior_cls, content, behavior = model(*tensors)
            content_cls_parts.append(raw_cls.float().cpu().numpy())
            behavior_cls_parts.append(behavior_cls.float().cpu().numpy())
            content_parts.append(content.float().cpu().numpy())
            behavior_parts.append(behavior.float().cpu().numpy())
            if batch_number == 1 or batch_number == batch_count or batch_number % 10 == 0:
                processed = min(batch_number * batch_size, indices.size)
                print(f"[{state_name}] {processed}/{indices.size} flows")

    return {
        "content_cls": np.concatenate(content_cls_parts, axis=0).astype(np.float32),
        "behavior_cls": np.concatenate(behavior_cls_parts, axis=0).astype(np.float32),
        "content": np.concatenate(content_parts, axis=0).astype(np.float32),
        "behavior": np.concatenate(behavior_parts, axis=0).astype(np.float32),
    }


def save_embeddings(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    dataset: Sequence[Mapping[str, object]],
    indices: np.ndarray,
    state_name: str,
) -> None:
    labels = np.asarray(
        [int(dataset[int(index)].get("label", -1)) for index in indices],
        dtype=np.int64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        flow_id=indices.astype(str),
        sample_index=indices,
        label=labels,
        state=np.asarray(state_name),
        content=arrays["content"],
        behavior=arrays["behavior"],
        content_cls=arrays["content_cls"],
        behavior_cls=arrays["behavior_cls"],
    )
    print(f"[write] {state_name} embeddings -> {path}")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    dataset_size: int,
    indices: np.ndarray,
) -> None:
    manifest = {
        "stage1_raw_checkpoint": str(args.stage1_raw_checkpoint.resolve()),
        "stage1_behavior_checkpoint": str(
            args.stage1_behavior_checkpoint.resolve()
        ),
        "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        "data_path": str(args.data_path.resolve()),
        "config_path_raw": str(args.config_path_raw.resolve()),
        "config_path_behavior": str(args.config_path_behavior.resolve()),
        "dataset_size": dataset_size,
        "selected_flow_count": int(indices.size),
        "sample_seed": args.sample_seed,
        "model_init_seed": args.model_init_seed,
        "representations": {
            "motivation": (
                "Within-modality centered-cosine geometry from independently "
                "pretrained Stage-1 encoder CLS representations only"
            ),
            "alignment_before": "Stage-1 pre-fusion encoder CLS",
            "alignment_after": "Stage-2 modality-specific CMC projection",
            "saved_diagnostics": (
                "Both NPZ files also retain encoder CLS and projected features"
            ),
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def ensure_output_policy(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite or --reuse-embeddings: "
            + ", ".join(existing)
        )


def release_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_plotter(
    args: argparse.Namespace,
    before_path: Path,
    after_path: Path,
    motivation_output_path: Path,
    alignment_output_path: Path,
) -> None:
    plotter = SCRIPT_DIR / "plot_representation_alignment.py"
    command = [
        sys.executable,
        str(plotter),
        "--before",
        str(before_path),
        "--after",
        str(after_path),
        "--output",
        str(motivation_output_path),
        "--alignment-output",
        str(alignment_output_path),
        "--before-content-key",
        "content_cls",
        "--before-behavior-key",
        "behavior_cls",
        "--after-content-key",
        "content",
        "--after-behavior-key",
        "behavior",
        "--sample-size",
        str(args.plot_sample_size),
        "--seed",
        str(args.plot_seed),
        "--colormap",
        args.colormap,
        "--geometry-colormap",
        args.geometry_colormap,
        "--difference-colormap",
        args.difference_colormap,
        "--font-scale",
        str(args.font_scale),
        "--fig-width",
        str(args.fig_width),
        "--fig-height",
        str(args.fig_height),
        "--dpi",
        str(args.dpi),
    ]
    if args.similarity_vmin is not None:
        command.extend(("--vmin", str(args.similarity_vmin)))
    if args.similarity_vmax is not None:
        command.extend(("--vmax", str(args.similarity_vmax)))
    print("[plot] " + " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    before_path = output_dir / "before_alignment.npz"
    after_path = output_dir / "after_alignment.npz"
    manifest_path = output_dir / "manifest.json"
    motivation_figure_path = output_dir / "representation_geometry_motivation.png"
    alignment_figure_path = output_dir / "representation_alignment_similarity.png"

    if args.reuse_embeddings:
        missing = [str(path) for path in (before_path, after_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "--reuse-embeddings requested but files are missing: "
                + ", ".join(missing)
            )
    else:
        protected_outputs = [before_path, after_path, manifest_path]
        if not args.skip_plot:
            protected_outputs.extend(
                (
                    motivation_figure_path,
                    motivation_figure_path.with_suffix(".pdf"),
                    alignment_figure_path,
                    alignment_figure_path.with_suffix(".pdf"),
                )
            )
        ensure_output_policy(protected_outputs, args.overwrite)
        print(f"[device] {device}")
        print(f"[data] loading {args.data_path}")
        dataset = load_processed_dataset(args.data_path)
        indices = select_dataset_indices(len(dataset), args.max_flows, args.sample_seed)
        validate_selected_records(
            dataset,
            indices,
            args.seq_length_raw,
            args.seq_length_behavior,
        )
        print(f"[data] selected {indices.size}/{len(dataset)} flows")

        model_args = build_model_args(args)
        vocab_sizes = (
            load_vocab_size(args.vocab_path_raw),
            load_vocab_size(args.vocab_path_behavior),
            load_vocab_size(args.vocab_path_temporal),
        )
        print(
            "[vocab] raw={} behavior={} temporal={}".format(*vocab_sizes)
        )
        print(
            f"[model] raw layers={model_args.layers_num_raw}, "
            f"behavior layers={model_args.layers_num_size}, "
            f"hidden={model_args.hidden_size}"
        )

        print("[before] loading independently pre-trained Stage-1 encoders")
        before_model = build_before_model(
            model_args,
            vocab_sizes,
            args.stage1_raw_checkpoint,
            args.stage1_behavior_checkpoint,
            args.model_init_seed,
        )
        before_arrays = extract_embeddings(
            before_model,
            dataset,
            indices,
            args.batch_size,
            device,
            "before",
        )
        save_embeddings(before_path, before_arrays, dataset, indices, "before")
        del before_arrays, before_model
        release_device_cache()

        print("[after] loading Stage-2 encoders and learned CMC projections")
        after_model = build_after_model(
            model_args,
            vocab_sizes,
            args.stage2_checkpoint,
            args.model_init_seed,
        )
        after_arrays = extract_embeddings(
            after_model,
            dataset,
            indices,
            args.batch_size,
            device,
            "after",
        )
        save_embeddings(after_path, after_arrays, dataset, indices, "after")
        del after_arrays, after_model
        release_device_cache()

        write_manifest(manifest_path, args, len(dataset), indices)
        print(f"[write] manifest -> {manifest_path}")

    if not args.skip_plot:
        figure_paths = (motivation_figure_path, alignment_figure_path)
        existing_figures = [str(path) for path in figure_paths if path.exists()]
        if existing_figures and not args.overwrite and not args.reuse_embeddings:
            raise FileExistsError(
                "figure outputs already exist: " + ", ".join(existing_figures)
            )
        run_plotter(
            args,
            before_path,
            after_path,
            motivation_figure_path,
            alignment_figure_path,
        )


if __name__ == "__main__":
    main()
