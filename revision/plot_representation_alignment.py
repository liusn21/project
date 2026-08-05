#!/usr/bin/env python3
"""Plot Stage-1 geometry mismatch and, optionally, CMC alignment effects.

This plot-only companion to ``run_representation_alignment.py`` produces two
conceptually separate visualizations from saved NPZ embeddings:

* a Stage-1-only motivation figure comparing the within-modality geometry of
  independently pretrained content and behavior representations; and
* an optional before/after CMC cross-modal similarity heatmap that illustrates
  why explicit alignment is needed and how CMC changes paired-flow retrieval.

The motivation figure never uses Stage-2 representations.  For the same
deterministic flow sample, it computes centered-cosine Gram matrices within the
content and behavior spaces, then displays their normalized absolute
difference.  Within-modality Gram matrices are invariant to rotations of their
respective coordinate systems, so the comparison does not assume that two
independently pretrained encoders already share a basis.  Self-similarity cells
are omitted from the two geometry panels because they are identically one and
carry no geometric evidence; the mismatch panel retains their true zero value.

If labels are present, selected flows are reordered by label only for display;
labels do not affect which flows are sampled.  A narrow class-color strip above
each matrix shows this ordering without drawing boundaries through the data.

Example::

    python3 revision/plot_representation_alignment.py \\
        --before revision/representation_alignment/before_alignment.npz \\
        --after revision/representation_alignment/after_alignment.npz \\
        --output revision/representation_alignment/representation_geometry_motivation.png \\
        --alignment-output revision/representation_alignment/representation_alignment_similarity.png
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mm_trafficbert_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass(frozen=True)
class EmbeddingSet:
    flow_ids: np.ndarray
    labels: np.ndarray
    content: np.ndarray
    behavior: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--before",
        type=Path,
        required=True,
        help="NPZ containing independently pretrained Stage-1 representations.",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=None,
        help="NPZ containing learned Stage-2 CMC representations.",
    )
    parser.add_argument("--flow-id-key", default="flow_id")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--before-content-key", default="content_cls")
    parser.add_argument("--before-behavior-key", default="behavior_cls")
    parser.add_argument("--after-content-key", default="content")
    parser.add_argument("--after-behavior-key", default="behavior")

    parser.add_argument(
        "--sample-size",
        type=int,
        default=48,
        help="Number of randomly selected flows displayed; 0 uses all flows.",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--preserve-sample-order",
        action="store_true",
        help="Do not group the randomly selected flows by their saved labels.",
    )

    parser.add_argument(
        "--content-title",
        default="(a) Content geometry",
    )
    parser.add_argument(
        "--behavior-title",
        default="(b) Behavior geometry",
    )
    parser.add_argument(
        "--difference-title",
        default="(c) Geometry mismatch",
    )
    parser.add_argument("--geometry-colormap", default="RdBu_r")
    parser.add_argument("--difference-colormap", default="magma")

    parser.add_argument(
        "--before-title",
        default="(a) Before alignment",
    )
    parser.add_argument(
        "--after-title",
        default="(b) After alignment",
    )
    parser.add_argument(
        "--colormap",
        default="magma",
        help="Colormap for the optional CMC-effect diagnostic.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional shared lower limit for the CMC-effect heatmap.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional shared upper limit for the CMC-effect heatmap.",
    )

    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--fig-width", type=float, default=7.1)
    parser.add_argument("--fig-height", type=float, default=2.75)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "representation_alignment"
        / "representation_geometry_motivation.png",
        help="Stage-1-only motivation figure (PNG).",
    )
    parser.add_argument(
        "--alignment-output",
        type=Path,
        default=None,
        help="Optional before/after CMC diagnostic figure (PNG).",
    )
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    missing = [str(args.before)] if not args.before.is_file() else []
    if args.after is not None and not args.after.is_file():
        missing.append(str(args.after))
    if missing:
        parser.error("input files do not exist: " + ", ".join(missing))
    if args.alignment_output is not None and args.after is None:
        parser.error("--alignment-output requires --after")
    if args.sample_size < 0 or 0 < args.sample_size < 3:
        parser.error("--sample-size must be 0 or at least 3")
    if args.vmin is not None and not -1.0 <= args.vmin <= 1.0:
        parser.error("--vmin must lie in [-1, 1]")
    if args.vmax is not None and not -1.0 <= args.vmax <= 1.0:
        parser.error("--vmax must lie in [-1, 1]")
    if args.vmin is not None and args.vmax is not None and args.vmin >= args.vmax:
        parser.error("--vmin must be smaller than --vmax")
    if args.font_scale <= 0.0:
        parser.error("--font-scale must be positive")
    if args.fig_width <= 0.0 or args.fig_height <= 0.0:
        parser.error("figure dimensions must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    outputs = [args.output]
    if args.alignment_output is not None:
        outputs.append(args.alignment_output)
    invalid_outputs = [str(path) for path in outputs if path.suffix.lower() != ".png"]
    if invalid_outputs:
        parser.error("figure outputs must end in .png: " + ", ".join(invalid_outputs))
    for name in (args.geometry_colormap, args.difference_colormap, args.colormap):
        try:
            plt.get_cmap(name)
        except ValueError:
            parser.error(f"unknown Matplotlib colormap: {name}")


def stringify_ids(values: np.ndarray, path: Path) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"{path}: flow IDs must be 1-D, got {values.shape}")
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return np.asarray(result, dtype=str)


def load_embeddings(
    path: Path,
    flow_id_key: str,
    label_key: str,
    content_key: str,
    behavior_key: str,
) -> EmbeddingSet:
    with np.load(path, allow_pickle=False) as archive:
        required = (flow_id_key, content_key, behavior_key)
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(
                f"{path}: missing keys {missing}; available: {sorted(archive.files)}"
            )
        flow_ids = stringify_ids(np.asarray(archive[flow_id_key]), path)
        content = np.asarray(archive[content_key], dtype=np.float32)
        behavior = np.asarray(archive[behavior_key], dtype=np.float32)
        if label_key in archive:
            labels = np.asarray(archive[label_key], dtype=np.int64)
        else:
            labels = np.full(flow_ids.size, -1, dtype=np.int64)

    if content.ndim != 2 or behavior.ndim != 2:
        raise ValueError(
            f"{path}: embeddings must be 2-D, got {content.shape}, {behavior.shape}"
        )
    if content.shape[0] != flow_ids.size or behavior.shape[0] != flow_ids.size:
        raise ValueError(f"{path}: flow and embedding row counts differ")
    if labels.ndim != 1 or labels.size != flow_ids.size:
        raise ValueError(f"{path}: labels must be one per flow, got {labels.shape}")
    if flow_ids.size < 3:
        raise ValueError(f"{path}: at least three paired flows are required")
    if np.unique(flow_ids).size != flow_ids.size:
        raise ValueError(f"{path}: flow IDs are not unique")
    if not np.isfinite(content).all() or not np.isfinite(behavior).all():
        raise ValueError(f"{path}: embeddings contain NaN or infinity")
    return EmbeddingSet(flow_ids, labels, content, behavior)


def align_after(before: EmbeddingSet, after: EmbeddingSet) -> EmbeddingSet:
    before_ids = set(before.flow_ids.tolist())
    after_ids = set(after.flow_ids.tolist())
    if before_ids != after_ids:
        raise ValueError("Before/after NPZ files must contain the same flow IDs")
    after_index = {flow_id: index for index, flow_id in enumerate(after.flow_ids)}
    order = np.asarray([after_index[flow_id] for flow_id in before.flow_ids])
    aligned_labels = after.labels[order]
    known_in_both = (before.labels >= 0) & (aligned_labels >= 0)
    if np.any(before.labels[known_in_both] != aligned_labels[known_in_both]):
        raise ValueError("Before/after labels disagree for at least one flow")
    return EmbeddingSet(
        before.flow_ids.copy(),
        before.labels.copy(),
        after.content[order],
        after.behavior[order],
    )


def select_display_indices(
    embeddings: EmbeddingSet,
    sample_size: int,
    seed: int,
    preserve_order: bool,
) -> np.ndarray:
    total = embeddings.flow_ids.size
    count = total if sample_size == 0 else min(sample_size, total)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=count, replace=False))
    if not preserve_order and np.any(embeddings.labels[selected] >= 0):
        selected = selected[
            np.lexsort((selected, embeddings.labels[selected]))
        ]
    return selected


def subset_embeddings(embeddings: EmbeddingSet, indices: np.ndarray) -> EmbeddingSet:
    return EmbeddingSet(
        embeddings.flow_ids[indices],
        embeddings.labels[indices],
        embeddings.content[indices],
        embeddings.behavior[indices],
    )


def l2_normalize(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name}: zero-norm embeddings found")
    return values / norms


def centered_unit_embeddings(values: np.ndarray, name: str) -> np.ndarray:
    centered = values.astype(np.float64) - np.mean(values, axis=0, keepdims=True)
    return l2_normalize(centered, name).astype(np.float32)


def stage1_geometry(
    before: EmbeddingSet,
    display_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Center on all extracted flows, not only the displayed subset, so changing
    # the display count does not redefine the representation-space origin.
    content = centered_unit_embeddings(before.content, "Stage-1/content")
    behavior = centered_unit_embeddings(before.behavior, "Stage-1/behavior")
    content = content[display_indices]
    behavior = behavior[display_indices]
    content_similarity = np.clip(content @ content.T, -1.0, 1.0)
    behavior_similarity = np.clip(behavior @ behavior.T, -1.0, 1.0)
    disagreement = np.abs(content_similarity - behavior_similarity) / 2.0
    return content_similarity, behavior_similarity, disagreement


def cross_modal_similarity(embeddings: EmbeddingSet, name: str) -> np.ndarray:
    if embeddings.content.shape[1] != embeddings.behavior.shape[1]:
        raise ValueError(
            f"{name}: cross-modal cosine requires equal dimensions, got "
            f"{embeddings.content.shape[1]} and {embeddings.behavior.shape[1]}"
        )
    content = l2_normalize(embeddings.content, f"{name}/content")
    behavior = l2_normalize(embeddings.behavior, f"{name}/behavior")
    return np.clip(content @ behavior.T, -1.0, 1.0)


def style_matrix_axis(
    axis: plt.Axes,
    title: str,
    show_y_label: bool,
) -> None:
    axis.set_title(title, pad=15.0)
    axis.set_ylabel("Flows" if show_y_label else "")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#AEB4B9")
        spine.set_linewidth(0.65)


def class_strip_colors(labels: np.ndarray) -> np.ndarray | None:
    known_labels = np.unique(labels[labels >= 0])
    if known_labels.size == 0:
        return None
    palette = plt.get_cmap("tab20", max(int(known_labels.size), 1))
    label_to_color = {
        int(label): palette(index)
        for index, label in enumerate(known_labels)
    }
    unknown_color = (0.72, 0.72, 0.72, 1.0)
    colors = np.asarray(
        [label_to_color.get(int(label), unknown_color) for label in labels],
        dtype=float,
    )
    return colors[np.newaxis, :, :]


def draw_class_strip(axis: plt.Axes, colors: np.ndarray | None) -> None:
    if colors is None:
        return
    strip = axis.inset_axes([0.0, 1.012, 1.0, 0.026])
    strip.imshow(colors, interpolation="nearest", aspect="auto")
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def geometry_color_limit(
    content_similarity: np.ndarray,
    behavior_similarity: np.ndarray,
) -> float:
    values = np.concatenate(
        (
            off_diagonal_values(content_similarity),
            off_diagonal_values(behavior_similarity),
        )
    )
    limit = float(np.max(np.abs(values)))
    if limit <= 1e-8:
        raise ValueError("All off-diagonal Stage-1 similarities are identical")
    return limit


def configure_matplotlib(font_scale: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.2 * font_scale,
            "axes.titlesize": 9.8 * font_scale,
            "axes.labelsize": 9.2 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_motivation_plot(
    content_similarity: np.ndarray,
    behavior_similarity: np.ndarray,
    disagreement: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> None:
    configure_matplotlib(args.font_scale)
    limit = geometry_color_limit(content_similarity, behavior_similarity)
    difference_max = float(np.max(off_diagonal_values(disagreement)))
    if difference_max <= 1e-8:
        raise ValueError(
            "The displayed Stage-1 geometries are effectively identical; "
            "a mismatch figure would be uninformative."
        )

    geometry_cmap = plt.get_cmap(args.geometry_colormap).copy()
    difference_cmap = plt.get_cmap(args.difference_colormap).copy()
    geometry_cmap.set_bad("#D9D9D9")
    diagonal_mask = np.eye(content_similarity.shape[0], dtype=bool)
    strip_colors = class_strip_colors(labels)

    figure = plt.figure(figsize=(args.fig_width, args.fig_height))
    grid = figure.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=(1.0, 0.065),
        hspace=0.42,
        wspace=0.16,
    )
    axes = np.asarray([figure.add_subplot(grid[0, index]) for index in range(3)])
    similarity_bar_axis = figure.add_subplot(grid[1, :2])
    difference_bar_axis = figure.add_subplot(grid[1, 2])
    geometry_image = axes[0].imshow(
        np.ma.array(content_similarity, mask=diagonal_mask),
        cmap=geometry_cmap,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
        rasterized=True,
    )
    axes[1].imshow(
        np.ma.array(behavior_similarity, mask=diagonal_mask),
        cmap=geometry_cmap,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
        rasterized=True,
    )
    difference_image = axes[2].imshow(
        disagreement,
        cmap=difference_cmap,
        vmin=0.0,
        vmax=difference_max,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
        rasterized=True,
    )
    titles = (args.content_title, args.behavior_title, args.difference_title)
    for index, (axis, title) in enumerate(zip(axes, titles)):
        style_matrix_axis(axis, title, show_y_label=index == 0)
        draw_class_strip(axis, strip_colors)

    similarity_bar = figure.colorbar(
        geometry_image,
        cax=similarity_bar_axis,
        orientation="horizontal",
    )
    similarity_bar.set_label("Centered cosine similarity", labelpad=3.0)
    similarity_bar.outline.set_linewidth(0.55)
    similarity_bar.ax.tick_params(
        labelsize=7.7 * args.font_scale,
        width=0.55,
        length=2.2,
    )
    difference_bar = figure.colorbar(
        difference_image,
        cax=difference_bar_axis,
        orientation="horizontal",
    )
    difference_bar.set_label("Normalized disagreement", labelpad=3.0)
    difference_bar.outline.set_linewidth(0.55)
    difference_bar.ax.tick_params(
        labelsize=7.7 * args.font_scale,
        width=0.55,
        length=2.2,
    )
    figure.supxlabel(
        "Flows (ordered by class)",
        x=0.52,
        y=0.205,
        fontsize=9.2 * args.font_scale,
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.12)
    save_figure(figure, args.output, args.dpi, args.no_pdf)


def resolve_alignment_limits(
    before_similarity: np.ndarray,
    after_similarity: np.ndarray,
    requested_min: float | None,
    requested_max: float | None,
) -> Tuple[float, float]:
    combined = np.concatenate(
        (before_similarity.ravel(), after_similarity.ravel())
    )
    lower = float(np.min(combined)) if requested_min is None else requested_min
    upper = float(np.max(combined)) if requested_max is None else requested_max
    if upper - lower <= 1e-8:
        raise ValueError(
            "All displayed cross-modal similarities are effectively identical"
        )
    return lower, upper


def paired_flow_recall_at_one(similarities: np.ndarray) -> float:
    """Return Recall@1 averaged over both cross-modal retrieval directions."""
    count = similarities.shape[0]
    expected = np.arange(count)
    content_to_behavior = np.count_nonzero(
        np.argmax(similarities, axis=1) == expected
    )
    behavior_to_content = np.count_nonzero(
        np.argmax(similarities, axis=0) == expected
    )
    return float(
        (content_to_behavior + behavior_to_content) / (2.0 * count)
    )


def draw_alignment_panel(
    axis: plt.Axes,
    similarities: np.ndarray,
    title: str,
    args: argparse.Namespace,
    color_limits: Tuple[float, float],
):
    image = axis.imshow(
        similarities,
        cmap=args.colormap,
        vmin=color_limits[0],
        vmax=color_limits[1],
        interpolation="nearest",
        origin="upper",
        aspect="equal",
        rasterized=True,
    )
    axis.set_title(title, pad=7.0)
    axis.set_xlabel("Behavior flows")
    axis.set_ylabel("Content flows")
    axis.set_xticks([])
    axis.set_yticks([])
    recall_at_one = paired_flow_recall_at_one(similarities)
    axis.text(
        0.965,
        0.965,
        f"Paired-flow R@1: {recall_at_one:.1%}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.2 * args.font_scale,
        color="#1A1A1A",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#BFC3C7",
            "linewidth": 0.55,
            "alpha": 0.92,
        },
    )
    for spine in axis.spines.values():
        spine.set_color("#AEB4B9")
        spine.set_linewidth(0.65)
    return image


def save_alignment_plot(
    before_similarity: np.ndarray,
    after_similarity: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if args.alignment_output is None:
        return
    configure_matplotlib(args.font_scale)
    color_limits = resolve_alignment_limits(
        before_similarity,
        after_similarity,
        args.vmin,
        args.vmax,
    )
    figure, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height))
    before_image = draw_alignment_panel(
        axes[0], before_similarity, args.before_title, args, color_limits
    )
    after_image = draw_alignment_panel(
        axes[1], after_similarity, args.after_title, args, color_limits
    )
    for axis, image in zip(axes, (before_image, after_image)):
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="3.6%", pad=0.055)
        colorbar = figure.colorbar(
            image,
            cax=colorbar_axis,
            orientation="vertical",
        )
        colorbar.set_label("Cosine similarity", rotation=90, labelpad=5.0)
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(
            labelsize=8.3 * args.font_scale,
            width=0.6,
            length=2.5,
        )
    figure.subplots_adjust(
        left=0.065,
        right=0.965,
        top=0.89,
        bottom=0.17,
        wspace=0.20,
    )
    save_figure(figure, args.alignment_output, args.dpi, args.no_pdf)


def save_figure(
    figure: plt.Figure,
    output: Path,
    dpi: int,
    no_pdf: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    if not no_pdf:
        figure.savefig(
            output.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def print_alignment_diagnostics(name: str, similarities: np.ndarray) -> None:
    recall_at_one = paired_flow_recall_at_one(similarities)
    print(f"[{name}] paired-flow bidirectional R@1={recall_at_one:.3f}")


def geometry_diagnostics(
    content_similarity: np.ndarray,
    behavior_similarity: np.ndarray,
) -> Tuple[float, float, int, float]:
    count = content_similarity.shape[0]
    triangle = np.triu_indices(count, k=1)
    content_pairs = content_similarity[triangle]
    behavior_pairs = behavior_similarity[triangle]
    if np.std(content_pairs) <= 1e-12 or np.std(behavior_pairs) <= 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(content_pairs, behavior_pairs)[0, 1])
    disagreement = float(np.mean(np.abs(content_pairs - behavior_pairs) / 2.0))

    neighbors = min(5, count - 1)
    content_for_rank = content_similarity.copy()
    behavior_for_rank = behavior_similarity.copy()
    np.fill_diagonal(content_for_rank, -np.inf)
    np.fill_diagonal(behavior_for_rank, -np.inf)
    content_top = np.argpartition(content_for_rank, -neighbors, axis=1)[:, -neighbors:]
    behavior_top = np.argpartition(behavior_for_rank, -neighbors, axis=1)[:, -neighbors:]
    overlaps = [
        len(set(content_top[row]).intersection(behavior_top[row])) / neighbors
        for row in range(count)
    ]
    return correlation, disagreement, neighbors, float(np.mean(overlaps))


def output_paths(output: Path, no_pdf: bool) -> list[str]:
    paths = [str(output)]
    if not no_pdf:
        paths.append(str(output.with_suffix(".pdf")))
    return paths


def main() -> None:
    args = parse_args()
    before = load_embeddings(
        args.before,
        args.flow_id_key,
        args.label_key,
        args.before_content_key,
        args.before_behavior_key,
    )
    display_indices = select_display_indices(
        before,
        args.sample_size,
        args.seed,
        args.preserve_sample_order,
    )
    displayed_before = subset_embeddings(before, display_indices)
    content_geometry, behavior_geometry, disagreement = stage1_geometry(
        before,
        display_indices,
    )
    correlation, mean_disagreement, neighbors, overlap = geometry_diagnostics(
        content_geometry,
        behavior_geometry,
    )
    print(
        f"[motivation/displayed] geometry correlation={correlation:.4f}, "
        f"mean normalized disagreement={mean_disagreement:.4f}, "
        f"neighbor overlap@{neighbors}={overlap:.3f}"
    )
    save_motivation_plot(
        content_geometry,
        behavior_geometry,
        disagreement,
        displayed_before.labels,
        args,
    )
    outputs = output_paths(args.output, args.no_pdf)

    if args.alignment_output is not None:
        assert args.after is not None
        after = load_embeddings(
            args.after,
            args.flow_id_key,
            args.label_key,
            args.after_content_key,
            args.after_behavior_key,
        )
        after = align_after(before, after)
        displayed_after = subset_embeddings(after, display_indices)
        before_similarity = cross_modal_similarity(displayed_before, "before")
        after_similarity = cross_modal_similarity(displayed_after, "after")
        print_alignment_diagnostics("before", before_similarity)
        print_alignment_diagnostics("after", after_similarity)
        save_alignment_plot(before_similarity, after_similarity, args)
        outputs.extend(output_paths(args.alignment_output, args.no_pdf))

    ordering = (
        "saved-label order"
        if not args.preserve_sample_order and np.any(displayed_before.labels >= 0)
        else "sample order"
    )
    print(
        f"Displayed {displayed_before.flow_ids.size} randomly selected flows in "
        f"{ordering}. Wrote: {', '.join(outputs)}"
    )


if __name__ == "__main__":
    main()
