#!/usr/bin/env python3
"""Summarize or select learned reliability corrections on compressed flows.

The input is ``flow_results.csv`` produced by
``compression_checkpoint_inference.py``.  This script is intentionally a pure
post-processing step: it does not load checkpoints, read PCAPs, or rerun model
inference.

Two modes are available:

* ``distribution`` summarizes and plots
  ``r_learned - r_calibrated`` for the selected flow population;
* ``topk`` writes the flows with the largest positive learned corrections.

Both ``--min-e-i`` and ``--content-useful-only`` are optional filters shared by
the two modes.  Content usefulness follows the inference script's continuous
definition: ``content_utility > 0``.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SCRIPT_VERSION = "1.0.0"
DEFAULT_TOP_K = 20

CORE_INPUT_FIELDS = {
    "inference_status",
    "relative_pcap",
    "r_calibrated",
    "r_learned",
}

SUMMARY_FIELDS = [
    "input_file",
    "dataset",
    "compression_level",
    "evaluation_scope",
    "min_e_i",
    "min_e_i_inclusive",
    "content_useful_only",
    "input_rows",
    "valid_reliability_rows",
    "invalid_status_rows",
    "invalid_reliability_rows",
    "e_i_filter_rejected_rows",
    "content_filter_rejected_rows",
    "selected_flows",
    "positive_flows",
    "positive_fraction",
    "zero_flows",
    "negative_flows",
    "mean_delta_learned",
    "std_delta_learned",
    "min_delta_learned",
    "p10_delta_learned",
    "p25_delta_learned",
    "median_delta_learned",
    "p75_delta_learned",
    "p90_delta_learned",
    "max_delta_learned",
    "mean_r_calibrated",
    "mean_r_learned",
]

TOPK_FIELDS = [
    "rank",
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
    "r_calibrated",
    "r_learned",
    "learned_delta",
    "r_mod",
    "applied_delta",
    "content_utility",
    "raw_pred",
    "raw_correct",
    "raw_p_true",
    "size_pred",
    "size_correct",
    "size_p_true",
    "itgca_pred",
    "itgca_correct",
    "itgca_p_true",
    "content_opportunity",
    "itgca_rescue",
    "source_row_number",
]


@dataclass
class SelectedFlow:
    """A successfully inferred flow after applying the optional filters."""

    source_row_number: int
    row: Dict[str, str]
    r_calibrated: float
    r_learned: float
    learned_delta: float
    r_mod: Optional[float]
    applied_delta: Optional[float]


@dataclass
class ScanResult:
    """Streaming scan counters plus the data needed by the selected mode."""

    input_rows: int = 0
    valid_reliability_rows: int = 0
    invalid_status_rows: int = 0
    invalid_reliability_rows: int = 0
    e_i_filter_rejected_rows: int = 0
    content_filter_rejected_rows: int = 0
    selected_flows: int = 0
    positive_flows: int = 0
    zero_flows: int = 0
    negative_flows: int = 0
    sum_r_calibrated: float = 0.0
    sum_r_learned: float = 0.0
    deltas: List[float] = field(default_factory=list)
    topk_heap: List[Tuple[float, int, SelectedFlow]] = field(default_factory=list)
    datasets: Set[str] = field(default_factory=set)
    compression_levels: Set[str] = field(default_factory=set)
    evaluation_scopes: Set[str] = field(default_factory=set)


def _maximize_csv_field_size_limit() -> None:
    """Raise csv's parser limit without overflowing platforms with a C long."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _join_metadata(values: Set[str]) -> str:
    return ";".join(sorted(value for value in values if value))


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


def _percentile(sorted_values: Sequence[float], percentile: float) -> Optional[float]:
    """Return a linearly interpolated percentile for an already sorted sample."""

    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _scan_flow_results(
    path: Path,
    mode: str,
    top_k: int,
    min_e_i: Optional[float],
    content_useful_only: bool,
) -> ScanResult:
    """Read the input once and retain only the data required by ``mode``."""

    result = ScanResult()
    required_fields = set(CORE_INPUT_FIELDS)
    if min_e_i is not None:
        required_fields.add("e_i")
    if content_useful_only:
        required_fields.add("content_utility")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(required_fields - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "input is not a compatible flow_results.csv; missing fields: "
                + ", ".join(missing)
            )

        for source_row_number, raw_row in enumerate(reader, start=2):
            result.input_rows += 1
            row = {key: value if value is not None else "" for key, value in raw_row.items()}

            dataset = row.get("dataset", "").strip()
            compression_level = row.get("compression_level", "").strip()
            evaluation_scope = row.get("evaluation_scope", "").strip()
            if dataset:
                result.datasets.add(dataset)
            if compression_level:
                result.compression_levels.add(compression_level)
            if evaluation_scope:
                result.evaluation_scopes.add(evaluation_scope)

            if row.get("inference_status", "").strip().lower() != "ok":
                result.invalid_status_rows += 1
                continue

            r_calibrated = _finite_float(row.get("r_calibrated"))
            r_learned = _finite_float(row.get("r_learned"))
            if r_calibrated is None or r_learned is None:
                result.invalid_reliability_rows += 1
                continue
            result.valid_reliability_rows += 1

            if min_e_i is not None:
                e_i = _finite_float(row.get("e_i"))
                if e_i is None or e_i < min_e_i:
                    result.e_i_filter_rejected_rows += 1
                    continue

            if content_useful_only:
                content_utility = _finite_float(row.get("content_utility"))
                if content_utility is None or content_utility <= 0.0:
                    result.content_filter_rejected_rows += 1
                    continue

            learned_delta = r_learned - r_calibrated
            r_mod = _finite_float(row.get("r_mod"))
            applied_delta = (
                r_mod - r_calibrated if r_mod is not None else None
            )

            result.selected_flows += 1
            result.sum_r_calibrated += r_calibrated
            result.sum_r_learned += r_learned
            if learned_delta > 0.0:
                result.positive_flows += 1
            elif learned_delta < 0.0:
                result.negative_flows += 1
            else:
                result.zero_flows += 1

            if mode == "distribution":
                result.deltas.append(learned_delta)
                continue

            if learned_delta <= 0.0:
                continue
            selected = SelectedFlow(
                source_row_number=source_row_number,
                row=row,
                r_calibrated=r_calibrated,
                r_learned=r_learned,
                learned_delta=learned_delta,
                r_mod=r_mod,
                applied_delta=applied_delta,
            )
            # A larger tuple is a better candidate.  Negating the source row
            # makes earlier input rows win deterministic ties in delta.
            heap_entry = (learned_delta, -source_row_number, selected)
            if len(result.topk_heap) < top_k:
                heapq.heappush(result.topk_heap, heap_entry)
            elif heap_entry[:2] > result.topk_heap[0][:2]:
                heapq.heapreplace(result.topk_heap, heap_entry)

    return result


def _summary_row(
    result: ScanResult,
    input_path: Path,
    min_e_i: Optional[float],
    content_useful_only: bool,
) -> Dict[str, Any]:
    sorted_deltas = sorted(result.deltas)
    count = result.selected_flows
    mean_delta = statistics.fmean(sorted_deltas) if sorted_deltas else None
    std_delta = statistics.pstdev(sorted_deltas) if sorted_deltas else None
    return {
        "input_file": str(input_path),
        "dataset": _join_metadata(result.datasets),
        "compression_level": _join_metadata(result.compression_levels),
        "evaluation_scope": _join_metadata(result.evaluation_scopes),
        "min_e_i": min_e_i,
        "min_e_i_inclusive": int(min_e_i is not None),
        "content_useful_only": int(content_useful_only),
        "input_rows": result.input_rows,
        "valid_reliability_rows": result.valid_reliability_rows,
        "invalid_status_rows": result.invalid_status_rows,
        "invalid_reliability_rows": result.invalid_reliability_rows,
        "e_i_filter_rejected_rows": result.e_i_filter_rejected_rows,
        "content_filter_rejected_rows": result.content_filter_rejected_rows,
        "selected_flows": count,
        "positive_flows": result.positive_flows,
        "positive_fraction": result.positive_flows / count if count else None,
        "zero_flows": result.zero_flows,
        "negative_flows": result.negative_flows,
        "mean_delta_learned": mean_delta,
        "std_delta_learned": std_delta,
        "min_delta_learned": sorted_deltas[0] if sorted_deltas else None,
        "p10_delta_learned": _percentile(sorted_deltas, 0.10),
        "p25_delta_learned": _percentile(sorted_deltas, 0.25),
        "median_delta_learned": _percentile(sorted_deltas, 0.50),
        "p75_delta_learned": _percentile(sorted_deltas, 0.75),
        "p90_delta_learned": _percentile(sorted_deltas, 0.90),
        "max_delta_learned": sorted_deltas[-1] if sorted_deltas else None,
        "mean_r_calibrated": (
            result.sum_r_calibrated / count if count else None
        ),
        "mean_r_learned": result.sum_r_learned / count if count else None,
    }


def _write_distribution_plot(
    path: Path,
    deltas: Sequence[float],
    positive_fraction: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "distribution mode requires matplotlib to write the PNG plot"
        ) from exc

    bin_count = min(60, max(10, int(math.ceil(math.sqrt(len(deltas))))))
    mean_delta = statistics.fmean(deltas)
    median_delta = statistics.median(deltas)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.hist(
        deltas,
        bins=bin_count,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    axis.axvline(0.0, color="black", linewidth=1.4, label="zero")
    axis.axvline(
        mean_delta,
        color="#F58518",
        linestyle="--",
        linewidth=1.4,
        label=f"mean = {mean_delta:.4f}",
    )
    axis.axvline(
        median_delta,
        color="#54A24B",
        linestyle=":",
        linewidth=1.6,
        label=f"median = {median_delta:.4f}",
    )
    axis.set_xlabel(r"$\Delta_{\mathrm{learned}} = r_{\mathrm{learned}} - r_{\mathrm{calibrated}}$")
    axis.set_ylabel("Flow count")
    axis.set_title("Distribution of learned reliability correction")
    axis.text(
        0.98,
        0.96,
        f"N = {len(deltas):,}\nP($\\Delta_{{\\mathrm{{learned}}}}>0$) = {positive_fraction:.2%}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()

    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        figure.savefig(temporary, dpi=220, bbox_inches="tight", format="png")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _topk_output_rows(result: ScanResult) -> Iterable[Dict[str, Any]]:
    selected = sorted(
        (entry[2] for entry in result.topk_heap),
        key=lambda flow: (-flow.learned_delta, flow.source_row_number),
    )
    for rank, flow in enumerate(selected, start=1):
        row = flow.row
        yield {
            "rank": rank,
            "dataset": row.get("dataset", ""),
            "compression_level": row.get("compression_level", ""),
            "evaluation_scope": row.get("evaluation_scope", ""),
            "label": row.get("label", ""),
            "relative_pcap": row.get("relative_pcap", ""),
            "e_i": row.get("e_i", ""),
            "e_bin": row.get("e_bin", ""),
            "model_entropy_bits": row.get("model_entropy_bits", ""),
            "audit_model_r_stat": row.get("audit_model_r_stat", ""),
            "inference_r_stat": row.get("inference_r_stat", ""),
            "r_calibrated": flow.r_calibrated,
            "r_learned": flow.r_learned,
            "learned_delta": flow.learned_delta,
            "r_mod": flow.r_mod,
            "applied_delta": flow.applied_delta,
            "content_utility": row.get("content_utility", ""),
            "raw_pred": row.get("raw_pred", ""),
            "raw_correct": row.get("raw_correct", ""),
            "raw_p_true": row.get("raw_p_true", ""),
            "size_pred": row.get("size_pred", ""),
            "size_correct": row.get("size_correct", ""),
            "size_p_true": row.get("size_p_true", ""),
            "itgca_pred": row.get("itgca_pred", ""),
            "itgca_correct": row.get("itgca_correct", ""),
            "itgca_p_true": row.get("itgca_p_true", ""),
            "content_opportunity": row.get("content_opportunity", ""),
            "itgca_rescue": row.get("itgca_rescue", ""),
            "source_row_number": flow.source_row_number,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze r_learned - r_calibrated from compression checkpoint "
            "flow_results.csv without rerunning inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "flow_results",
        type=Path,
        help="flow_results.csv produced by compression_checkpoint_inference.py",
    )
    parser.add_argument(
        "--mode",
        choices=("distribution", "topk"),
        required=True,
        help="distribution summary/plot or Top-K positive corrections",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_integer,
        default=DEFAULT_TOP_K,
        help="number of positive corrections retained in topk mode",
    )
    parser.add_argument(
        "--min-e-i",
        type=_probability,
        default=None,
        help="optional inclusive lower bound: retain flows with e_i >= this value",
    )
    parser.add_argument(
        "--content-useful-only",
        action="store_true",
        help="retain only flows with content_utility > 0",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; defaults to INPUT_DIR/compression_pick",
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.flow_results.expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"flow_results.csv not found: {input_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / "compression_pick"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _maximize_csv_field_size_limit()
    result = _scan_flow_results(
        input_path,
        args.mode,
        args.top_k,
        args.min_e_i,
        args.content_useful_only,
    )

    print(
        f"[filter] input={result.input_rows} "
        f"valid={result.valid_reliability_rows} selected={result.selected_flows}"
    )
    if args.min_e_i is not None:
        print(
            f"[filter] e_i >= {args.min_e_i:g}; "
            f"rejected={result.e_i_filter_rejected_rows}"
        )
    if args.content_useful_only:
        print(
            "[filter] content_utility > 0; "
            f"rejected={result.content_filter_rejected_rows}"
        )

    if args.mode == "distribution":
        summary_path = output_dir / "delta_summary.csv"
        plot_path = output_dir / "delta_distribution.png"
        summary = _summary_row(
            result,
            input_path,
            args.min_e_i,
            args.content_useful_only,
        )
        _atomic_write_csv(summary_path, [summary], SUMMARY_FIELDS)
        print(f"[write] {summary_path}")
        if result.deltas:
            positive_fraction = result.positive_flows / result.selected_flows
            _write_distribution_plot(plot_path, result.deltas, positive_fraction)
            print(f"[write] {plot_path}")
            print(
                f"[result] positive={result.positive_flows}/{result.selected_flows} "
                f"({positive_fraction:.2%})"
            )
        else:
            print("[warning] no flow passed the filters; distribution plot skipped")
        return 0

    topk_path = output_dir / "topk_positive.csv"
    _atomic_write_csv(topk_path, _topk_output_rows(result), TOPK_FIELDS)
    retained = len(result.topk_heap)
    print(f"[write] {topk_path}")
    print(
        f"[result] positive_candidates={result.positive_flows} "
        f"requested_top_k={args.top_k} written={retained}"
    )
    if retained < args.top_k:
        print(
            f"[warning] only {retained} positive flow(s) passed the filters; "
            "all available positive flows were written"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
