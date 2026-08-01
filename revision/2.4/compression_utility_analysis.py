#!/usr/bin/env python3
"""Analyze utility-conditioned ITGCA correction across three exposure bins.

This is a pure CSV post-processing script.  It never loads checkpoints or
PCAPs.  Its input is one or more ``flow_results.csv`` files produced by
``compression_checkpoint_inference.py`` v2.

The main output keeps exactly the three positive compression-exposure rows:

    0 < e_i <= 0.25
    0.25 < e_i <= 0.50
    0.50 < e_i <= 1.00

For each row it reports:

* the entropy prior and conditional content utility;
* the fraction of flows with u_i > epsilon;
* upward gate correction among those helpful flows;
* continuous utility recovery by the full model;
* strict concat-rescues-behavior classification opportunities and recoveries.

Confidence intervals use a label-stratified percentile bootstrap.  A second
CSV reports full-flow monotonic trends and high-minus-low exposure contrasts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score


SCRIPT_VERSION = "1.0.0"
DEFAULT_UTILITY_THRESHOLD = 0.01
DEFAULT_BOOTSTRAP_REPETITIONS = 2000

REQUIRED_FIELDS = {
    "dataset",
    "evaluation_scope",
    "label",
    "e_i",
    "inference_r_stat",
    "content_utility",
    "full_gain",
    "correction_gain",
    "delta_learned",
    "delta_mod",
    "concat_correct",
    "behavior_correct",
    "itgca_correct",
    "stat_only_correct",
    "inference_status",
}

BIN_DEFINITIONS = (
    ("e_gt_0_le_0_25", "0 < e_i <= 0.25", lambda value: 0.0 < value <= 0.25),
    (
        "e_gt_0_25_le_0_50",
        "0.25 < e_i <= 0.50",
        lambda value: 0.25 < value <= 0.50,
    ),
    ("e_gt_0_50_le_1", "0.50 < e_i <= 1.00", lambda value: 0.50 < value <= 1.0),
)

CI_METRICS = (
    "mean_r_stat",
    "median_content_utility",
    "content_helpful_rate",
    "helpful_gate_up_rate",
    "helpful_median_delta_learned",
    "helpful_median_delta_mod",
    "helpful_median_full_gain",
    "helpful_median_correction_gain",
    "helpful_stat_only_improvement_rate",
    "utility_recovery_rate",
    "gate_supported_recovery_rate",
    "hard_rescue_rate",
    "itgca_accuracy",
    "itgca_minus_stat_only_accuracy",
)

SUMMARY_FIELDS = [
    "input_file",
    "dataset",
    "evaluation_scope",
    "group",
    "group_display",
    "flow_count",
    "mean_r_stat",
    "mean_r_stat_ci_lower",
    "mean_r_stat_ci_upper",
    "mean_content_utility",
    "median_content_utility",
    "median_content_utility_ci_lower",
    "median_content_utility_ci_upper",
    "content_helpful_count",
    "content_helpful_rate",
    "content_helpful_rate_ci_lower",
    "content_helpful_rate_ci_upper",
    "helpful_gate_up_count",
    "helpful_gate_up_rate",
    "helpful_gate_up_rate_ci_lower",
    "helpful_gate_up_rate_ci_upper",
    "helpful_median_delta_learned",
    "helpful_median_delta_learned_ci_lower",
    "helpful_median_delta_learned_ci_upper",
    "helpful_median_delta_mod",
    "helpful_median_delta_mod_ci_lower",
    "helpful_median_delta_mod_ci_upper",
    "helpful_median_full_gain",
    "helpful_median_full_gain_ci_lower",
    "helpful_median_full_gain_ci_upper",
    "helpful_median_correction_gain",
    "helpful_median_correction_gain_ci_lower",
    "helpful_median_correction_gain_ci_upper",
    "helpful_stat_only_improvement_count",
    "helpful_stat_only_improvement_rate",
    "helpful_stat_only_improvement_rate_ci_lower",
    "helpful_stat_only_improvement_rate_ci_upper",
    "utility_recovery_count",
    "utility_recovery_rate",
    "utility_recovery_rate_ci_lower",
    "utility_recovery_rate_ci_upper",
    "gate_supported_recovery_count",
    "gate_supported_recovery_rate",
    "gate_supported_recovery_rate_ci_lower",
    "gate_supported_recovery_rate_ci_upper",
    "hard_opportunity_count",
    "hard_rescue_count",
    "hard_rescue_rate",
    "hard_rescue_rate_ci_lower",
    "hard_rescue_rate_ci_upper",
    "concat_accuracy",
    "behavior_accuracy",
    "itgca_accuracy",
    "itgca_accuracy_ci_lower",
    "itgca_accuracy_ci_upper",
    "stat_only_accuracy",
    "itgca_minus_stat_only_accuracy",
    "itgca_minus_stat_only_accuracy_ci_lower",
    "itgca_minus_stat_only_accuracy_ci_upper",
    "utility_threshold",
]

TREND_FIELDS = [
    "input_file",
    "dataset",
    "evaluation_scope",
    "positive_exposure_flows",
    "spearman_e_r_stat",
    "spearman_e_r_stat_pvalue",
    "spearman_e_r_stat_ci_lower",
    "spearman_e_r_stat_ci_upper",
    "kendall_e_r_stat",
    "kendall_e_r_stat_pvalue",
    "kendall_e_r_stat_ci_lower",
    "kendall_e_r_stat_ci_upper",
    "spearman_e_content_utility",
    "spearman_e_content_utility_pvalue",
    "spearman_e_content_utility_ci_lower",
    "spearman_e_content_utility_ci_upper",
    "kendall_e_content_utility",
    "kendall_e_content_utility_pvalue",
    "kendall_e_content_utility_ci_lower",
    "kendall_e_content_utility_ci_upper",
    "high_minus_low_mean_r_stat",
    "high_minus_low_mean_r_stat_ci_lower",
    "high_minus_low_mean_r_stat_ci_upper",
    "high_minus_low_median_content_utility",
    "high_minus_low_median_content_utility_ci_lower",
    "high_minus_low_median_content_utility_ci_upper",
    "high_minus_low_helpful_rate",
    "high_minus_low_helpful_rate_ci_lower",
    "high_minus_low_helpful_rate_ci_upper",
    "median_delta_learned_helpful_minus_harmful",
    "median_delta_learned_helpful_minus_harmful_ci_lower",
    "median_delta_learned_helpful_minus_harmful_ci_upper",
    "delta_learned_auroc_helpful_vs_harmful",
    "utility_threshold",
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


def _binary(value: Any, field: str, row_number: int, path: Path) -> int:
    parsed = _finite_float(value, field, row_number, path)
    if parsed not in (0.0, 1.0):
        raise ValueError(
            f"{path}: row {row_number} has non-binary {field}={value!r}"
        )
    return int(parsed)


def _read_flow_results(path: Path) -> Tuple[str, str, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"{path} is not a v2 flow_results.csv; missing: "
                + ", ".join(missing)
            )

        for row_number, raw in enumerate(reader, start=2):
            if str(raw.get("inference_status", "")).strip().lower() != "ok":
                continue
            e_i = _finite_float(raw.get("e_i"), "e_i", row_number, path)
            if not 0.0 <= e_i <= 1.0:
                raise ValueError(f"{path}: row {row_number} has e_i outside [0,1]")
            rows.append(
                {
                    "label": str(raw.get("label", "")).strip(),
                    "e_i": e_i,
                    "r_stat": _finite_float(
                        raw.get("inference_r_stat"),
                        "inference_r_stat",
                        row_number,
                        path,
                    ),
                    "u": _finite_float(
                        raw.get("content_utility"),
                        "content_utility",
                        row_number,
                        path,
                    ),
                    "full_gain": _finite_float(
                        raw.get("full_gain"), "full_gain", row_number, path
                    ),
                    "correction_gain": _finite_float(
                        raw.get("correction_gain"),
                        "correction_gain",
                        row_number,
                        path,
                    ),
                    "delta_learned": _finite_float(
                        raw.get("delta_learned"),
                        "delta_learned",
                        row_number,
                        path,
                    ),
                    "delta_mod": _finite_float(
                        raw.get("delta_mod"), "delta_mod", row_number, path
                    ),
                    "concat_correct": _binary(
                        raw.get("concat_correct"),
                        "concat_correct",
                        row_number,
                        path,
                    ),
                    "behavior_correct": _binary(
                        raw.get("behavior_correct"),
                        "behavior_correct",
                        row_number,
                        path,
                    ),
                    "itgca_correct": _binary(
                        raw.get("itgca_correct"),
                        "itgca_correct",
                        row_number,
                        path,
                    ),
                    "stat_only_correct": _binary(
                        raw.get("stat_only_correct"),
                        "stat_only_correct",
                        row_number,
                        path,
                    ),
                    "dataset": str(raw.get("dataset", "")).strip(),
                    "scope": str(raw.get("evaluation_scope", "")).strip(),
                }
            )

    if not rows:
        raise ValueError(f"{path} contains no successfully evaluated flows")
    datasets = {row["dataset"] for row in rows if row["dataset"]}
    scopes = {row["scope"] for row in rows if row["scope"]}
    if len(datasets) > 1:
        raise ValueError(f"{path} contains multiple datasets: {sorted(datasets)}")
    if len(scopes) > 1:
        raise ValueError(f"{path} contains multiple evaluation scopes: {sorted(scopes)}")
    return (
        next(iter(datasets), path.parent.name),
        next(iter(scopes), ""),
        rows,
    )


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else math.nan


def _rate(mask: np.ndarray, denominator_mask: Optional[np.ndarray] = None) -> float:
    selected = mask if denominator_mask is None else mask[denominator_mask]
    return float(np.mean(selected)) if len(selected) else math.nan


def _metric_values(
    rows: Sequence[Mapping[str, Any]],
    utility_threshold: float,
) -> Dict[str, float]:
    if not rows:
        return {
            "flow_count": 0,
            "mean_r_stat": math.nan,
            "mean_content_utility": math.nan,
            "median_content_utility": math.nan,
            "content_helpful_count": 0,
            "content_helpful_rate": math.nan,
            "helpful_gate_up_count": 0,
            "helpful_gate_up_rate": math.nan,
            "helpful_median_delta_learned": math.nan,
            "helpful_median_delta_mod": math.nan,
            "helpful_median_full_gain": math.nan,
            "helpful_median_correction_gain": math.nan,
            "helpful_stat_only_improvement_count": 0,
            "helpful_stat_only_improvement_rate": math.nan,
            "utility_recovery_count": 0,
            "utility_recovery_rate": math.nan,
            "gate_supported_recovery_count": 0,
            "gate_supported_recovery_rate": math.nan,
            "hard_opportunity_count": 0,
            "hard_rescue_count": 0,
            "hard_rescue_rate": math.nan,
            "concat_accuracy": math.nan,
            "behavior_accuracy": math.nan,
            "itgca_accuracy": math.nan,
            "stat_only_accuracy": math.nan,
            "itgca_minus_stat_only_accuracy": math.nan,
        }

    r_stat = np.asarray([row["r_stat"] for row in rows], dtype=np.float64)
    utility = np.asarray([row["u"] for row in rows], dtype=np.float64)
    delta_learned = np.asarray(
        [row["delta_learned"] for row in rows], dtype=np.float64
    )
    delta_mod = np.asarray([row["delta_mod"] for row in rows], dtype=np.float64)
    full_gain = np.asarray([row["full_gain"] for row in rows], dtype=np.float64)
    correction_gain = np.asarray(
        [row["correction_gain"] for row in rows], dtype=np.float64
    )
    concat_correct = np.asarray(
        [row["concat_correct"] for row in rows], dtype=bool
    )
    behavior_correct = np.asarray(
        [row["behavior_correct"] for row in rows], dtype=bool
    )
    itgca_correct = np.asarray(
        [row["itgca_correct"] for row in rows], dtype=bool
    )
    stat_only_correct = np.asarray(
        [row["stat_only_correct"] for row in rows], dtype=bool
    )

    helpful = utility > utility_threshold
    gate_up = delta_learned > 0.0
    utility_recovered = helpful & (full_gain > 0.0)
    gate_supported = utility_recovered & gate_up
    stat_only_improved = helpful & (correction_gain > 0.0)
    opportunity = concat_correct & ~behavior_correct
    hard_rescue = opportunity & itgca_correct

    helpful_count = int(helpful.sum())
    opportunity_count = int(opportunity.sum())
    return {
        "flow_count": len(rows),
        "mean_r_stat": _mean(r_stat),
        "mean_content_utility": _mean(utility),
        "median_content_utility": _median(utility),
        "content_helpful_count": helpful_count,
        "content_helpful_rate": _rate(helpful),
        "helpful_gate_up_count": int((helpful & gate_up).sum()),
        "helpful_gate_up_rate": _rate(gate_up, helpful),
        "helpful_median_delta_learned": _median(delta_learned[helpful]),
        "helpful_median_delta_mod": _median(delta_mod[helpful]),
        "helpful_median_full_gain": _median(full_gain[helpful]),
        "helpful_median_correction_gain": _median(correction_gain[helpful]),
        "helpful_stat_only_improvement_count": int(stat_only_improved.sum()),
        "helpful_stat_only_improvement_rate": (
            int(stat_only_improved.sum()) / helpful_count
            if helpful_count
            else math.nan
        ),
        "utility_recovery_count": int(utility_recovered.sum()),
        "utility_recovery_rate": (
            int(utility_recovered.sum()) / helpful_count
            if helpful_count
            else math.nan
        ),
        "gate_supported_recovery_count": int(gate_supported.sum()),
        "gate_supported_recovery_rate": (
            int(gate_supported.sum()) / helpful_count
            if helpful_count
            else math.nan
        ),
        "hard_opportunity_count": opportunity_count,
        "hard_rescue_count": int(hard_rescue.sum()),
        "hard_rescue_rate": (
            int(hard_rescue.sum()) / opportunity_count
            if opportunity_count
            else math.nan
        ),
        "concat_accuracy": _rate(concat_correct),
        "behavior_accuracy": _rate(behavior_correct),
        "itgca_accuracy": _rate(itgca_correct),
        "stat_only_accuracy": _rate(stat_only_correct),
        "itgca_minus_stat_only_accuracy": (
            _rate(itgca_correct) - _rate(stat_only_correct)
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
    for label_rows in by_label.values():
        indices = rng.integers(0, len(label_rows), size=len(label_rows))
        sampled.extend(label_rows[int(index)] for index in indices)
    return sampled


def _percentile_interval(values: Iterable[float]) -> Tuple[float, float]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=np.float64,
    )
    if not len(finite):
        return math.nan, math.nan
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def _bootstrap_summary(
    rows: Sequence[Mapping[str, Any]],
    utility_threshold: float,
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    replicates: Dict[str, List[float]] = {metric: [] for metric in CI_METRICS}
    for _ in range(repetitions):
        metrics = _metric_values(
            _stratified_sample(rows, rng),
            utility_threshold,
        )
        for metric in CI_METRICS:
            replicates[metric].append(float(metrics[metric]))
    return {
        metric: _percentile_interval(values)
        for metric, values in replicates.items()
    }


def _correlation(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    method: str,
) -> Tuple[float, float]:
    x = np.asarray([row["e_i"] for row in rows], dtype=np.float64)
    y = np.asarray([row[target] for row in rows], dtype=np.float64)
    result = spearmanr(x, y) if method == "spearman" else kendalltau(x, y)
    statistic = getattr(result, "statistic", result[0])
    pvalue = getattr(result, "pvalue", result[1])
    return float(statistic), float(pvalue)


def _high_low_contrasts(
    rows: Sequence[Mapping[str, Any]],
    utility_threshold: float,
) -> Dict[str, float]:
    low = [row for row in rows if 0.0 < row["e_i"] <= 0.25]
    high = [row for row in rows if 0.50 < row["e_i"] <= 1.0]
    low_metrics = _metric_values(low, utility_threshold)
    high_metrics = _metric_values(high, utility_threshold)

    helpful_delta = np.asarray(
        [
            row["delta_learned"]
            for row in rows
            if row["u"] > utility_threshold
        ],
        dtype=np.float64,
    )
    harmful_delta = np.asarray(
        [
            row["delta_learned"]
            for row in rows
            if row["u"] < -utility_threshold
        ],
        dtype=np.float64,
    )
    return {
        "high_minus_low_mean_r_stat": (
            high_metrics["mean_r_stat"] - low_metrics["mean_r_stat"]
        ),
        "high_minus_low_median_content_utility": (
            high_metrics["median_content_utility"]
            - low_metrics["median_content_utility"]
        ),
        "high_minus_low_helpful_rate": (
            high_metrics["content_helpful_rate"]
            - low_metrics["content_helpful_rate"]
        ),
        "median_delta_learned_helpful_minus_harmful": (
            _median(helpful_delta) - _median(harmful_delta)
        ),
    }


def _correction_auroc(
    rows: Sequence[Mapping[str, Any]],
    utility_threshold: float,
) -> float:
    selected = [
        row
        for row in rows
        if row["u"] > utility_threshold or row["u"] < -utility_threshold
    ]
    targets = np.asarray(
        [int(row["u"] > utility_threshold) for row in selected],
        dtype=np.int64,
    )
    if len(np.unique(targets)) != 2:
        return math.nan
    scores = np.asarray(
        [row["delta_learned"] for row in selected],
        dtype=np.float64,
    )
    return float(roc_auc_score(targets, scores))


def _build_trend_row(
    input_path: Path,
    dataset: str,
    scope: str,
    rows: Sequence[Mapping[str, Any]],
    utility_threshold: float,
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    positive = [row for row in rows if row["e_i"] > 0.0]
    if not positive:
        raise ValueError(f"{input_path} has no flow with e_i > 0")

    correlations = {}
    for method in ("spearman", "kendall"):
        for target, output_name in (
            ("r_stat", "r_stat"),
            ("u", "content_utility"),
        ):
            statistic, pvalue = _correlation(positive, target, method)
            correlations[f"{method}_e_{output_name}"] = statistic
            correlations[f"{method}_e_{output_name}_pvalue"] = pvalue

    bootstrap_values: Dict[str, List[float]] = defaultdict(list)
    contrast_names = tuple(_high_low_contrasts(positive, utility_threshold))
    for _ in range(repetitions):
        sampled = _stratified_sample(positive, rng)
        for method in ("spearman", "kendall"):
            for target, output_name in (
                ("r_stat", "r_stat"),
                ("u", "content_utility"),
            ):
                statistic, _ = _correlation(sampled, target, method)
                bootstrap_values[f"{method}_e_{output_name}"].append(statistic)
        contrasts = _high_low_contrasts(sampled, utility_threshold)
        for name in contrast_names:
            bootstrap_values[name].append(contrasts[name])

    row: Dict[str, Any] = {
        "input_file": str(input_path),
        "dataset": dataset,
        "evaluation_scope": scope,
        "positive_exposure_flows": len(positive),
        "utility_threshold": utility_threshold,
        "delta_learned_auroc_helpful_vs_harmful": _correction_auroc(
            positive, utility_threshold
        ),
    }
    row.update(correlations)
    row.update(_high_low_contrasts(positive, utility_threshold))
    for name, values in bootstrap_values.items():
        lower, upper = _percentile_interval(values)
        row[f"{name}_ci_lower"] = lower
        row[f"{name}_ci_upper"] = upper
    return row


def _atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


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
            "Build the three-row compression/utility/correction table from "
            "one or more v2 flow_results.csv files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("flow_results", nargs="+", type=Path)
    parser.add_argument(
        "--utility-threshold",
        type=_nonnegative_float,
        default=DEFAULT_UTILITY_THRESHOLD,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=_positive_integer,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing utility_summary.csv/trend_statistics.csv",
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_paths = [path.expanduser().resolve() for path in args.flow_results]
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise ValueError("input file(s) not found: " + ", ".join(map(str, missing)))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "utility_summary.csv"
    trend_path = output_dir / "trend_statistics.csv"
    existing = [path for path in (summary_path, trend_path) if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(
            "output file(s) already exist; pass --overwrite: "
            + ", ".join(map(str, existing))
        )

    rng = np.random.default_rng(args.seed)
    summary_rows: List[Dict[str, Any]] = []
    trend_rows: List[Dict[str, Any]] = []
    seen_datasets = set()

    for input_path in input_paths:
        dataset, scope, rows = _read_flow_results(input_path)
        if dataset in seen_datasets:
            raise ValueError(
                f"dataset {dataset!r} appears in more than one input; "
                "refusing to silently duplicate flows"
            )
        seen_datasets.add(dataset)

        for group, display, predicate in BIN_DEFINITIONS:
            selected = [row for row in rows if predicate(row["e_i"])]
            point = _metric_values(selected, args.utility_threshold)
            intervals = _bootstrap_summary(
                selected,
                args.utility_threshold,
                args.bootstrap_repetitions,
                rng,
            ) if selected else {metric: (math.nan, math.nan) for metric in CI_METRICS}

            output_row: Dict[str, Any] = {
                "input_file": str(input_path),
                "dataset": dataset,
                "evaluation_scope": scope,
                "group": group,
                "group_display": display,
                "utility_threshold": args.utility_threshold,
            }
            output_row.update(point)
            for metric, (lower, upper) in intervals.items():
                output_row[f"{metric}_ci_lower"] = lower
                output_row[f"{metric}_ci_upper"] = upper
            summary_rows.append(output_row)

        trend_rows.append(
            _build_trend_row(
                input_path,
                dataset,
                scope,
                rows,
                args.utility_threshold,
                args.bootstrap_repetitions,
                rng,
            )
        )

    _atomic_write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _atomic_write_csv(trend_path, trend_rows, TREND_FIELDS)
    print(f"[write] {summary_path}")
    print(f"[write] {trend_path}")
    print(
        f"[done] datasets={len(seen_datasets)} "
        f"rows={len(summary_rows)} threshold={args.utility_threshold:g}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
