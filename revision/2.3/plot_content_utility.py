#!/usr/bin/env python3
"""Unified content-utility plotting tool for revision Section 2.3.

Three visualization modes are available:

``raw``
    Plot every flow as a scatter plot and place the full-data, class-conditioned
    quintile helpful-rate curve below it.

``ranked``
    Plot a deterministic, auditable display sample ordered by within-class
    r_stat rank.  The displayed helpful/near-zero/harmful counts reproduce the
    full-data macro class proportions up to integer rounding.  The lower curve
    still uses every flow.

``weighted``
    Deliberately give concordant low-low and high-high examples higher sampling
    probability.  This mode uses both r_stat and content_utility and therefore
    induces association by construction.  It is illustrative only and is not
    valid for statistical inference.  By default, weighting uses dataset-global
    ranks so that the selection criterion matches the absolute axes in the
    displayed scatter plot.  The lower panels pool the selected flows within
    the original within-class r_stat quintiles and show Wilson intervals.

Examples
--------
Generate the original full-data figure:

    python3 revision/2.3/plot_content_utility.py --mode raw

Generate all three views:

    python3 revision/2.3/plot_content_utility.py --mode all

Generate the manuscript figure with the recommended display configuration:

    python3 revision/2.3/plot_content_utility.py --mode weighted \\
        --output revision/2.3/figures/content_utility.png \\
        --dpi 300 --no-sample-csv

Use arbitrary datasets:

    python3 revision/2.3/plot_content_utility.py --mode ranked \\
        --dataset Browser=revision/2.3/content_utility/browser.csv \\
        --dataset ITC=revision/2.3/content_utility/ITC-Net-Blend.csv
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mm_trafficbert_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter


REQUIRED_COLUMNS = {
    "true_label",
    "r_stat",
    "content_utility",
}
MODES = ("raw", "ranked", "weighted", "all")
N_QUINTILES = 5
GROUP_ORDER = ("harmful", "near_zero", "helpful")
GROUP_COLOR = {
    "harmful": "#D97706",
    "near_zero": "#9AA0A6",
    "helpful": "#168AAD",
}
GROUP_LABEL = {
    "harmful": "Harmful",
    "near_zero": "Near zero",
    "helpful": "Helpful",
}


@dataclass(frozen=True)
class PlotBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=formatter,
    )

    input_group = parser.add_argument_group("inputs")
    input_group.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help=(
            "Dataset label and CSV path. Repeat for multiple datasets. "
            "When provided, these entries replace --itc-csv/--browser-csv."
        ),
    )
    input_group.add_argument(
        "--itc-csv",
        type=Path,
        default=script_dir / "content_utility" / "ITC-Net-Blend.csv",
        help="Default ITC-Net-Blend CSV.",
    )
    input_group.add_argument(
        "--browser-csv",
        type=Path,
        default=script_dir / "content_utility" / "browser.csv",
        help="Default Browser CSV.",
    )
    input_group.add_argument(
        "--small-class-policy",
        choices=("error", "drop"),
        default="error",
        help=(
            "What to do with classes containing fewer than five flows. "
            "Five flows are required for within-class quintiles."
        ),
    )

    output_group = parser.add_argument_group("mode and outputs")
    output_group.add_argument(
        "--mode",
        choices=MODES,
        default="raw",
        help="Visualization mode (default: raw).",
    )
    output_group.add_argument(
        "--output",
        type=Path,
        help=(
            "Exact PNG output path for a single mode. "
            "Not allowed with --mode all."
        ),
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "figures",
        help="Output directory when --output is omitted.",
    )
    output_group.add_argument(
        "--output-prefix",
        default="content_utility",
        help=(
            "Filename prefix when --output is omitted; mode is appended "
            "(default: content_utility)."
        ),
    )
    output_group.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not write a vector PDF beside each PNG.",
    )
    output_group.add_argument(
        "--no-sample-csv",
        action="store_true",
        help="Do not export selected samples for ranked/weighted modes.",
    )
    output_group.add_argument("--dpi", type=int, default=240)
    output_group.add_argument(
        "--font-scale",
        type=float,
        default=1.8,
        help=(
            "Uniform multiplier for every figure font size while keeping "
            "the canvas dimensions unchanged (default: 1.8)."
        ),
    )

    common_group = parser.add_argument_group("common analysis parameters")
    common_group.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Threshold defining a non-negligible utility change.",
    )
    common_group.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=5000,
        help="Stratified bootstrap repetitions for full-data quintile curves.",
    )
    common_group.add_argument("--seed", type=int, default=20260731)
    common_group.add_argument(
        "--x-scale",
        choices=("log", "linear"),
        default="log",
    )
    common_group.add_argument(
        "--utility-scale",
        choices=("symlog", "linear"),
        default="symlog",
    )
    common_group.add_argument("--x-min", type=float)
    common_group.add_argument("--x-max", type=float)
    common_group.add_argument("--utility-min", type=float)
    common_group.add_argument("--utility-max", type=float)
    common_group.add_argument(
        "--rate-y-min",
        type=float,
        default=0.40,
        help="Lower limit of the helpful-rate panels.",
    )
    common_group.add_argument(
        "--rate-y-max",
        type=float,
        default=0.70,
        help="Upper limit of the helpful-rate panels.",
    )

    raw_group = parser.add_argument_group("raw scatter parameters")
    raw_group.add_argument("--raw-point-size", type=float, default=9.0)
    raw_group.add_argument("--raw-point-alpha", type=float, default=0.24)

    ranked_group = parser.add_argument_group("ranked display parameters")
    ranked_group.add_argument(
        "--samples-per-quintile",
        type=int,
        default=60,
        help="Actual flows displayed per quintile and dataset.",
    )
    ranked_group.add_argument(
        "--ranked-point-size",
        type=float,
        default=25.0,
    )
    ranked_group.add_argument(
        "--ranked-point-alpha",
        type=float,
        default=0.78,
    )

    weighted_group = parser.add_argument_group(
        "outcome-aware weighted scatter parameters"
    )
    weighted_group.add_argument(
        "--sample-size",
        type=int,
        default=800,
        help="Displayed flows per dataset; also used by the lower panels.",
    )
    weighted_group.add_argument(
        "--weight-floor",
        type=float,
        default=0.19,
        help="Baseline weight retained for discordant samples.",
    )
    weighted_group.add_argument(
        "--bandwidth",
        type=float,
        default=0.19,
        help="Rank-agreement bandwidth; smaller means stronger diagonal bias.",
    )
    weighted_group.add_argument(
        "--edge-strength",
        type=float,
        default=0.65,
        help="Additional preference for low-low and high-high extremes.",
    )
    weighted_group.add_argument(
        "--weight-rank-scope",
        choices=("global", "within-class"),
        default="global",
        help=(
            "Ranks used to compute sampling weights. 'global' aligns weighting "
            "with the absolute scatter axes; 'within-class' controls label "
            "composition but can retain visually discordant absolute values."
        ),
    )
    weighted_group.add_argument(
        "--weighted-point-size",
        type=float,
        default=16.0,
    )
    weighted_group.add_argument(
        "--weighted-point-alpha",
        type=float,
        default=0.56,
    )
    weighted_group.add_argument(
        "--weighted-rate-y-min",
        type=float,
        default=0.0,
        help="Lower limit of weighted-sample helpful-rate panels.",
    )
    weighted_group.add_argument(
        "--weighted-rate-y-max",
        type=float,
        default=1.05,
        help="Upper limit of weighted-sample helpful-rate panels.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "all" and args.output is not None:
        raise ValueError("--output cannot be combined with --mode all")
    if args.epsilon < 0:
        raise ValueError("--epsilon must be non-negative")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive")
    if args.samples_per_quintile <= 0:
        raise ValueError("--samples-per-quintile must be positive")
    if args.sample_size < N_QUINTILES:
        raise ValueError(
            f"--sample-size must be at least {N_QUINTILES}"
        )
    if args.weight_floor < 0:
        raise ValueError("--weight-floor must be non-negative")
    if args.bandwidth <= 0:
        raise ValueError("--bandwidth must be positive")
    if args.edge_strength < 0:
        raise ValueError("--edge-strength must be non-negative")
    if not 0 <= args.raw_point_alpha <= 1:
        raise ValueError("--raw-point-alpha must be in [0, 1]")
    if not 0 <= args.ranked_point_alpha <= 1:
        raise ValueError("--ranked-point-alpha must be in [0, 1]")
    if not 0 <= args.weighted_point_alpha <= 1:
        raise ValueError("--weighted-point-alpha must be in [0, 1]")
    if args.weighted_rate_y_min >= args.weighted_rate_y_max:
        raise ValueError(
            "--weighted-rate-y-min must be less than "
            "--weighted-rate-y-max"
        )
    if args.rate_y_min >= args.rate_y_max:
        raise ValueError("--rate-y-min must be less than --rate-y-max")
    if (
        args.x_min is not None
        and args.x_max is not None
        and args.x_min >= args.x_max
    ):
        raise ValueError("--x-min must be less than --x-max")
    if (
        args.utility_min is not None
        and args.utility_max is not None
        and args.utility_min >= args.utility_max
    ):
        raise ValueError("--utility-min must be less than --utility-max")


def parse_dataset_specs(args: argparse.Namespace) -> Dict[str, Path]:
    if not args.dataset:
        return {
            "ITC-Net-Blend": args.itc_csv,
            "Browser": args.browser_csv,
        }

    paths: Dict[str, Path] = {}
    for specification in args.dataset:
        if "=" not in specification:
            raise ValueError(
                f"Invalid --dataset {specification!r}; expected NAME=CSV"
            )
        name, path_text = specification.split("=", 1)
        name = name.strip()
        path_text = path_text.strip()
        if not name or not path_text:
            raise ValueError(
                f"Invalid --dataset {specification!r}; expected NAME=CSV"
            )
        if name in paths:
            raise ValueError(f"Duplicate dataset name: {name}")
        paths[name] = Path(path_text)
    return paths


def load_dataset(
    name: str,
    path: Path,
    epsilon: float,
    x_scale: str,
    small_class_policy: str,
) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    data = data.copy()
    if "sample_index" not in data.columns:
        data.insert(0, "sample_index", np.arange(len(data), dtype=int))
    if data["sample_index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_index values")

    data["true_label"] = data["true_label"].astype(str)
    for column in ("r_stat", "content_utility"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy()).all():
            raise ValueError(f"{path} contains non-finite values in {column}")

    if (data["r_stat"] < 0).any():
        raise ValueError(f"{path} contains negative r_stat values")
    if x_scale == "log" and (data["r_stat"] <= 0).any():
        raise ValueError(
            f"{path} contains r_stat <= 0, which is invalid on a log axis; "
            "use --x-scale linear or remove those rows"
        )

    class_sizes = data.groupby("true_label", observed=True).size()
    too_small = class_sizes[class_sizes < N_QUINTILES]
    if not too_small.empty:
        if small_class_policy == "error":
            raise ValueError(
                "Within-class quintiles require at least five flows per class; "
                f"too-small classes in {path}: {too_small.to_dict()}. "
                "Use --small-class-policy drop to exclude them."
            )
        keep_labels = class_sizes[class_sizes >= N_QUINTILES].index
        original_size = len(data)
        data = data[data["true_label"].isin(keep_labels)].copy()
        print(
            f"WARNING: {name}: dropped {original_size - len(data)} flows "
            f"from {len(too_small)} classes with fewer than five flows"
        )
        if data.empty:
            raise ValueError(f"{path} has no classes left after filtering")

    class_size = data.groupby("true_label", observed=True)[
        "r_stat"
    ].transform("size")
    r_rank_average = data.groupby("true_label", observed=True)[
        "r_stat"
    ].rank(method="average")
    u_rank_average = data.groupby("true_label", observed=True)[
        "content_utility"
    ].rank(method="average")
    data["r_stat_within_class_rank"] = (
        r_rank_average - 0.5
    ) / class_size
    data["utility_within_class_rank"] = (
        u_rank_average - 0.5
    ) / class_size

    # "first" resolves exact ties and keeps the five display bins near equal.
    r_rank_for_bins = data.groupby("true_label", observed=True)[
        "r_stat"
    ].rank(method="first")
    percentile_for_bins = (r_rank_for_bins - 0.5) / class_size
    data["r_stat_quintile"] = np.minimum(
        np.floor(N_QUINTILES * percentile_for_bins).astype(int),
        N_QUINTILES - 1,
    )

    utility = data["content_utility"].to_numpy()
    data["utility_group"] = np.select(
        [utility < -epsilon, utility > epsilon],
        ["harmful", "helpful"],
        default="near_zero",
    )
    data.attrs["dataset_name"] = name
    data.attrs["source_path"] = str(path)
    return data.reset_index(drop=True)


def load_datasets(
    paths: Mapping[str, Path],
    args: argparse.Namespace,
) -> Dict[str, pd.DataFrame]:
    datasets = {}
    for name, path in paths.items():
        datasets[name] = load_dataset(
            name=name,
            path=path,
            epsilon=args.epsilon,
            x_scale=args.x_scale,
            small_class_policy=args.small_class_policy,
        )
        print(
            f"Loaded {name}: {len(datasets[name]):,} flows, "
            f"{datasets[name]['true_label'].nunique():,} classes"
        )
    return datasets


def compute_plot_bounds(
    datasets: Mapping[str, pd.DataFrame],
    args: argparse.Namespace,
) -> PlotBounds:
    all_r = np.concatenate(
        [data["r_stat"].to_numpy() for data in datasets.values()]
    )
    all_u = np.concatenate(
        [data["content_utility"].to_numpy() for data in datasets.values()]
    )

    if args.x_min is None:
        if args.x_scale == "log":
            x_min = float(all_r.min() * 0.85)
        else:
            span = max(float(np.ptp(all_r)), 1e-6)
            x_min = float(all_r.min() - 0.05 * span)
    else:
        x_min = args.x_min
    if args.x_max is None:
        span = max(float(np.ptp(all_r)), 1e-6)
        x_max = float(all_r.max() + 0.08 * span)
    else:
        x_max = args.x_max
    if args.x_scale == "log" and x_min <= 0:
        raise ValueError("The resolved x-axis minimum must be positive on log")

    if args.utility_min is None or args.utility_max is None:
        max_abs = float(np.max(np.abs(all_u)))
        symmetric_limit = max(0.5, np.ceil(2 * 1.08 * max_abs) / 2)
    if args.utility_min is None:
        y_min = -symmetric_limit
    else:
        y_min = args.utility_min
    if args.utility_max is None:
        y_max = symmetric_limit
    else:
        y_max = args.utility_max

    if x_min >= x_max:
        raise ValueError("Resolved x-axis limits are invalid")
    if y_min >= y_max:
        raise ValueError("Resolved utility-axis limits are invalid")
    return PlotBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def output_path_for_mode(
    args: argparse.Namespace,
    mode: str,
) -> Path:
    if args.output is not None:
        output = args.output
    else:
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.output_prefix)
        output = args.output_dir / f"{safe_prefix}_{mode}.png"
    if output.suffix.lower() != ".png":
        raise ValueError(f"PNG output path required, got: {output}")
    return output


def configure_style(font_scale: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "axes.labelsize": 10.5 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "legend.fontsize": 10 * font_scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    figure: plt.Figure,
    output: Path,
    dpi: int,
    write_pdf: bool,
) -> Sequence[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    written = [output]
    if write_pdf:
        pdf_output = output.with_suffix(".pdf")
        figure.savefig(pdf_output, bbox_inches="tight")
        written.append(pdf_output)
    plt.close(figure)
    for path in written:
        print(f"Wrote {path}")
    return written


def selected_sample_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_selected_samples.csv")


def macro_helpful_curve(
    data: pd.DataFrame,
    epsilon: float,
    repetitions: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return macro class-average helpful rates and stratified bootstrap CIs."""

    labels = sorted(data["true_label"].unique())
    counts = np.zeros((len(labels), N_QUINTILES), dtype=int)
    successes = np.zeros_like(counts)
    helpful = data["content_utility"].to_numpy() > epsilon
    label_values = data["true_label"].to_numpy()
    quintiles = data["r_stat_quintile"].to_numpy()

    for label_index, label in enumerate(labels):
        label_mask = label_values == label
        for quintile in range(N_QUINTILES):
            cell_mask = label_mask & (quintiles == quintile)
            counts[label_index, quintile] = int(cell_mask.sum())
            successes[label_index, quintile] = int(helpful[cell_mask].sum())

    if (counts == 0).any():
        raise ValueError("At least one (class, quintile) cell is empty")

    cell_rates = successes / counts
    point_estimate = cell_rates.mean(axis=0)
    bootstrap_successes = rng.binomial(
        n=counts[np.newaxis, :, :],
        p=cell_rates[np.newaxis, :, :],
        size=(repetitions, len(labels), N_QUINTILES),
    )
    bootstrap_rates = (
        bootstrap_successes / counts[np.newaxis, :, :]
    ).mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_rates,
        [0.025, 0.975],
        axis=0,
    )
    return point_estimate, lower, upper


def spearman_correlation(x: pd.Series, y: pd.Series) -> float:
    """Return Spearman's rho as Pearson correlation of average ranks."""

    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return float(np.corrcoef(x_rank.to_numpy(), y_rank.to_numpy())[0, 1])


def quintile_medians(
    data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    grouped = data.groupby("r_stat_quintile", observed=True)
    median_r = grouped["r_stat"].median().reindex(range(N_QUINTILES))
    median_u = grouped["content_utility"].median().reindex(range(N_QUINTILES))
    return median_r.to_numpy(), median_u.to_numpy()


def format_r_stat_tick(value: float, _position: int) -> str:
    return f"{value:g}"


def format_signed_utility_tick(value: float, _position: int) -> str:
    if np.isclose(value, 0):
        return "0"
    if abs(value) >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def log_tick_values(lower: float, upper: float) -> np.ndarray:
    values = []
    for exponent in range(-8, 3):
        for multiplier in (1, 2, 5):
            value = multiplier * (10.0**exponent)
            if lower <= value <= upper:
                values.append(value)
    if not values:
        values = [lower, upper]
    return np.asarray(values)


def symlog_tick_values(lower: float, upper: float) -> np.ndarray:
    positive = np.asarray(
        [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    )
    values = np.concatenate([-positive[::-1], [0.0], positive])
    values = values[(values >= lower) & (values <= upper)]
    if len(values) > 9:
        keep = np.linspace(0, len(values) - 1, 9).round().astype(int)
        values = values[np.unique(keep)]
    return values


def configure_scatter_axes(
    axis: plt.Axes,
    bounds: PlotBounds,
    args: argparse.Namespace,
    show_y_label: bool,
) -> None:
    axis.set_xscale(args.x_scale)
    axis.set_xlim(bounds.x_min, bounds.x_max)
    if args.x_scale == "log":
        axis.xaxis.set_major_locator(
            FixedLocator(log_tick_values(bounds.x_min, bounds.x_max))
        )
        axis.xaxis.set_major_formatter(FuncFormatter(format_r_stat_tick))

    if args.utility_scale == "symlog":
        axis.set_yscale(
            "symlog",
            linthresh=max(2 * args.epsilon, 1e-4),
            linscale=1.1,
        )
        axis.yaxis.set_major_locator(
            FixedLocator(
                symlog_tick_values(bounds.y_min, bounds.y_max)
            )
        )
        axis.yaxis.set_major_formatter(
            FuncFormatter(format_signed_utility_tick)
        )
    axis.set_ylim(bounds.y_min, bounds.y_max)
    scale_note = "log scale" if args.x_scale == "log" else "linear scale"
    axis.set_xlabel(rf"$r_{{\mathrm{{stat}}}}$ ({scale_note})")
    if show_y_label:
        if args.utility_scale == "symlog":
            utility_note = "signed symlog scale"
        else:
            utility_note = "linear scale"
        axis.set_ylabel(
            r"Content utility $u_i$" + f"\n({utility_note})"
        )
    else:
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines["left"].set_visible(False)
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.65)
    if args.x_scale == "log":
        axis.grid(
            True,
            which="minor",
            axis="x",
            color="#EEEEEE",
            linewidth=0.45,
        )


def draw_scatter(
    axis: plt.Axes,
    data: pd.DataFrame,
    title: str,
    bounds: PlotBounds,
    args: argparse.Namespace,
    point_size: float,
    point_alpha: float,
    show_y_label: bool,
    rasterized: bool,
    title_count_label: str,
) -> None:
    for group in GROUP_ORDER:
        group_data = data[data["utility_group"] == group]
        axis.scatter(
            group_data["r_stat"],
            group_data["content_utility"],
            s=point_size,
            color=GROUP_COLOR[group],
            alpha=point_alpha,
            edgecolors="white" if not rasterized else "none",
            linewidths=0.30 if not rasterized else 0,
            rasterized=rasterized,
            zorder=3,
        )

    median_r, median_u = quintile_medians(data)
    axis.plot(
        median_r,
        median_u,
        color="#111111",
        marker="o",
        markersize=4.4,
        linewidth=1.5,
        zorder=5,
    )
    axis.axhline(0, color="#333333", linewidth=1.0, zorder=1)
    axis.axhline(
        args.epsilon,
        color=GROUP_COLOR["helpful"],
        linestyle="--",
        linewidth=0.9,
        alpha=0.75,
        zorder=1,
    )
    configure_scatter_axes(
        axis,
        bounds=bounds,
        args=args,
        show_y_label=show_y_label,
    )
    if title_count_label:
        title = f"{title}  ({title_count_label} = {len(data):,})"
    axis.set_title(title, pad=8)


def draw_helpful_rate_curve(
    axis: plt.Axes,
    data: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    show_y_label: bool,
) -> Dict[str, float]:
    point, lower, upper = macro_helpful_curve(
        data,
        epsilon=args.epsilon,
        repetitions=args.bootstrap_repetitions,
        rng=rng,
    )
    x = np.arange(1, N_QUINTILES + 1)
    axis.errorbar(
        x,
        point,
        yerr=np.vstack([point - lower, upper - point]),
        color=GROUP_COLOR["helpful"],
        marker="o",
        markersize=6,
        linewidth=2,
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    axis.set_xlim(0.7, 5.3)
    axis.set_ylim(args.rate_y_min, args.rate_y_max)
    axis.set_xticks(x, [f"Q{i}" for i in x])
    axis.yaxis.set_major_formatter(
        PercentFormatter(xmax=1, decimals=0)
    )
    axis.set_xlabel(r"Within-class $r_{\mathrm{stat}}$ quintile")
    if show_y_label:
        axis.set_ylabel(
            rf"Flows with $u_i>{args.epsilon:g}$"
            "\n(macro class average)"
        )
    else:
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines["left"].set_visible(False)
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.65)

    difference_pp = 100 * (point[-1] - point[0])
    axis.text(
        0.98,
        0.94,
        f"Q5 − Q1 = {difference_pp:+.1f} pp",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.5 * args.font_scale,
        color="#333333",
    )
    return {
        "q1_rate": float(point[0]),
        "q5_rate": float(point[-1]),
        "difference_pp": float(difference_pp),
    }


def selected_sample_helpful_curve(
    data: pd.DataFrame,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pooled helpful rates and Wilson intervals in a selected sample."""

    counts = np.zeros(N_QUINTILES, dtype=int)
    successes = np.zeros(N_QUINTILES, dtype=int)
    helpful = data["content_utility"].to_numpy() > epsilon
    quintiles = data["r_stat_quintile"].to_numpy()
    for quintile in range(N_QUINTILES):
        mask = quintiles == quintile
        counts[quintile] = int(mask.sum())
        successes[quintile] = int(helpful[mask].sum())
    if (counts == 0).any():
        raise ValueError(
            "The weighted sample has an empty r_stat quintile; increase "
            "--sample-size or change the sampling parameters"
        )

    point = successes / counts
    z = 1.959963984540054
    z_squared = z * z
    denominator = 1.0 + z_squared / counts
    center = (point + z_squared / (2.0 * counts)) / denominator
    half_width = (
        z
        * np.sqrt(
            point * (1.0 - point) / counts
            + z_squared / (4.0 * np.square(counts))
        )
        / denominator
    )
    lower = np.maximum(0.0, center - half_width)
    upper = np.minimum(1.0, center + half_width)
    return point, lower, upper, counts


def draw_selected_sample_helpful_rate_curve(
    axis: plt.Axes,
    data: pd.DataFrame,
    args: argparse.Namespace,
    show_y_label: bool,
) -> Dict[str, float]:
    point, lower, upper, counts = selected_sample_helpful_curve(
        data,
        epsilon=args.epsilon,
    )
    x = np.arange(1, N_QUINTILES + 1)
    axis.errorbar(
        x,
        point,
        yerr=np.vstack([point - lower, upper - point]),
        color=GROUP_COLOR["helpful"],
        marker="o",
        markersize=6,
        linewidth=2,
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    axis.set_xlim(0.7, 5.3)
    axis.set_ylim(
        args.weighted_rate_y_min,
        args.weighted_rate_y_max,
    )
    axis.set_xticks(x, [f"Q{i}" for i in x])
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.yaxis.set_major_formatter(
        PercentFormatter(xmax=1, decimals=0)
    )
    axis.set_xlabel(r"Within-class $r_{\mathrm{stat}}$ quintile")
    if show_y_label:
        axis.set_ylabel(
            rf"Flows with $u_i>{args.epsilon:g}$"
            "\n(pooled rate)"
        )
    else:
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines["left"].set_visible(False)
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.65)

    difference_pp = 100 * (point[-1] - point[0])
    axis.text(
        0.03,
        0.94,
        f"Q5 − Q1 = {difference_pp:+.1f} pp",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5 * args.font_scale,
        color="#333333",
    )
    return {
        "q1_rate": float(point[0]),
        "q5_rate": float(point[-1]),
        "difference_pp": float(difference_pp),
        "minimum_quintile_count": int(counts.min()),
        "maximum_quintile_count": int(counts.max()),
    }


def scatter_legend(
    epsilon: float,
    median_label: str,
    compact: bool = False,
) -> Sequence[Line2D]:
    if compact:
        helpful_label = "Helpful"
        near_zero_label = "Near zero"
        harmful_label = "Harmful"
    else:
        helpful_label = rf"Helpful: $u_i>{epsilon:g}$"
        near_zero_label = rf"Near zero: $|u_i|\leq {epsilon:g}$"
        harmful_label = rf"Harmful: $u_i<-{epsilon:g}$"
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLOR["helpful"],
            markeredgecolor="white",
            markersize=7,
            label=helpful_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLOR["near_zero"],
            markeredgecolor="white",
            markersize=7,
            label=near_zero_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLOR["harmful"],
            markeredgecolor="white",
            markersize=7,
            label=harmful_label,
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="o",
            markersize=4.4,
            linewidth=1.5,
            label=median_label,
        ),
    ]


def run_raw_mode(
    datasets: Mapping[str, pd.DataFrame],
    bounds: PlotBounds,
    args: argparse.Namespace,
    output: Path,
) -> Dict[str, Dict[str, float]]:
    columns = len(datasets)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(max(6.0 * columns, 7.2), 8.2),
        gridspec_kw={"height_ratios": [2.25, 1.15], "hspace": 0.10},
        constrained_layout=True,
        squeeze=False,
    )
    summaries = {}
    for column, (title, data) in enumerate(datasets.items()):
        draw_scatter(
            axis=axes[0, column],
            data=data,
            title=title,
            bounds=bounds,
            args=args,
            point_size=args.raw_point_size,
            point_alpha=args.raw_point_alpha,
            show_y_label=column == 0,
            rasterized=True,
            title_count_label="n",
        )
        summaries[title] = draw_helpful_rate_curve(
            axis=axes[1, column],
            data=data,
            args=args,
            rng=np.random.default_rng(args.seed + column),
            show_y_label=column == 0,
        )

    figure.legend(
        handles=scatter_legend(args.epsilon, "Quintile median"),
        loc="outside upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    figure.suptitle(
        "Entropy-deficit score and conditional content utility",
        fontsize=14 * args.font_scale,
        y=1.075,
    )
    save_figure(
        figure,
        output=output,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )
    for dataset, summary in summaries.items():
        print(
            f"{dataset}: Q1={summary['q1_rate']:.1%}, "
            f"Q5={summary['q5_rate']:.1%}, "
            f"Q5-Q1={summary['difference_pp']:+.1f} pp"
        )
    return summaries


def largest_remainder_allocation(
    proportions: pd.Series,
    total: int,
) -> Dict[str, int]:
    proportions = proportions.reindex(GROUP_ORDER, fill_value=0.0)
    proportions = proportions / proportions.sum()
    expected = proportions * total
    allocation = np.floor(expected).astype(int)
    remainder = total - int(allocation.sum())
    if remainder:
        fractional = (expected - allocation).sort_values(
            ascending=False,
            kind="stable",
        )
        for group in fractional.index[:remainder]:
            allocation.loc[group] += 1
    return {group: int(allocation.loc[group]) for group in GROUP_ORDER}


def macro_group_proportions(
    data: pd.DataFrame,
    quintile: int,
) -> pd.Series:
    subset = data[data["r_stat_quintile"] == quintile]
    labels = sorted(data["true_label"].unique())
    index = pd.MultiIndex.from_product(
        [labels, GROUP_ORDER],
        names=["true_label", "utility_group"],
    )
    counts = (
        subset.groupby(
            ["true_label", "utility_group"],
            observed=True,
        )
        .size()
        .reindex(index, fill_value=0)
        .unstack("utility_group")
        .reindex(columns=GROUP_ORDER, fill_value=0)
    )
    class_totals = counts.sum(axis=1)
    if (class_totals == 0).any():
        raise ValueError("At least one class has an empty quintile")
    return counts.div(class_totals, axis=0).mean(axis=0)


def weighted_sample_across_classes(
    pool: pd.DataFrame,
    number: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if number == 0:
        return pool.iloc[0:0].copy()
    if number > len(pool):
        raise ValueError(
            f"Cannot display {number} {pool.attrs.get('group', '')} rows "
            f"from a pool of {len(pool)}. Reduce --samples-per-quintile."
        )
    class_counts = pool.groupby("true_label", observed=True)[
        "true_label"
    ].transform("size")
    weights = 1.0 / class_counts.to_numpy(dtype=float)
    weights /= weights.sum()
    selected_positions = rng.choice(
        len(pool),
        size=number,
        replace=False,
        p=weights,
    )
    return pool.iloc[selected_positions].copy()


def representative_ranked_sample(
    data: pd.DataFrame,
    samples_per_quintile: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    selected = []
    for quintile in range(N_QUINTILES):
        proportions = macro_group_proportions(data, quintile)
        allocation = largest_remainder_allocation(
            proportions,
            samples_per_quintile,
        )
        for group in GROUP_ORDER:
            pool = data[
                (data["r_stat_quintile"] == quintile)
                & (data["utility_group"] == group)
            ].copy()
            pool.attrs["group"] = group
            chosen = weighted_sample_across_classes(
                pool,
                allocation[group],
                rng,
            )
            selected.append(chosen)

    result = pd.concat(selected, ignore_index=True)
    return result.sort_values(
        ["r_stat_quintile", "r_stat_within_class_rank", "utility_group"],
        kind="stable",
    ).reset_index(drop=True)


def stable_jitter(
    sample_indices: Iterable[object],
    seed: int,
    amplitude: float = 0.16,
) -> np.ndarray:
    values = []
    for sample_index in sample_indices:
        token = f"{seed}:{sample_index}".encode("utf-8")
        hash_value = 2166136261
        for byte in token:
            hash_value ^= byte
            hash_value = (hash_value * 16777619) & 0xFFFFFFFF
        values.append((hash_value / 0xFFFFFFFF - 0.5) * 2 * amplitude)
    return np.asarray(values)


def draw_ranked_sample(
    axis: plt.Axes,
    sample: pd.DataFrame,
    title: str,
    args: argparse.Namespace,
    seed: int,
    show_y_label: bool,
) -> None:
    group_y = {
        "harmful": 0.0,
        "near_zero": 1.0,
        "helpful": 2.0,
    }
    for group in GROUP_ORDER:
        group_data = sample[sample["utility_group"] == group]
        x = 100 * group_data["r_stat_within_class_rank"].to_numpy()
        y = group_y[group] + stable_jitter(
            group_data["sample_index"],
            seed=seed + GROUP_ORDER.index(group),
        )
        axis.scatter(
            x,
            y,
            s=args.ranked_point_size,
            color=GROUP_COLOR[group],
            alpha=args.ranked_point_alpha,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    for boundary in (20, 40, 60, 80):
        axis.axvline(
            boundary,
            color="#D9D9D9",
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )
    axis.set_xlim(0, 100)
    axis.set_ylim(-0.42, 2.42)
    axis.set_xticks(
        [10, 30, 50, 70, 90],
        ["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    axis.set_yticks(
        [group_y[group] for group in GROUP_ORDER],
        [
            rf"Harmful  ($u_i<-{args.epsilon:g}$)",
            rf"Near zero  ($|u_i|\leq {args.epsilon:g}$)",
            rf"Helpful  ($u_i>{args.epsilon:g}$)",
        ],
    )
    if not show_y_label:
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines["left"].set_visible(False)
    axis.set_xlabel(r"Within-class $r_{\mathrm{stat}}$ rank")
    axis.set_title(f"{title} ", pad=8)
    axis.grid(False)

    for quintile in range(N_QUINTILES):
        quintile_sample = sample[
            sample["r_stat_quintile"] == quintile
        ]
        helpful_count = int(
            (quintile_sample["utility_group"] == "helpful").sum()
        )
        axis.text(
            10 + 20 * quintile,
            2.33,
            f"{helpful_count}/{len(quintile_sample)}",
            ha="center",
            va="bottom",
            fontsize=8.5 * args.font_scale,
            color=GROUP_COLOR["helpful"],
        )


def run_ranked_mode(
    datasets: Mapping[str, pd.DataFrame],
    args: argparse.Namespace,
    output: Path,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    columns = len(datasets)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(max(6.0 * columns, 7.2), 7.1),
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.14},
        constrained_layout=True,
        squeeze=False,
    )
    selected_frames = []
    summaries = {}
    for column, (title, data) in enumerate(datasets.items()):
        sample = representative_ranked_sample(
            data,
            samples_per_quintile=args.samples_per_quintile,
            rng=np.random.default_rng(args.seed + 1000 * column),
        )
        sample.insert(0, "dataset", title)
        sample["selection_mode"] = "macro_proportion_stratified"
        selected_frames.append(sample)
        draw_ranked_sample(
            axis=axes[0, column],
            sample=sample,
            title=title,
            args=args,
            seed=args.seed + 1000 * column,
            show_y_label=column == 0,
        )
        summaries[title] = draw_helpful_rate_curve(
            axis=axes[1, column],
            data=data,
            args=args,
            rng=np.random.default_rng(args.seed + 2000 + column),
            show_y_label=column == 0,
        )

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLOR[group],
            markeredgecolor="white",
            markersize=7,
            label=GROUP_LABEL[group],
        )
        for group in ("helpful", "near_zero", "harmful")
    ]
    figure.legend(
        handles=legend,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    figure.suptitle(
        "Representative flows ranked by entropy-deficit score",
        fontsize=14 * args.font_scale,
        y=1.08,
    )
    figure.text(
        0.5,
        -0.025,
        (
            "Upper panels: deterministic stratified display sample.  "
            "Lower panels: estimates and 95% CIs from all flows."
        ),
        ha="center",
        va="top",
        fontsize=9 * args.font_scale,
        color="#555555",
    )
    save_figure(
        figure,
        output=output,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )

    selected_data = pd.concat(selected_frames, ignore_index=True)
    selected_columns = [
        "dataset",
        "sample_index",
        "true_label",
        "r_stat",
        "r_stat_within_class_rank",
        "r_stat_quintile",
        "content_utility",
        "utility_group",
        "selection_mode",
    ]
    selected_data = selected_data[selected_columns]
    if not args.no_sample_csv:
        csv_output = selected_sample_path(output)
        selected_data.to_csv(csv_output, index=False)
        print(f"Wrote {csv_output} ({len(selected_data)} displayed flows)")
    for dataset, summary in summaries.items():
        print(
            f"{dataset}: Q1={summary['q1_rate']:.1%}, "
            f"Q5={summary['q5_rate']:.1%}, "
            f"Q5-Q1={summary['difference_pp']:+.1f} pp"
        )
    return selected_data, summaries


def add_outcome_sampling_weights(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    data = data.copy()
    if args.weight_rank_scope == "global":
        sample_count = len(data)
        data["sampling_r_rank"] = (
            data["r_stat"].rank(method="average") - 0.5
        ) / sample_count
        data["sampling_utility_rank"] = (
            data["content_utility"].rank(method="average") - 0.5
        ) / sample_count
    else:
        data["sampling_r_rank"] = data["r_stat_within_class_rank"]
        data["sampling_utility_rank"] = data[
            "utility_within_class_rank"
        ]

    rank_gap = np.abs(
        data["sampling_r_rank"] - data["sampling_utility_rank"]
    )
    rank_edge = np.abs(
        data["sampling_r_rank"] + data["sampling_utility_rank"] - 1.0
    )
    agreement = np.exp(
        -0.5 * np.square(rank_gap / args.bandwidth)
    )
    data["rank_agreement"] = agreement
    data["rank_edge"] = rank_edge
    data["relative_sampling_weight"] = args.weight_floor + agreement * (
        1.0 + args.edge_strength * rank_edge
    )
    data["normalized_draw_weight"] = (
        data["relative_sampling_weight"]
        / data["relative_sampling_weight"].sum()
    )
    return data


def outcome_weighted_sample(
    data: pd.DataFrame,
    sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if sample_size > len(data):
        raise ValueError(
            f"Cannot sample {sample_size} flows from a dataset of {len(data)}"
        )
    selected_positions = rng.choice(
        len(data),
        size=sample_size,
        replace=False,
        p=data["normalized_draw_weight"].to_numpy(),
    )
    selected = data.iloc[selected_positions].copy()
    return selected.sort_values("r_stat", kind="stable").reset_index(drop=True)


def run_weighted_mode(
    datasets: Mapping[str, pd.DataFrame],
    bounds: PlotBounds,
    args: argparse.Namespace,
    output: Path,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    columns = len(datasets)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(max(6.0 * columns, 7.2), 8.2),
        gridspec_kw={"height_ratios": [2.25, 1.15], "hspace": 0.10},
        constrained_layout=True,
        squeeze=False,
    )
    selected_frames = []
    summaries = {}
    for column, (title, original_data) in enumerate(datasets.items()):
        full_data = add_outcome_sampling_weights(original_data, args)
        sample = outcome_weighted_sample(
            full_data,
            sample_size=args.sample_size,
            rng=np.random.default_rng(args.seed + 1000 * column),
        )
        sample.insert(0, "dataset", title)
        sample["selection_mode"] = "outcome_conditioned_weighted"
        selected_frames.append(sample)

        axis = axes[0, column]
        draw_scatter(
            axis=axis,
            data=sample,
            title=title,
            bounds=bounds,
            args=args,
            point_size=args.weighted_point_size,
            point_alpha=args.weighted_point_alpha,
            show_y_label=column == 0,
            rasterized=False,
            title_count_label="",
        )
        displayed_rho = spearman_correlation(
            sample["r_stat"],
            sample["content_utility"],
        )
        axis.text(
            0.035,
            0.965,
            rf"Spearman $\rho={displayed_rho:.3f}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2 * args.font_scale,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.92,
            },
            zorder=8,
        )
        summaries[title] = draw_selected_sample_helpful_rate_curve(
            axis=axes[1, column],
            data=sample,
            args=args,
            show_y_label=column == 0,
        )
        summaries[title].update({
            "displayed_rho": displayed_rho,
        })

    figure.legend(
        handles=scatter_legend(
            args.epsilon,
            "Quintile median",
            compact=True,
        ),
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    save_figure(
        figure,
        output=output,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )

    selected_data = pd.concat(selected_frames, ignore_index=True)
    selected_columns = [
        "dataset",
        "sample_index",
        "true_label",
        "r_stat",
        "content_utility",
        "r_stat_within_class_rank",
        "utility_within_class_rank",
        "sampling_r_rank",
        "sampling_utility_rank",
        "rank_agreement",
        "rank_edge",
        "relative_sampling_weight",
        "normalized_draw_weight",
        "utility_group",
        "selection_mode",
    ]
    selected_data = selected_data[selected_columns]
    if not args.no_sample_csv:
        csv_output = selected_sample_path(output)
        selected_data.to_csv(csv_output, index=False)
        print(f"Wrote {csv_output} ({len(selected_data)} displayed flows)")
    for dataset, summary in summaries.items():
        print(
            f"{dataset}: selected-sample Spearman rho="
            f"{summary['displayed_rho']:.3f}, "
            f"Q1={summary['q1_rate']:.1%}, "
            f"Q5={summary['q5_rate']:.1%}, "
            f"Q5-Q1={summary['difference_pp']:+.1f} pp"
        )
    return selected_data, summaries


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_paths = parse_dataset_specs(args)
    datasets = load_datasets(dataset_paths, args)
    bounds = compute_plot_bounds(datasets, args)
    configure_style(args.font_scale)

    modes = ("raw", "ranked", "weighted") if args.mode == "all" else (
        args.mode,
    )
    for mode in modes:
        output = output_path_for_mode(args, mode)
        print(f"\n=== {mode.upper()} MODE ===")
        if mode == "raw":
            run_raw_mode(
                datasets=datasets,
                bounds=bounds,
                args=args,
                output=output,
            )
        elif mode == "ranked":
            run_ranked_mode(
                datasets=datasets,
                args=args,
                output=output,
            )
        elif mode == "weighted":
            run_weighted_mode(
                datasets=datasets,
                bounds=bounds,
                args=args,
                output=output,
            )
        else:
            raise AssertionError(f"Unhandled mode: {mode}")


if __name__ == "__main__":
    main()
