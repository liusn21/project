"""
ITGCA Gate Visualization (Single Dataset)

Extracts gate values from a Stage2Classifier via forward hooks.
Supports multiple plot types: scatter, heatmap, bar.
Run on different datasets separately, then compare output figures.

Usage:
    # Scatter: r_stat vs gate (recommended — shows gate adapts to entropy)
    python fine-tuning/visualize_gate.py \
        --model_path model.bin --data_path test.pkl --plot_type scatter \
        --use_itgca --output_path results/scatter.pdf ...

    # Heatmap: per-position per-layer gate for a representative sample
    python fine-tuning/visualize_gate.py \
        --model_path model.bin --data_path test.pkl --plot_type heatmap \
        --use_itgca --output_path results/heatmap.pdf ...

    # Bar: per-layer average gate
    python fine-tuning/visualize_gate.py \
        --model_path model.bin --data_path test.pkl --plot_type bar \
        --use_itgca --output_path results/bar.pdf ...
"""

import os
import sys
sys.path.append(os.getcwd())

import argparse
import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from run_classifier_stage2 import Stage2Classifier
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import model_opts
from collections import defaultdict
from uer.models.multimodal_model import compute_flow_reliability_raw, compute_local_entropy


# ============================================================
# Gate extraction via forward hooks
# ============================================================

class GateCollector:
    """Collect gate_info from BidirectionalFusionLayer via forward hooks."""

    def __init__(self, model):
        self.records = []
        self._hooks = []
        self._current = {}
        self._register(model)

    def _register(self, model):
        fusion = model.module.fusion if hasattr(model, 'module') else model.fusion
        for i, layer in enumerate(fusion.fusion_layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            _, _, gate_info = output
            if gate_info is not None:
                self._current[layer_idx] = {
                    'gate_size': gate_info['gate_size'].detach().cpu(),
                    'r_mod_s2r': gate_info['r_mod_s2r'].detach().cpu(),
                }
        return hook_fn

    def flush_batch(self, seq_lens_size, r_stat_batch,
                    seq_lens_raw=None, local_rel_raw_batch=None):
        """Store per-sample gate values + r_stat (+ optional local raw reliability)."""
        if not self._current:
            return
        B = self._current[0]['gate_size'].shape[0]
        for b in range(B):
            L = seq_lens_size[b].item()
            sample = {
                'seq_len_size': L,
                'r_stat': r_stat_batch[b].item(),
                'layers': {},
            }
            for layer_idx in self._current:
                sample['layers'][layer_idx] = {
                    'gate_size': self._current[layer_idx]['gate_size'][b, :L].numpy(),
                    'r_mod_s2r': self._current[layer_idx]['r_mod_s2r'][b].item(),
                }
            if local_rel_raw_batch is not None and seq_lens_raw is not None:
                L_raw = seq_lens_raw[b].item()
                sample['seq_len_raw'] = L_raw
                sample['local_rel_raw'] = local_rel_raw_batch[b, :L_raw].numpy()
            self.records.append(sample)
        self._current = {}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def clear(self):
        self.records = []
        self._current = {}


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def collect_gate_values(model, dataset, collector, device, vocab_size_raw,
                        max_samples=0, batch_size=32, window_size=16):
    """Run inference and collect gate values + r_stat + per-byte local reliability."""
    model.eval()
    collector.clear()

    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    data = dataset[:n]

    raw_src = torch.LongTensor([s['raw_src'] for s in data])
    packet_ids = torch.LongTensor([s['packet_ids'] for s in data])
    directions = torch.LongTensor([s['directions'] for s in data])
    size_src = torch.LongTensor([s['size_src'] for s in data])
    iat_src = torch.LongTensor([s['iat_src'] for s in data])

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        r = raw_src[start:end].to(device)
        p = packet_ids[start:end].to(device)
        d = directions[start:end].to(device)
        s = size_src[start:end].to(device)
        ia = iat_src[start:end].to(device)

        seq_lens = (s != 0).sum(dim=1).cpu()
        seq_lens_raw = (r != 0).sum(dim=1).cpu()
        r_stat_batch = compute_flow_reliability_raw(r, vocab_size=vocab_size_raw).cpu()
        local_rel_batch = compute_local_entropy(r, window_size=window_size).cpu()

        _ = model(r, p, d, s, ia)
        collector.flush_batch(seq_lens, r_stat_batch,
                              seq_lens_raw=seq_lens_raw,
                              local_rel_raw_batch=local_rel_batch)


# ============================================================
# Plot: Scatter (r_stat vs gate)
# ============================================================

def plot_scatter(records, output_path, title=None, num_layers=6):
    """Scatter plot: X=r_stat, Y=mean gate_size per sample."""
    r_stats = np.array([rec['r_stat'] for rec in records])
    gate_means = np.array([
        np.mean([rec['layers'][l]['gate_size'].mean() for l in range(num_layers)])
        for rec in records
    ])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(r_stats, gate_means, s=12, alpha=0.5, edgecolors='none', c='#2196F3')

    # Trend line
    if len(r_stats) > 2 and r_stats.std() > 1e-6:
        z = np.polyfit(r_stats, gate_means, 1)
        p = np.poly1d(z)
        x_line = np.linspace(r_stats.min(), r_stats.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#F44336', linewidth=1.5, label=f'slope={z[0]:.3f}')

        # Pearson correlation
        corr = np.corrcoef(r_stats, gate_means)[0, 1]
        ax.text(0.03, 0.97, f'r = {corr:.3f}\nn = {len(r_stats)}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.legend(fontsize=7, loc='lower right')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('r_stat (Content Entropy Prior)', fontsize=9)
    ax.set_ylabel('Mean Gate Value (Behavior←Content)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# Plot: Heatmap (averaged across flows, position-normalized)
# ============================================================

def build_avg_heatmap_normalized(records, num_layers=6, num_bins=16):
    """
    Aggregate per-flow gate values into a (num_layers, num_bins) matrix.

    Each flow's valid token range (0..seq_len_size-1) is mapped onto num_bins
    equal-width bins; the mean gate within each bin contributes to the global
    accumulator. This makes flows of different lengths comparable position-wise.
    """
    accum = np.zeros((num_layers, num_bins))
    counts = np.zeros((num_layers, num_bins))
    for rec in records:
        L = rec['seq_len_size']
        if L <= 0:
            continue
        bin_idx = np.minimum(np.arange(L) * num_bins // L, num_bins - 1).astype(int)
        for layer_idx in range(num_layers):
            gate = rec['layers'][layer_idx]['gate_size'][:L]
            accum[layer_idx] += np.bincount(bin_idx, weights=gate, minlength=num_bins)
            counts[layer_idx] += np.bincount(bin_idx, minlength=num_bins)
    return accum / np.maximum(counts, 1)


def plot_heatmap(records, output_path, title=None, num_layers=6,
                 vmin=0.0, vmax=None, num_bins=16):
    """Averaged heatmap over all records, with normalized token position bins."""
    if not records:
        print("ERROR: No records to plot heatmap.")
        return
    matrix = build_avg_heatmap_normalized(records, num_layers=num_layers, num_bins=num_bins)

    if vmax is None:
        vmax = max(matrix.max(), 0.3)

    fig, ax = plt.subplots(figsize=(5, 2.2))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r',
                   norm=Normalize(vmin=vmin, vmax=vmax),
                   interpolation='nearest', origin='lower')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Token Position (normalized bins)', fontsize=9)
    ax.set_ylabel('Fusion Layer', fontsize=9)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'L{i+1}' for i in range(num_layers)], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)

    mean_val = matrix.mean()
    ax.text(0.98, 0.92, f'n = {len(records)}\nmean = {mean_val:.3f}',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Gate Value (Behavior←Content)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save_fig(fig, output_path)

    print(f"  Heatmap: averaged over {len(records)} flows, "
          f"{num_bins} bins, mean_gate={mean_val:.4f}")


# ============================================================
# Plot: Heatmap Compare (low vs high r_stat side-by-side)
# ============================================================

def plot_heatmap_compare(records, output_path, title=None, num_layers=6,
                         num_bins=16, top_k=50, vmax=None):
    """
    Side-by-side averaged heatmap comparing flows in the low vs high r_stat
    extremes of the current dataset. Shared colorbar makes the contrast direct.

    Partition is done AFTER collect: sort all records by r_stat and take the
    bottom-K (low) and top-K (high). This is dataset-self-consistent — works
    regardless of whether the dataset has class-level entropy diversity.
    """
    if len(records) < 2:
        print("ERROR: Need >= 2 records for heatmap_compare.")
        return

    # Non-overlapping split: cap K at len/2
    actual_k = min(top_k, len(records) // 2)
    if actual_k < top_k:
        print(f"  WARNING: only {len(records)} records, using k={actual_k} per side")

    sorted_recs = sorted(records, key=lambda r: r['r_stat'])
    low_recs = sorted_recs[:actual_k]
    high_recs = sorted_recs[-actual_k:]

    low_mat = build_avg_heatmap_normalized(low_recs, num_layers, num_bins)
    high_mat = build_avg_heatmap_normalized(high_recs, num_layers, num_bins)

    if vmax is None:
        vmax = max(low_mat.max(), high_mat.max(), 0.3)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.4), constrained_layout=True)

    low_r = np.array([r['r_stat'] for r in low_recs])
    high_r = np.array([r['r_stat'] for r in high_recs])

    panels = [
        (axes[0], low_mat, f"Low r_stat  (n={len(low_recs)}, r̄={low_r.mean():.3f})"),
        (axes[1], high_mat, f"High r_stat (n={len(high_recs)}, r̄={high_r.mean():.3f})"),
    ]

    im = None
    for ax, mat, label in panels:
        im = ax.imshow(mat, aspect='auto', cmap='RdYlBu_r',
                       norm=Normalize(vmin=0.0, vmax=vmax),
                       interpolation='nearest', origin='lower')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Token Position (normalized)', fontsize=9)
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f'L{i+1}' for i in range(num_layers)], fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.text(0.97, 0.93, f'mean={mat.mean():.3f}',
                transform=ax.transAxes, fontsize=7, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    axes[0].set_ylabel('Fusion Layer', fontsize=9)

    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('Gate Value (Behavior←Content)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold')

    _save_fig(fig, output_path)

    print(f"  Heatmap compare: low(k={len(low_recs)}, r̄={low_r.mean():.4f}) "
          f"mean_gate={low_mat.mean():.4f}  |  "
          f"high(k={len(high_recs)}, r̄={high_r.mean():.4f}) "
          f"mean_gate={high_mat.mean():.4f}  |  "
          f"delta={high_mat.mean() - low_mat.mean():+.4f}")


# ============================================================
# Plot: Local Entropy (raw byte position × flow)
# ============================================================

def plot_local_entropy(records, output_path, title=None,
                       first_k=256, max_flows=80, vmax=None):
    """
    Stacked per-flow local reliability heatmap.

    Each row = one flow, each column = byte position (0..first_k-1).
    Rows are sorted by flow-level r_stat descending (high-reliability flows on top).
    Flows shorter than first_k are NaN-padded; cells render as white background.

    Visual narrative: structured regions (TLS handshake header, HTTP plaintext)
    appear as warm vertical bands; encrypted payload regions stay cold.
    Top-to-bottom should naturally show a high→low gradient.
    """
    recs = [r for r in records
            if 'local_rel_raw' in r and len(r['local_rel_raw']) > 0]
    if not recs:
        print("ERROR: no local_rel_raw in records (run collect first)")
        return

    # Sort by flow-level r_stat descending (highest at top)
    recs = sorted(recs, key=lambda r: r['r_stat'], reverse=True)

    # Subsample evenly if too many flows (keep top, bottom and a stride in between)
    if len(recs) > max_flows:
        idx = np.linspace(0, len(recs) - 1, max_flows).astype(int)
        recs = [recs[i] for i in idx]

    n = len(recs)
    matrix = np.full((n, first_k), np.nan)
    for i, rec in enumerate(recs):
        local = rec['local_rel_raw'][:first_k]
        matrix[i, :len(local)] = local

    masked = np.ma.array(matrix, mask=np.isnan(matrix))

    # Adaptive vmax: percentile-based, with a small floor so noise data still
    # spans the colormap (encrypted-only datasets have local_rel near 0).
    if vmax is None:
        vmax = max(float(np.nanpercentile(matrix, 99)), 0.05)

    # Auto-scale figure height to row count
    fig_h = max(2.5, min(6.5, 0.06 * n + 1.5))
    fig, ax = plt.subplots(figsize=(6.5, fig_h))

    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad('white')

    im = ax.imshow(masked, aspect='auto', cmap=cmap,
                   vmin=0.0, vmax=vmax,
                   interpolation='nearest', origin='upper')

    # Draw a horizontal split line at the median r_stat row, to visually anchor
    # "above = high-reliability flows, below = low-reliability flows"
    if n >= 4:
        split = n // 2
        ax.axhline(y=split - 0.5, color='black', linewidth=0.8,
                   linestyle='--', alpha=0.6)

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Byte Position', fontsize=9)
    ax.set_ylabel('Flow (sorted by r_stat ↓)', fontsize=9)
    ax.tick_params(labelsize=8)

    r_stats = np.array([r['r_stat'] for r in recs])
    ax.text(0.02, 0.98,
            f'n = {n}\nr_stat: [{r_stats.min():.3f}, {r_stats.max():.3f}]',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Local Reliability (1 - H/log n, w=16)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save_fig(fig, output_path)

    top_half = matrix[:n // 2]
    bot_half = matrix[n // 2:]
    print(f"  Local entropy heatmap: {n} flows × {first_k} bytes")
    print(f"    top  half mean: {np.nanmean(top_half):.4f}  (high r_stat)")
    print(f"    bot  half mean: {np.nanmean(bot_half):.4f}  (low  r_stat)")
    print(f"    delta: {np.nanmean(top_half) - np.nanmean(bot_half):+.4f}")


# ============================================================
# Plot: Per-flow Gate Heatmap (rows sorted by r_stat)
# ============================================================

def plot_gate_per_flow(records, output_path, title=None, num_layers=6,
                       max_flows=80, max_positions=None, vmax=None):
    """
    Per-flow gate heatmap, gate-side analog of plot_local_entropy.

    Each row = one flow, each column = packet position (un-normalized).
    Rows are sorted by flow-level r_stat descending (highest at top), so the
    visual ordering matches plot_local_entropy and the per-flow correspondence
    becomes immediately readable: top rows (high r_stat) should be warmer
    overall, bottom rows (low r_stat) cooler.

    Cell color = mean gate_size across L fusion layers at that (flow, position).
    Flows shorter than max_positions are NaN-padded; cells render as white.
    """
    if not records:
        print("ERROR: No records to plot.")
        return

    # Sort by r_stat descending (highest at top, matches local_ent)
    recs = sorted(records, key=lambda r: r['r_stat'], reverse=True)

    if len(recs) > max_flows:
        idx = np.linspace(0, len(recs) - 1, max_flows).astype(int)
        recs = [recs[i] for i in idx]

    n = len(recs)
    if max_positions is None:
        max_positions = max(rec['seq_len_size'] for rec in recs)

    matrix = np.full((n, max_positions), np.nan)
    for i, rec in enumerate(recs):
        L = rec['seq_len_size']
        if L <= 0:
            continue
        per_pos = np.mean(
            [rec['layers'][l]['gate_size'][:L] for l in range(num_layers)],
            axis=0,
        )
        matrix[i, :min(L, max_positions)] = per_pos[:max_positions]

    masked = np.ma.array(matrix, mask=np.isnan(matrix))

    if vmax is None:
        vmax = max(float(np.nanpercentile(matrix, 99)), 0.1)

    fig_h = max(2.5, min(6.5, 0.06 * n + 1.5))
    fig, ax = plt.subplots(figsize=(6.5, fig_h))

    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad('white')

    im = ax.imshow(masked, aspect='auto', cmap=cmap,
                   vmin=0.0, vmax=vmax,
                   interpolation='nearest', origin='upper')

    if n >= 4:
        split = n // 2
        ax.axhline(y=split - 0.5, color='black', linewidth=0.8,
                   linestyle='--', alpha=0.6)

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Packet Position', fontsize=9)
    ax.set_ylabel('Flow (sorted by r_stat ↓)', fontsize=9)
    ax.tick_params(labelsize=8)

    r_stats = np.array([r['r_stat'] for r in recs])
    ax.text(0.02, 0.98,
            f'n = {n}\nr_stat: [{r_stats.min():.3f}, {r_stats.max():.3f}]',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Mean Gate Value (Behavior←Content)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save_fig(fig, output_path)

    top_half = matrix[:n // 2]
    bot_half = matrix[n // 2:]
    print(f"  Per-flow gate heatmap: {n} flows × {max_positions} positions")
    print(f"    top half mean: {np.nanmean(top_half):.4f}  (high r_stat)")
    print(f"    bot half mean: {np.nanmean(bot_half):.4f}  (low  r_stat)")
    print(f"    delta: {np.nanmean(top_half) - np.nanmean(bot_half):+.4f}")


# ============================================================
# Plot: Bar (per-layer mean gate)
# ============================================================

def plot_bar(records, output_path, title=None, num_layers=6):
    """Bar chart of mean gate_size per fusion layer."""
    per_layer = {l: [] for l in range(num_layers)}
    for rec in records:
        for l in range(num_layers):
            per_layer[l].append(rec['layers'][l]['gate_size'].mean())

    means = [np.mean(per_layer[l]) for l in range(num_layers)]
    stds = [np.std(per_layer[l]) for l in range(num_layers)]
    labels = [f'L{i+1}' for i in range(num_layers)]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(labels, means, yerr=stds, capsize=3,
                  color='#2196F3', edgecolor='white', alpha=0.85)

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fusion Layer', fontsize=9)
    ax.set_ylabel('Mean Gate Value (Behavior←Content)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)

    # Annotate overall mean
    overall = np.mean(means)
    ax.axhline(y=overall, color='#F44336', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.97, overall + 0.01, f'avg={overall:.3f}',
            transform=ax.get_yaxis_transform(), fontsize=7, ha='right', color='#F44336')

    plt.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# Util
# ============================================================

def _save_fig(fig, output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    base, ext = os.path.splitext(output_path)
    other_ext = '.png' if ext == '.pdf' else '.pdf'
    fig.savefig(base + other_ext, dpi=300, bbox_inches='tight')
    print(f"Saved: {base + other_ext}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ITGCA Gate Visualization — single dataset, multiple plot types",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_path_raw", type=str, default="./test/vocab_raw.txt")
    parser.add_argument("--vocab_path_size", type=str, default="./test/vocab_size.txt")
    parser.add_argument("--vocab_path_temporal", type=str, default="./test/vocab_temporal.txt")
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json")
    parser.add_argument("--output_path", type=str, default="results/gate.pdf")
    parser.add_argument("--title", type=str, default=None)

    parser.add_argument("--plot_type",
                        choices=["scatter", "heatmap", "heatmap_compare",
                                 "local_ent", "gate_per_flow", "bar"],
                        default="scatter",
                        help="scatter: r_stat vs gate; heatmap: averaged per-position; "
                             "heatmap_compare: low vs high r_stat side-by-side; "
                             "local_ent: per-flow local raw reliability stack; "
                             "gate_per_flow: per-flow gate stack (gate-side analog of local_ent); "
                             "bar: per-layer")

    model_opts(parser)
    parser.add_argument("--num_fusion_layers", type=int, default=6)
    parser.add_argument("--use_itgca", action="store_true")
    parser.add_argument("--itgca_window_size", type=int, default=16)
    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples (0 = all)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Class selection (like plot.py's auto_select_label)
    parser.add_argument("--label_rule", choices=["all", "low", "high"],
                        default="all",
                        help="all: use all classes; low: lowest r_stat class; "
                             "high: highest r_stat class")
    parser.add_argument("--label2id_path", type=str, default=None,
                        help="Path to label2id.pkl for class name display")

    # Flow-level selection (after class selection)
    parser.add_argument("--flow_rule", choices=["all", "low", "high"],
                        default="all",
                        help="Flow-level filter inside (or across) classes by r_stat")
    parser.add_argument("--flow_top_k", type=int, default=0,
                        help="Keep top-K flows by r_stat under flow_rule (0=disabled)")

    # Heatmap specific
    parser.add_argument("--heatmap_bins", type=int, default=16,
                        help="Number of normalized position bins for averaged heatmap")
    parser.add_argument("--vmax", type=float, default=None)

    # Compat
    parser.add_argument("--is_moe", action="store_true")
    parser.add_argument("--vocab_size", type=int, required=False)
    parser.add_argument("--moebert_expert_dim", type=int, default=3072)
    parser.add_argument("--moebert_expert_num", type=int, required=False)
    parser.add_argument("--moebert_route_method", default="hash-random")
    parser.add_argument("--moebert_route_hash_list", default=None)
    parser.add_argument("--moebert_load_balance", type=float, default=0.0)

    args = parser.parse_args()

    if args.config_path:
        args = load_hyperparam(args)
    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)
    args.num_dropouts = 1
    args.use_scl = False
    args.simple_classifier = False
    args.use_attn_pooling = False
    args.dropout = getattr(args, 'dropout', 0.1)
    set_seed(args.seed)

    # ---- Vocabularies ----
    print("Loading vocabularies...")
    vocab_raw = Vocab()
    vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab()
    vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab()
    vocab_temporal.load(args.vocab_path_temporal)
    print(f"  Raw: {len(vocab_raw)}, Size: {len(vocab_size)}, Temporal: {len(vocab_temporal)}")

    # ---- Model ----
    print(f"Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu')

    if 'classifier.3.weight' in state_dict:
        labels_num = state_dict['classifier.3.weight'].shape[0]
    elif 'classifier.weight' in state_dict:
        labels_num = state_dict['classifier.weight'].shape[0]
    else:
        labels_num = 10
    print(f"  Detected labels_num: {labels_num}")

    model = Stage2Classifier(
        args, len(vocab_raw), len(vocab_size), len(vocab_temporal), labels_num
    )
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"  Device: {device}, ITGCA: {args.use_itgca}")

    if not args.use_itgca:
        print("ERROR: --use_itgca not set.")
        return

    # ---- Load data ----
    print(f"\nProcessing: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} samples")

    # ---- Class selection ----
    id2label = None
    if args.label2id_path:
        with open(args.label2id_path, 'rb') as f:
            label2id = pickle.load(f)
        id2label = {v: k for k, v in label2id.items()}

    # Always print per-class r_stat stats
    if data and 'label' in data[0]:
        label_groups = defaultdict(list)
        for idx, sample in enumerate(data):
            label_groups[sample['label']].append(idx)

        class_stats = {}
        for label, indices in label_groups.items():
            sample_indices = indices[:50]
            batch = torch.LongTensor([data[i]['raw_src'] for i in sample_indices])
            r = compute_flow_reliability_raw(batch, vocab_size=len(vocab_raw))
            class_stats[label] = r.mean().item()

        print(f"\n  Per-class r_stat ({len(class_stats)} classes):")
        for label, mean_r in sorted(class_stats.items(), key=lambda x: x[1]):
            lname = id2label[label] if id2label and label in id2label else str(label)
            print(f"    {lname:25s}: r_stat={mean_r:.4f} (n={len(label_groups[label])})")

        if args.label_rule != 'all':
            if args.label_rule == 'low':
                selected = min(class_stats, key=class_stats.get)
            else:
                selected = max(class_stats, key=class_stats.get)
            lname = id2label[selected] if id2label and selected in id2label else str(selected)
            print(f"  -> Selected [{args.label_rule}]: {lname} "
                  f"(r_stat={class_stats[selected]:.4f})")
            data = [data[i] for i in label_groups[selected]]
            print(f"  Filtered to {len(data)} samples")

    # ---- Flow selection (within current data, after class filter) ----
    # heatmap_compare partitions records *after* collect, so skip pre-filter.
    if args.plot_type == 'heatmap_compare' and args.flow_rule != 'all':
        print("  NOTE: heatmap_compare ignores --flow_rule "
              "(low and high are derived from the same record set)")
    if (args.plot_type != 'heatmap_compare' and
            args.flow_rule != 'all' and args.flow_top_k > 0 and len(data) > 0):
        all_raw = torch.LongTensor([s['raw_src'] for s in data])
        all_r = compute_flow_reliability_raw(all_raw, vocab_size=len(vocab_raw))
        k = min(args.flow_top_k, len(data))
        largest = (args.flow_rule == 'high')
        topk_idx = torch.topk(all_r, k=k, largest=largest).indices.tolist()
        selected_r = all_r[topk_idx]
        print(f"  Flow filter [{args.flow_rule}, top-{k}]: "
              f"r_stat range [{selected_r.min().item():.4f}, {selected_r.max().item():.4f}], "
              f"mean {selected_r.mean().item():.4f}")
        data = [data[i] for i in topk_idx]

    # ---- Collect ----

    collector = GateCollector(model)
    collect_gate_values(model, data, collector, device,
                        vocab_size_raw=len(vocab_raw),
                        max_samples=args.max_samples, batch_size=args.batch_size,
                        window_size=args.itgca_window_size)
    records = collector.records
    collector.remove_hooks()
    print(f"  Collected {len(records)} samples")

    # ---- Summary ----
    r_stats = [rec['r_stat'] for rec in records]
    gate_means = [np.mean([rec['layers'][l]['gate_size'].mean()
                           for l in range(args.num_fusion_layers)]) for rec in records]
    print(f"\n  r_stat:    mean={np.mean(r_stats):.4f} ± {np.std(r_stats):.4f}")
    print(f"  gate_size: mean={np.mean(gate_means):.4f} ± {np.std(gate_means):.4f}")
    if len(r_stats) > 2 and np.std(r_stats) > 1e-6:
        corr = np.corrcoef(r_stats, gate_means)[0, 1]
        print(f"  correlation(r_stat, gate): {corr:.4f}")

    # ---- Plot ----
    print(f"\nGenerating {args.plot_type} plot...")
    if args.plot_type == 'scatter':
        plot_scatter(records, args.output_path, title=args.title,
                     num_layers=args.num_fusion_layers)
    elif args.plot_type == 'heatmap':
        plot_heatmap(records, args.output_path, title=args.title,
                     num_layers=args.num_fusion_layers, vmax=args.vmax,
                     num_bins=args.heatmap_bins)
    elif args.plot_type == 'heatmap_compare':
        # flow_top_k doubles as per-side k; default to 50 if user left it 0
        per_side_k = args.flow_top_k if args.flow_top_k > 0 else 50
        plot_heatmap_compare(records, args.output_path, title=args.title,
                             num_layers=args.num_fusion_layers,
                             num_bins=args.heatmap_bins,
                             top_k=per_side_k, vmax=args.vmax)
    elif args.plot_type == 'local_ent':
        plot_local_entropy(records, args.output_path, title=args.title,
                           first_k=256, max_flows=80, vmax=args.vmax)
    elif args.plot_type == 'gate_per_flow':
        plot_gate_per_flow(records, args.output_path, title=args.title,
                           num_layers=args.num_fusion_layers,
                           max_flows=80, vmax=args.vmax)
    elif args.plot_type == 'bar':
        plot_bar(records, args.output_path, title=args.title,
                 num_layers=args.num_fusion_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()
