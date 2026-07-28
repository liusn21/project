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
from matplotlib.gridspec import GridSpec

from run_classifier_stage2 import Stage2Classifier
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam, apply_modality_configs
from uer.utils.seed import set_seed
from uer.opts import model_opts
from collections import defaultdict
from uer.models.multimodal_model import compute_flow_reliability_raw, compute_local_entropy


# ============================================================
# Gate extraction via forward hooks
# ============================================================

class GateCollector:
    """Collect gate_info + cross-attention probs from BidirectionalFusionLayer."""

    def __init__(self, model):
        self.records = []
        self._hooks = []
        self._current = {}
        self._register(model)

    def _register(self, model):
        fusion = model.module.fusion if hasattr(model, 'module') else model.fusion
        for i, layer in enumerate(fusion.fusion_layers):
            # Hook 1: BidirectionalFusionLayer → gate_info (gate_size, r_mod_s2r)
            h1 = layer.register_forward_hook(self._make_fusion_hook(i))
            self._hooks.append(h1)
            # Hook 2: cross_attn_size submodule → post-softmax attention probs
            # cross_attn_size is the Size←Raw cross-attention; query=size, key=raw,
            # so probs.shape = [B, n_heads, L_size, L_raw]
            h2 = layer.cross_attn_size.register_forward_hook(self._make_attn_hook(i))
            self._hooks.append(h2)

    def _make_fusion_hook(self, layer_idx):
        def hook_fn(module, input, output):
            _, _, gate_info = output
            if gate_info is not None:
                self._current.setdefault(layer_idx, {})
                self._current[layer_idx]['gate_size'] = gate_info['gate_size'].detach().cpu()
                self._current[layer_idx]['r_mod_s2r'] = gate_info['r_mod_s2r'].detach().cpu()
        return hook_fn

    def _make_attn_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # MultiHeadedAttention.forward returns (attn_output, probs)
            # probs: [B, n_heads, L_q, L_k] = [B, n_heads, L_size, L_raw]
            probs = output[1].detach().mean(dim=1).cpu()  # mean over heads → [B, L_size, L_raw]
            self._current.setdefault(layer_idx, {})
            self._current[layer_idx]['cross_attn_probs'] = probs
        return hook_fn

    def flush_batch(self, seq_lens_size, r_stat_batch,
                    seq_lens_raw=None, local_rel_raw_batch=None):
        """Store per-sample gate values + r_stat + per-byte effective attention."""
        if not self._current:
            return
        B = self._current[0]['gate_size'].shape[0]
        num_layers = len(self._current)
        for b in range(B):
            L_size = seq_lens_size[b].item()
            L_raw = seq_lens_raw[b].item() if seq_lens_raw is not None else None

            # Per-layer gate-weighted attention received by each raw byte:
            #   eff_l[j] = sum_k(gate_size[l, k] * probs[l, k, j])
            # Then average across the L fusion layers.
            # NOTE: NO normalization by sum(g). Normalization would cancel out the
            # absolute magnitude of gate_size and erase the dataset-level r_mod
            # × g_token contrast (the very thing we want to visualize). The byte-level
            # source_bias pattern survives anyway because it's encoded in p, not in g.
            eff_per_layer = []
            for layer_idx in range(num_layers):
                g = self._current[layer_idx]['gate_size'][b, :L_size]            # [L_size]
                if 'cross_attn_probs' not in self._current[layer_idx] or L_raw is None:
                    continue
                p = self._current[layer_idx]['cross_attn_probs'][b, :L_size, :L_raw]  # [L_size, L_raw]
                weighted = (g.unsqueeze(-1) * p).sum(dim=0)                       # [L_raw]
                eff_per_layer.append(weighted.numpy())
            eff_attn = np.mean(eff_per_layer, axis=0) if eff_per_layer else None  # [L_raw] or None

            r_mod_mean = float(np.mean([
                self._current[l]['r_mod_s2r'][b].item() for l in range(num_layers)
            ]))

            sample = {
                'seq_len_size': L_size,
                'r_stat': r_stat_batch[b].item(),
                'r_mod_mean': r_mod_mean,
                'layers': {},
            }
            if eff_attn is not None:
                sample['eff_attn'] = eff_attn
            for layer_idx in self._current:
                sample['layers'][layer_idx] = {
                    'gate_size': self._current[layer_idx]['gate_size'][b, :L_size].numpy(),
                    'r_mod_s2r': self._current[layer_idx]['r_mod_s2r'][b].item(),
                }
            if local_rel_raw_batch is not None and seq_lens_raw is not None:
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

    Visual narrative: structurally low-entropy regions of the payload
    (typically the leading bytes of each per-packet 64-byte block, where
    application-layer framing such as protocol record / message headers sits)
    appear as warm vertical bands; the remainder of encrypted payloads stays cold.
    Top-to-bottom should naturally show a high→low gradient by flow-level r_stat.
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

    # Auto-scale figure height to row count (compact version)
    fig_h = max(2.0, min(4.0, 0.04 * n + 1.0))
    fig, ax = plt.subplots(figsize=(4.0, fig_h))

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
        ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlabel('Byte Position', fontsize=7)
    ax.set_ylabel('Flow (sorted by r_stat ↓)', fontsize=7)
    ax.tick_params(labelsize=6)

    r_stats = np.array([r['r_stat'] for r in recs])
    ax.text(0.02, 0.98,
            f'n = {n}\nr_stat: [{r_stats.min():.3f}, {r_stats.max():.3f}]',
            transform=ax.transAxes, fontsize=6, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Local Reliability (1 - H/log n, w=16)', fontsize=6)
    cb.ax.tick_params(labelsize=6)

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
# Plot: Effective Attention (gate side, byte axis)
# ============================================================

def plot_effective_attention(records, output_path, title=None,
                             first_k=256, max_flows=80, vmax=None):
    """
    Per-flow effective gated cross-attention on the raw byte axis.

    Each row = one flow, sorted by r_stat descending (matches plot_local_entropy).
    Each column = byte position 0..first_k-1 (matches plot_local_entropy x-axis).
    Cell value = mean_l( sum_k(gate_size[l, k] * cross_attn_probs[l, k, j]) ).

    No post-hoc normalization or r_mod multiplication: gate_size already carries
    r_mod * g_token, so its absolute magnitude propagates naturally to row brightness
    and to dataset-level contrast. The byte-level source_bias pattern survives
    because it's encoded in the attention weights p, independent of g.

    Visualizes both:
      - flow-level r_stat effect: top rows brighter via r_mod * g_token magnitude
      - byte-level local_ent effect: warm columns at structurally reliable bytes,
        because source_bias inside the softmax skews attention toward those bytes

    Designed to be column-aligned with plot_local_entropy so the data side and
    model side can be compared cell-by-cell.
    """
    recs = [r for r in records
            if 'eff_attn' in r and r['eff_attn'] is not None]
    if not recs:
        print("ERROR: no eff_attn in records (cross_attn hooks may have failed)")
        return

    # Sort by flow-level r_stat descending (highest at top), matches local_ent
    recs = sorted(recs, key=lambda r: r['r_stat'], reverse=True)

    if len(recs) > max_flows:
        idx = np.linspace(0, len(recs) - 1, max_flows).astype(int)
        recs = [recs[i] for i in idx]

    n = len(recs)
    matrix = np.full((n, first_k), np.nan)
    for i, rec in enumerate(recs):
        eff = rec['eff_attn'][:first_k]
        matrix[i, :len(eff)] = eff

    masked = np.ma.array(matrix, mask=np.isnan(matrix))

    if vmax is None:
        # Adaptive vmax: 99th percentile, with a small floor for very-cool datasets
        vmax = max(float(np.nanpercentile(matrix, 99)), 1e-4)

    # Auto-scale figure height to row count (compact version)
    fig_h = max(2.0, min(4.0, 0.04 * n + 1.0))
    fig, ax = plt.subplots(figsize=(4.0, fig_h))

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
        ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlabel('Byte Position', fontsize=7)
    ax.set_ylabel('Flow (sorted by r_stat ↓)', fontsize=7)
    ax.tick_params(labelsize=6)

    r_stats = np.array([r['r_stat'] for r in recs])
    ax.text(0.02, 0.98,
            f'n = {n}\nr_stat: [{r_stats.min():.3f}, {r_stats.max():.3f}]',
            transform=ax.transAxes, fontsize=6, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Effective gated attention (per byte)', fontsize=6)
    cb.ax.tick_params(labelsize=6)

    plt.tight_layout()
    _save_fig(fig, output_path)

    top_half = matrix[:n // 2]
    bot_half = matrix[n // 2:]
    print(f"  Effective attention: {n} flows × {first_k} bytes")
    print(f"    top half mean: {np.nanmean(top_half):.6f}  (high r_stat)")
    print(f"    bot half mean: {np.nanmean(bot_half):.6f}  (low  r_stat)")
    print(f"    delta: {np.nanmean(top_half) - np.nanmean(bot_half):+.6f}")
    if r_stats.std() > 1e-6:
        # Pearson correlation between row r_stat and row mean eff_attn
        row_means = np.nanmean(matrix, axis=1)
        corr = np.corrcoef(r_stats, row_means)[0, 1]
        print(f"    pearson(r_stat, row mean eff_attn): {corr:+.4f}")

    # ===== Diagnostics: r_mod distribution and per-byte concentration =====
    # These tell us whether the flow-level gate (r_mod) is doing meaningful work
    # or whether the byte-level source_bias is the dominant mechanism.
    r_mods = np.array([r['r_mod_mean'] for r in recs])
    print(f"  r_mod distribution across {n} sampled flows:")
    print(f"    min={r_mods.min():.4f}  mean={r_mods.mean():.4f}  max={r_mods.max():.4f}  std={r_mods.std():.4f}")
    print(f"    pearson(r_stat, r_mod): {np.corrcoef(r_stats, r_mods)[0, 1]:+.4f}")
    # Per-byte concentration ratio: max byte / mean byte (per row, then averaged).
    # Large value → strong byte-level concentration via source_bias.
    # We use the raw matrix (no r_mod division) since matrix already has r_mod baked in
    # uniformly across each row, so dividing it out wouldn't change the ratio.
    row_max = np.nanmax(matrix, axis=1)
    row_mean = np.nanmean(matrix, axis=1)
    concentration = np.nanmean(row_max / np.where(row_mean > 1e-12, row_mean, 1.0))
    print(f"  byte-level concentration ratio (max/mean per row, averaged): {concentration:.2f}x")
    print(f"    (>3x means source_bias steers attention strongly toward specific bytes)")


# ============================================================
# Plot: Combined 2x2 Panel for §4.5.1 (the paper figure)
# ============================================================

def _draw_panel_cell(fig, gridspec_pos, mat, r_stats, vmax,
                     cb_label, packet_size=64,
                     show_xlabel=True, show_ylabel=True,
                     median_split=False,
                     strip_vmin=None, strip_vmax=None,
                     top_value_text=None):
    """Draw one panel cell: r_stat scale strip on the left + heatmap on the right.

    The strip is a thin gray gradient that visually anchors the row sorting.
    When ``strip_vmin`` / ``strip_vmax`` are provided, the strip is rendered on
    a SHARED scale across panels (so cross-dataset r_stat magnitudes become
    visually comparable — a uniformly bright strip means a globally-low-r_stat
    dataset, not "this dataset has an internal range that fills [0,1]").
    The actual numeric r_stat range for this panel is annotated as a second
    line of the y-label so readers can map rows back to absolute values.
    Returns (ax_main, ax_strip, im).
    """
    inner = gridspec_pos.subgridspec(1, 2, width_ratios=[0.035, 1.0], wspace=0.015)
    ax_strip = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[0, 1])

    n = mat.shape[0]

    # r_stat strip — shared cross-panel vmin/vmax preserves magnitude difference
    s_vmin = float(r_stats.min()) if strip_vmin is None else float(strip_vmin)
    s_vmax = float(r_stats.max()) if strip_vmax is None else float(strip_vmax)
    if s_vmax <= s_vmin:
        s_vmax = s_vmin + 1e-6
    ax_strip.imshow(r_stats[:, None], aspect='auto', cmap='gray_r',
                    vmin=s_vmin, vmax=s_vmax,
                    interpolation='nearest', origin='upper')
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    if show_ylabel and n >= 2:
        # Two-line ylabel: row semantics + actual numeric r_stat range
        ax_strip.set_ylabel(
            f'Flow (sorted by $r_{{\\mathrm{{stat}}}}$ $\\downarrow$)\n'
            f'$r_{{\\mathrm{{stat}}}}\\in[{r_stats[-1]:.2f},\\;{r_stats[0]:.2f}]$',
            fontsize=12
        )

    # Main heatmap
    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad('white')
    masked = np.ma.array(mat, mask=np.isnan(mat))
    im = ax_main.imshow(masked, aspect='auto', cmap=cmap,
                        vmin=0.0, vmax=vmax,
                        interpolation='nearest', origin='upper')

    # Packet boundary lines on the byte axis: bytes 64, 128, 192, ...
    # Lets the reader visually associate warm columns with header positions.
    for k in range(packet_size, mat.shape[1], packet_size):
        ax_main.axvline(x=k - 0.5, color='black', linewidth=0.35,
                        alpha=0.30, linestyle=':')

    ax_main.set_yticks([])
    ax_main.tick_params(axis='x', labelsize=10)
    ax_main.set_xlabel('Byte Position', fontsize=12)

    cb = fig.colorbar(im, ax=ax_main, fraction=0.030, pad=0.020)
    cb.set_label(cb_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.4)

    # Optional bold callout above the colorbar so the cross-cell magnitude
    # difference is readable at a glance (e.g. 0.595 vs 0.105 across the two
    # encryption regimes — finding (ii) becomes self-evident from the figure).
    if top_value_text is not None:
        cb.ax.text(0.5, 1.015, top_value_text, transform=cb.ax.transAxes,
                   fontsize=10, fontweight='bold',
                   ha='center', va='bottom', color='black')

    return ax_main, ax_strip, im


def plot_gate_panel_2x2(records_a, records_b, label_a, label_b, output_path,
                        first_k=256, max_flows=80, num_layers=6,
                        packet_size=64):
    """Combined 2x2 figure for §4.5.1 Gate Behavior Across the Encryption Spectrum.

    Layout:
        Row 1: dataset A — col (a) input local_ent | col (b) model effective_attn
        Row 2: dataset B — col (a) input local_ent | col (b) model effective_attn
        Row 3: small horizontal bar comparing 99th-percentile effective attention
               between the two datasets — quantifies the cross-dataset attenuation.

    Each cell uses its OWN vmax so byte-pattern contrast is preserved (no
    "all-cold CSTNet" failure mode). The cross-dataset magnitude difference
    that an absolute-vmax view would show is communicated by the bottom bar
    instead, decoupling the two observations onto separate visual channels.
    """

    def build_matrices(records):
        recs = sorted(records, key=lambda r: r['r_stat'], reverse=True)
        if len(recs) > max_flows:
            idx = np.linspace(0, len(recs) - 1, max_flows).astype(int)
            recs = [recs[i] for i in idx]
        n = len(recs)
        le = np.full((n, first_k), np.nan)
        ea = np.full((n, first_k), np.nan)
        for i, rec in enumerate(recs):
            local = rec.get('local_rel_raw')
            if local is not None and len(local) > 0:
                le[i, :min(len(local), first_k)] = local[:first_k]
            eff = rec.get('eff_attn')
            if eff is not None and len(eff) > 0:
                ea[i, :min(len(eff), first_k)] = eff[:first_k]
        rs = np.array([r['r_stat'] for r in recs])
        rm = np.array([r['r_mod_mean'] for r in recs])
        return le, ea, rs, rm, n

    def _diag_stats(ea, le, rs, rm):
        """Per-panel diagnostics tied to paper findings (i) and (iii).

        - top/bot mean ratio + Pearson(r_stat, row mean eff_attn): finding (i)
        - byte concentration (max/mean per row, averaged) +
          Pearson(local_rel, eff_attn) per-row averaged: finding (iii)
        - mean r_mod for cross-dataset reporting.
        """
        n = ea.shape[0]
        half = max(1, n // 2)
        top_mean = float(np.nanmean(ea[:half]))
        bot_mean = float(np.nanmean(ea[half:]))
        ratio_tb = top_mean / max(bot_mean, 1e-12)
        row_means = np.nanmean(ea, axis=1)
        valid = ~np.isnan(row_means)
        if valid.sum() >= 2 and rs[valid].std() > 1e-6 and row_means[valid].std() > 1e-12:
            pearson_eff = float(np.corrcoef(rs[valid], row_means[valid])[0, 1])
        else:
            pearson_eff = float('nan')
        if rs.std() > 1e-6 and rm.std() > 1e-6:
            pearson_rmod = float(np.corrcoef(rs, rm)[0, 1])
        else:
            pearson_rmod = float('nan')
        row_max = np.nanmax(ea, axis=1)
        safe_mean = np.where(row_means > 1e-12, row_means, 1.0)
        concentration = float(np.nanmean(row_max / safe_mean))
        corrs = []
        for i in range(n):
            le_row = le[i]
            ea_row = ea[i]
            m = ~(np.isnan(le_row) | np.isnan(ea_row))
            if m.sum() >= 2 and le_row[m].std() > 1e-9 and ea_row[m].std() > 1e-9:
                corrs.append(np.corrcoef(le_row[m], ea_row[m])[0, 1])
        corr_local_eff = float(np.nanmean(corrs)) if corrs else float('nan')
        return {
            'top_mean': top_mean,
            'bot_mean': bot_mean,
            'ratio_tb': ratio_tb,
            'pearson_eff': pearson_eff,
            'pearson_rmod': pearson_rmod,
            'concentration': concentration,
            'corr_local_eff': corr_local_eff,
            'rmod_mean': float(np.nanmean(rm)),
        }

    le_a, ea_a, rs_a, rm_a, n_a = build_matrices(records_a)
    le_b, ea_b, rs_b, rm_b, n_b = build_matrices(records_b)
    diag_a = _diag_stats(ea_a, le_a, rs_a, rm_a)
    diag_b = _diag_stats(ea_b, le_b, rs_b, rm_b)

    # Per-cell vmax: input panels stay at p95 (suppress hot-byte outliers so
    # the bulk byte structure stays readable). Model panels use p99 so the
    # colorbar top tick equals the same number quoted in the paper text
    # (0.595 vs 0.105) — this turns finding (ii) into a one-glance read of
    # the two colorbar tops.
    le_vmax_a = max(float(np.nanpercentile(le_a, 95)), 0.05)
    le_vmax_b = max(float(np.nanpercentile(le_b, 95)), 0.05)
    ea_99_a = float(np.nanpercentile(ea_a, 99))
    ea_99_b = float(np.nanpercentile(ea_b, 99))
    ea_vmax_a = max(ea_99_a, 1e-4)
    ea_vmax_b = max(ea_99_b, 1e-4)
    if ea_99_a >= ea_99_b:
        ratio = ea_99_a / max(ea_99_b, 1e-12)
        ratio_label = f'{label_a} / {label_b}'
    else:
        ratio = ea_99_b / max(ea_99_a, 1e-12)
        ratio_label = f'{label_b} / {label_a}'

    # Shared strip scale across all 4 cells: anchored at 0, top = global max.
    # This is what makes the two strips' brightness comparable, so a uniformly
    # near-white strip immediately reads as "this dataset has globally-low r_stat".
    strip_vmin = 0.0
    strip_vmax = max(float(rs_a.max()), float(rs_b.max()))

    # 2-row grid. Cross-dataset attenuation (finding ii) is already readable
    # from the per-cell colorbar tick magnitudes — no separate footer needed.
    fig = plt.figure(figsize=(9.0, 7.0))
    outer = GridSpec(2, 2, figure=fig,
                     hspace=0.24, wspace=0.30,
                     left=0.075, right=0.97, top=0.93, bottom=0.08)

    # Row labels (dataset names, vertical, outside the gridspec)
    fig.text(0, 0.730, label_a, fontsize=14, fontweight='bold',
             rotation=90, va='center', ha='center')
    fig.text(0, 0.250, label_b, fontsize=14, fontweight='bold',
             rotation=90, va='center', ha='center')

    # Column labels (above row 1)
    fig.text(0.250, 0.955, '(a) Input side: local reliability',
             fontsize=14, fontweight='bold', ha='center')
    fig.text(0.765, 0.955, '(b) Model side: gated attention',
             fontsize=14, fontweight='bold', ha='center')

    # Cells: row 1 = dataset A
    _draw_panel_cell(fig, outer[0, 0], le_a, rs_a, le_vmax_a,
                     'Local rel. ($1{-}H/\\log n$, $w{=}16$)',
                     packet_size=packet_size,
                     show_xlabel=False, show_ylabel=True,
                     strip_vmin=strip_vmin, strip_vmax=strip_vmax)
    ax_a_model, _, _ = _draw_panel_cell(
        fig, outer[0, 1], ea_a, rs_a, ea_vmax_a,
        'Effective gated attn (per byte)',
        packet_size=packet_size,
        show_xlabel=False, show_ylabel=False,
        strip_vmin=strip_vmin, strip_vmax=strip_vmax,
        top_value_text=f'{ea_99_a:.3f}')
    # Cells: row 2 = dataset B
    _draw_panel_cell(fig, outer[1, 0], le_b, rs_b, le_vmax_b,
                     'Local rel. ($1{-}H/\\log n$, $w{=}16$)',
                     packet_size=packet_size,
                     show_xlabel=True, show_ylabel=True,
                     strip_vmin=strip_vmin, strip_vmax=strip_vmax)
    ax_b_model, _, _ = _draw_panel_cell(
        fig, outer[1, 1], ea_b, rs_b, ea_vmax_b,
        'Effective gated attn (per byte)',
        packet_size=packet_size,
        show_xlabel=True, show_ylabel=False,
        strip_vmin=strip_vmin, strip_vmax=strip_vmax,
        top_value_text=f'{ea_99_b:.3f}')

    # Per-panel diagnostics overlaid on the model-side heatmaps.
    # One line, two numbers: top/bot mean ratio (finding i — how much brighter
    # the high-r_stat upper half of rows is than the low-r_stat lower half)
    # and per-row mean Pearson(local_rel, eff_attn) (finding iii — spatial
    # co-occurrence of input warmth with model warmth). Placed in the
    # bottom-right corner where the heatmap is always cool (low r_stat × late
    # bytes), so the box never occludes signal.
    def _annotate_diag(ax, d):
        text = (
            f"top/bot ${d['ratio_tb']:.2f}\\times$"
            f"     $\\rho(\\mathrm{{local}},\\!\\mathrm{{attn}}){{=}}{d['corr_local_eff']:+.2f}$"
        )
        ax.text(
            0.985, 0.030, text, transform=ax.transAxes,
            fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.30', facecolor='white',
                      edgecolor='black', linewidth=0.5, alpha=0.92),
        )

    _annotate_diag(ax_a_model, diag_a)
    _annotate_diag(ax_b_model, diag_b)

    _save_fig(fig, output_path)

    def _stats(m, name):
        return (f"    {name:14s}: mean={np.nanmean(m):.4f}  "
                f"p50={np.nanpercentile(m, 50):.4f}  "
                f"p95={np.nanpercentile(m, 95):.4f}  "
                f"p99={np.nanpercentile(m, 99):.4f}")
    print(f"  Panel 2x2: {label_a}({n_a} flows)  |  {label_b}({n_b} flows)")
    print(f"  -- {label_a} --")
    print(_stats(le_a, 'local_ent'))
    print(_stats(ea_a, 'eff_attn'))
    print(f"  -- {label_b} --")
    print(_stats(le_b, 'local_ent'))
    print(_stats(ea_b, 'eff_attn'))
    print(f"  cmap vmax: local_ent {label_a}={le_vmax_a:.4f}  "
          f"{label_b}={le_vmax_b:.4f}  (95th pct, outlier-suppressed)")
    print(f"  cmap vmax: eff_attn  {label_a}={ea_vmax_a:.4f}  "
          f"{label_b}={ea_vmax_b:.4f}  (95th pct, outlier-suppressed)")
    print(f"  caption number: 99th-pct eff_attn  {label_a}={ea_99_a:.4f}  "
          f"{label_b}={ea_99_b:.4f}  ratio={ratio:.2f}x ({ratio_label})")
    print(f"  -> add to caption: 'cross-dataset attenuation {ratio:.1f}x'")

    def _diag_print(label, d):
        print(f"  -- {label} per-panel diagnostics --")
        print(f"    flow modulation (finding i):")
        print(f"      top half mean eff_attn   : {d['top_mean']:.6f}")
        print(f"      bot half mean eff_attn   : {d['bot_mean']:.6f}")
        print(f"      top/bot ratio            : {d['ratio_tb']:.2f}x")
        print(f"      pearson(r_stat, row mean): {d['pearson_eff']:+.4f}")
        print(f"      pearson(r_stat, r_mod)   : {d['pearson_rmod']:+.4f}")
        print(f"      mean r_mod               : {d['rmod_mean']:.4f}")
        print(f"    byte focusing (finding iii):")
        print(f"      byte concentration (max/mean per row, avg): {d['concentration']:.2f}x")
        print(f"      pearson(local_rel, eff_attn) per-row mean : {d['corr_local_eff']:+.4f}")
    _diag_print(label_a, diag_a)
    _diag_print(label_b, diag_b)


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


def plot_gate_panel_1x4(records_a, records_b, label_a, label_b, output_path,
                        first_k=256, max_flows=80, num_layers=6,
                        packet_size=64):
    """Compact 1x4 horizontal layout for figure* with minimal vertical footprint.

    Layout (left to right):
        [ A-input ][ A-model ][ B-input ][ B-model ]
    Designed to save vertical space relative to the 2x2 panel: same information
    width-wise, ~2.5" tall instead of ~5.5" when set to \\textwidth.
    """
    def build_matrices(records):
        recs = sorted(records, key=lambda r: r['r_stat'], reverse=True)
        if len(recs) > max_flows:
            idx = np.linspace(0, len(recs) - 1, max_flows).astype(int)
            recs = [recs[i] for i in idx]
        n = len(recs)
        le = np.full((n, first_k), np.nan)
        ea = np.full((n, first_k), np.nan)
        for i, rec in enumerate(recs):
            local = rec.get('local_rel_raw')
            if local is not None and len(local) > 0:
                le[i, :min(len(local), first_k)] = local[:first_k]
            eff = rec.get('eff_attn')
            if eff is not None and len(eff) > 0:
                ea[i, :min(len(eff), first_k)] = eff[:first_k]
        rs = np.array([r['r_stat'] for r in recs])
        return le, ea, rs, n

    le_a, ea_a, rs_a, n_a = build_matrices(records_a)
    le_b, ea_b, rs_b, n_b = build_matrices(records_b)

    le_vmax_a = max(float(np.nanpercentile(le_a, 95)), 0.05)
    le_vmax_b = max(float(np.nanpercentile(le_b, 95)), 0.05)
    ea_vmax_a = max(float(np.nanpercentile(ea_a, 95)), 1e-4)
    ea_vmax_b = max(float(np.nanpercentile(ea_b, 95)), 1e-4)
    ea_99_a = float(np.nanpercentile(ea_a, 99))
    ea_99_b = float(np.nanpercentile(ea_b, 99))
    if ea_99_a >= ea_99_b:
        ratio = ea_99_a / max(ea_99_b, 1e-12)
        ratio_label = f'{label_a} / {label_b}'
    else:
        ratio = ea_99_b / max(ea_99_a, 1e-12)
        ratio_label = f'{label_b} / {label_a}'

    strip_vmin = 0.0
    strip_vmax = max(float(rs_a.max()), float(rs_b.max()))

    fig = plt.figure(figsize=(11.0, 3.2))
    outer = GridSpec(1, 4, figure=fig,
                     wspace=0.50,
                     left=0.05, right=0.985, top=0.80, bottom=0.20)

    # Group titles (dataset names) over each pair of panels
    fig.text(0.265, 0.92, label_a, fontsize=10, fontweight='bold', ha='center')
    fig.text(0.755, 0.92, label_b, fontsize=10, fontweight='bold', ha='center')
    # A vertical separator between the two dataset groups
    fig.add_artist(plt.Line2D([0.510, 0.510], [0.20, 0.85],
                              transform=fig.transFigure,
                              color='black', linewidth=0.5, alpha=0.4))

    # Per-cell sub-letter labels above each panel
    fig.text(0.140, 0.855, '(a) Input',  fontsize=9, ha='center')
    fig.text(0.385, 0.855, '(b) Model',  fontsize=9, ha='center')
    fig.text(0.635, 0.855, '(c) Input',  fontsize=9, ha='center')
    fig.text(0.880, 0.855, '(d) Model',  fontsize=9, ha='center')

    _draw_panel_cell(fig, outer[0, 0], le_a, rs_a, le_vmax_a,
                     'Local rel.',
                     packet_size=packet_size,
                     show_xlabel=True, show_ylabel=True,
                     strip_vmin=strip_vmin, strip_vmax=strip_vmax)
    _draw_panel_cell(fig, outer[0, 1], ea_a, rs_a, ea_vmax_a,
                     'Eff. gated attn',
                     packet_size=packet_size,
                     show_xlabel=True, show_ylabel=False,
                     strip_vmin=strip_vmin, strip_vmax=strip_vmax)
    _draw_panel_cell(fig, outer[0, 2], le_b, rs_b, le_vmax_b,
                     'Local rel.',
                     packet_size=packet_size,
                     show_xlabel=True, show_ylabel=True,
                     strip_vmin=strip_vmin, strip_vmax=strip_vmax)
    _draw_panel_cell(fig, outer[0, 3], ea_b, rs_b, ea_vmax_b,
                     'Eff. gated attn',
                     packet_size=packet_size,
                     show_xlabel=True, show_ylabel=False,
                     strip_vmin=strip_vmin, strip_vmax=strip_vmax)

    _save_fig(fig, output_path)

    print(f"  Panel 1x4: {label_a}({n_a}) | {label_b}({n_b})")
    print(f"  cmap vmax: local_ent A={le_vmax_a:.4f}  B={le_vmax_b:.4f}  "
          f"eff_attn A={ea_vmax_a:.4f}  B={ea_vmax_b:.4f}")
    print(f"  caption: 99th-pct eff_attn  {label_a}={ea_99_a:.4f}  "
          f"{label_b}={ea_99_b:.4f}  -> {ratio:.1f}x ({ratio_label})")


# ============================================================
# Helper for panel_2x2: load model + data and collect records
# ============================================================

def _collect_for_dataset(args, model_path, data_path, vocab_raw, vocab_size,
                         vocab_temporal, device, label2id_path=None,
                         label_rule=None, flow_rule=None, flow_top_k=None):
    """Load a fine-tuned Stage2Classifier checkpoint, run inference on the
    given pickle, and return the list of per-flow gate records.

    Class-level (label_rule) and flow-level (flow_rule, flow_top_k) filtering
    are applied per-dataset. When not explicitly provided, falls back to
    args.label_rule / args.flow_rule / args.flow_top_k so single-dataset usage
    keeps the original behavior.
    """
    # Per-dataset overrides (None → use shared args defaults)
    label_rule = label_rule if label_rule is not None else args.label_rule
    flow_rule = flow_rule if flow_rule is not None else args.flow_rule
    flow_top_k = flow_top_k if flow_top_k is not None else args.flow_top_k
    print(f"\n[Collect] model = {model_path}")
    print(f"[Collect] data  = {data_path}")

    state_dict = torch.load(model_path, map_location='cpu')
    if 'classifier.3.weight' in state_dict:
        labels_num = state_dict['classifier.3.weight'].shape[0]
    elif 'classifier.weight' in state_dict:
        labels_num = state_dict['classifier.weight'].shape[0]
    else:
        labels_num = 10
    print(f"  labels_num: {labels_num}")

    model = Stage2Classifier(args, len(vocab_raw), len(vocab_size),
                             len(vocab_temporal), labels_num)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  loaded {len(data)} samples")

    # ---- Class-level selection (--label_rule) ----
    id2label = None
    if label2id_path:
        with open(label2id_path, 'rb') as f:
            label2id = pickle.load(f)
        id2label = {v: k for k, v in label2id.items()}

    if data and 'label' in data[0] and label_rule != 'all':
        label_groups = defaultdict(list)
        for idx, sample in enumerate(data):
            label_groups[sample['label']].append(idx)
        class_stats = {}
        for label, indices in label_groups.items():
            sample_indices = indices[:50]
            batch = torch.LongTensor([data[i]['raw_src'] for i in sample_indices])
            r = compute_flow_reliability_raw(batch, vocab_size=len(vocab_raw))
            class_stats[label] = r.mean().item()
        if label_rule == 'low':
            selected = min(class_stats, key=class_stats.get)
        else:
            selected = max(class_stats, key=class_stats.get)
        lname = id2label[selected] if id2label and selected in id2label else str(selected)
        print(f"  class filter [{label_rule}]: {lname} "
              f"(r_stat={class_stats[selected]:.4f})")
        data = [data[i] for i in label_groups[selected]]
        print(f"  -> {len(data)} samples after class filter")

    # ---- Flow-level selection (flow_rule, flow_top_k) ----
    if flow_rule != 'all' and flow_top_k > 0 and len(data) > 0:
        all_raw = torch.LongTensor([s['raw_src'] for s in data])
        all_r = compute_flow_reliability_raw(all_raw, vocab_size=len(vocab_raw))
        k = min(flow_top_k, len(data))
        largest = (flow_rule == 'high')
        topk_idx = torch.topk(all_r, k=k, largest=largest).indices.tolist()
        sel_r = all_r[topk_idx]
        print(f"  flow filter [{flow_rule}, top-{k}]: "
              f"r_stat range [{sel_r.min().item():.4f}, {sel_r.max().item():.4f}], "
              f"mean {sel_r.mean().item():.4f}")
        data = [data[i] for i in topk_idx]

    collector = GateCollector(model)
    collect_gate_values(model, data, collector, device,
                        vocab_size_raw=len(vocab_raw),
                        max_samples=args.max_samples,
                        batch_size=args.batch_size,
                        window_size=args.itgca_window_size)
    records = collector.records
    collector.remove_hooks()
    print(f"  collected {len(records)} records")

    # Free GPU memory before loading the next checkpoint
    del model, state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records


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
    parser.add_argument("--config_path_raw", type=str, default=None,
                        help="Optional per-modality config for the Raw encoder (layers_num override).")
    parser.add_argument("--config_path_size", type=str, default=None,
                        help="Optional per-modality config for the Size encoder (layers_num override).")
    parser.add_argument("--output_path", type=str, default="results/gate.pdf")
    parser.add_argument("--title", type=str, default=None)

    parser.add_argument("--plot_type",
                        choices=["scatter", "heatmap", "heatmap_compare",
                                 "local_ent", "gate_per_flow", "effective_attn",
                                 "panel_2x2", "panel_1x4", "bar"],
                        default="scatter",
                        help="scatter: r_stat vs gate; heatmap: averaged per-position; "
                             "heatmap_compare: low vs high r_stat side-by-side; "
                             "local_ent: per-flow local raw reliability stack; "
                             "gate_per_flow: per-flow gate stack (gate-side analog of local_ent); "
                             "effective_attn: per-byte gated cross-attention (gate side, byte axis); "
                             "panel_2x2: combined §4.5.1 figure (2 datasets × {input,model}, "
                             "requires --model_path_b/--data_path_b); "
                             "bar: per-layer")

    # ---- panel_2x2 second-dataset args ----
    parser.add_argument("--model_path_b", type=str, default=None,
                        help="Second fine-tuned checkpoint (panel_2x2 only)")
    parser.add_argument("--data_path_b", type=str, default=None,
                        help="Second dataset pickle (panel_2x2 only)")
    parser.add_argument("--label_a", type=str, default=None,
                        help="Display name for dataset A in panel_2x2")
    parser.add_argument("--label_b", type=str, default=None,
                        help="Display name for dataset B in panel_2x2")
    parser.add_argument("--label2id_path_b", type=str, default=None,
                        help="label2id.pkl for dataset B (panel_2x2 only). "
                             "Dataset A reuses --label2id_path.")
    parser.add_argument("--label_rule_b", choices=["all", "low", "high"],
                        default=None,
                        help="Per-dataset class rule for B (panel_2x2 only). "
                             "Defaults to --label_rule when unset.")
    parser.add_argument("--flow_rule_b", choices=["all", "low", "high"],
                        default=None,
                        help="Per-dataset flow rule for B (panel_2x2 only). "
                             "Defaults to --flow_rule when unset.")
    parser.add_argument("--flow_top_k_b", type=int, default=None,
                        help="Per-dataset flow top-K for B (panel_2x2 only). "
                             "Defaults to --flow_top_k when unset.")

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
    args = apply_modality_configs(args)
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

    # ---- panel_2x2 / panel_1x4: self-contained dispatch ----
    if args.plot_type in ('panel_2x2', 'panel_1x4'):
        if not args.use_itgca:
            print("ERROR: --use_itgca required for panel_2x2.")
            return
        if not args.model_path_b or not args.data_path_b:
            print("ERROR: panel_2x2 requires --model_path_b and --data_path_b.")
            return
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        records_a = _collect_for_dataset(
            args, args.model_path, args.data_path,
            vocab_raw, vocab_size, vocab_temporal, device,
            label2id_path=args.label2id_path,
            label_rule=args.label_rule,
            flow_rule=args.flow_rule,
            flow_top_k=args.flow_top_k,
        )
        records_b = _collect_for_dataset(
            args, args.model_path_b, args.data_path_b,
            vocab_raw, vocab_size, vocab_temporal, device,
            label2id_path=args.label2id_path_b,
            label_rule=args.label_rule_b if args.label_rule_b is not None else args.label_rule,
            flow_rule=args.flow_rule_b if args.flow_rule_b is not None else args.flow_rule,
            flow_top_k=args.flow_top_k_b if args.flow_top_k_b is not None else args.flow_top_k,
        )
        print(f"\nGenerating {args.plot_type} plot...")
        plot_fn = plot_gate_panel_1x4 if args.plot_type == 'panel_1x4' else plot_gate_panel_2x2
        plot_fn(
            records_a, records_b,
            label_a=args.label_a or os.path.splitext(os.path.basename(args.data_path))[0],
            label_b=args.label_b or os.path.splitext(os.path.basename(args.data_path_b))[0],
            output_path=args.output_path,
            num_layers=args.num_fusion_layers,
        )
        print("\nDone.")
        return

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

    # ---- Diagnostic: print learned ITGCA calibration parameters per layer ----
    fusion = model.module.fusion if hasattr(model, 'module') else model.fusion
    print(f"\nLearned ITGCA calibration parameters (per fusion layer):")
    print(f"  {'L':>3}  {'stat_scale':>11}  {'stat_shift':>11}  {'local_scale':>11}  {'local_shift':>11}  {'alpha_mod':>10}")
    for i, layer in enumerate(fusion.fusion_layers):
        gate = layer.gate_size  # Size←Raw branch (the one with stat prior)
        stat_scale = float(gate.stat_scale.detach()) if hasattr(gate, 'stat_scale') else float('nan')
        stat_shift = float(gate.stat_shift.detach()) if hasattr(gate, 'stat_shift') else float('nan')
        alpha_mod = float(gate.alpha_modality.detach()) if hasattr(gate, 'alpha_modality') else float('nan')
        local_scale = float(layer.local_stat_scale.detach()) if hasattr(layer, 'local_stat_scale') else float('nan')
        local_shift = float(layer.local_stat_shift.detach()) if hasattr(layer, 'local_stat_shift') else float('nan')
        print(f"  L{i+1:<2}  {stat_scale:>11.4f}  {stat_shift:>11.4f}  {local_scale:>11.4f}  {local_shift:>11.4f}  {alpha_mod:>10.4f}")
    print("  (init values: stat_scale=5.0, stat_shift=-0.5, local_scale=5.0, local_shift=-0.5, alpha_mod=-2.0)")

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
    elif args.plot_type == 'effective_attn':
        plot_effective_attention(records, args.output_path, title=args.title,
                                 first_k=256, max_flows=80, vmax=args.vmax)
    elif args.plot_type == 'bar':
        plot_bar(records, args.output_path, title=args.title,
                 num_layers=args.num_fusion_layers)

    print("\nDone.")


if __name__ == "__main__":
    main() 