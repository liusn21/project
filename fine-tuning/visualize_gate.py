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
from uer.models.multimodal_model import compute_flow_reliability_raw


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

    def flush_batch(self, seq_lens_size, r_stat_batch):
        """Store per-sample gate values + r_stat."""
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
                        max_samples=0, batch_size=32):
    """Run inference and collect gate values + r_stat per sample."""
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
        r_stat_batch = compute_flow_reliability_raw(r, vocab_size=vocab_size_raw).cpu()

        _ = model(r, p, d, s, ia)
        collector.flush_batch(seq_lens, r_stat_batch)


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
# Plot: Heatmap (per-position per-layer)
# ============================================================

def build_gate_matrix(sample, num_layers=6):
    L = sample['seq_len_size']
    matrix = np.zeros((num_layers, L))
    for layer_idx in range(num_layers):
        gate = sample['layers'][layer_idx]['gate_size']
        matrix[layer_idx, :len(gate)] = gate
    return matrix


def pick_representative(records, strategy='median'):
    if not records:
        return None
    avg_gates = np.array([
        np.mean([rec['layers'][l]['gate_size'].mean() for l in rec['layers']])
        for rec in records
    ])
    if strategy == 'median':
        idx = np.argmin(np.abs(avg_gates - np.median(avg_gates)))
    elif strategy == 'min':
        idx = np.argmin(avg_gates)
    elif strategy == 'max':
        idx = np.argmax(avg_gates)
    else:
        idx = 0
    return records[idx]


def plot_heatmap(records, output_path, title=None, num_layers=6,
                 vmin=0.0, vmax=None, strategy='median'):
    """Heatmap of gate values for a representative sample."""
    sample = pick_representative(records, strategy=strategy)
    if sample is None:
        print("ERROR: No valid samples for heatmap.")
        return
    matrix = build_gate_matrix(sample, num_layers)

    if vmax is None:
        vmax = max(matrix.max(), 0.3)

    fig, ax = plt.subplots(figsize=(5, 2.2))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r',
                   norm=Normalize(vmin=vmin, vmax=vmax),
                   interpolation='nearest', origin='lower')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Token Position', fontsize=9)
    ax.set_ylabel('Fusion Layer', fontsize=9)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'L{i+1}' for i in range(num_layers)], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)

    mean_val = matrix.mean()
    ax.text(0.98, 0.92, f'mean = {mean_val:.3f}',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Gate Value (Behavior←Content)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save_fig(fig, output_path)

    print(f"  Heatmap sample: seq_len={sample['seq_len_size']}, "
          f"r_stat={sample['r_stat']:.4f}, mean_gate={mean_val:.4f}")


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
    parser.add_argument("--vocab_path_raw", type=str, required=True)
    parser.add_argument("--vocab_path_size", type=str, required=True)
    parser.add_argument("--vocab_path_temporal", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json")
    parser.add_argument("--output_path", type=str, default="results/gate.pdf")
    parser.add_argument("--title", type=str, default=None)

    parser.add_argument("--plot_type", choices=["scatter", "heatmap", "bar"],
                        default="scatter",
                        help="scatter: r_stat vs gate; heatmap: per-position; bar: per-layer")

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

    # Heatmap specific
    parser.add_argument("--sample_strategy", choices=["median", "min", "max"],
                        default="median")
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

    # ---- Collect ----
    print(f"\nProcessing: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} samples")

    collector = GateCollector(model)
    collect_gate_values(model, data, collector, device,
                        vocab_size_raw=len(vocab_raw),
                        max_samples=args.max_samples, batch_size=args.batch_size)
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
                     strategy=args.sample_strategy)
    elif args.plot_type == 'bar':
        plot_bar(records, args.output_path, title=args.title,
                 num_layers=args.num_fusion_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()
