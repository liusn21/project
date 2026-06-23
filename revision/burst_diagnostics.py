#!/usr/bin/env python3
"""
burst_diagnostics.py — training-free diagnostic for the next-burst pre-training task.

Reads the behavior-side corpus (the `corpus_size.txt` produced by
multimodal_data_gen.py), applies PURE-DIRECTION burst segmentation, and reports
whether the next-burst summary objective is viable BEFORE spending any GPU.

No model, no GPU, single streaming pass. Three groups (the locked diagnostic spec):

  A. Segmentation health   — K (bursts/flow) distribution, dark-flow fraction,
                             supervision count, truncation rate.
  B. Target distributions  — per-burst n (packets), v (payload bytes), g (lead gap),
                             with bin occupancy, so we can set the prediction bins.
  C. Signal health         — lag-1 (cross-direction) and lag-2 (same-direction)
                             mutual information + persistence-vs-marginal baselines,
                             ranking which of n / v / g carry learnable sequential signal.

Corpus format (one flow per block, see write_flow() in multimodal_data_gen.py):
    ||
    <protocol:int>                       # 6=TCP, 17=UDP
    <size_token size_token ...>          # size_token = payload_size * direction + 1500
    <iat_token iat_token ...>            # iat_token  = int(sigmoid(log10(IAT))*1000), 0..999

Decisions baked in (discussed in design):
  * Pure-direction segmentation: new burst iff packet direction flips.
  * Mirror training truncation: stats computed on the first (seq_length_size - 2)
    packets, since that is what the model actually sees. Full-flow stats also shown.
  * Truncation handling = option (2): the last burst is dropped as a TARGET only when
    the flow is truncated (length hit the cap); complete flows keep their last burst.
  * g target = the IAT token at a burst's first packet (the inter-burst gap). Burst 0's
    g is the epsilon artifact (first packet has no predecessor) and is excluded.

Usage:
    python data_generation/burst_diagnostics.py \
        --corpus_path data/corpus_size.txt \
        --seq_length_size 256 \
        [--max_flows 200000] [--plot diag_out/]

NOTE: mutual information is bin-sensitive and the bins here are provisional. If a
target lands near NMI~0, re-check with finer bins before declaring "no signal".
"""

import argparse
import math
from bisect import bisect_right
from collections import Counter


# --------------------------------------------------------------------------- #
# IAT token <-> seconds (mirror _compute_iat_tokens in multimodal_data_gen.py) #
# --------------------------------------------------------------------------- #
def iat_seconds_to_token(sec):
    """Forward map used by the data generator (for building readable g-bin edges)."""
    sec = max(sec, 1e-6)
    normalized = 1.0 / (1.0 + math.exp(-math.log10(sec)))  # sigmoid(log10(iat))
    return min(max(int(normalized * 1000), 0), 999)


def iat_token_to_seconds(token):
    """Approximate inverse (bin-center), for human-readable labels only."""
    p = (token + 0.5) / 1000.0
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    log10_iat = math.log(p / (1.0 - p))  # logit(p) == log10(iat)
    return 10.0 ** log10_iat


# --------------------------------------------------------------------------- #
#  Fixed, interpretable bins (used for both occupancy and MI joints)          #
# --------------------------------------------------------------------------- #
def n_bin(count):
    """8 ordinal bins: {1,2,3,4,5-8,9-16,17-32,33+}."""
    if count <= 4:
        return count - 1            # 1->0, 2->1, 3->2, 4->3
    if count <= 8:
        return 4
    if count <= 16:
        return 5
    if count <= 32:
        return 6
    return 7


N_LABELS = ["1", "2", "3", "4", "5-8", "9-16", "17-32", "33+"]

# payload bytes per burst (upper bounds -> 11 bins)
V_BOUNDS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]
V_LABELS = ["<64", "64-128", "128-256", "256-512", "512-1K", "1-2K",
            "2-4K", "4-8K", "8-16K", "16-64K", ">=64K"]

# inter-burst gap, by real-time decade (token edges derived from the forward map)
_G_SEC_EDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
G_BOUNDS = [iat_seconds_to_token(s) for s in _G_SEC_EDGES]   # token thresholds
G_LABELS = ["<0.1ms", "0.1-1ms", "1-10ms", "10-100ms", "0.1-1s", "1-10s", ">10s"]


def v_bin(v):
    return bisect_right(V_BOUNDS, v)


def g_bin(token):
    return bisect_right(G_BOUNDS, token)


# --------------------------------------------------------------------------- #
#  Corpus parsing                                                             #
# --------------------------------------------------------------------------- #
def iter_flows(path, max_flows=0):
    """Yield (protocol, size_tokens, iat_tokens) per flow, streaming."""
    proto = sizes = iats = None
    field = -1          # -1: before first '||'; 0: proto; 1: size; 2: iat
    count = 0
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line == "||":
                if proto is not None and sizes is not None and iats is not None:
                    yield proto, sizes, iats
                    count += 1
                    if max_flows and count >= max_flows:
                        return
                proto = sizes = iats = None
                field = 0
                continue
            if field == 0:
                try:
                    proto = int(line)
                except ValueError:
                    proto = -1
                field = 1
            elif field == 1:
                sizes = [int(x) for x in line.split()] if line else []
                field = 2
            elif field == 2:
                iats = [int(x) for x in line.split()] if line else []
                field = 3
        if proto is not None and sizes is not None and iats is not None:
            yield proto, sizes, iats


def recover_dir_size(size_tokens):
    """size_token = size*dir + 1500  ->  (dir in {+1,-1,0}, payload_size>=0)."""
    dirs, sizes = [], []
    for t in size_tokens:
        d = t - 1500
        if d > 0:
            dirs.append(1)
            sizes.append(d)
        elif d < 0:
            dirs.append(-1)
            sizes.append(-d)
        else:                       # size 0 == zero payload: should not occur
            dirs.append(0)
            sizes.append(0)
    return dirs, sizes


def segment_pure_direction(dirs):
    """burst_id per packet; new burst whenever direction flips."""
    if not dirs:
        return []
    bid = [0] * len(dirs)
    cur = 0
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            cur += 1
        bid[i] = cur
    return bid


def flow_bursts(size_tokens, iat_tokens, max_num_tokens):
    """Segment one flow's truncated view; return per-burst summaries + flags."""
    L = min(len(size_tokens), len(iat_tokens))          # mirror data.py min-align
    if L == 0:
        return None
    full_len = L
    trunc = full_len > max_num_tokens                   # flow hit the cap
    st = size_tokens[:max_num_tokens]
    it = iat_tokens[:max_num_tokens]

    dirs, sizes = recover_dir_size(st)
    bid = segment_pure_direction(dirs)
    if not bid:
        return None
    K = bid[-1] + 1

    n = [0] * K
    v = [0] * K
    g = [None] * K
    zero_anom = 0
    for i, b in enumerate(bid):
        n[b] += 1
        v[b] += sizes[i]
        if dirs[i] == 0:
            zero_anom += 1
        if i == 0 or bid[i] != bid[i - 1]:              # first packet of burst b
            g[b] = it[i]
    # direction per burst (first packet) — for the alternation sanity check
    burst_dir = []
    seen = -1
    for i, b in enumerate(bid):
        if b != seen:
            burst_dir.append(dirs[i])
            seen = b
    return {"K": K, "n": n, "v": v, "g": g, "trunc": trunc,
            "full_len": full_len, "burst_dir": burst_dir, "zero_anom": zero_anom}


# --------------------------------------------------------------------------- #
#  Mutual information helpers (from a joint Counter of (x_bin, y_bin))         #
# --------------------------------------------------------------------------- #
def mi_report(joint):
    """Return dict with MI, H(next), NMI, persistence acc, marginal acc, n."""
    total = sum(joint.values())
    if total == 0:
        return None
    px, py = Counter(), Counter()
    for (x, y), c in joint.items():
        px[x] += c
        py[y] += c
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / total
        mi += pxy * math.log2(pxy / ((px[x] / total) * (py[y] / total)))
    hy = -sum((c / total) * math.log2(c / total) for c in py.values())
    persist = sum(c for (x, y), c in joint.items() if x == y) / total
    marginal = max(py.values()) / total
    return {"mi": mi, "hy": hy, "nmi": (mi / hy if hy > 0 else 0.0),
            "persist": persist, "marginal": marginal, "n": total}


# --------------------------------------------------------------------------- #
#  Pretty printing                                                            #
# --------------------------------------------------------------------------- #
def show_dist(title, labels, counts, note=""):
    total = sum(counts) or 1
    print(f"  {title}{('   ' + note) if note else ''}")
    for lab, c in zip(labels, counts):
        f = c / total
        print(f"    {lab:>10} {c:>12,} {f * 100:6.2f}%  {'#' * int(round(f * 40))}")
    print()


def banner(text):
    print("=" * 78)
    print(text)
    print("=" * 78)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus_path", required=True,
                    help="Path to the behavior-side corpus (corpus_size.txt).")
    ap.add_argument("--seq_length_size", type=int, default=256,
                    help="Behavior sequence length used in training (default 256). "
                         "Truncation mirror uses seq_length_size - 2 packets.")
    ap.add_argument("--max_flows", type=int, default=0,
                    help="Sample only the first N flows (0 = all).")
    ap.add_argument("--plot", type=str, default="",
                    help="Directory to save PNG histograms (needs matplotlib).")
    args = ap.parse_args()

    max_num_tokens = args.seq_length_size - 2           # CLS + SEP reserved

    # ---- accumulators ----------------------------------------------------- #
    n_flows = 0
    proto_counter = Counter()
    K_trunc = Counter()
    K_full = Counter()
    transitions = Counter()                              # option (2) per-flow
    n_dark = n_dark_complete_k1 = n_dark_trunc = 0
    n_trunc = 0
    total_transitions = 0
    alt_violations = 0
    zero_anom_total = 0

    nbin_counter = Counter()
    vbin_counter = Counter()
    gbin_counter = Counter()

    joint = {f"{feat}_lag{lag}": Counter()
             for feat in ("n", "v", "g") for lag in (1, 2)}

    # ---- single streaming pass ------------------------------------------- #
    for proto, sizes, iats in iter_flows(args.corpus_path, args.max_flows):
        res = flow_bursts(sizes, iats, max_num_tokens)
        if res is None:
            continue
        n_flows += 1
        proto_counter[proto] += 1
        zero_anom_total += res["zero_anom"]

        K = res["K"]
        trunc = res["trunc"]
        K_trunc[K] += 1
        n_trunc += int(trunc)

        # full-flow K (no truncation) for comparison
        dirs_full, _ = recover_dir_size(sizes[:min(len(sizes), len(iats))])
        bid_full = segment_pure_direction(dirs_full)
        K_full[(bid_full[-1] + 1) if bid_full else 0] += 1

        # direction-alternation sanity (pure-direction => strictly alternating)
        bd = res["burst_dir"]
        if any(bd[i] == bd[i - 1] for i in range(1, len(bd))):
            alt_violations += 1

        # option (2): last clean/valid target burst index (inclusive)
        last_clean = (K - 1) if not trunc else (K - 2)
        n_trans = max(0, last_clean)                     # #targets == #transitions
        transitions[n_trans] += 1
        total_transitions += n_trans
        if n_trans == 0:
            n_dark += 1
            if not trunc and K == 1:
                n_dark_complete_k1 += 1
            elif trunc and K <= 2:
                n_dark_trunc += 1

        if last_clean < 0:
            continue

        n_arr, v_arr, g_arr = res["n"], res["v"], res["g"]

        # ---- B: occupancy over clean bursts ------------------------------ #
        for k in range(0, last_clean + 1):
            nbin_counter[n_bin(n_arr[k])] += 1
            vbin_counter[v_bin(v_arr[k])] += 1
        for k in range(1, last_clean + 1):               # skip burst-0 epsilon g
            if g_arr[k] is not None:
                gbin_counter[g_bin(g_arr[k])] += 1

        # ---- C: lag-1 / lag-2 joints over consecutive clean bursts ------- #
        # n, v : sources from 0 ; g : sources from 1 (skip epsilon g_0)
        for k in range(0, last_clean):                   # lag-1: (k, k+1)
            joint["n_lag1"][(n_bin(n_arr[k]), n_bin(n_arr[k + 1]))] += 1
            joint["v_lag1"][(v_bin(v_arr[k]), v_bin(v_arr[k + 1]))] += 1
        for k in range(1, last_clean):
            if g_arr[k] is not None and g_arr[k + 1] is not None:
                joint["g_lag1"][(g_bin(g_arr[k]), g_bin(g_arr[k + 1]))] += 1
        for k in range(0, last_clean - 1):               # lag-2: (k, k+2)
            joint["n_lag2"][(n_bin(n_arr[k]), n_bin(n_arr[k + 2]))] += 1
            joint["v_lag2"][(v_bin(v_arr[k]), v_bin(v_arr[k + 2]))] += 1
        for k in range(1, last_clean - 1):
            if g_arr[k] is not None and g_arr[k + 2] is not None:
                joint["g_lag2"][(g_bin(g_arr[k]), g_bin(g_arr[k + 2]))] += 1

    if n_flows == 0:
        print("No flows parsed — check --corpus_path and format.")
        return

    # ---- helpers for K stats --------------------------------------------- #
    def pct(x):
        return 100.0 * x / n_flows

    def median_from_counter(c):
        items = sorted(c.items())
        tot = sum(v for _, v in items)
        half, run = tot / 2, 0
        for k, v in items:
            run += v
            if run >= half:
                return k
        return 0

    # =====================  REPORT  ======================================= #
    banner(f"CORPUS: {args.corpus_path}")
    print(f"  flows parsed         : {n_flows:,}")
    print(f"  protocol mix         : " +
          ", ".join(f"{'TCP' if p == 6 else 'UDP' if p == 17 else p}={c:,}"
                    for p, c in proto_counter.most_common()))
    print(f"  truncation cap       : first {max_num_tokens} packets "
          f"(seq_length_size={args.seq_length_size})")
    print(f"  zero-payload anomalies: {zero_anom_total:,} (expected 0)")
    print(f"  dir-alternation viols : {alt_violations:,} flows (expected 0)")
    print()

    banner("A. SEGMENTATION HEALTH")

    # collapse K into readable buckets
    def kbucket(K):
        if K <= 5:
            return str(K)
        if K <= 8:
            return "6-8"
        if K <= 16:
            return "9-16"
        if K <= 32:
            return "17-32"
        return "33+"
    KB_ORDER = ["1", "2", "3", "4", "5", "6-8", "9-16", "17-32", "33+"]
    kb_t, kb_f = Counter(), Counter()
    for K, c in K_trunc.items():
        kb_t[kbucket(K)] += c
    for K, c in K_full.items():
        kb_f[kbucket(K)] += c
    print("  K = bursts per flow  (truncated view = what training sees)")
    show_dist("truncated", KB_ORDER, [kb_t.get(l, 0) for l in KB_ORDER])
    show_dist("full flow", KB_ORDER, [kb_f.get(l, 0) for l in KB_ORDER])
    print(f"  median K (trunc) = {median_from_counter(K_trunc)} | "
          f"median K (full) = {median_from_counter(K_full)}")
    print(f"  P(K=1) = {pct(K_trunc.get(1, 0)):.2f}%   "
          f"P(K=2) = {pct(K_trunc.get(2, 0)):.2f}%   "
          f"P(K<=2) = {pct(K_trunc.get(1, 0) + K_trunc.get(2, 0)):.2f}%")
    print(f"  truncation rate      : {pct(n_trunc):.2f}% of flows hit the cap")
    print()
    print("  Supervision under option (2)  [drop last burst as target iff truncated]")
    TB_ORDER = ["0", "1", "2", "3", "4", "5+"]
    tb = Counter()
    for t, c in transitions.items():
        tb["5+" if t >= 5 else str(t)] += c
    show_dist("#transitions/flow", TB_ORDER, [tb.get(l, 0) for l in TB_ORDER])
    print(f"  total transitions    : {total_transitions:,}  "
          f"(mean {total_transitions / n_flows:.2f}/flow)")
    print(f"  DARK flows (0 transitions): {pct(n_dark):.2f}%  "
          f"[complete&K=1: {pct(n_dark_complete_k1):.2f}%  "
          f"truncated&K<=2: {pct(n_dark_trunc):.2f}%]")
    print()
    print("  >>> DECISION: pure-direction is viable if DARK% is small and median K healthy.")
    print("      If DARK% is large, the single-direction tail needs gap-split (deferred).")
    print()

    banner("B. TARGET DISTRIBUTIONS  (clean bursts; corrupt truncated-last excluded)")
    show_dist("n  (packets/burst)", N_LABELS,
              [nbin_counter.get(i, 0) for i in range(len(N_LABELS))],
              note="-> watch n=1 share (collapse = weak n target)")
    show_dist("v  (payload bytes/burst)", V_LABELS,
              [vbin_counter.get(i, 0) for i in range(len(V_LABELS))])
    g_counts = [gbin_counter.get(i, 0) for i in range(len(G_LABELS))]
    show_dist("g  (lead gap, by real time)", G_LABELS, g_counts,
              note="-> top-bin pile-up = gap saturation")
    gt = sum(g_counts) or 1
    print(f"  g saturation check: top bin '{G_LABELS[-1]}' = "
          f"{100.0 * g_counts[-1] / gt:.2f}% of gaps "
          f"(pure-direction should spread g across decades, unlike gap-split)")
    print()

    banner("C. SIGNAL HEALTH  (does burst k predict burst k+1 ?)")
    print("  NMI = MI / H(next)  in [0,1]; persist = acc of predicting next=this;")
    print("  marginal = acc of always predicting the modal bin; lift = persist - marginal.")
    print("  lag-1 = cross-direction (request<->response); lag-2 = same-direction (rhythm).")
    print()
    header = f"  {'target':>8} {'NMI':>7} {'MI(b)':>7} {'H(next)':>8} {'persist':>8} {'marg':>7} {'lift':>7} {'pairs':>12}"
    for lag in (1, 2):
        print(f"  --- lag-{lag} {'(cross-dir)' if lag == 1 else '(same-dir)'} ---")
        print(header)
        for feat in ("n", "v", "g"):
            r = mi_report(joint[f"{feat}_lag{lag}"])
            if r is None:
                print(f"  {feat:>8}  (no pairs)")
                continue
            print(f"  {feat:>8} {r['nmi']:>7.3f} {r['mi']:>7.3f} {r['hy']:>8.3f} "
                  f"{r['persist']:>8.3f} {r['marginal']:>7.3f} "
                  f"{r['persist'] - r['marginal']:>+7.3f} {r['n']:>12,}")
        print()
    print("  >>> READ: NMI>0 and lift>0  => exploitable signal (model uses full history, so")
    print("      this is a LOWER bound). NMI~0 on all targets => premise at risk before GPU.")
    print("      Expected ranking ~ n,v > g. MI is bin-sensitive; re-check near-zero with finer bins.")
    print()

    # ---- optional plots --------------------------------------------------- #
    if args.plot:
        try:
            import os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(args.plot, exist_ok=True)

            def save_bar(labels, counts, title, fname):
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(labels)), counts)
                plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
                plt.title(title)
                plt.tight_layout()
                plt.savefig(os.path.join(args.plot, fname), dpi=110)
                plt.close()

            save_bar(KB_ORDER, [kb_t.get(l, 0) for l in KB_ORDER],
                     "K per flow (truncated)", "A_K_dist.png")
            save_bar(N_LABELS, [nbin_counter.get(i, 0) for i in range(len(N_LABELS))],
                     "n per burst", "B_n_dist.png")
            save_bar(V_LABELS, [vbin_counter.get(i, 0) for i in range(len(V_LABELS))],
                     "v per burst", "B_v_dist.png")
            save_bar(G_LABELS, g_counts, "g (lead gap)", "B_g_dist.png")
            print(f"  plots saved to {args.plot}/")
        except ImportError:
            print("  [plot skipped: matplotlib not available]")

    print("done.")


if __name__ == "__main__":
    main()
