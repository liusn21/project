#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
high_entropy_informative.py
===========================

Rebuttal experiment for the reviewer concern:

    "How does the model distinguish highly compressed plaintext payloads
     (high entropy, high information) from heavily encrypted payloads
     (high entropy, low information)?"

Our answer is NOT that the entropy statistic distinguishes them (no unsupervised
statistic can). It is that an entropy *misfire* on informative content does not
cost us information: the learned, CMC-shaped compatibility signal r_learned
overrides the low statistical prior and keeps the gate open. This script turns
that §3.3.2 safety-valve *claim* into a *demonstration*, on REAL data, with no
synthetic/OOD construction.

------------------------------------------------------------------------------
WHAT IT DOES
------------------------------------------------------------------------------
Over a test set, for every flow it records two independent axes:

  * HIGH-ENTROPY axis  : r_stat = 1 - H/Hmax of the content bytes (low r_stat =
                         high entropy). Computed with the *exact* training-time
                         function compute_flow_reliability_raw.
  * INFORMATIVE axis   : whether a *content-only* model (Stage1Classifier,
                         modality='raw') predicts the flow correctly and
                         confidently. This is an oracle for "content alone
                         carries the label", independent of entropy AND
                         independent of the full model's gate.

It then defines (quantile threshold on entropy):

  P  = high-entropy  AND  content-informative      (the reviewer's worry case)
  HU = high-entropy  AND  content-UNinformative     (contrast = looks encrypted)

and, by monkey-patching the ITGCA gate to expose its internals, reports for the
full model on each group:

  r_stat_calib : sigmoid(stat_scale * r_stat + stat_shift) -- the calibrated
                 statistical prior, i.e. the gate value the model WOULD use if
                 it trusted only entropy (== the gate under the beta=0 below).
  r_learned    : sigmoid(c_q^T W c_k + b)  -- the learned pair-compatibility.
  r_mod        : r_calib + sigmoid(alpha)*(r_learned - r_calib) -- actual gate.
  gap          : r_mod - r_stat_calib  -- how far the learned term opens the gate.

Finally it runs a COUNTERFACTUAL: force beta = sigmoid(alpha) -> 0 (set
alpha_modality to a large negative value) so r_mod collapses to r_stat_calib,
i.e. "trust only the statistic", and measures how many P flows that the full
model gets right would be LOST. full_acc >> beta0_acc on P is the proof that
the learned override (not the statistic) is what preserves the information.

The story to read off the summary:
  * On P : r_calib low (entropy says distrust), r_learned high, r_mod > r_calib
           (gate stays open), full_acc high, beta0_acc lower.
  * On HU: r_calib low AND r_learned low -> r_mod stays low (gate correctly
           closed). Same entropy, opposite gate behaviour -> r_learned is doing
           real, task-aligned work, not leaking open.

------------------------------------------------------------------------------
INPUTS
------------------------------------------------------------------------------
  --test_path           test.pkl  (list[dict] with raw_src/packet_ids/
                        directions/size_src/iat_src/label) -- same pickle the
                        Stage 2 fine-tune used.
  --label2id_path       label2id.pkl
  --full_ckpt           fine-tuned Stage2Classifier .bin (full ITGCA model)
  --content_only_ckpt   fine-tuned Stage1Classifier(modality='raw') .bin
  --behavior_only_ckpt  (optional, currently unused; reserved for the P_B subset)
  --vocab_path_{raw,size,temporal}, --config_path[/_raw/_size]
  --use_itgca (required), --num_fusion_layers, --itgca_window_size
  selection: --entropy_pct (default 0.25), --conf_thresh (default 0.9)
  --beta0_alpha (default -20.0), --mode {count,full}
  --out_dir (default 'revision'), --tag (dataset name for filenames)

OUTPUTS  (all under --out_dir)
  perflow_<tag>.csv     per-flow table (reproducible, auditable)
  summary_<tag>.json    aggregate stats for groups P and HU  (-> the rebuttal table)
  examples_<tag>.csv     P flows sorted by gap (pick the figure example here)

USAGE (run from project/ root)
  python revision/high_entropy_informative.py \
      --test_path           data/finetune/<task>/processed/test.pkl \
      --label2id_path       data/finetune/<task>/processed/label2id.pkl \
      --full_ckpt           models/<task>_classifier.bin \
      --content_only_ckpt   models/<task>_content_only.bin \
      --vocab_path_raw      models/bert/vocab_raw.txt \
      --vocab_path_size     models/bert/vocab_size.txt \
      --vocab_path_temporal models/bert/vocab_temporal.txt \
      --config_path         models/bert/base_config.json \
      --config_path_size    models/bert/behavior_6_config.json \
      --use_itgca --num_fusion_layers 6 \
      --tag <task> --gpu_ranks 0

NOTE: requires the FULL ITGCA model (--use_itgca, NOT --ablate_r_stat); the
experiment is about the r_stat prior vs the learned override, which only exist
in that configuration.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# --- make `run_classifier_stage{1,2}` and `uer...` importable ---------------
_HERE = os.path.dirname(os.path.abspath(__file__))            # revision/
_PROJECT_ROOT = os.path.dirname(_HERE)                        # project/
_FINETUNE_DIR = os.path.join(_PROJECT_ROOT, "fine-tuning")
for _p in (_FINETUNE_DIR, _PROJECT_ROOT, os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from run_classifier_stage2 import Stage2Classifier, load_dataset, pre_tensorize  # noqa: E402
from run_classifier_stage1 import Stage1Classifier  # noqa: E402
from uer.models.multimodal_model import compute_flow_reliability_raw  # noqa: E402
from uer.layers.multimodal_fusion import ITGCrossAttentionGate  # noqa: E402
from uer.utils.vocab import Vocab  # noqa: E402
from uer.utils.config import load_hyperparam, apply_modality_configs  # noqa: E402
from uer.utils.seed import set_seed  # noqa: E402
from uer.opts import model_opts  # noqa: E402


# ===========================================================================
# 1. Traced ITGCA gate forward (monkey-patch): identical math, stashes internals
# ===========================================================================
# KEEP IN SYNC with uer/layers/multimodal_fusion.py :: ITGCrossAttentionGate.forward
# The ONLY change is the `self._itgca_trace = {...}` side-channel; the returned
# (gate, r_mod) and all arithmetic are byte-for-byte the deployed computation,
# so the numbers we read out are exactly what the model uses.
def _traced_gate_forward(self, query_feat, sa_delta, encoder_cls_q, encoder_cls_k, r_stat):
    B, L_q, H = query_feat.shape

    # Modality gate -- r_learned (always computed)
    r_logit = torch.sum(
        torch.matmul(encoder_cls_q, self.bilinear_W) * encoder_cls_k, dim=-1
    ) + self.bilinear_bias.squeeze()
    r_learned = torch.sigmoid(r_logit)

    if self.uses_r_stat and r_stat is not None:
        r_calibrated = torch.sigmoid(self.stat_scale * r_stat + self.stat_shift)
        beta_1 = torch.sigmoid(self.alpha_modality)
        r_mod = r_calibrated + beta_1 * (r_learned - r_calibrated)
    else:
        r_calibrated = None
        r_mod = r_learned

    # Token gate (pure learned)
    if self.ablate_g_token:
        g_token = torch.ones(B, L_q, device=query_feat.device, dtype=query_feat.dtype)
    else:
        q_proj = self.W_k(query_feat)
        d_proj = self.W_v(sa_delta)
        t_logit = (q_proj * d_proj).sum(dim=-1) / math.sqrt(self.bottleneck) + self.token_gate_bias
        g_token = torch.sigmoid(t_logit)

    gate = (r_mod.unsqueeze(1) * g_token).unsqueeze(-1)

    # ---- side-channel: stash flow-level signals for this forward ----
    self._itgca_trace = {
        "has_prior": bool(self.has_modality_prior),
        "r_learned": r_learned.detach().float().cpu(),
        "r_mod": r_mod.detach().float().cpu(),
        "r_calibrated": (None if r_calibrated is None
                         else r_calibrated.detach().float().cpu()),
    }
    return gate, r_mod


def _enable_gate_tracing():
    ITGCrossAttentionGate.forward = _traced_gate_forward


def _fusion_of(model):
    m = model.module if hasattr(model, "module") else model
    return m.fusion


def _prior_gates(model):
    """The per-layer Size<-Raw gates (the ones carrying the r_stat prior)."""
    return [layer.gate_size for layer in _fusion_of(model).fusion_layers]


# ===========================================================================
# 2. Checkpoint loading (follows run_classifier_stage1 / run_inference convention)
# ===========================================================================
def _safe_load(path):
    """Load a checkpoint to CPU and return the bare param dict (unwrap common
    wrappers, strip a DataParallel 'module.' prefix)."""
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _remap_generic_to_raw(sd, model_keys):
    """
    Stage-1 convention (cf. run_classifier_stage1.load_pretrained_encoder, which
    strips 'embedding.'/'encoder.' and loads into embedding_raw/encoder_raw): a
    single-modality checkpoint may store its tower under the GENERIC prefixes
    'embedding.' / 'encoder.' instead of the Stage1Classifier names
    'embedding_raw.' / 'encoder_raw.'. If remapping those prefixes increases the
    overlap with the model's keys, apply it. Returns (sd, remapped_bool).
    """
    direct = len(model_keys & set(sd))
    remapped = {}
    for k, v in sd.items():
        if k.startswith("embedding.") and not k.startswith("embedding_raw."):
            remapped["embedding_raw." + k[len("embedding."):]] = v
        elif k.startswith("encoder.") and not k.startswith("encoder_raw."):
            remapped["encoder_raw." + k[len("encoder."):]] = v
        else:
            remapped[k] = v
    if len(model_keys & set(remapped)) > direct:
        return remapped, True
    return sd, False


def _diff_keys(model, sd):
    """Return (missing, unexpected, shape_mismatch) between model and ckpt dict."""
    msd = model.state_dict()
    mkeys, ckeys = set(msd), set(sd)
    missing = sorted(mkeys - ckeys)        # model expects, ckpt lacks
    unexpected = sorted(ckeys - mkeys)     # ckpt has, model lacks
    shape_mismatch = []
    for k in (mkeys & ckeys):
        v = sd[k]
        if hasattr(v, "shape") and tuple(v.shape) != tuple(msd[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(msd[k].shape)))
    return missing, unexpected, sorted(shape_mismatch)


def _itgca_keyish(keys):
    return [k for k in keys if any(s in k for s in
            ("gate_raw", "gate_size", "alpha_modality",
             "stat_scale", "stat_shift", "local_stat_", "bilinear", "token_gate"))]


def _load_classifier_ckpt(model, path, name, allow_remap=True):
    """
    Load a FINE-TUNED classifier checkpoint (Stage1 content-only, or the full
    Stage2 model) the way run_inference.py does: build the right architecture,
    auto-detect the generic-prefix layout (stage-1 convention), print an EXACT
    key diff, and FAIL LOUDLY rather than silently random-initialising real
    parameters -- an oracle/full model with random weights is worthless and
    would quietly corrupt the whole experiment. Unexpected checkpoint keys
    (e.g. a stored size tower, or pretraining heads) are reported and dropped.
    """
    sd = _safe_load(path)
    model_keys = set(model.state_dict().keys())
    remapped = False
    if allow_remap:
        sd, remapped = _remap_generic_to_raw(sd, model_keys)

    missing, unexpected, shape_mismatch = _diff_keys(model, sd)
    common = len(model_keys) - len(missing)

    print(f"[{name}] {path}")
    if remapped:
        print("   (remapped generic 'embedding.'/'encoder.' -> '*_raw.' per stage-1 convention)")
    print(f"   model_keys={len(model_keys)}  ckpt_keys={len(sd)}  common={common}  "
          f"missing={len(missing)}  unexpected={len(unexpected)}  shape_mismatch={len(shape_mismatch)}")
    if unexpected:
        print(f"   [drop] unexpected ckpt keys ({len(unexpected)}), e.g.: {unexpected[:6]}")
    if missing:
        print(f"   [warn] missing keys ({len(missing)}), e.g.: {missing[:6]}")
    for k, cs, ms in shape_mismatch[:6]:
        print(f"   [warn] shape mismatch {k}: ckpt{cs} vs model{ms}")

    if missing or shape_mismatch:
        cls_shape = [t for t in shape_mismatch if t[0].startswith("classifier")]
        size_tower = [k for k in unexpected if k.startswith(("embedding_size", "encoder_size"))]
        if size_tower or cls_shape:
            print("   HINT: this checkpoint carries a SIZE tower / a wider classifier -> "
                  "it is a modality='both' (or full) model, not a content-only one. The "
                  "informativeness oracle MUST be modality='raw'; train one with "
                  "`run_classifier_stage1.py --modality raw`, or point --content_only_ckpt at it.")
        if _itgca_keyish(missing) or _itgca_keyish(unexpected):
            print("   HINT: ITGCA gate keys are out of sync -> check --use_itgca and the "
                  "--ablate_r_stat/--ablate_g_token/--ablate_source_bias flags match the ckpt.")
        if missing and not size_tower and not cls_shape:
            print("   HINT: wholesale key/shape mismatch usually means the model geometry "
                  "(hidden_size/layers_num/emb_size/heads_num) differs from --config_path. "
                  "Pass the SAME config the checkpoint was trained with "
                  "(base/small/medium/... under models/bert/).")
        print(f"   [FATAL] refusing to run with an incompletely-loaded '{name}'.")
        sys.exit(1)

    filtered = {k: v for k, v in sd.items() if k in model_keys}
    model.load_state_dict(filtered, strict=True)
    print(f"   [ok] loaded '{name}' cleanly ({len(filtered)} params; "
          f"dropped {len(unexpected)} unexpected).")


# ===========================================================================
# 3. Inference passes
# ===========================================================================
def _iter_batches(tensors, bs, device):
    n = tensors["label"].size(0)
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        yield (
            tensors["raw_src"][sl].to(device),
            tensors["packet_ids"][sl].to(device),
            tensors["directions"][sl].to(device),
            tensors["size_src"][sl].to(device),
            tensors["iat_src"][sl].to(device),
            tensors["label"][sl],
        )


@torch.inference_mode()
def pass_main(full_model, content_model, tensors, args, vocab_size_raw):
    """One pass: content-only preds, full preds, r_stat, and gate traces."""
    device = args.device
    L = args.num_fusion_layers
    n = tensors["label"].size(0)

    labels = tensors["label"].numpy()
    r_stat = np.empty(n, dtype=np.float64)
    co_pred = np.empty(n, dtype=np.int64); co_conf = np.empty(n, dtype=np.float64)
    full_pred = np.empty(n, dtype=np.int64); full_conf = np.empty(n, dtype=np.float64)
    g_learned = np.empty((n, L), dtype=np.float64)
    g_mod = np.empty((n, L), dtype=np.float64)
    g_calib = np.full((n, L), np.nan, dtype=np.float64)

    pos = 0
    for raw, pid, dirs, size, iat, _tgt in _iter_batches(tensors, args.batch_size, device):
        b = raw.size(0)

        # content-only oracle for "informative"
        _, co_logits = content_model(raw, pid, dirs, size, iat, None)
        co_p = F.softmax(co_logits, dim=-1)
        co_conf[pos:pos + b] = co_p.max(dim=-1).values.cpu().numpy()
        co_pred[pos:pos + b] = co_logits.argmax(dim=-1).cpu().numpy()

        # full model (stashes gate traces)
        f_logits = full_model(raw, pid, dirs, size, iat)
        f_p = F.softmax(f_logits, dim=-1)
        full_conf[pos:pos + b] = f_p.max(dim=-1).values.cpu().numpy()
        full_pred[pos:pos + b] = f_logits.argmax(dim=-1).cpu().numpy()

        # entropy axis (independent recompute, identical function)
        r_stat[pos:pos + b] = compute_flow_reliability_raw(
            raw, vocab_size=vocab_size_raw).detach().float().cpu().numpy()

        # gate traces, per fusion layer
        for li, gate in enumerate(_prior_gates(full_model)):
            tr = gate._itgca_trace
            g_learned[pos:pos + b, li] = tr["r_learned"].numpy()
            g_mod[pos:pos + b, li] = tr["r_mod"].numpy()
            if tr["r_calibrated"] is not None:
                g_calib[pos:pos + b, li] = tr["r_calibrated"].numpy()

        pos += b

    return dict(labels=labels, r_stat=r_stat,
                co_pred=co_pred, co_conf=co_conf,
                full_pred=full_pred, full_conf=full_conf,
                g_learned=g_learned, g_mod=g_mod, g_calib=g_calib)


@torch.inference_mode()
def pass_beta0(full_model, tensors, args):
    """Counterfactual: force beta=sigmoid(alpha)->0 so r_mod == r_stat_calib."""
    device = args.device
    n = tensors["label"].size(0)
    gates = _prior_gates(full_model)

    # save & override alpha_modality
    saved = []
    for g in gates:
        if hasattr(g, "alpha_modality"):
            saved.append((g, g.alpha_modality.data.clone()))
            g.alpha_modality.data.fill_(float(args.beta0_alpha))
    if not saved:
        raise RuntimeError("No alpha_modality found -- need the full ITGCA model "
                           "(--use_itgca and NOT --ablate_r_stat).")

    pred = np.empty(n, dtype=np.int64)
    try:
        pos = 0
        for raw, pid, dirs, size, iat, _tgt in _iter_batches(tensors, args.batch_size, device):
            b = raw.size(0)
            logits = full_model(raw, pid, dirs, size, iat)
            pred[pos:pos + b] = logits.argmax(dim=-1).cpu().numpy()
            pos += b
    finally:
        for g, orig in saved:
            g.alpha_modality.data.copy_(orig)
    return pred


# ===========================================================================
# 4. Selection + reporting
# ===========================================================================
def group_stats(idx, R, beta0_correct=None):
    """Aggregate metrics over a flow index set."""
    if idx.sum() == 0:
        return {"n": 0}
    sel = lambda a: a[idx]
    co_correct = (R["co_pred"] == R["labels"])
    full_correct = (R["full_pred"] == R["labels"])
    learned = R["g_learned"].mean(axis=1)
    mod = R["g_mod"].mean(axis=1)
    calib = R["g_calib"].mean(axis=1)
    gap = mod - calib
    out = {
        "n": int(idx.sum()),
        "mean_r_stat": float(np.mean(sel(R["r_stat"]))),
        "mean_r_calib": float(np.nanmean(sel(calib))),
        "mean_r_learned": float(np.mean(sel(learned))),
        "mean_r_mod": float(np.mean(sel(mod))),
        "mean_gap_rmod_minus_rcalib": float(np.nanmean(sel(gap))),
        "frac_gap_positive": float(np.mean(sel(gap) > 0)),
        "acc_content_only": float(np.mean(sel(co_correct))),
        "acc_full": float(np.mean(sel(full_correct))),
        "mean_full_conf": float(np.mean(sel(R["full_conf"]))),
    }
    if beta0_correct is not None:
        out["acc_full_beta0"] = float(np.mean(sel(beta0_correct)))
        out["delta_full_minus_beta0"] = out["acc_full"] - out["acc_full_beta0"]
    return out


def write_outputs(R, idx_P, idx_HU, beta0_pred, args, r_stat_thr):
    os.makedirs(args.out_dir, exist_ok=True)
    n = len(R["labels"])
    learned = R["g_learned"].mean(axis=1)
    mod = R["g_mod"].mean(axis=1)
    calib = R["g_calib"].mean(axis=1)
    gap = mod - calib
    co_correct = (R["co_pred"] == R["labels"])
    full_correct = (R["full_pred"] == R["labels"])
    beta0_correct = None if beta0_pred is None else (beta0_pred == R["labels"])

    group = np.full(n, "other", dtype=object)
    group[idx_HU] = "HU"
    group[idx_P] = "P"

    # ---- per-flow CSV ----
    perflow_path = os.path.join(args.out_dir, f"perflow_{args.tag}.csv")
    cols = ["flow_id", "label", "group", "r_stat",
            "co_pred", "co_correct", "co_conf",
            "full_pred", "full_correct", "full_conf",
            "r_calib_mean", "r_learned_mean", "r_mod_mean", "gap_mean"]
    if beta0_pred is not None:
        cols += ["full_beta0_pred", "full_beta0_correct"]
    with open(perflow_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            row = [i, int(R["labels"][i]), group[i], f"{R['r_stat'][i]:.6f}",
                   int(R["co_pred"][i]), int(co_correct[i]), f"{R['co_conf'][i]:.6f}",
                   int(R["full_pred"][i]), int(full_correct[i]), f"{R['full_conf'][i]:.6f}",
                   f"{calib[i]:.6f}", f"{learned[i]:.6f}", f"{mod[i]:.6f}", f"{gap[i]:.6f}"]
            if beta0_pred is not None:
                row += [int(beta0_pred[i]), int(beta0_correct[i])]
            w.writerow(row)

    # ---- summary JSON ----
    summary = {
        "tag": args.tag,
        "n_total": n,
        "entropy_pct": args.entropy_pct,
        "r_stat_threshold": float(r_stat_thr),
        "conf_thresh": args.conf_thresh,
        "num_fusion_layers": args.num_fusion_layers,
        "beta0_alpha": args.beta0_alpha,
        "groups": {
            "P_high_entropy_informative": group_stats(idx_P, R, beta0_correct),
            "HU_high_entropy_uninformative": group_stats(idx_HU, R, beta0_correct),
        },
        "per_layer_over_P": ({} if idx_P.sum() == 0 else {
            "r_calib": np.nanmean(R["g_calib"][idx_P], axis=0).tolist(),
            "r_learned": R["g_learned"][idx_P].mean(axis=0).tolist(),
            "r_mod": R["g_mod"][idx_P].mean(axis=0).tolist(),
        }),
    }
    summary_path = os.path.join(args.out_dir, f"summary_{args.tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- examples CSV (P flows, sorted by gap desc) ----
    examples_path = os.path.join(args.out_dir, f"examples_{args.tag}.csv")
    P_ids = np.where(idx_P)[0]
    P_ids = P_ids[np.argsort(-gap[P_ids])][: args.topk_examples]
    with open(examples_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["flow_id", "label", "r_stat", "r_calib_mean", "r_learned_mean",
                "r_mod_mean", "gap_mean", "co_conf", "full_correct"]
        if beta0_pred is not None:
            head.append("full_beta0_correct")
        w.writerow(head)
        for i in P_ids:
            row = [i, int(R["labels"][i]), f"{R['r_stat'][i]:.6f}",
                   f"{calib[i]:.6f}", f"{learned[i]:.6f}", f"{mod[i]:.6f}",
                   f"{gap[i]:.6f}", f"{R['co_conf'][i]:.6f}", int(full_correct[i])]
            if beta0_pred is not None:
                row.append(int(beta0_correct[i]))
            w.writerow(row)

    return perflow_path, summary_path, examples_path, summary


def print_report(summary):
    print("\n" + "=" * 78)
    print(f"  high-entropy-but-informative analysis  --  {summary['tag']}")
    print("=" * 78)
    print(f"  n_total={summary['n_total']}  entropy_pct={summary['entropy_pct']} "
          f"(r_stat<= {summary['r_stat_threshold']:.4f})  conf_thresh={summary['conf_thresh']}")
    for gname, g in summary["groups"].items():
        print(f"\n  [{gname}]  n={g.get('n', 0)}")
        if g.get("n", 0) == 0:
            continue
        print(f"    mean r_stat            : {g['mean_r_stat']:.4f}   (low = high entropy)")
        print(f"    mean r_stat_calib      : {g['mean_r_calib']:.4f}   (gate if it trusted only entropy)")
        print(f"    mean r_learned         : {g['mean_r_learned']:.4f}")
        print(f"    mean r_mod (actual)    : {g['mean_r_mod']:.4f}")
        print(f"    mean gap (rmod-rcalib) : {g['mean_gap_rmod_minus_rcalib']:+.4f}   "
              f"(frac>0: {g['frac_gap_positive']:.2f})")
        print(f"    acc content-only       : {g['acc_content_only']:.4f}")
        print(f"    acc full               : {g['acc_full']:.4f}")
        if "acc_full_beta0" in g:
            print(f"    acc full @ beta=0      : {g['acc_full_beta0']:.4f}   "
                  f"(delta full-beta0: {g['delta_full_minus_beta0']:+.4f})")
    print("\n  Read P: r_calib low (entropy distrusts content) but r_learned high and")
    print("  r_mod>r_calib (gate stays open); full acc >> beta=0 acc => the learned")
    print("  override, not the statistic, preserves the information. HU is the control:")
    print("  same entropy, content uninformative -> r_learned low -> gate stays closed.")
    print("=" * 78 + "\n")


# ===========================================================================
# 5. main
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Surface high-entropy-but-informative flows and show the ITGCA "
                    "gate stays open via the learned override (rebuttal experiment).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # paths
    p.add_argument("--test_path", required=True)
    p.add_argument("--label2id_path", required=True)
    p.add_argument("--full_ckpt", required=True, help="fine-tuned Stage2Classifier .bin")
    p.add_argument("--content_only_ckpt", required=True,
                   help="fine-tuned Stage1Classifier(modality='raw') .bin")
    p.add_argument("--vocab_path_raw", default="test/vocab_raw_bytes.txt")
    p.add_argument("--vocab_path_size", default="test/vocab_size.txt")
    p.add_argument("--vocab_path_temporal", default="test/vocab_temporal.txt")
    p.add_argument("--config_path", default="models/bert/base_config.json")
    p.add_argument("--config_path_raw", default=None)
    p.add_argument("--config_path_size", default=None)

    # architecture (filled from config by load_hyperparam)
    model_opts(p)
    p.add_argument("--num_fusion_layers", type=int, default=6)
    p.add_argument("--use_itgca", action="store_true")
    p.add_argument("--itgca_window_size", type=int, default=16)
    p.add_argument("--ablate_r_stat", action="store_true")
    p.add_argument("--ablate_g_token", action="store_true")
    p.add_argument("--ablate_source_bias", action="store_true")
    p.add_argument("--seq_length_raw", type=int, default=512)
    p.add_argument("--seq_length_size", type=int, default=256)

    p.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    p.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    p.add_argument("--moebert_expert_dim", type=int, required=False, default=3072, help="Dim of expert,default is ffn.")
    p.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    p.add_argument("--moebert_route_method", choices=["gate-token", "gate-sentence", "hash-random", "hash-balance","proto"], default="hash-random",
                   help="moebert route method.")
    p.add_argument("--moebert_route_hash_list", default=None, type=str, help="Path of moebert hash list file.")
    p.add_argument("--moebert_load_balance", type=float, default=0.0, help="gate loss weight.")

    # selection / counterfactual
    p.add_argument("--entropy_pct", type=float, default=0.25,
                   help="High-entropy = lowest this fraction of r_stat (e.g. 0.25 = top-25%% entropy).")
    p.add_argument("--conf_thresh", type=float, default=0.9,
                   help="Content-only confidence threshold for 'informative'.")
    p.add_argument("--beta0_alpha", type=float, default=-20.0,
                   help="alpha_modality override for the beta->0 counterfactual.")
    p.add_argument("--mode", choices=["count", "full"], default="full",
                   help="'count' = selection + gate stats only (skip beta=0 pass).")
    p.add_argument("--topk_examples", type=int, default=20)

    # run
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", default="revision")
    p.add_argument("--tag", default="dataset")
    p.add_argument("--limit", type=int, default=None, help="only first N flows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_ranks", default=[], nargs="+", type=int)

    args = p.parse_args()
    if args.config_path:
        args = load_hyperparam(args)
    args = apply_modality_configs(args)
    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)
    args.dropout = getattr(args, "dropout", 0.1)
    return args


def setup_device(args):
    if len(args.gpu_ranks) >= 1:
        assert torch.cuda.is_available(), "No available GPUs."
        args.device = torch.device(f"cuda:{args.gpu_ranks[0]}")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")


def main():
    args = parse_args()
    assert args.use_itgca, "This experiment requires the full ITGCA model (--use_itgca)."
    assert not args.ablate_r_stat, "Needs the r_stat prior present (drop --ablate_r_stat)."
    set_seed(args.seed)
    setup_device(args)
    device = args.device
    print(f"[device] {device}")

    # vocab / labels / data
    vocab_raw = Vocab(); vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab(); vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab(); vocab_temporal.load(args.vocab_path_temporal)
    label2id = load_dataset(args.label2id_path)
    args.labels_num = len(label2id)
    print(f"[vocab] raw={len(vocab_raw)} size={len(vocab_size)} temporal={len(vocab_temporal)} "
          f"labels={args.labels_num}")

    data = load_dataset(args.test_path)
    if args.limit is not None:
        data = data[: args.limit]
    tensors = pre_tensorize(data)
    print(f"[data] flows={tensors['label'].size(0)}")

    # ---- build models ----
    # content-only first, while args.layers_num == base (12) from base_config.
    args.modality = "raw"
    content_model = Stage1Classifier(args, len(vocab_raw), len(vocab_size),
                                     len(vocab_temporal), args.labels_num)
    _load_classifier_ckpt(content_model, args.content_only_ckpt, "content-only")
    content_model.to(device).eval()

    full_model = Stage2Classifier(args, len(vocab_raw), len(vocab_size),
                                  len(vocab_temporal), args.labels_num)
    _load_into(full_model, args.full_ckpt, "full")
    full_model.to(device).eval()

    _enable_gate_tracing()  # monkey-patch the gate to stash internals

    # ---- pass 1: content-only + full + r_stat + gate traces ----
    R = pass_main(full_model, content_model, tensors, args, len(vocab_raw))

    # ---- selection ----
    r_stat_thr = float(np.quantile(R["r_stat"], args.entropy_pct))
    high_ent = R["r_stat"] <= r_stat_thr
    co_correct = (R["co_pred"] == R["labels"])
    informative = co_correct & (R["co_conf"] >= args.conf_thresh)
    idx_P = high_ent & informative
    idx_HU = high_ent & (~informative)
    print(f"[select] r_stat<= {r_stat_thr:.4f} -> high-entropy={int(high_ent.sum())}; "
          f"P(informative)={int(idx_P.sum())}  HU(uninformative)={int(idx_HU.sum())}")

    # ---- pass 2: beta=0 counterfactual (skipped in count mode) ----
    beta0_pred = None
    if args.mode == "full":
        if idx_P.sum() == 0:
            print("[beta0] population P is empty; skipping counterfactual. "
                  "Consider raising --entropy_pct, lowering --conf_thresh, or another dataset.")
        else:
            beta0_pred = pass_beta0(full_model, tensors, args)

    # ---- outputs ----
    perflow_path, summary_path, examples_path, summary = write_outputs(
        R, idx_P, idx_HU, beta0_pred, args, r_stat_thr)
    print_report(summary)
    print(f"[write] {perflow_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {examples_path}")


if __name__ == "__main__":
    main()
