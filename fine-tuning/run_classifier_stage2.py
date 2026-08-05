"""
Stage 2 Multi-Modal Classifier (paper §4.1.4 fine-tuning recipe).

Architecture:
    - Raw encoder (loaded from Stage 2 pretrained checkpoint)
    - Size encoder (loaded from Stage 2 pretrained checkpoint)
    - Fusion module (loaded from Stage 2 pretrained checkpoint, with optional gating)
    - Classification head (concat fused [CLS] + non-CLS mean-pool, two-layer MLP)

Training (paper §4.1.4):
    Phase 1 (--phase1_epochs, default 5): freeze the pretrained backbone, train only
        the classifier head with AdamW.
    Phase 2 (--epochs_num,    default 10): unfreeze everything with layer-wise LRs:
        encoder = base_lr * --llrd_encoder_ratio (paper: 0.3)
        fusion  = base_lr * --llrd_fusion_ratio  (paper: 0.7)
        classifier = base_lr (paper: 5e-5)

Cross-entropy loss with --label_smoothing (default 0.1) and gradient clipping at
--max_grad_norm (default 1.0). Optional AMP via --fp16. Few-shot subsampling for
the §4.3 label-efficiency experiments via --few_shot.

Usage example (Stage 2 fine-tune, single GPU):
    python fine-tuning/run_classifier_stage2.py \
        --train_path datasets/processed/train.pkl \
        --dev_path   datasets/processed/val.pkl \
        --test_path  datasets/processed/test.pkl \
        --label2id_path datasets/processed/label2id.pkl \
        --vocab_path_raw      models/bert/vocab_raw.txt \
        --vocab_path_size     models/bert/vocab_size.txt \
        --vocab_path_temporal models/bert/vocab_temporal.txt \
        --pretrained_model_path models/mm_trafficbert.bin \
        --output_model_path     models/<task>_classifier.bin \
        --config_path           models/bert/base_config.json \
        --config_path_size      models/bert/behavior_6_config.json \
        --use_itgca \
        --batch_size 32 --learning_rate 5e-5 \
        --phase1_epochs 5 --phase1_lr 1e-3 \
        --epochs_num 10 \
        --llrd_encoder_ratio 0.3 --llrd_fusion_ratio 0.7 \
        --label_smoothing 0.1 --max_grad_norm 1.0 \
        --reserve_gpu_memory auto \
        --seed 42 --gpu_ranks 0
"""

import os
import sys
sys.path.append(os.getcwd())

import gc
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from uer.layers import RawPacketEmbedding, PacketSizeEmbedding
from uer.layers.multimodal_fusion import MultiModalFusionEncoder
from uer.encoders import str2encoder
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam, apply_modality_configs
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.utils.constants import PAD_ID
from uer.utils import *
from uer.opts import finetune_opts
from uer.models.multimodal_model import (
    compute_flow_reliability_raw, compute_local_entropy
)


BYTES_PER_MIB = 1024 ** 2
BYTES_PER_GIB = 1024 ** 3
MEMORY_PROBE_STEPS = 2
MEMORY_RESERVATION_CHUNK_BYTES = 64 * BYTES_PER_MIB
MEMORY_RESERVATION_TOLERANCE_BYTES = 64 * BYTES_PER_MIB


class ClassifierHead(nn.Module):
    """
    Two-layer MLP head with optional multi-sample dropout (item 5).

    With msd_num > 1 at train time, the (Dropout -> output Linear) is resampled
    msd_num times from the SAME pre-dropout features and the per-sample logits are
    returned stacked as [batch, msd_num, labels]; the caller averages the per-sample
    cross-entropy (Inoue 2019). At eval time (dropout off) it returns [batch, labels].
    The batch dim stays first so DataParallel gathers correctly. Kept under the
    attribute name `classifier` so existing name-based grouping/freezing is unchanged.
    """

    def __init__(self, in_dim, hidden, labels_num, dropout, msd_num=1):
        super(ClassifierHead, self).__init__()
        self.dense = nn.Linear(in_dim, hidden)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, labels_num)
        self.msd_num = msd_num

    def forward(self, x):
        h = self.act(self.dense(x))
        if self.training and self.msd_num > 1:
            return torch.stack(
                [self.out(self.dropout(h)) for _ in range(self.msd_num)], dim=1
            )  # [batch, msd_num, labels]
        return self.out(self.dropout(h))


class Stage2Classifier(nn.Module):
    """
    Stage 2 Multi-Modal Classifier.

    Architecture:
        - Raw encoder (pretrained): embedding + transformer
        - Size encoder (pretrained): embedding + transformer
        - Fusion module (pretrained): bidirectional cross-attention with optional gating
        - Classifier head: concat([fused_raw_cls, fused_size_cls,
                                   mean_pool(fused_raw), mean_pool(fused_size)])
                          -> Linear -> Tanh -> Dropout -> Linear -> labels
    """

    def __init__(self, args, vocab_size_raw, vocab_size_size, vocab_size_temporal, labels_num):
        super(Stage2Classifier, self).__init__()

        self.hidden_size = args.hidden_size
        self.labels_num = labels_num

        # Fusion gate mode (must match the pretrained checkpoint).
        self.use_itgca = getattr(args, 'use_itgca', False)
        self.use_mlp_gate = getattr(args, 'use_mlp_gate', False)
        if self.use_itgca and self.use_mlp_gate:
            raise ValueError("--use_itgca and --use_mlp_gate are mutually exclusive.")
        self.itgca_window_size = getattr(args, 'itgca_window_size', 16)
        self.vocab_size_raw = vocab_size_raw
        self.ablate_r_stat = getattr(args, 'ablate_r_stat', False)
        self.ablate_r_learned = getattr(args, 'ablate_r_learned', False)
        self.ablate_source_bias = getattr(args, 'ablate_source_bias', False)

        # Per-modality encoder depth (falls back to args.layers_num if not set
        # by apply_modality_configs).
        base_layers = args.layers_num
        layers_num_raw = getattr(args, 'layers_num_raw', base_layers) or base_layers
        layers_num_size = getattr(args, 'layers_num_size', base_layers) or base_layers

        # Raw modality encoder.
        self.embedding_raw = RawPacketEmbedding(args, vocab_size_raw)
        args.layers_num = layers_num_raw
        self.encoder_raw = str2encoder[args.encoder](args)

        # Size modality encoder (with temporal/IAT support).
        self.embedding_size = PacketSizeEmbedding(args, vocab_size_size, vocab_size_temporal)
        args.layers_num = layers_num_size
        self.encoder_size = str2encoder[args.encoder](args)

        # Restore so fusion / classifier head use the original base depth.
        args.layers_num = base_layers

        # Fusion module.
        num_fusion_layers = getattr(args, 'num_fusion_layers', 6)
        self.fusion = MultiModalFusionEncoder(args, num_layers=num_fusion_layers, use_itgca=self.use_itgca)

        # Classification head: 4 * hidden_size -> hidden_size -> labels_num.
        # Multi-sample dropout (item 5) is folded into the head; msd_num == 1 keeps
        # the original single-sample behavior.
        msd_num = getattr(args, 'msd_num', 1) if getattr(args, 'use_msd', False) else 1
        self.classifier = ClassifierHead(
            4 * args.hidden_size, args.hidden_size, labels_num, args.dropout, msd_num=msd_num
        )

    def forward(self, raw_src, packet_ids, directions, size_src, iat_src):
        """
        Returns:
            logits: [batch, labels_num]
        """
        # Raw encoder.
        raw_emb = self.embedding_raw(raw_src, packet_ids, directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)

        # Size encoder.
        size_emb = self.embedding_size(size_src, iat_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)

        # ITGCA signals (no detach -- preserve the existing fine-tuning behavior).
        if self.use_itgca:
            r_stat_raw = (None if self.ablate_r_stat
                          else compute_flow_reliability_raw(raw_src, vocab_size=self.vocab_size_raw))
            local_ent_raw = (None if self.ablate_source_bias
                             else compute_local_entropy(raw_src, self.itgca_window_size))
            gate_kwargs = {
                'raw_cls_enc': raw_output[:, 0, :],
                'size_cls_enc': size_output[:, 0, :],
                'r_stat_raw': r_stat_raw,
                'local_ent_raw': local_ent_raw,
            }
        elif self.use_mlp_gate:
            gate_kwargs = {
                'raw_cls_enc': raw_output[:, 0, :],
                'size_cls_enc': size_output[:, 0, :],
            }
        else:
            gate_kwargs = {}

        # Fusion.
        raw_fused, size_fused, _ = self.fusion(
            raw_output, size_output, raw_seg, size_seg, **gate_kwargs
        )

        # Fused [CLS] tokens + mean pool over non-CLS, non-PAD positions.
        raw_cls = raw_fused[:, 0, :]
        size_cls = size_fused[:, 0, :]

        raw_mask = raw_seg[:, 1:].unsqueeze(-1).float()
        raw_pool = (raw_fused[:, 1:, :] * raw_mask).sum(1) / (raw_mask.sum(1).clamp(min=1.0))
        size_mask = size_seg[:, 1:].unsqueeze(-1).float()
        size_pool = (size_fused[:, 1:, :] * size_mask).sum(1) / (size_mask.sum(1).clamp(min=1.0))

        combined = torch.cat([raw_cls, size_cls, raw_pool, size_pool], dim=-1)
        logits = self.classifier(combined)
        return logits


def load_pretrained_model(model, pretrained_path):
    """
    Load Stage 2 multimodal pretrained weights into the classifier model.

    The pretrained checkpoint contains the two encoders, the fusion module, plus
    momentum encoders / ITC projections / target heads / feature queues that are
    only used during pretraining. We filter the latter out here.
    """
    if pretrained_path is None:
        return

    print(f"Loading pretrained multimodal model from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')

    exclude_prefixes = ['embedding_raw_m', 'encoder_raw_m',
                        'embedding_size_m', 'encoder_size_m',
                        'itc_proj_raw_m', 'itc_proj_size_m',
                        'target', 'raw_queue', 'size_queue', 'queue_ptr']

    filtered_state = {k: v for k, v in state_dict.items()
                      if not any(k.startswith(prefix) for prefix in exclude_prefixes)}

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    classifier_missing = [k for k in missing if k.startswith('classifier')]
    other_missing = [k for k in missing if not k.startswith('classifier')]
    excluded_count = len(state_dict) - len(filtered_state)

    print(f"  Checkpoint total keys: {len(state_dict)}")
    print(f"  Excluded keys (momentum/ITC/target/queues): {excluded_count}")
    print(f"  Filtered keys (should be loaded): {len(filtered_state)}")
    print(f"  Missing keys: {len(missing)} (classifier: {len(classifier_missing)}, other: {len(other_missing)})")
    print(f"  Unexpected keys: {len(unexpected)}")

    # Fusion-gate config mismatch detection.
    has_itgca_in_checkpoint = any(
        k.startswith('fusion.fusion_layers.0.gate_') or
        k.startswith('fusion.fusion_layers.0.local_stat_')
        for k in filtered_state.keys()
    )
    has_mlp_gate_in_checkpoint = any(
        k.startswith('fusion.mlp_gate.') for k in filtered_state.keys()
    )
    model_uses_itgca = getattr(model, 'use_itgca', False)
    model_uses_mlp_gate = getattr(model, 'use_mlp_gate', False)
    checkpoint_gate = ('ITGCA' if has_itgca_in_checkpoint else
                       'MLP' if has_mlp_gate_in_checkpoint else 'none')
    model_gate = ('ITGCA' if model_uses_itgca else
                  'MLP' if model_uses_mlp_gate else 'none')
    if checkpoint_gate != model_gate:
        print(f"  WARNING: Pretrained gate mode is {checkpoint_gate}, but "
              f"fine-tuning gate mode is {model_gate}. Use the matching "
              f"--use_itgca or --use_mlp_gate flag.")

    if len(other_missing) == 0 and len(unexpected) == 0:
        print(f"  All encoder/fusion parameters loaded successfully.")
        print(f"  Classifier randomly initialized ({len(classifier_missing)} params).")
    else:
        print(f"  Load incomplete:")
        if other_missing:
            print(f"    Missing non-classifier keys ({len(other_missing)}): {other_missing[:10]}...")
            if 'embedding_size.temporal_embedding.weight' in other_missing:
                raise ValueError(
                    "Pretrained model missing temporal_embedding! "
                    "Use a Stage 2 model trained with IAT.")
        if unexpected:
            print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")


def load_dataset(path):
    """Load a pickled dataset."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def few_shot_sample(dataset, ratio, seed=42):
    """
    Stratified sampling: keep `ratio` fraction of training samples per class,
    minimum 1 sample per class. Used for the §4.3 label-efficiency curves.
    """
    from collections import defaultdict
    rng = random.Random(seed)

    class_samples = defaultdict(list)
    for sample in dataset:
        class_samples[sample['label']].append(sample)

    sampled = []
    for label, samples in class_samples.items():
        k = max(1, int(len(samples) * ratio))
        k = min(k, len(samples))
        sampled.extend(rng.sample(samples, k))

    rng.shuffle(sampled)

    print(f"  Few-shot sampling: ratio={ratio}, "
          f"{len(dataset)} -> {len(sampled)} samples "
          f"({len(sampled) / max(len(dataset), 1) * 100:.1f}%)")
    return sampled


def pre_tensorize(dataset, pin=False):
    """Pre-convert list-of-dicts dataset into contiguous tensors (one-time cost)."""
    tensors = {
        'raw_src': torch.LongTensor([s['raw_src'] for s in dataset]),
        'packet_ids': torch.LongTensor([s['packet_ids'] for s in dataset]),
        'directions': torch.LongTensor([s['directions'] for s in dataset]),
        'size_src': torch.LongTensor([s['size_src'] for s in dataset]),
        'iat_src': torch.LongTensor([s['iat_src'] for s in dataset]),
        'label': torch.LongTensor([s['label'] for s in dataset]),
    }
    if pin and torch.cuda.is_available():
        tensors = {k: v.pin_memory() for k, v in tensors.items()}
    return tensors


def batch_loader(batch_size, dataset_tensors, shuffle=False):
    """Index-slice batches over the pre-built tensors."""
    n = dataset_tensors['label'].size(0)
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    num_batches = (n + batch_size - 1) // batch_size

    for i in range(num_batches):
        idx = indices[i * batch_size: (i + 1) * batch_size]
        yield (dataset_tensors['raw_src'][idx],
               dataset_tensors['packet_ids'][idx],
               dataset_tensors['directions'][idx],
               dataset_tensors['size_src'][idx],
               dataset_tensors['iat_src'][idx],
               dataset_tensors['label'][idx])


class ModelEMA:
    """
    Exponential moving average of model parameters (item 1; always on in Phase 2).

    decay is ramped as min(decay, (1+t)/(10+t)) so the average is not pinned near the
    post-Phase-1 weights early in the short fine-tuning schedule. The model has only
    LayerNorm (no BatchNorm), so no running-stat recomputation is needed. The shadow
    is held in fp32 (params are fp32 even under AMP).
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        m = model.module if hasattr(model, 'module') else model
        self.shadow = {n: p.detach().clone().float() for n, p in m.named_parameters()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        m = model.module if hasattr(model, 'module') else model
        for n, p in m.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[n].copy_(p.detach().float())

    @torch.no_grad()
    def store(self, model):
        """Back up the current (online) params so they can be restored after EMA eval."""
        m = model.module if hasattr(model, 'module') else model
        self.backup = {n: p.detach().clone() for n, p in m.named_parameters()}

    @torch.no_grad()
    def copy_to(self, model):
        """Load EMA params into the model (for eval / saving)."""
        m = model.module if hasattr(model, 'module') else model
        for n, p in m.named_parameters():
            p.data.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, model):
        m = model.module if hasattr(model, 'module') else model
        for n, p in m.named_parameters():
            p.data.copy_(self.backup[n])
        self.backup = {}


class FGM:
    """
    Fast Gradient Method adversarial training (item 4; Phase 2 only).

    Perturbs BOTH modalities' token embeddings along the normalized gradient
    direction (epsilon * g / ||g||). The normalization makes the perturbation
    scale-invariant, so it is correct under AMP/GradScaler (the loss-scale cancels).
    ITGCA priors are computed from token IDs, not embeddings, so they stay identical
    across the clean and adversarial passes.
    """

    DEFAULT_TARGETS = ('embedding_raw.token_embedding', 'embedding_size.token_embedding')

    def __init__(self, model, epsilon, targets=DEFAULT_TARGETS):
        self.model = model.module if hasattr(model, 'module') else model
        self.epsilon = epsilon
        self.targets = targets
        self.backup = {}

    @torch.no_grad()
    def attack(self):
        for name, p in self.model.named_parameters():
            if (p.requires_grad and p.grad is not None
                    and any(t in name for t in self.targets)):
                norm = p.grad.norm()
                if torch.isfinite(norm) and norm > 0:
                    self.backup[name] = p.data.clone()
                    p.data.add_(self.epsilon * p.grad / norm)

    @torch.no_grad()
    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def rdrop_kl_loss(logits1, logits2):
    """Symmetric KL between two dropout views (item 6, R-Drop)."""
    lp1 = F.log_softmax(logits1, dim=-1)
    lp2 = F.log_softmax(logits2, dim=-1)
    return 0.5 * (F.kl_div(lp1, lp2, log_target=True, reduction='batchmean')
                  + F.kl_div(lp2, lp1, log_target=True, reduction='batchmean'))


def compute_loss(args, model, inputs, tgt, criterion):
    """
    Unified per-batch loss honoring the item-5/6 switches.

    - R-Drop (--use_rdrop): two independent dropout passes + alpha * symmetric KL.
    - Multi-sample dropout (--use_msd): head returns [B, K, C]; average per-sample CE.
    - Otherwise: plain cross-entropy.
    MSD and R-Drop are mutually exclusive (asserted in main()).
    """
    if getattr(args, 'use_rdrop', False) and model.training:
        logits1 = model(*inputs)
        logits2 = model(*inputs)
        ce = 0.5 * (criterion(logits1, tgt) + criterion(logits2, tgt))
        return ce + args.rdrop_alpha * rdrop_kl_loss(logits1, logits2)

    logits = model(*inputs)
    if logits.dim() == 3:  # multi-sample dropout: [B, K, C]
        k = logits.size(1)
        loss = sum(criterion(logits[:, j, :], tgt) for j in range(k)) / k
    else:
        loss = criterion(logits, tgt)
    if loss.dim() > 0:  # DataParallel reduction safety
        loss = loss.mean()
    return loss


def train_epoch(args, model, optimizer, scheduler, train_data_tensors, criterion,
                scaler=None, fgm=None, ema=None):
    """Train for one epoch (optional AMP, FGM adversarial, and EMA update)."""
    model.train()
    total_loss = 0.0
    step = 0
    use_amp = scaler is not None

    for raw_src, packet_ids, directions, size_src, iat_src, tgt in batch_loader(
            args.batch_size, train_data_tensors, shuffle=True):
        raw_src = raw_src.to(args.device, non_blocking=True)
        packet_ids = packet_ids.to(args.device, non_blocking=True)
        directions = directions.to(args.device, non_blocking=True)
        size_src = size_src.to(args.device, non_blocking=True)
        iat_src = iat_src.to(args.device, non_blocking=True)
        tgt = tgt.to(args.device, non_blocking=True)
        inputs = (raw_src, packet_ids, directions, size_src, iat_src)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss = compute_loss(args, model, inputs, tgt, criterion)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # FGM: perturb token embeddings, accumulate a second (adversarial) gradient.
        if fgm is not None:
            fgm.attack()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss_adv = compute_loss(args, model, inputs, tgt, criterion)
            if use_amp:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            fgm.restore()

        if use_amp:
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                # Only clip when no inf/nan is present, to avoid inf*0 = NaN inside clip_grad_norm.
                found_inf = sum(v.item() for state in scaler._per_optimizer_states.values()
                                for v in state["found_inf_per_device"].values()) > 0
                if not found_inf:
                    params = (model.module if hasattr(model, 'module') else model).parameters()
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.max_grad_norm > 0:
                params = (model.module if hasattr(model, 'module') else model).parameters()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        step += 1

        if step % args.report_steps == 0:
            print(f"  Step {step}, Avg loss: {total_loss / step:.4f}")

    return total_loss / max(step, 1)


def evaluate(args, model, eval_tensors, print_confusion=False):
    """Evaluate the model on the given dataset."""
    model.eval()

    y_true, y_pred = [], []
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    with torch.no_grad():
        for raw_src, packet_ids, directions, size_src, iat_src, tgt in batch_loader(
                args.batch_size, eval_tensors):
            raw_src = raw_src.to(args.device, non_blocking=True)
            packet_ids = packet_ids.to(args.device, non_blocking=True)
            directions = directions.to(args.device, non_blocking=True)
            size_src = size_src.to(args.device, non_blocking=True)
            iat_src = iat_src.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)

            logits = model(raw_src, packet_ids, directions, size_src, iat_src)
            pred = torch.argmax(logits, dim=-1)

            for p, g in zip(pred.cpu().tolist(), tgt.cpu().tolist()):
                confusion[g, p] += 1  # [ground_truth, predicted] - standard convention
                y_pred.append(p)
                y_true.append(g)

    correct = sum(1 for p, g in zip(y_pred, y_true) if p == g)
    acc = correct / len(y_true)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    micro_p = precision_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_p = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    micro_r = recall_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Acc: {acc:.4f} ({correct}/{len(y_true)})")
    print(f"Precision - Macro: {macro_p:.4f}, Micro: {micro_p:.4f}, Weighted: {weighted_p:.4f}")
    print(f"Recall    - Macro: {macro_r:.4f}, Micro: {micro_r:.4f}, Weighted: {weighted_r:.4f}")
    print(f"F1        - Macro: {macro_f1:.4f}, Micro: {micro_f1:.4f}, Weighted: {weighted_f1:.4f}")

    if print_confusion:
        print("\nConfusion Matrix:")
        print(confusion)
        print("\nPer-class metrics:")
        eps = 1e-9
        for i in range(args.labels_num):
            # confusion[g, p]: row=ground_truth, col=predicted
            # Precision = TP / (TP + FP) = correct fraction among items predicted as i
            # Recall    = TP / (TP + FN) = correct fraction among items truly i
            p = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)  # column sum
            r = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)  # row sum
            f1 = 2 * p * r / (p + r + eps)
            print(f"  Label {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    return macro_f1, confusion


def freeze_backbone(model, verbose=True):
    """Phase 1: freeze everything except the classifier head."""
    actual_model = model.module if hasattr(model, 'module') else model
    for name, param in actual_model.named_parameters():
        param.requires_grad = name.startswith('classifier')
    trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in actual_model.parameters())
    if verbose:
        print(f"  Backbone frozen: {trainable}/{total} params trainable")


def unfreeze_backbone(model, verbose=True):
    """Phase 2: unfreeze every parameter."""
    actual_model = model.module if hasattr(model, 'module') else model
    for param in actual_model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    if verbose:
        print(f"  All params unfrozen: {trainable} params trainable")


def build_phase1_optimizer(args, model, num_train_samples, verbose=True):
    """Build the Phase 1 optimizer/scheduler: classifier-only with AdamW."""
    actual_model = model.module if hasattr(model, 'module') else model
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    decay_params, no_decay_params = [], []
    for name, param in actual_model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = str2optimizer["adamw"](
        optimizer_grouped_parameters, lr=args.phase1_lr, correct_bias=True
    )

    train_steps = max(1, int(num_train_samples * args.phase1_epochs / args.batch_size) + 1)
    scheduler = str2scheduler[args.scheduler](
        optimizer, int(train_steps * args.warmup), train_steps
    )

    if verbose:
        print(f"  Phase 1 optimizer: AdamW, lr={args.phase1_lr:.1e}, steps={train_steps}")
    return optimizer, scheduler


def build_phase2_optimizer(args, model, verbose=True):
    """
    Phase 2 optimizer/scheduler with two LR anchors + optional per-layer LLRD (item 3).

        classifier      LR = base_lr
        fusion  stack   top-layer LR = base_lr * --llrd_fusion_ratio  (0.7)
        encoder stacks  top-layer LR = base_lr * --llrd_encoder_ratio (0.3)

    With --use_llrd, each stack is geometrically decayed downward by --llrd_decay
    per layer (top layer = anchor, embedding = anchor * decay^depth), applied
    INDEPENDENTLY within the raw encoder, the size encoder, and the fusion stack.
    --llrd_decay == 1.0 (or --use_llrd off) reproduces the original flat
    encoder/fusion/classifier grouping exactly.

    ITGCA calibration parameters (stat_scale / stat_shift / local_stat_scale /
    local_stat_shift) stay on a separate faster group (5 * fusion anchor, no decay).
    """
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    calibration_names = ('stat_scale', 'stat_shift', 'local_stat_scale', 'local_stat_shift')

    actual_model = model.module if hasattr(model, 'module') else model
    base_lr = args.learning_rate
    encoder_anchor = base_lr * args.llrd_encoder_ratio
    fusion_anchor = base_lr * args.llrd_fusion_ratio
    decay = args.llrd_decay if getattr(args, 'use_llrd', False) else 1.0

    base_layers = args.layers_num
    l_raw = getattr(args, 'layers_num_raw', base_layers) or base_layers
    l_size = getattr(args, 'layers_num_size', base_layers) or base_layers
    l_fus = getattr(args, 'num_fusion_layers', 6)

    def _idx(name, marker):
        return int(name.split(marker, 1)[1].split('.', 1)[0])

    def param_lr(name):
        # Depth measured from the top of each stack; deeper => more decay.
        if name.startswith('classifier'):
            return base_lr
        if 'encoder_raw.transformer.' in name:
            return encoder_anchor * (decay ** (l_raw - 1 - _idx(name, 'encoder_raw.transformer.')))
        if 'encoder_size.transformer.' in name:
            return encoder_anchor * (decay ** (l_size - 1 - _idx(name, 'encoder_size.transformer.')))
        if name.startswith('embedding_raw'):
            return encoder_anchor * (decay ** l_raw)
        if name.startswith('embedding_size'):
            return encoder_anchor * (decay ** l_size)
        if 'fusion.fusion_layers.' in name:
            return fusion_anchor * (decay ** (l_fus - 1 - _idx(name, 'fusion.fusion_layers.')))
        if name.startswith('fusion'):
            return fusion_anchor
        return base_lr

    # Bucket params into groups keyed by (lr, weight_decay); calibration params separate.
    groups = {}
    for name, param in actual_model.named_parameters():
        if not param.requires_grad:
            continue
        is_calib = ('fusion' in name) and any(cn in name for cn in calibration_names)
        if is_calib:
            key, lr, wd = ('calib',), fusion_anchor * 5, 0.0
        else:
            lr = param_lr(name)
            wd = 0.0 if any(nd in name for nd in no_decay) else 0.01
            key = (round(lr, 12), wd)
        g = groups.setdefault(key, {'params': [], 'lr': lr, 'weight_decay': wd})
        g['params'].append(param)

    optimizer_grouped_parameters = list(groups.values())
    optimizer = str2optimizer["adamw"](
        optimizer_grouped_parameters, lr=base_lr, correct_bias=True
    )
    scheduler = str2scheduler[args.scheduler](
        optimizer, args.train_steps * args.warmup, args.train_steps
    )

    n_calib = sum(len(g['params']) for k, g in groups.items() if k == ('calib',))
    if verbose:
        print(f"  LLRD: {'ON' if decay < 1.0 else 'OFF (flat)'} (decay={decay}), "
              f"encoder_anchor={encoder_anchor:.2e}, fusion_anchor={fusion_anchor:.2e}, "
              f"classifier_lr={base_lr:.2e}")
        print(f"  Phase 2 param groups: {len(optimizer_grouped_parameters)} "
              f"(calibration params: {n_calib}, lr={fusion_anchor * 5:.2e})")
    return optimizer, scheduler


def _gib(num_bytes):
    return num_bytes / BYTES_PER_GIB


def _capture_rng_state(device):
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(device),
    }


def _restore_rng_state(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'], device)


def _capture_model_probe_state(model):
    actual_model = model.module if hasattr(model, 'module') else model
    return {
        'state_dict': {
            name: tensor.detach().cpu().clone()
            for name, tensor in actual_model.state_dict().items()
        },
        'requires_grad': [(param, param.requires_grad)
                          for param in actual_model.parameters()],
        'training_modes': [(module, module.training)
                           for module in actual_model.modules()],
    }


def _restore_model_probe_state(model, state):
    actual_model = model.module if hasattr(model, 'module') else model
    model.zero_grad(set_to_none=True)
    actual_model.load_state_dict(state['state_dict'], strict=True)
    for param, requires_grad in state['requires_grad']:
        param.requires_grad = requires_grad
    for module, training in state['training_modes']:
        module.training = training


def _build_probe_batch(args, train_tensors):
    samples_num = train_tensors['label'].size(0)
    if samples_num == 0:
        raise ValueError("Cannot probe GPU memory with an empty training dataset.")

    end = min(args.batch_size, samples_num)
    keys = ('raw_src', 'packet_ids', 'directions', 'size_src', 'iat_src', 'label')
    return tuple(
        train_tensors[key][:end].to(args.device, non_blocking=True)
        for key in keys
    )


def _run_memory_probe_step(args, model, optimizer, inputs, tgt, criterion,
                           scaler=None, fgm=None, ema=None):
    """Run the same GPU-heavy lifecycle as one real training step."""
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        loss = compute_loss(args, model, inputs, tgt, criterion)
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if fgm is not None:
        fgm.attack()
        try:
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss_adv = compute_loss(args, model, inputs, tgt, criterion)
            if use_amp:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
        finally:
            fgm.restore()

    if use_amp:
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            found_inf = sum(
                value.item()
                for scaler_state in scaler._per_optimizer_states.values()
                for value in scaler_state["found_inf_per_device"].values()
            ) > 0
            if not found_inf:
                actual_model = model.module if hasattr(model, 'module') else model
                torch.nn.utils.clip_grad_norm_(actual_model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if args.max_grad_norm > 0:
            actual_model = model.module if hasattr(model, 'module') else model
            torch.nn.utils.clip_grad_norm_(actual_model.parameters(), args.max_grad_norm)
        optimizer.step()

    if ema is not None:
        ema.update(model)


def _run_memory_probe_evaluation(model, inputs, ema=None):
    """Cover the online and optional EMA validation memory lifecycle."""
    model.eval()
    with torch.no_grad():
        logits = model(*inputs)
        del logits

    if ema is None:
        return

    ema.store(model)
    try:
        ema.copy_to(model)
        with torch.no_grad():
            logits = model(*inputs)
            del logits
    finally:
        ema.restore(model)


def _measure_training_memory_peak(args, model, train_tensors, criterion, phase):
    """Measure one phase with real tensor shapes without retaining training state."""
    optimizer = scheduler = scaler = ema = fgm = probe_batch = None
    device = args.device

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        if phase == 1:
            freeze_backbone(model, verbose=False)
            optimizer, scheduler = build_phase1_optimizer(
                args, model, train_tensors['label'].size(0), verbose=False
            )
        elif phase == 2:
            unfreeze_backbone(model, verbose=False)
            optimizer, scheduler = build_phase2_optimizer(args, model, verbose=False)
            ema = ModelEMA(model, decay=0.999)
            fgm = FGM(model, args.adv_epsilon) if args.use_fgm else None
        else:
            raise ValueError(f"Unknown memory-probe phase: {phase}")

        # A zero LR exercises AdamW's real allocation/update path without changing
        # parameters. A full CPU snapshot is restored after both probes as a second
        # guard against any future optimizer/model changes.
        for group in optimizer.param_groups:
            group['lr'] = 0.0

        use_amp = getattr(args, 'fp16', False)
        if use_amp:
            scaler = torch.amp.GradScaler(device="cuda", init_scale=1.0)

        probe_batch = _build_probe_batch(args, train_tensors)
        inputs, tgt = probe_batch[:-1], probe_batch[-1]
        model.train()

        for _ in range(MEMORY_PROBE_STEPS):
            _run_memory_probe_step(
                args, model, optimizer, inputs, tgt, criterion,
                scaler=scaler, fgm=fgm, ema=ema
            )

        _run_memory_probe_evaluation(model, inputs, ema=ema)
        torch.cuda.synchronize(device)
        return {
            'allocated': torch.cuda.max_memory_allocated(device),
            'reserved': torch.cuda.max_memory_reserved(device),
        }
    finally:
        if fgm is not None and fgm.backup:
            fgm.restore()
        if ema is not None and ema.backup:
            ema.restore(model)
        model.zero_grad(set_to_none=True)
        optimizer = scheduler = scaler = ema = fgm = probe_batch = None
        gc.collect()
        torch.cuda.synchronize(device)


class Phase1GpuMemoryReservation:
    """Live, computation-independent CUDA tensors held only during Phase 1."""

    def __init__(self, device, tensors, target_reserved, phase1_peak, phase2_peak):
        self.device = device
        self.tensors = tensors
        self.target_reserved = target_reserved
        self.phase1_peak = phase1_peak
        self.phase2_peak = phase2_peak
        self.reserved_bytes = sum(tensor.numel() * tensor.element_size()
                                  for tensor in tensors)

    def verify(self):
        torch.cuda.synchronize(self.device)
        current_reserved = torch.cuda.memory_reserved(self.device)
        if current_reserved + MEMORY_RESERVATION_TOLERANCE_BYTES < self.target_reserved:
            raise RuntimeError(
                "Phase 1 GPU memory reservation was unexpectedly released: "
                f"current_reserved={_gib(current_reserved):.2f} GiB, "
                f"target={_gib(self.target_reserved):.2f} GiB."
            )

    def release_for_phase2(self):
        """Return live blocks to PyTorch's cache so Phase 2 can reuse them."""
        self.verify()
        self.tensors.clear()
        self.reserved_bytes = 0
        gc.collect()
        torch.cuda.synchronize(self.device)
        print(f"  Phase 1 reservation released to the PyTorch cache; "
              f"process still reserves {_gib(torch.cuda.memory_reserved(self.device)):.2f} GiB.")


def _allocate_phase1_reservation(device, num_bytes, target_reserved,
                                 phase1_peak, phase2_peak):
    tensors = []
    remaining = num_bytes
    try:
        while remaining > 0:
            chunk_bytes = min(remaining, MEMORY_RESERVATION_CHUNK_BYTES)
            tensors.append(torch.empty(chunk_bytes, dtype=torch.uint8, device=device))
            remaining -= chunk_bytes
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            "Phase 2 memory probing succeeded, but the Phase 1 reservation could "
            f"not retain {_gib(num_bytes):.2f} GiB on {device}. Another process may "
            "have claimed memory during setup; training was not started.\n"
            f"Original PyTorch OOM: {error}"
        ) from None

    reservation = Phase1GpuMemoryReservation(
        device, tensors, target_reserved, phase1_peak, phase2_peak
    )
    reservation.verify()
    return reservation


def setup_phase1_gpu_memory_reservation(args, model, train_tensors, criterion):
    """Probe real single-GPU lifecycles and reserve the Phase 1/2 memory gap."""
    if args.reserve_gpu_memory == 'none':
        return None
    if args.device.type != 'cuda':
        raise RuntimeError("--reserve_gpu_memory auto requires CUDA.")
    if isinstance(model, torch.nn.DataParallel):
        raise RuntimeError(
            "--reserve_gpu_memory auto currently supports one GPU only. "
            "Use --world_size 1 with exactly one --gpu_ranks entry."
        )

    device = args.device
    print("  GPU memory reservation: probing real single-GPU Phase 1/Phase 2 peaks...")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')
    print(f"    process pid={os.getpid()}, CUDA_VISIBLE_DEVICES={visible_devices}, "
          f"device={device}, batch_size={args.batch_size}, "
          f"AMP={'on' if args.fp16 else 'off'}, "
          f"global_free={_gib(free_bytes):.2f}/{_gib(total_bytes):.2f} GiB")

    rng_state = _capture_rng_state(device)
    model_state = _capture_model_probe_state(model)
    phase1_peak = phase2_peak = retained_phase1_peak = None
    reservation = None
    failed_phase = 'Phase 1'
    oom_message = None
    setup_succeeded = False

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    try:
        phase1_peak = _measure_training_memory_peak(
            args, model, train_tensors, criterion, phase=1
        )

        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        failed_phase = 'Phase 2'
        phase2_peak = _measure_training_memory_peak(
            args, model, train_tensors, criterion, phase=2
        )

        target_reserved = max(phase1_peak['reserved'], phase2_peak['reserved'])
        reservation_bytes = max(0, target_reserved - phase1_peak['allocated'])
        reservation = _allocate_phase1_reservation(
            device, reservation_bytes, target_reserved, phase1_peak, phase2_peak
        )

        # The live tensors can alter allocator block layout. Replaying Phase 1
        # while they are held proves that the remaining workspace is usable,
        # rather than relying only on an arithmetic byte difference.
        failed_phase = 'Phase 1 reservation verification'
        retained_phase1_peak = _measure_training_memory_peak(
            args, model, train_tensors, criterion, phase=1
        )
        setup_succeeded = True
    except torch.OutOfMemoryError as error:
        oom_message = str(error)
    finally:
        if not setup_succeeded and reservation is not None:
            reservation.tensors.clear()
            reservation = None
        _restore_model_probe_state(model, model_state)
        _restore_rng_state(rng_state, device)
        del model_state
        gc.collect()
        torch.cuda.synchronize(device)

    if oom_message is not None:
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        raise RuntimeError(
            f"Automatic GPU memory probe ran out of memory during {failed_phase}. "
            "The corresponding real training phase is likely to OOM as well, so "
            "training was not started.\n"
            f"  device={device}, free={_gib(free_bytes):.2f}/{_gib(total_bytes):.2f} GiB, "
            f"batch_size={args.batch_size}, AMP={'on' if args.fp16 else 'off'}\n"
            f"Original PyTorch OOM: {oom_message}"
        ) from None

    print(f"    Phase 1 peak: allocated={_gib(phase1_peak['allocated']):.2f} GiB, "
          f"reserved={_gib(phase1_peak['reserved']):.2f} GiB")
    print(f"    Phase 2 peak: allocated={_gib(phase2_peak['allocated']):.2f} GiB, "
          f"reserved={_gib(phase2_peak['reserved']):.2f} GiB")
    print(f"    Phase 1 with reservation: "
          f"allocated={_gib(retained_phase1_peak['allocated']):.2f} GiB, "
          f"reserved={_gib(retained_phase1_peak['reserved']):.2f} GiB")

    target_reserved = reservation.target_reserved
    reservation_bytes = reservation.reserved_bytes
    current_allocated = torch.cuda.memory_allocated(device)
    current_reserved = torch.cuda.memory_reserved(device)
    print(f"    Phase 1 live reservation tensors: {_gib(reservation_bytes):.2f} GiB")
    print(f"    After reservation: allocated={_gib(current_allocated):.2f} GiB, "
          f"reserved={_gib(current_reserved):.2f} GiB "
          f"(target={_gib(target_reserved):.2f} GiB)")
    return reservation


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)

    # Per-modality config overrides (asymmetric encoder depths).
    parser.add_argument("--config_path_raw", type=str, default=None,
                        help="Optional per-modality config for the Raw (content) encoder. "
                             "Only 'layers_num' is applied; shared dims must match --config_path.")
    parser.add_argument("--config_path_size", type=str, default=None,
                        help="Optional per-modality config for the Size (behavior) encoder.")

    # Path options.
    parser.add_argument("--label2id_path", type=str, required=True,
                        help="Path to label2id mapping (pickle)")
    parser.add_argument("--vocab_path_raw", type=str, default="./test/vocab_raw_bytes.txt",
                        help="Path to raw modality vocabulary")
    parser.add_argument("--vocab_path_size", type=str, default="./test/vocab_size.txt",
                        help="Path to size modality vocabulary")
    parser.add_argument("--vocab_path_temporal", type=str, default="./test/vocab_temporal.txt",
                        help="Path to temporal (IAT) modality vocabulary")

    # Training options.
    parser.add_argument("--earlystop", type=int, default=5,
                        help="Patience for Phase 2 early stopping.")

    # Few-shot (paper §4.3 label efficiency).
    parser.add_argument("--few_shot", type=float, default=None,
                        help="Few-shot ratio: fraction of training data to use (e.g. 0.1 = 10%%). "
                             "Stratified sampling preserves class distribution; min 1 sample/class.")

    # Model options.
    parser.add_argument("--num_fusion_layers", type=int, default=6,
                        help="Number of fusion layers (must match the pretrained model).")
    parser.add_argument("--use_itgca", action="store_true",
                        help="Enable ITGCA gate mechanism. MUST match the pretrained model's configuration.")
    parser.add_argument("--use_mlp_gate", action="store_true",
                        help="Enable the shared lightweight MLP gate. MUST match the "
                             "pretrained model's configuration.")
    parser.add_argument("--itgca_window_size", type=int, default=16,
                        help="Sliding window size for ITGCA local entropy.")

    # ITGCA component-level ablation flags -- MUST match the pretrained checkpoint.
    parser.add_argument("--ablate_r_stat", action="store_true",
                        help="Disable r_stat prior. Must match the pretrained checkpoint.")
    parser.add_argument("--ablate_r_learned", action="store_true",
                        help="Disable learned compatibility r_learned. Must match the pretrained checkpoint.")
    parser.add_argument("--ablate_g_token", action="store_true",
                        help="Disable token-level gate. Must match the pretrained checkpoint.")
    parser.add_argument("--ablate_source_bias", action="store_true",
                        help="Disable source-side attention bias. Must match the pretrained checkpoint.")

    # Sequence lengths.
    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

    # Loss / optimization.
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0.0 = no smoothing).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disable).")

    # Layer-wise LR Decay (Phase 2; paper §4.1.4 uses 0.3 / 0.7).
    parser.add_argument("--llrd_encoder_ratio", type=float, default=0.3,
                        help="Phase 2 encoder LR = base_LR * this ratio (paper: 0.3).")
    parser.add_argument("--llrd_fusion_ratio", type=float, default=0.7,
                        help="Phase 2 fusion LR = base_LR * this ratio (paper: 0.7).")

    # Two-Phase training (paper §4.1.4).
    parser.add_argument("--phase1_epochs", type=int, default=5,
                        help="Phase 1 epochs: backbone frozen, classifier head only (paper: 5).")
    parser.add_argument("--phase1_lr", type=float, default=1e-3,
                        help="Phase 1 learning rate (classifier-only).")

    # ---- Fine-tuning tricks (items 3/4/5/6; default OFF -> reproduces the baseline) ----
    # Item 3: per-layer LLRD. Off -> current flat encoder/fusion/classifier LRs.
    parser.add_argument("--use_llrd", action="store_true",
                        help="Enable per-layer LLRD within each stack (anchored on "
                             "--llrd_encoder_ratio / --llrd_fusion_ratio).")
    parser.add_argument("--llrd_decay", type=float, default=0.9,
                        help="Geometric LR decay per layer, top=anchor going down. "
                             "Only used with --use_llrd; 1.0 == flat.")
    # Item 4: FGM adversarial training on token embeddings (Phase 2 only).
    parser.add_argument("--use_fgm", action="store_true",
                        help="Enable FGM adversarial training (perturbs both modalities' "
                             "token embeddings). Phase 2 only.")
    parser.add_argument("--adv_epsilon", type=float, default=1.0,
                        help="FGM perturbation magnitude (normalized-gradient units).")
    # Item 5: multi-sample dropout in the classifier head.
    parser.add_argument("--use_msd", action="store_true",
                        help="Enable multi-sample dropout in the head (averages per-sample "
                             "CE). Mutually exclusive with --use_rdrop.")
    parser.add_argument("--msd_num", type=int, default=4,
                        help="Number of dropout samples for --use_msd.")
    # Item 6: R-Drop consistency regularization.
    parser.add_argument("--use_rdrop", action="store_true",
                        help="Enable R-Drop (two dropout passes + symmetric KL). "
                             "Mutually exclusive with --use_msd.")
    parser.add_argument("--rdrop_alpha", type=float, default=1.0,
                        help="Weight of the R-Drop symmetric-KL term.")

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int,
                        help="List of GPU ranks to use (e.g., --gpu_ranks 0 1).")
    parser.add_argument("--reserve_gpu_memory", choices=("none", "auto"), default="none",
                        help="Single-GPU Phase 1 memory occupancy matching. 'auto' "
                             "probes the real Phase 1/Phase 2 lifecycle and retains "
                             "the measured difference; 'none' disables it.")

    args = parser.parse_args()

    if args.use_itgca and args.use_mlp_gate:
        parser.error("--use_itgca and --use_mlp_gate are mutually exclusive.")

    # Guardrail: items 5 and 6 both reshape dropout/loss; keep them mutually exclusive.
    assert not (args.use_msd and args.use_rdrop), \
        "--use_msd and --use_rdrop are mutually exclusive (item-5 vs item-6)."

    # Load hyperparameters from config.
    if args.config_path:
        args = load_hyperparam(args)
    args = apply_modality_configs(args)

    # Set max_seq_length so that embedding tables cover whichever modality is longer.
    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)

    set_seed(args.seed)

    # Vocabularies.
    print("Loading vocabularies...")
    vocab_raw = Vocab(); vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab(); vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab(); vocab_temporal.load(args.vocab_path_temporal)
    print(f"  Raw vocab size:      {len(vocab_raw)}")
    print(f"  Size vocab size:     {len(vocab_size)}")
    print(f"  Temporal vocab size: {len(vocab_temporal)}")

    # Label mapping.
    print("Loading label mapping...")
    label2id = load_dataset(args.label2id_path)
    args.labels_num = len(label2id)
    print(f"Number of labels: {args.labels_num}")

    # Datasets.
    print("Loading datasets...")
    train_data = load_dataset(args.train_path)
    dev_data = load_dataset(args.dev_path)
    test_data = load_dataset(args.test_path) if args.test_path else None

    # Few-shot subsampling (training set only).
    if args.few_shot is not None:
        assert 0.0 < args.few_shot <= 1.0, f"--few_shot must be in (0, 1], got {args.few_shot}"
        train_data = few_shot_sample(train_data, args.few_shot, seed=args.seed)

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, "
          f"Test: {len(test_data) if test_data else 0}")

    # Pre-tensorize.
    use_pin = torch.cuda.is_available()
    print("Pre-tensorizing datasets...")
    train_tensors = pre_tensorize(train_data, pin=use_pin)
    dev_tensors = pre_tensorize(dev_data, pin=use_pin)
    test_tensors = pre_tensorize(test_data, pin=use_pin) if test_data else None

    # Build model.
    print("Building model...")
    model = Stage2Classifier(args, len(vocab_raw), len(vocab_size),
                             len(vocab_temporal), args.labels_num)
    gate_mode = ('ITGCA' if args.use_itgca else
                 'lightweight MLP' if args.use_mlp_gate else 'none')
    print(f"  Fusion gate: {gate_mode}")

    # Load pretrained Stage 2 weights.
    load_pretrained_model(model, args.pretrained_model_path)

    # GPU setup.
    ranks_num = len(args.gpu_ranks)
    if args.world_size > 1 and ranks_num > 1:
        assert torch.cuda.is_available(), "No available GPUs."
        assert len(set(args.gpu_ranks)) == ranks_num, \
            f"Duplicate GPU IDs are not allowed: {args.gpu_ranks}"
        invalid_ranks = [
            rank for rank in args.gpu_ranks
            if rank < 0 or rank >= torch.cuda.device_count()
        ]
        assert not invalid_ranks, \
            f"GPU IDs not available: {invalid_ranks}"
        primary_gpu = args.gpu_ranks[0]
        args.device = torch.device(f"cuda:{primary_gpu}")
        model = model.to(args.device)
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ranks)
        print(f"Using DataParallel on GPUs: {args.gpu_ranks}")
    elif ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs."
        gpu_id = args.gpu_ranks[0]
        assert 0 <= gpu_id < torch.cuda.device_count(), \
            f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs)."
        args.device = torch.device(f"cuda:{gpu_id}")
        model = model.to(args.device)
        print(f"Using single GPU: {gpu_id}")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        model = model.to(args.device)
        print("Using default GPU: cuda:0")
    else:
        args.device = torch.device("cpu")
        model = model.to(args.device)
        print("Using CPU mode")

    # Loss: standard CrossEntropy with label smoothing (paper §4.1.4 default).
    print("\nSetting up loss function...")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f"  Loss: CrossEntropy (label_smoothing={args.label_smoothing})")
    print(f"  Gradient clipping: {args.max_grad_norm if args.max_grad_norm > 0 else 'Disabled'}")

    # AMP (mixed precision): off by default, enable with --fp16.
    use_amp = getattr(args, 'fp16', False) and args.device.type == 'cuda'
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp) if use_amp else None
    print(f"  AMP: {'Enabled' if use_amp else 'Disabled'}")

    best_f1 = 0.0
    best_epoch = 0
    args.train_steps = max(1, int(len(train_data) * args.epochs_num / args.batch_size) + 1)

    # ===== Phase 1: Classifier head warmup (backbone frozen) =====
    print("\n" + "=" * 50)
    print("Two-Phase Training (paper §4.1.4)")
    print("=" * 50)

    print(f"\n--- Phase 1: Classifier head warmup ({args.phase1_epochs} epochs) ---")
    phase1_memory_reservation = setup_phase1_gpu_memory_reservation(
        args, model, train_tensors, criterion
    )
    freeze_backbone(model)
    p1_optimizer, p1_scheduler = build_phase1_optimizer(args, model, len(train_data))

    for epoch in range(1, args.phase1_epochs + 1):
        print(f"\n[Phase 1] Epoch {epoch}/{args.phase1_epochs}")
        print("-" * 30)
        avg_loss = train_epoch(args, model, p1_optimizer, p1_scheduler, train_tensors,
                               criterion, scaler)
        print(f"Training loss: {avg_loss:.4f}")
        print("Validation:")
        f1, _ = evaluate(args, model, dev_tensors)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            save_model(model, args.output_model_path)
            print(f"New best model saved! F1: {best_f1:.4f}")
        if phase1_memory_reservation is not None:
            phase1_memory_reservation.verify()

    print(f"\nPhase 1 complete. Best F1 so far: {best_f1:.4f}")

    # ===== Phase 2: Full fine-tuning with LLRD =====
    print(f"\n--- Phase 2: Full fine-tuning with LLRD ({args.epochs_num} epochs) ---")
    model.zero_grad(set_to_none=True)
    del p1_optimizer, p1_scheduler
    if phase1_memory_reservation is not None:
        phase1_memory_reservation.release_for_phase2()
        phase1_memory_reservation = None
    unfreeze_backbone(model)

    optimizer, scheduler = build_phase2_optimizer(args, model)

    # Item 1: EMA (always on in Phase 2, initialized from the post-Phase-1 weights).
    # Item 4: FGM (Phase 2 only).
    ema = ModelEMA(model, decay=0.999)
    fgm = FGM(model, args.adv_epsilon) if args.use_fgm else None
    print(f"  EMA: ON (decay=0.999, warmup-ramped)")
    print(f"  FGM adversarial: {('ON (eps=%.2f)' % args.adv_epsilon) if fgm else 'OFF'}")
    if args.use_msd:
        print(f"  Multi-sample dropout: ON (k={args.msd_num})")
    if args.use_rdrop:
        print(f"  R-Drop: ON (alpha={args.rdrop_alpha})")
    if args.use_rdrop and args.use_fgm:
        print("  WARNING: --use_rdrop + --use_fgm => 4 forward passes/step (2x R-Drop x 2x FGM).")

    patience_counter = 0
    for epoch in range(1, args.epochs_num + 1):
        print(f"\n[Phase 2] Epoch {epoch}/{args.epochs_num}")
        print("-" * 30)
        avg_loss = train_epoch(args, model, optimizer, scheduler, train_tensors,
                               criterion, scaler, fgm=fgm, ema=ema)
        print(f"Training loss: {avg_loss:.4f}")

        # Dual validation: online weights vs EMA weights; keep the better of the two.
        print("Validation [online]:")
        f1_online, _ = evaluate(args, model, dev_tensors)
        ema.store(model)
        ema.copy_to(model)
        print("Validation [EMA]:")
        f1_ema, _ = evaluate(args, model, dev_tensors)
        ema.restore(model)

        use_ema = f1_ema >= f1_online
        f1 = max(f1_online, f1_ema)
        print(f"  -> online F1={f1_online:.4f}, EMA F1={f1_ema:.4f}; "
              f"selected={'EMA' if use_ema else 'online'} ({f1:.4f})")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = args.phase1_epochs + epoch
            patience_counter = 0
            if use_ema:
                ema.store(model)
                ema.copy_to(model)
                save_model(model, args.output_model_path)
                ema.restore(model)
            else:
                save_model(model, args.output_model_path)
            print(f"New best model saved ({'EMA' if use_ema else 'online'})! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.earlystop}")
            if patience_counter >= args.earlystop:
                print(f"Early stopping at Phase 2 epoch {epoch}")
                break

    # ===== Final evaluation on the test set =====
    if test_data:
        print("\n" + "=" * 50)
        print("Test Set Evaluation")
        print("=" * 50)
        ckpt = torch.load(args.output_model_path, map_location=args.device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)
        evaluate(args, model, test_tensors, print_confusion=True)

    print("\nTraining complete!")
    print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
