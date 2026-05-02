"""
Stage 2 Multi-Modal Classifier (paper §4.1.4 fine-tuning recipe).

Architecture:
    - Raw encoder (loaded from Stage 2 pretrained checkpoint)
    - Size encoder (loaded from Stage 2 pretrained checkpoint)
    - Fusion module (loaded from Stage 2 pretrained checkpoint, with optional ITGCA)
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
        --seed 42 --gpu_ranks 0
"""

import os
import sys
sys.path.append(os.getcwd())

import random
import argparse
import torch
import torch.nn as nn
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


class Stage2Classifier(nn.Module):
    """
    Stage 2 Multi-Modal Classifier.

    Architecture:
        - Raw encoder (pretrained): embedding + transformer
        - Size encoder (pretrained): embedding + transformer
        - Fusion module (pretrained): bidirectional cross-attention with optional ITGCA
        - Classifier head: concat([fused_raw_cls, fused_size_cls,
                                   mean_pool(fused_raw), mean_pool(fused_size)])
                          -> Linear -> Tanh -> Dropout -> Linear -> labels
    """

    def __init__(self, args, vocab_size_raw, vocab_size_size, vocab_size_temporal, labels_num):
        super(Stage2Classifier, self).__init__()

        self.hidden_size = args.hidden_size
        self.labels_num = labels_num

        # ITGCA support (must match the pretrained checkpoint).
        self.use_itgca = getattr(args, 'use_itgca', False)
        self.itgca_window_size = getattr(args, 'itgca_window_size', 16)
        self.vocab_size_raw = vocab_size_raw
        self.ablate_r_stat = getattr(args, 'ablate_r_stat', False)
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
        self.classifier = nn.Sequential(
            nn.Linear(4 * args.hidden_size, args.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, labels_num)
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

        # ITGCA signals (no detach -- let gate gradients flow back into the encoder).
        if self.use_itgca:
            r_stat_raw = (None if self.ablate_r_stat
                          else compute_flow_reliability_raw(raw_src, vocab_size=self.vocab_size_raw))
            local_ent_raw = (None if self.ablate_source_bias
                             else compute_local_entropy(raw_src, self.itgca_window_size))
            itgca_kwargs = {
                'raw_cls_enc': raw_output[:, 0, :],
                'size_cls_enc': size_output[:, 0, :],
                'r_stat_raw': r_stat_raw,
                'local_ent_raw': local_ent_raw,
            }
        else:
            itgca_kwargs = {}

        # Fusion.
        raw_fused, size_fused, _ = self.fusion(
            raw_output, size_output, raw_seg, size_seg, **itgca_kwargs
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

    # ITGCA config mismatch detection.
    has_itgca_in_checkpoint = any(
        k.startswith('fusion.fusion_layers.0.gate_') for k in filtered_state.keys()
    )
    model_uses_itgca = getattr(model, 'use_itgca', False)
    if has_itgca_in_checkpoint and not model_uses_itgca:
        print(f"  WARNING: Pretrained model was trained WITH ITGCA, "
              f"but fine-tuning does NOT use --use_itgca. Gate weights discarded.")
    elif not has_itgca_in_checkpoint and model_uses_itgca:
        print(f"  WARNING: Pretrained model was trained WITHOUT ITGCA, "
              f"but fine-tuning uses --use_itgca. Gate parameters randomly initialized!")

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


def train_epoch(args, model, optimizer, scheduler, train_data_tensors, criterion, scaler=None):
    """Train for one epoch (with optional AMP via scaler)."""
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

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(raw_src, packet_ids, directions, size_src, iat_src)
            loss = criterion(logits, tgt)
            if loss.dim() > 0:  # DataParallel reduction.
                loss = torch.mean(loss)

        if use_amp:
            scaler.scale(loss).backward()
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
            loss.backward()
            if args.max_grad_norm > 0:
                params = (model.module if hasattr(model, 'module') else model).parameters()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()

        scheduler.step()

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


def freeze_backbone(model):
    """Phase 1: freeze everything except the classifier head."""
    actual_model = model.module if hasattr(model, 'module') else model
    for name, param in actual_model.named_parameters():
        param.requires_grad = name.startswith('classifier')
    trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in actual_model.parameters())
    print(f"  Backbone frozen: {trainable}/{total} params trainable")


def unfreeze_backbone(model):
    """Phase 2: unfreeze every parameter."""
    actual_model = model.module if hasattr(model, 'module') else model
    for param in actual_model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    print(f"  All params unfrozen: {trainable} params trainable")


def build_phase1_optimizer(args, model, num_train_samples):
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
        optimizer_grouped_parameters, lr=args.phase1_lr, correct_bias=False
    )

    train_steps = max(1, int(num_train_samples * args.phase1_epochs / args.batch_size) + 1)
    scheduler = str2scheduler[args.scheduler](
        optimizer, int(train_steps * args.warmup), train_steps
    )

    print(f"  Phase 1 optimizer: AdamW, lr={args.phase1_lr:.1e}, steps={train_steps}")
    return optimizer, scheduler


def build_phase2_optimizer(args, model):
    """
    Build the Phase 2 optimizer/scheduler with layer-wise LR decay (paper §4.1.4):
        encoder LR = base_lr * --llrd_encoder_ratio
        fusion  LR = base_lr * --llrd_fusion_ratio
        classifier LR = base_lr

    ITGCA calibration parameters (`stat_scale / stat_shift / local_stat_scale /
    local_stat_shift`) are kept on a separate, faster LR group (5 x fusion_lr,
    no weight decay) since they parameterize a sigmoid mapping and need a higher
    LR to escape saturation.
    """
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    calibration_names = ('stat_scale', 'stat_shift', 'local_stat_scale', 'local_stat_shift')

    actual_model = model.module if hasattr(model, 'module') else model
    encoder_lr = args.learning_rate * args.llrd_encoder_ratio
    fusion_lr = args.learning_rate * args.llrd_fusion_ratio
    classifier_lr = args.learning_rate

    encoder_decay, encoder_no_decay = [], []
    fusion_decay, fusion_no_decay, calibration_params = [], [], []
    classifier_decay, classifier_no_decay = [], []

    for name, param in actual_model.named_parameters():
        if 'embedding_raw' in name or 'encoder_raw' in name or \
           'embedding_size' in name or 'encoder_size' in name:
            (encoder_no_decay if any(nd in name for nd in no_decay) else encoder_decay).append(param)
        elif 'fusion' in name:
            if any(cn in name for cn in calibration_names):
                calibration_params.append(param)
            elif any(nd in name for nd in no_decay):
                fusion_no_decay.append(param)
            else:
                fusion_decay.append(param)
        elif 'classifier' in name:
            (classifier_no_decay if any(nd in name for nd in no_decay) else classifier_decay).append(param)

    optimizer_grouped_parameters = [
        {'params': encoder_decay,     'lr': encoder_lr,    'weight_decay': 0.01},
        {'params': encoder_no_decay,  'lr': encoder_lr,    'weight_decay': 0.0},
        {'params': fusion_decay,      'lr': fusion_lr,     'weight_decay': 0.01},
        {'params': fusion_no_decay,   'lr': fusion_lr,     'weight_decay': 0.0},
        {'params': classifier_decay,    'lr': classifier_lr, 'weight_decay': 0.01},
        {'params': classifier_no_decay, 'lr': classifier_lr, 'weight_decay': 0.0},
    ]
    if calibration_params:
        optimizer_grouped_parameters.append({
            'params': calibration_params,
            'lr': fusion_lr * 5,
            'weight_decay': 0.0,
        })

    optimizer = str2optimizer["adamw"](
        optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
    )

    scheduler = str2scheduler[args.scheduler](
        optimizer, args.train_steps * args.warmup, args.train_steps
    )

    print(f"  LLRD: encoder_lr={encoder_lr:.2e}, fusion_lr={fusion_lr:.2e}, "
          f"classifier_lr={classifier_lr:.2e}")
    if calibration_params:
        print(f"  Calibration params ({len(calibration_params)}): lr={fusion_lr * 5:.2e}, no weight_decay")

    return optimizer, scheduler


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
    parser.add_argument("--vocab_path_raw", type=str, required=True,
                        help="Path to raw modality vocabulary")
    parser.add_argument("--vocab_path_size", type=str, required=True,
                        help="Path to size modality vocabulary")
    parser.add_argument("--vocab_path_temporal", type=str, required=True,
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
    parser.add_argument("--itgca_window_size", type=int, default=16,
                        help="Sliding window size for ITGCA local entropy.")

    # ITGCA component-level ablation flags -- MUST match the pretrained checkpoint.
    parser.add_argument("--ablate_r_stat", action="store_true",
                        help="Disable r_stat prior. Must match the pretrained checkpoint.")
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

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int,
                        help="List of GPU ranks to use (e.g., --gpu_ranks 0 1).")

    args = parser.parse_args()

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

    # Load pretrained Stage 2 weights.
    load_pretrained_model(model, args.pretrained_model_path)

    # GPU setup.
    ranks_num = len(args.gpu_ranks)
    if args.world_size > 1 and ranks_num > 1:
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= torch.cuda.device_count(), "Specified GPUs exceed available GPUs."
        primary_gpu = args.gpu_ranks[0]
        args.device = torch.device(f"cuda:{primary_gpu}")
        model = model.to(args.device)
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ranks)
        print(f"Using DataParallel on GPUs: {args.gpu_ranks}")
    elif ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs."
        gpu_id = args.gpu_ranks[0]
        assert gpu_id < torch.cuda.device_count(), \
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

    # ===== Phase 1: Classifier head warmup (backbone frozen) =====
    print("\n" + "=" * 50)
    print("Two-Phase Training (paper §4.1.4)")
    print("=" * 50)

    print(f"\n--- Phase 1: Classifier head warmup ({args.phase1_epochs} epochs) ---")
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

    print(f"\nPhase 1 complete. Best F1 so far: {best_f1:.4f}")

    # ===== Phase 2: Full fine-tuning with LLRD =====
    print(f"\n--- Phase 2: Full fine-tuning with LLRD ({args.epochs_num} epochs) ---")
    unfreeze_backbone(model)

    args.train_steps = max(1, int(len(train_data) * args.epochs_num / args.batch_size) + 1)
    optimizer, scheduler = build_phase2_optimizer(args, model)

    patience_counter = 0
    for epoch in range(1, args.epochs_num + 1):
        print(f"\n[Phase 2] Epoch {epoch}/{args.epochs_num}")
        print("-" * 30)
        avg_loss = train_epoch(args, model, optimizer, scheduler, train_tensors,
                               criterion, scaler)
        print(f"Training loss: {avg_loss:.4f}")

        print("Validation:")
        f1, _ = evaluate(args, model, dev_tensors)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = args.phase1_epochs + epoch
            patience_counter = 0
            save_model(model, args.output_model_path)
            print(f"New best model saved! F1: {best_f1:.4f}")
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
