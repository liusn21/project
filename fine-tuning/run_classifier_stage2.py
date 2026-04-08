"""
Stage 2 Multi-Modal Classifier

Uses the full multimodal model with fusion (ALBEF-style).
Classification is done by concatenating fused CLS tokens from both modalities.

Usage:
    python fine-tuning/run_classifier_stage2.py \
        --train_path datasets/processed/train.pkl \
        --dev_path datasets/processed/val.pkl \
        --test_path datasets/processed/test.pkl \
        --label2id_path datasets/processed/label2id.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --pretrained_model_path models/multimodal_stage2.bin \
        --output_model_path models/classifier_stage2.bin \
        --config_path models/bert/base_config.json \
        --epochs_num 10 \
        --batch_size 32 \
        --use_itgca  # Add this if pretrained model was trained with ITGCA
"""

import os
import sys
sys.path.append(os.getcwd())

import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from sklearn.metrics import f1_score, precision_score, recall_score


# ============ EMA (Exponential Moving Average) ============
class EMA:
    """
    Exponential Moving Average: Smooth model weights for better generalization
    Initialized after first epoch to avoid being dragged by initial weights
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.initialized = False

    def init_shadow(self):
        """Initialize shadow weights (call after first epoch)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.initialized = True

    def update(self):
        if not self.initialized:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """Use EMA weights for evaluation"""
        if not self.initialized:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights"""
        if not self.initialized:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


from uer.layers import RawPacketEmbedding, PacketSizeEmbedding
from uer.layers.multimodal_fusion import MultiModalFusionEncoder
from uer.encoders import str2encoder
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.utils.constants import PAD_ID
from uer.utils import *
from uer.opts import finetune_opts
from uer.models.multimodal_model import (
    compute_flow_reliability_raw, compute_local_entropy
)


# ============ Attention Pooling ============
class AttentionPooling(nn.Module):
    """Learnable attention pooling replacing mean pooling"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden, mask):
        scores = self.attention(hidden).squeeze(-1)       # [B, L]
        scores = scores.masked_fill(~mask.bool(), -1e4)
        weights = F.softmax(scores, dim=-1)               # [B, L]
        return (weights.unsqueeze(-1) * hidden).sum(1)    # [B, H]


# ============ Supervised Contrastive Loss ============
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
    with optional feature queue (MoCo-style) for small batch sizes.

    Queue stores (feature, label) pairs from recent batches as extra
    positives/negatives. Gradients only flow through current batch features.
    """
    def __init__(self, temperature=0.1, queue_size=0):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        # Queue lazily initialized on first forward (auto-device)
        self._queue_feat = None
        self._queue_labels = None
        self._queue_ptr = 0
        self._queue_full = False

    @torch.no_grad()
    def _enqueue(self, features, labels):
        if self.queue_size <= 0:
            return
        B = features.size(0)
        if self._queue_feat is None:
            self._queue_feat = torch.zeros(self.queue_size, features.size(1),
                                           device=features.device, dtype=torch.float32)
            self._queue_labels = torch.full((self.queue_size,), -1,
                                            dtype=torch.long, device=features.device)
        ptr = self._queue_ptr
        if ptr + B <= self.queue_size:
            self._queue_feat[ptr:ptr + B] = features
            self._queue_labels[ptr:ptr + B] = labels
        else:
            tail = self.queue_size - ptr
            self._queue_feat[ptr:] = features[:tail]
            self._queue_labels[ptr:] = labels[:tail]
            head = B - tail
            self._queue_feat[:head] = features[tail:]
            self._queue_labels[:head] = labels[tail:]
        new_ptr = (ptr + B) % self.queue_size
        if not self._queue_full and (new_ptr <= ptr or B >= self.queue_size):
            self._queue_full = True
        self._queue_ptr = new_ptr

    def forward(self, features, labels):
        # Force FP32: FP16 下 exp/log 精度不足会产生 NaN
        features = features.float()
        device = features.device
        B = features.shape[0]
        if B <= 1:
            self._enqueue(features.detach(), labels)
            return torch.tensor(0.0, device=device)

        # Combine current batch with queue for more positive/negative pairs
        if self.queue_size > 0 and self._queue_feat is not None:
            valid_len = self.queue_size if self._queue_full else self._queue_ptr
            if valid_len > 0:
                q_feat = self._queue_feat[:valid_len]    # detached (stored without grad)
                q_labels = self._queue_labels[:valid_len]
                all_feat = torch.cat([features, q_feat], dim=0)    # [B+Q, D]
                all_labels = torch.cat([labels, q_labels], dim=0)  # [B+Q]
            else:
                all_feat, all_labels = features, labels
        else:
            all_feat, all_labels = features, labels

        N = all_feat.shape[0]

        # Similarity: [B, N] — gradients only for current batch (left side)
        sim = torch.matmul(features, all_feat.T) / self.temperature

        # Positive mask: same class (exclude self in batch×batch block)
        mask_pos = (labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()  # [B, N]
        mask_pos[:, :B].fill_diagonal_(0)

        if mask_pos.sum() == 0:
            self._enqueue(features.detach(), labels)
            return torch.tensor(0.0, device=device)

        # Numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # Denominator: all pairs except self
        mask_denom = torch.ones(B, N, device=device)
        mask_denom[:, :B].fill_diagonal_(0)
        exp_sim = torch.exp(sim) * mask_denom
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-prob of positives
        num_pos = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-8)
        valid = num_pos > 0
        if valid.sum() == 0:
            self._enqueue(features.detach(), labels)
            return torch.tensor(0.0, device=device)

        loss = -mean_log_prob[valid].mean()

        # Update queue
        self._enqueue(features.detach(), labels)
        return loss


# ============ SWA (Stochastic Weight Averaging) ============
class SWA:
    """Stochastic Weight Averaging for flatter minima"""
    def __init__(self, model):
        self.model = model
        self.swa_state = {}
        self.n_averaged = 0

    def update(self):
        actual = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in actual.named_parameters():
            if name not in self.swa_state:
                self.swa_state[name] = param.data.clone()
            else:
                self.swa_state[name] = (
                    self.swa_state[name] * self.n_averaged + param.data
                ) / (self.n_averaged + 1)
        self.n_averaged += 1

    def apply(self):
        actual = self.model.module if hasattr(self.model, 'module') else self.model
        self.backup = {}
        for name, param in actual.named_parameters():
            self.backup[name] = param.data.clone()
            if name in self.swa_state:
                param.data.copy_(self.swa_state[name])

    def restore(self):
        actual = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in actual.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017)

    Downweights well-classified samples, focuses training on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Compatible with class weights and label smoothing.
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        # Compute standard CE (with label smoothing) per sample
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # p_t: probability of the true class
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class Stage2Classifier(nn.Module):
    """
    Stage 2 Multi-Modal Classifier

    Architecture:
        - Raw encoder (pretrained): embedding + transformer
        - Size encoder (pretrained): embedding + transformer
        - Fusion module (pretrained): 6-layer bidirectional cross-attention
        - Concat fused CLS + mean pooling -> Classification head
    """

    def __init__(self, args, vocab_size_raw, vocab_size_size, vocab_size_temporal, labels_num):
        super(Stage2Classifier, self).__init__()

        self.hidden_size = args.hidden_size
        self.labels_num = labels_num

        # ITGCA support
        self.use_itgca = getattr(args, 'use_itgca', False)
        self.itgca_window_size = getattr(args, 'itgca_window_size', 16)
        self.vocab_size_raw = vocab_size_raw
        # Component-level ablation flags (must match pretrained checkpoint)
        self.ablate_r_stat = getattr(args, 'ablate_r_stat', False)
        self.ablate_source_bias = getattr(args, 'ablate_source_bias', False)

        # Raw modality encoder
        self.embedding_raw = RawPacketEmbedding(args, vocab_size_raw)
        self.encoder_raw = str2encoder[args.encoder](args)

        # Size modality encoder (with temporal/IAT support)
        self.embedding_size = PacketSizeEmbedding(args, vocab_size_size, vocab_size_temporal)
        self.encoder_size = str2encoder[args.encoder](args)

        # Fusion module
        num_fusion_layers = getattr(args, 'num_fusion_layers', 6)
        use_itgca = getattr(args, 'use_itgca', False)
        self.fusion = MultiModalFusionEncoder(args, num_layers=num_fusion_layers, use_itgca=use_itgca)

        # Multi-Sample Dropout
        self.num_dropouts = getattr(args, 'num_dropouts', 1)
        self.dropout_rate = args.dropout

        # Attention Pooling
        self.use_attn_pooling = getattr(args, 'use_attn_pooling', False)
        if self.use_attn_pooling:
            self.attn_pool_raw = AttentionPooling(args.hidden_size)
            self.attn_pool_size = AttentionPooling(args.hidden_size)

        # Classification head
        self.simple_classifier = getattr(args, 'simple_classifier', False)
        if self.simple_classifier:
            self.classifier = nn.Linear(2 * args.hidden_size, labels_num)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(4 * args.hidden_size, args.hidden_size),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, labels_num)
            )

        # Supervised Contrastive Learning projection head
        self.use_scl = getattr(args, 'use_scl', False)
        if self.use_scl:
            proj_in = 2 * args.hidden_size if self.simple_classifier else 4 * args.hidden_size
            self.scl_projection = nn.Sequential(
                nn.Linear(proj_in, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, 128)
            )

    def forward(self, raw_src, packet_ids, directions, size_src, iat_src):
        """
        Returns:
            logits: [batch, labels_num]
            — or (logits, scl_feat) during training when use_scl=True
        """
        # Raw encoder
        raw_emb = self.embedding_raw(raw_src, packet_ids, directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)

        # Size encoder
        size_emb = self.embedding_size(size_src, iat_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)

        # ITGCA signals (no detach — let gate gradients flow to encoder)
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

        # Fusion
        raw_fused, size_fused, _ = self.fusion(
            raw_output, size_output, raw_seg, size_seg, **itgca_kwargs
        )

        # Fused CLS tokens
        raw_cls = raw_fused[:, 0, :]
        size_cls = size_fused[:, 0, :]

        if self.simple_classifier:
            combined = torch.cat([raw_cls, size_cls], dim=-1)
        else:
            # Pooling over non-CLS, non-PAD positions
            if self.use_attn_pooling:
                raw_pool = self.attn_pool_raw(raw_fused[:, 1:, :], raw_seg[:, 1:])
                size_pool = self.attn_pool_size(size_fused[:, 1:, :], size_seg[:, 1:])
            else:
                raw_mask = raw_seg[:, 1:].unsqueeze(-1).float()
                raw_pool = (raw_fused[:, 1:, :] * raw_mask).sum(1) / (raw_mask.sum(1).clamp(min=1.0))
                size_mask = size_seg[:, 1:].unsqueeze(-1).float()
                size_pool = (size_fused[:, 1:, :] * size_mask).sum(1) / (size_mask.sum(1).clamp(min=1.0))
            combined = torch.cat([raw_cls, size_cls, raw_pool, size_pool], dim=-1)

        # Multi-Sample Dropout
        if self.training and self.num_dropouts > 1:
            logits = torch.mean(torch.stack(
                [self.classifier(F.dropout(combined, p=self.dropout_rate, training=True))
                 for _ in range(self.num_dropouts)]
            ), dim=0)
        else:
            logits = self.classifier(combined)

        # Supervised Contrastive Learning (FP32: 防止 FP16 反向梯度溢出)
        if self.use_scl and self.training:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                proj = self.scl_projection(combined.float())
            return logits, F.normalize(proj, dim=-1)
        return logits


def load_pretrained_model(model, pretrained_path):
    """
    Load pretrained Stage 2 multimodal model weights

    The pretrained model contains:
        - embedding_raw, encoder_raw (main encoders)
        - embedding_size, encoder_size (main encoders with temporal_embedding)
        - fusion (fusion layers)
        - momentum encoders (*_m), ITC projections, target layers, queues (excluded)

    Total params in multimodal.bin: 1226
        - Loaded: 803 (embedding_raw, encoder_raw, embedding_size, encoder_size, fusion)
        - Excluded: 423 (momentum, ITC, target, queues)
    """
    if pretrained_path is None:
        return

    print(f"Loading pretrained multimodal model from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')

    # Filter out momentum encoders, ITC projections, target layers, and queues
    # These are only used during pretraining and not needed for classification
    exclude_prefixes = ['embedding_raw_m', 'encoder_raw_m',
                        'embedding_size_m', 'encoder_size_m',
                        'itc_proj_raw_m', 'itc_proj_size_m',
                        'target', 'raw_queue', 'size_queue', 'queue_ptr']

    filtered_state = {}
    for k, v in state_dict.items():
        if not any(k.startswith(prefix) for prefix in exclude_prefixes):
            filtered_state[k] = v

    # Load into model
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    # Classifier and new fine-tuning modules should be missing (randomly initialized)
    ft_prefixes = ('classifier', 'scl_projection', 'attn_pool')
    classifier_missing = [k for k in missing if any(k.startswith(p) for p in ft_prefixes)]
    other_missing = [k for k in missing if not any(k.startswith(p) for p in ft_prefixes)]

    # Count excluded parameters by category
    excluded_count = len(state_dict) - len(filtered_state)

    print(f"  Checkpoint total keys: {len(state_dict)}")
    print(f"  Excluded keys (momentum/ITC/target/queues): {excluded_count}")
    print(f"  Filtered keys (should be loaded): {len(filtered_state)}")
    print(f"  Missing keys: {len(missing)} (classifier: {len(classifier_missing)}, other: {len(other_missing)})")
    print(f"  Unexpected keys: {len(unexpected)}")

    # ===== ITGCA config mismatch detection =====
    has_itgca_in_checkpoint = any(
        k.startswith('fusion.fusion_layers.0.gate_') for k in filtered_state.keys()
    )
    model_uses_itgca = getattr(model, 'use_itgca', False)

    if has_itgca_in_checkpoint and not model_uses_itgca:
        print(f"  WARNING: Pretrained model was trained WITH ITGCA, "
              f"but fine-tuning does NOT use --use_itgca.")
        print(f"           ITGCA gate weights will be discarded. "
              f"Consider adding --use_itgca to match pretrained config.")
    elif not has_itgca_in_checkpoint and model_uses_itgca:
        print(f"  WARNING: Pretrained model was trained WITHOUT ITGCA, "
              f"but fine-tuning uses --use_itgca.")
        print(f"           ITGCA gate parameters will be randomly initialized!")

    if len(other_missing) == 0 and len(unexpected) == 0:
        # Count loaded parameters by module
        from collections import defaultdict
        loaded_by_module = defaultdict(int)
        for k in filtered_state.keys():
            module = k.split('.')[0]
            loaded_by_module[module] += 1

        print(f"  All encoder/fusion parameters loaded successfully!")
        print(f"    Loaded modules:")
        for module in ['embedding_raw', 'encoder_raw', 'embedding_size', 'encoder_size', 'fusion']:
            if module in loaded_by_module:
                print(f"      - {module}: {loaded_by_module[module]} params")

        # Verify temporal_embedding was loaded
        if 'embedding_size.temporal_embedding.weight' in filtered_state:
            print(f"    temporal_embedding loaded (IAT support enabled)")
        if has_itgca_in_checkpoint and model_uses_itgca:
            print(f"    ITGCA gate parameters loaded")

        print(f"  Classifier randomly initialized ({len(classifier_missing)} params)")
    else:
        print(f"  Load incomplete:")
        if other_missing:
            print(f"    Missing non-classifier keys ({len(other_missing)}): {other_missing[:10]}...")

            # CRITICAL: Check for temporal_embedding in Size encoder
            if 'embedding_size.temporal_embedding.weight' in other_missing:
                print(f"    CRITICAL: embedding_size.temporal_embedding.weight is missing!")
                print(f"    This indicates the pretrained model was trained WITHOUT IAT temporal information.")
                print(f"    The temporal_embedding layer will use random initialization,")
                print(f"    which will significantly hurt performance. Please use a pretrained model that includes IAT.")
                raise ValueError("Pretrained model missing temporal_embedding! Use a Stage 2 model trained with IAT.")

        if unexpected:
            print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")


def load_dataset(path):
    """Load dataset from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_class_weights(dataset, num_classes, method='sqrt'):
    """
    Compute class weights for imbalanced data

    Args:
        dataset: list of samples with 'label' key
        num_classes: number of classes
        method: weighting method
            - 'inverse': 1/count (original, may over-amplify minority classes)
            - 'sqrt': 1/sqrt(count) (recommended, balanced effect)
            - 'log': 1/log(count+1) (more gentle)
            - 'effective': based on effective number of samples (CVPR 2019)

    Returns:
        torch.Tensor of shape [num_classes] with class weights
    """
    label_counts = torch.zeros(num_classes)
    for sample in dataset:
        label_counts[sample['label']] += 1

    print(f"Class distribution: {label_counts.tolist()}")

    if method == 'inverse':
        # Original method: full inverse frequency
        total = len(dataset)
        weights = total / (num_classes * label_counts + 1e-6)

    elif method == 'sqrt':
        # Square root method: moderate weight differences
        # 400 vs 40 -> weight ratio changes from 10:1 to ~3.16:1
        weights = 1.0 / (torch.sqrt(label_counts) + 1e-6)

    elif method == 'log':
        # Logarithm method: more gentle adjustment
        weights = 1.0 / (torch.log(label_counts + 1) + 1e-6)

    elif method == 'effective':
        # Effective Number of Samples (CVPR 2019)
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.999  # Tunable param, closer to 1 = closer to inverse frequency
        effective_num = 1.0 - torch.pow(beta, label_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    # Clip extreme weights (optional: limit max weight ratio)
    max_weight = weights.max()
    min_weight = weights.min()
    if max_weight / min_weight > 10:
        print(f"  Warning: weight ratio {max_weight/min_weight:.1f}x, clipping to 10x")
        weights = torch.clamp(weights, min=max_weight / 10)
        weights = weights / weights.mean()  # Re-normalize

    print(f"Class weights ({method}): {[f'{w:.3f}' for w in weights.tolist()]}")

    return weights


def few_shot_sample(dataset, ratio, seed=42):
    """
    Stratified sampling: keep `ratio` fraction of training data per class.
    Guarantees at least 1 sample per class.
    """
    from collections import defaultdict
    rng = random.Random(seed)

    # Group by label
    class_samples = defaultdict(list)
    for sample in dataset:
        class_samples[sample['label']].append(sample)

    sampled = []
    for label, samples in class_samples.items():
        k = max(1, int(len(samples) * ratio))
        k = min(k, len(samples))
        sampled.extend(rng.sample(samples, k))

    rng.shuffle(sampled)

    # Print per-class stats
    original_total = len(dataset)
    sampled_total = len(sampled)
    print(f"  Few-shot sampling: ratio={ratio}, {original_total} -> {sampled_total} samples "
          f"({sampled_total/original_total*100:.1f}%)")
    print(f"  Classes: {len(class_samples)}, min samples/class: "
          f"{min(max(1, int(len(v)*ratio)) for v in class_samples.values())}")

    return sampled


def pre_tensorize(dataset, pin=False):
    """
    Pre-convert list-of-dicts dataset into contiguous tensors (one-time cost).
    Optionally pin memory for faster CPU->GPU transfer.
    """
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
    """Generate batches via index slicing on pre-built tensors"""
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


def train_epoch(args, model, optimizer, scheduler, train_data_tensors, criterion, ema, scaler=None, scl_criterion=None):
    """Train for one epoch (with optional AMP via scaler)"""
    model.train()
    total_loss = 0.0
    step = 0
    use_amp = scaler is not None

    for raw_src, packet_ids, directions, size_src, iat_src, tgt in batch_loader(args.batch_size, train_data_tensors, shuffle=True):
        raw_src = raw_src.to(args.device, non_blocking=True)
        packet_ids = packet_ids.to(args.device, non_blocking=True)
        directions = directions.to(args.device, non_blocking=True)
        size_src = size_src.to(args.device, non_blocking=True)
        iat_src = iat_src.to(args.device, non_blocking=True)
        tgt = tgt.to(args.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass (with optional AMP autocast)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            output = model(raw_src, packet_ids, directions, size_src, iat_src)
            if isinstance(output, tuple):
                logits, scl_feat = output
            else:
                logits, scl_feat = output, None
            loss = criterion(logits, tgt)
            # Handle DataParallel case
            if loss.dim() > 0:
                loss = torch.mean(loss)

        # Supervised Contrastive Loss (outside autocast — full FP32 forward+backward)
        if scl_feat is not None and scl_criterion is not None:
            loss = loss + args.scl_weight * scl_criterion(scl_feat, tgt)

        if use_amp:
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                # 只在无 inf/nan 时裁剪，避免 clip_grad_norm 中 inf*0=NaN
                found_inf = sum(v.item() for state in scaler._per_optimizer_states.values()
                                for v in state["found_inf_per_device"].values()) > 0
                if not found_inf:
                    if hasattr(model, 'module'):
                        torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.max_grad_norm > 0:
                if hasattr(model, 'module'):
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        scheduler.step()

        # EMA update
        ema.update()

        total_loss += loss.item()
        step += 1

        if step % args.report_steps == 0:
            print(f"  Step {step}, Avg loss: {total_loss / step:.4f}")

    return total_loss / step


def evaluate(args, model, eval_tensors, print_confusion=False):
    """Evaluate model on dataset"""
    model.eval()

    y_true, y_pred = [], []
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    with torch.no_grad():
        for raw_src, packet_ids, directions, size_src, iat_src, tgt in batch_loader(args.batch_size, eval_tensors):
            raw_src = raw_src.to(args.device, non_blocking=True)
            packet_ids = packet_ids.to(args.device, non_blocking=True)
            directions = directions.to(args.device, non_blocking=True)
            size_src = size_src.to(args.device, non_blocking=True)
            iat_src = iat_src.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)

            output = model(raw_src, packet_ids, directions, size_src, iat_src)
            logits = output[0] if isinstance(output, tuple) else output
            pred = torch.argmax(logits, dim=-1)

            for p, g in zip(pred.cpu().tolist(), tgt.cpu().tolist()):
                confusion[g, p] += 1  # [ground_truth, predicted] - standard convention
                y_pred.append(p)
                y_true.append(g)

    # Compute metrics
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
    print(f"Recall - Macro: {macro_r:.4f}, Micro: {micro_r:.4f}, Weighted: {weighted_r:.4f}")
    print(f"F1 - Macro: {macro_f1:.4f}, Micro: {micro_f1:.4f}, Weighted: {weighted_f1:.4f}")

    if print_confusion:
        print("\nConfusion Matrix:")
        print(confusion)

        print("\nPer-class metrics:")
        eps = 1e-9
        for i in range(args.labels_num):
            # confusion[g, p]: row=ground_truth, col=predicted
            # Precision = TP / (TP + FP) = correct among predicted as i
            # Recall = TP / (TP + FN) = correct among actual i
            p = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)  # col sum
            r = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)  # row sum
            f1 = 2 * p * r / (p + r + eps)
            print(f"  Label {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    return macro_f1, confusion


def build_optimizer(args, model):
    """Build optimizer and scheduler with optional Layer-wise LR Decay"""
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    # Get the actual model (handle DataParallel wrapper)
    actual_model = model.module if hasattr(model, 'module') else model

    if args.use_llrd:
        # Layer-wise Learning Rate Decay: encoder < fusion < classifier
        encoder_lr = args.learning_rate * args.llrd_encoder_ratio
        fusion_lr = args.learning_rate * args.llrd_fusion_ratio
        classifier_lr = args.learning_rate

        optimizer_grouped_parameters = []

        # Encoder parameters (lowest LR)
        encoder_params = []
        encoder_params_no_decay = []
        for name, param in actual_model.named_parameters():
            if 'embedding_raw' in name or 'encoder_raw' in name or \
               'embedding_size' in name or 'encoder_size' in name:
                if any(nd in name for nd in no_decay):
                    encoder_params_no_decay.append(param)
                else:
                    encoder_params.append(param)

        optimizer_grouped_parameters.append({
            'params': encoder_params,
            'lr': encoder_lr,
            'weight_decay': 0.01
        })
        optimizer_grouped_parameters.append({
            'params': encoder_params_no_decay,
            'lr': encoder_lr,
            'weight_decay': 0.0
        })

        # Fusion parameters (middle LR), with calibration params separated
        calibration_names = ('stat_scale', 'stat_shift', 'local_stat_scale', 'local_stat_shift')
        fusion_params = []
        fusion_params_no_decay = []
        calibration_params = []
        for name, param in actual_model.named_parameters():
            if 'fusion' in name:
                if any(cn in name for cn in calibration_names):
                    calibration_params.append(param)
                elif any(nd in name for nd in no_decay):
                    fusion_params_no_decay.append(param)
                else:
                    fusion_params.append(param)

        optimizer_grouped_parameters.append({
            'params': fusion_params,
            'lr': fusion_lr,
            'weight_decay': 0.01
        })
        optimizer_grouped_parameters.append({
            'params': fusion_params_no_decay,
            'lr': fusion_lr,
            'weight_decay': 0.0
        })
        if calibration_params:
            calibration_lr = fusion_lr * 5
            optimizer_grouped_parameters.append({
                'params': calibration_params,
                'lr': calibration_lr,
                'weight_decay': 0.0
            })

        # Classifier parameters (highest LR)
        classifier_params = []
        classifier_params_no_decay = []
        for name, param in actual_model.named_parameters():
            if 'classifier' in name:
                if any(nd in name for nd in no_decay):
                    classifier_params_no_decay.append(param)
                else:
                    classifier_params.append(param)

        optimizer_grouped_parameters.append({
            'params': classifier_params,
            'lr': classifier_lr,
            'weight_decay': 0.01
        })
        optimizer_grouped_parameters.append({
            'params': classifier_params_no_decay,
            'lr': classifier_lr,
            'weight_decay': 0.0
        })

        print(f"  LLRD: encoder_lr={encoder_lr:.2e}, fusion_lr={fusion_lr:.2e}, classifier_lr={classifier_lr:.2e}")
        if calibration_params:
            print(f"  Calibration params ({len(calibration_params)}): lr={calibration_lr:.2e}, no weight_decay")

    else:
        # Standard optimizer: same LR for all parameters
        param_optimizer = list(model.named_parameters())
        calibration_names = ('stat_scale', 'stat_shift', 'local_stat_scale', 'local_stat_shift')

        calibration_params = []
        other_params_decay = []
        other_params_no_decay = []
        for n, p in param_optimizer:
            if any(cn in n for cn in calibration_names):
                calibration_params.append(p)
            elif any(nd in n for nd in no_decay):
                other_params_no_decay.append(p)
            else:
                other_params_decay.append(p)

        optimizer_grouped_parameters = [
            {'params': other_params_decay, 'weight_decay': 0.01},
            {'params': other_params_no_decay, 'weight_decay': 0.0},
        ]
        if calibration_params:
            calibration_lr = args.learning_rate * 5
            optimizer_grouped_parameters.append({
                'params': calibration_params,
                'lr': calibration_lr,
                'weight_decay': 0.0
            })
            print(f"  Calibration params ({len(calibration_params)}): lr={calibration_lr:.2e}, no weight_decay")

    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
        )
    else:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate,
            scale_parameter=False, relative_step=False
        )

    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps * args.warmup, args.train_steps
        )

    return optimizer, scheduler


def freeze_backbone(model):
    """Freeze all parameters except classifier head and fine-tuning modules"""
    actual_model = model.module if hasattr(model, 'module') else model
    trainable_prefixes = ('classifier', 'scl_projection', 'attn_pool')
    for name, param in actual_model.named_parameters():
        if not any(name.startswith(p) for p in trainable_prefixes):
            param.requires_grad = False
    trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in actual_model.parameters())
    print(f"  Backbone frozen: {trainable}/{total} params trainable")


def unfreeze_backbone(model):
    """Unfreeze all parameters"""
    actual_model = model.module if hasattr(model, 'module') else model
    for param in actual_model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    print(f"  All params unfrozen: {trainable} params trainable")


def build_phase1_optimizer(args, model):
    """Build optimizer for Phase 1: only classifier parameters"""
    actual_model = model.module if hasattr(model, 'module') else model
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    classifier_params = []
    classifier_params_no_decay = []
    for name, param in actual_model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in no_decay):
                classifier_params_no_decay.append(param)
            else:
                classifier_params.append(param)

    optimizer_grouped_parameters = [
        {'params': classifier_params, 'weight_decay': 0.01},
        {'params': classifier_params_no_decay, 'weight_decay': 0.0}
    ]

    optimizer = str2optimizer["adamw"](
        optimizer_grouped_parameters, lr=args.phase1_lr, correct_bias=False
    )

    train_steps = int(len(args._phase1_train_data) * args.phase1_epochs / args.batch_size) + 1
    scheduler = str2scheduler["cosine"](
        optimizer, int(train_steps * args.warmup), train_steps
    )

    print(f"  Phase 1 optimizer: AdamW, lr={args.phase1_lr:.1e}, steps={train_steps}")
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)

    # Path options
    parser.add_argument("--label2id_path", type=str, required=True,
                        help="Path to label2id mapping (pickle)")
    parser.add_argument("--vocab_path_raw", type=str, required=True,
                        help="Path to raw modality vocabulary")
    parser.add_argument("--vocab_path_size", type=str, required=True,
                        help="Path to size modality vocabulary")
    parser.add_argument("--vocab_path_temporal", type=str, required=True,
                        help="Path to temporal (IAT) modality vocabulary")

    # Training options
    parser.add_argument("--earlystop", type=int, default=5)

    # Few-shot
    parser.add_argument("--few_shot", type=float, default=None,
                        help="Few-shot ratio: fraction of training data to use (e.g. 0.1 = 10%%). "
                             "Stratified sampling preserves class distribution, min 1 sample/class.")

    # Model options
    parser.add_argument("--num_fusion_layers", type=int, default=6,
                        help="Number of fusion layers (should match pretrained model)")
    parser.add_argument("--use_itgca", action="store_true",
                        help="Enable ITGCA gate mechanism. "
                             "MUST match the pretrained model's configuration.")
    parser.add_argument("--itgca_window_size", type=int, default=16,
                        help="Sliding window size for ITGCA local entropy.")

    # ITGCA component-level ablation flags — MUST match the pretrained checkpoint config.
    # Defaults: all False = full ITGCA.
    parser.add_argument("--ablate_r_stat", action="store_true",
                        help="Disable r_stat prior. Must match the pretrained checkpoint.")
    parser.add_argument("--ablate_g_token", action="store_true",
                        help="Disable token-level gate. Must match the pretrained checkpoint.")
    parser.add_argument("--ablate_source_bias", action="store_true",
                        help="Disable source-side attention bias. Must match the pretrained checkpoint.")

    # Sequence lengths
    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

    # Fine-tuning options
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0.0 = no smoothing)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = no clipping)")
    parser.add_argument("--simple_classifier", action="store_true",
                        help="Use simple CLS-only linear classifier instead of CLS+mean MLP")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="Use class weights for imbalanced data")
    parser.add_argument("--class_weight_method", type=str, default="sqrt",
                        choices=["inverse", "sqrt", "log", "effective"],
                        help="Class weight method: inverse, sqrt (recommended), log, effective")

    # Layer-wise LR Decay
    parser.add_argument("--use_llrd", action="store_true",
                        help="Use Layer-wise Learning Rate Decay (encoder LR < fusion LR < classifier LR)")
    parser.add_argument("--llrd_encoder_ratio", type=float, default=0.1,
                        help="Encoder LR = base_LR * this ratio (default 0.1)")
    parser.add_argument("--llrd_fusion_ratio", type=float, default=0.5,
                        help="Fusion LR = base_LR * this ratio (default 0.5)")

    # Two-Phase Training
    parser.add_argument("--two_phase", action="store_true",
                        help="Enable two-phase training: Phase 1 freezes backbone, Phase 2 unfreezes all")
    parser.add_argument("--phase1_epochs", type=int, default=3,
                        help="Number of epochs for Phase 1 (classifier warmup)")
    parser.add_argument("--phase1_lr", type=float, default=1e-3,
                        help="Learning rate for Phase 1 (only classifier)")
    parser.add_argument("--phase2_scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"],
                        help="Scheduler for Phase 2 (default: cosine)")

    # Multi-Sample Dropout
    parser.add_argument("--num_dropouts", type=int, default=1,
                        help="Number of dropout samples for Multi-Sample Dropout (1=disabled, 5 recommended)")

    # Supervised Contrastive Learning
    parser.add_argument("--use_scl", action="store_true",
                        help="Enable Supervised Contrastive Learning loss")
    parser.add_argument("--scl_weight", type=float, default=0.1,
                        help="Weight for SupCon loss (typical range: 0.05-0.5)")
    parser.add_argument("--scl_temperature", type=float, default=0.1,
                        help="Temperature for SupCon loss")
    parser.add_argument("--scl_queue_size", type=int, default=1024,
                        help="Feature queue size for SupCon (0=no queue, 1024 recommended for small batch)")

    # Focal Loss
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Focal Loss instead of CrossEntropy (focuses on hard samples)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma: higher = stronger focus on hard samples (typical: 1.0-5.0)")

    # Attention Pooling
    parser.add_argument("--use_attn_pooling", action="store_true",
                        help="Use learnable attention pooling instead of mean pooling")

    # Stochastic Weight Averaging
    parser.add_argument("--use_swa", action="store_true",
                        help="Enable Stochastic Weight Averaging")
    parser.add_argument("--swa_start_epoch", type=int, default=-1,
                        help="Epoch to start SWA collection (-1 = auto: 50%% of epochs)")

    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072, help="Dim of expert,default is ffn.")
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    parser.add_argument("--moebert_route_method", choices=["gate-token", "gate-sentence", "hash-random", "hash-balance","proto"], default="hash-random",
                        help="moebert route method.")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str, help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.0, help="gate loss weight.")

    # GPU options
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int,
                        help="List of GPU ranks to use. E.g., --gpu_ranks 2 3 to use GPU 2 and 3.")

    args = parser.parse_args()

    # Load hyperparameters from config
    if args.config_path:
        args = load_hyperparam(args)

    # Set max_seq_length for embedding layers
    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)

    set_seed(args.seed)

    # Load vocabularies
    print("Loading vocabularies...")
    vocab_raw = Vocab()
    vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab()
    vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab()
    vocab_temporal.load(args.vocab_path_temporal)
    print(f"  Raw vocab size: {len(vocab_raw)}")
    print(f"  Size vocab size: {len(vocab_size)}")
    print(f"  Temporal vocab size: {len(vocab_temporal)}")

    # Load label mapping
    print("Loading label mapping...")
    label2id = load_dataset(args.label2id_path)
    args.labels_num = len(label2id)
    print(f"Number of labels: {args.labels_num}")

    # Load datasets
    print("Loading datasets...")
    train_data = load_dataset(args.train_path)
    dev_data = load_dataset(args.dev_path)
    test_data = load_dataset(args.test_path) if args.test_path else None

    # Few-shot sampling (only affects training set)
    if args.few_shot is not None:
        assert 0.0 < args.few_shot <= 1.0, f"--few_shot must be in (0, 1], got {args.few_shot}"
        train_data = few_shot_sample(train_data, args.few_shot, seed=args.seed)

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data) if test_data else 0}")

    # Pre-tensorize datasets (one-time cost, avoids per-batch list comprehension)
    use_pin = torch.cuda.is_available()
    print("Pre-tensorizing datasets...")
    train_tensors = pre_tensorize(train_data, pin=use_pin)
    dev_tensors = pre_tensorize(dev_data, pin=use_pin)
    test_tensors = pre_tensorize(test_data, pin=use_pin) if test_data else None
    print("  Done.")

    # Build model
    print("Building model...")
    model = Stage2Classifier(args, len(vocab_raw), len(vocab_size), len(vocab_temporal), args.labels_num)

    # Load pretrained model
    load_pretrained_model(model, args.pretrained_model_path)

    # Setup GPU device(s)
    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1 and ranks_num > 1:
        # Multi-GPU mode with DataParallel
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= torch.cuda.device_count(), "Specified GPUs exceed available GPUs."

        # Set the primary device to the first specified GPU
        primary_gpu = args.gpu_ranks[0]
        args.device = torch.device(f"cuda:{primary_gpu}")
        model = model.to(args.device)

        # Use DataParallel with specified GPUs
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ranks)
        print(f"Using DataParallel on GPUs: {args.gpu_ranks}")

    elif ranks_num == 1:
        # Single GPU mode with specified GPU
        assert torch.cuda.is_available(), "No available GPUs."
        gpu_id = args.gpu_ranks[0]
        assert gpu_id < torch.cuda.device_count(), f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs)."

        args.device = torch.device(f"cuda:{gpu_id}")
        model = model.to(args.device)
        print(f"Using single GPU: {gpu_id}")

    elif torch.cuda.is_available():
        # Default: use cuda:0
        args.device = torch.device("cuda:0")
        model = model.to(args.device)
        print(f"Using default GPU: cuda:0")

    else:
        # CPU mode
        args.device = torch.device("cpu")
        model = model.to(args.device)
        print("Using CPU mode")

    # Build loss criterion
    print("\nSetting up loss function...")
    class_weights = None
    if args.use_class_weight:
        class_weights = compute_class_weights(train_data, args.labels_num, method=args.class_weight_method)
        class_weights = class_weights.to(args.device)

    if getattr(args, 'use_focal_loss', False):
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            weight=class_weights,
            label_smoothing=args.label_smoothing
        )
        print(f"  Loss: FocalLoss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing
        )
        print(f"  Loss: CrossEntropy")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Class weights: {'Enabled (' + args.class_weight_method + ')' if args.use_class_weight else 'Disabled'}")
    print(f"  Gradient clipping: {args.max_grad_norm if args.max_grad_norm > 0 else 'Disabled'}")

    # AMP (mixed precision): off by default, enable with --fp16
    use_amp = getattr(args, 'fp16', False) and args.device.type == 'cuda'
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp) if use_amp else None
    print(f"  AMP: {'Enabled' if use_amp else 'Disabled'}")

    # SCL setup
    scl_criterion = None
    if getattr(args, 'use_scl', False):
        scl_criterion = SupConLoss(temperature=args.scl_temperature,
                                   queue_size=args.scl_queue_size)
        print(f"  SupCon: weight={args.scl_weight}, temp={args.scl_temperature}, queue={args.scl_queue_size}")

    # SWA setup
    swa = None
    swa_start = 0
    if getattr(args, 'use_swa', False):
        swa = SWA(model)
        swa_start = args.swa_start_epoch if args.swa_start_epoch >= 0 else max(1, args.epochs_num // 2)
        print(f"  SWA: start collecting from epoch {swa_start}")

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    if args.two_phase:
        # ========== TWO-PHASE TRAINING ==========
        print("\n" + "=" * 50)
        print("Two-Phase Training Mode")
        print("=" * 50)

        # ===== Phase 1: Classifier Warmup (backbone frozen) =====
        print(f"\n--- Phase 1: Classifier Warmup ({args.phase1_epochs} epochs) ---")
        freeze_backbone(model)

        # Store train_data reference for build_phase1_optimizer
        args._phase1_train_data = train_data
        p1_optimizer, p1_scheduler = build_phase1_optimizer(args, model)
        del args._phase1_train_data

        # No EMA in Phase 1 (weights changing too fast)
        ema_dummy = EMA(model, decay=0.999)  # placeholder, never initialized

        for epoch in range(1, args.phase1_epochs + 1):
            print(f"\n[Phase 1] Epoch {epoch}/{args.phase1_epochs}")
            print("-" * 30)

            avg_loss = train_epoch(args, model, p1_optimizer, p1_scheduler, train_tensors, criterion, ema_dummy, scaler, scl_criterion=None)
            print(f"Training loss: {avg_loss:.4f}")

            print("Validation:")
            f1, _ = evaluate(args, model, dev_tensors)

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                save_model(model, args.output_model_path)
                print(f"New best model saved! F1: {best_f1:.4f}")

        print(f"\nPhase 1 complete. Best F1: {best_f1:.4f}")

        # ===== Phase 2: Full Fine-Tuning (all params unfrozen) =====
        print(f"\n--- Phase 2: Full Fine-Tuning ({args.epochs_num} epochs) ---")
        unfreeze_backbone(model)

        # Build Phase 2 optimizer with LLRD
        args.train_steps = int(len(train_data) * args.epochs_num / args.batch_size) + 1
        # Override scheduler for Phase 2
        original_scheduler = args.scheduler
        args.scheduler = args.phase2_scheduler
        optimizer, scheduler = build_optimizer(args, model)
        args.scheduler = original_scheduler

        if args.use_llrd:
            print(f"  LLRD: encoder={args.llrd_encoder_ratio}x, fusion={args.llrd_fusion_ratio}x, classifier=1x")

        # Setup EMA for Phase 2 (initialized from Phase 1 trained weights)
        ema = EMA(model, decay=0.999)
        ema.init_shadow()
        print(f"  EMA: Initialized from Phase 1 weights")

        patience_counter = 0

        for epoch in range(1, args.epochs_num + 1):
            print(f"\n[Phase 2] Epoch {epoch}/{args.epochs_num}")
            print("-" * 30)

            avg_loss = train_epoch(args, model, optimizer, scheduler, train_tensors, criterion, ema, scaler, scl_criterion=scl_criterion)
            print(f"Training loss: {avg_loss:.4f}")

            # Evaluate with both original and EMA weights
            print("Validation:")
            print("  [Original weights]")
            f1_orig, _ = evaluate(args, model, dev_tensors)

            ema.apply_shadow()
            print("  [EMA weights]")
            f1_ema, _ = evaluate(args, model, dev_tensors)
            ema.restore()

            if f1_ema >= f1_orig:
                f1 = f1_ema
                use_ema = True
                print(f"  -> EMA wins ({f1_ema:.4f} >= {f1_orig:.4f})")
            else:
                f1 = f1_orig
                use_ema = False
                print(f"  -> Original wins ({f1_orig:.4f} > {f1_ema:.4f})")

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = args.phase1_epochs + epoch
                patience_counter = 0
                if use_ema:
                    ema.apply_shadow()
                    save_model(model, args.output_model_path)
                    ema.restore()
                else:
                    save_model(model, args.output_model_path)
                print(f"New best model saved! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{args.earlystop}")

            # SWA update
            if swa is not None and epoch >= swa_start:
                swa.update()

            if patience_counter >= args.earlystop:
                print(f"Early stopping at Phase 2 epoch {epoch}")
                break

    else:
        # ========== SINGLE-PHASE TRAINING (original behavior) ==========
        args.train_steps = int(len(train_data) * args.epochs_num / args.batch_size) + 1
        optimizer, scheduler = build_optimizer(args, model)

        if args.use_llrd:
            print(f"  LLRD: encoder={args.llrd_encoder_ratio}x, fusion={args.llrd_fusion_ratio}x, classifier=1x")

        # Setup EMA (initialized after first epoch)
        ema = EMA(model, decay=0.999)
        print(f"  EMA: Enabled (decay=0.999, init after epoch 1)")

        # Training loop
        print("\n" + "=" * 50)
        print("Starting training...")
        print("=" * 50)

        for epoch in range(1, args.epochs_num + 1):
            print(f"\nEpoch {epoch}/{args.epochs_num}")
            print("-" * 30)

            # Train
            avg_loss = train_epoch(args, model, optimizer, scheduler, train_tensors, criterion, ema, scaler, scl_criterion=scl_criterion)
            print(f"Training loss: {avg_loss:.4f}")

            # Initialize EMA after first epoch
            if epoch == 1:
                ema.init_shadow()
                print("  EMA initialized with epoch 1 weights")

            # Evaluate on dev set
            print("\nValidation:")
            print("  [Original weights]")
            f1_orig, _ = evaluate(args, model, dev_tensors)

            # EMA evaluation (only after initialization)
            use_ema_for_save = False
            if ema.initialized:
                ema.apply_shadow()
                print("  [EMA weights]")
                f1_ema, _ = evaluate(args, model, dev_tensors)
                ema.restore()

                if f1_ema >= f1_orig:
                    f1 = f1_ema
                    use_ema_for_save = True
                    print(f"  -> EMA wins ({f1_ema:.4f} >= {f1_orig:.4f})")
                else:
                    f1 = f1_orig
                    print(f"  -> Original wins ({f1_orig:.4f} > {f1_ema:.4f})")
            else:
                f1 = f1_orig

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                patience_counter = 0
                if use_ema_for_save:
                    ema.apply_shadow()
                    save_model(model, args.output_model_path)
                    ema.restore()
                else:
                    save_model(model, args.output_model_path)
                print(f"New best model saved! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{args.earlystop}")

            # SWA update
            if swa is not None and epoch >= swa_start:
                swa.update()

            if patience_counter >= args.earlystop:
                print(f"Early stopping at epoch {epoch}")
                break

    # SWA final evaluation
    if swa is not None and swa.n_averaged > 0:
        print(f"\nSWA Evaluation (averaged {swa.n_averaged} checkpoints)")
        swa.apply()
        print("Validation [SWA weights]:")
        f1_swa, _ = evaluate(args, model, dev_tensors)
        if f1_swa > best_f1:
            print(f"  SWA wins! F1: {f1_swa:.4f} > {best_f1:.4f}")
            best_f1 = f1_swa
            save_model(model, args.output_model_path)
        else:
            print(f"  SWA F1: {f1_swa:.4f} <= best {best_f1:.4f}, keeping original")
        swa.restore()

    # Final evaluation on test set
    if test_data:
        print("\n" + "=" * 50)
        print("Test Set Evaluation")
        print("=" * 50)

        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(args.output_model_path, map_location=args.device))
        else:
            model.load_state_dict(torch.load(args.output_model_path, map_location=args.device))

        evaluate(args, model, test_tensors, print_confusion=True)

    print("\nTraining complete!")
    print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
