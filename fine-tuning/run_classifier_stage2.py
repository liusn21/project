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
        --use_fusion_gate  # Add this if pretrained model was trained with gate
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

        # Raw modality encoder
        self.embedding_raw = RawPacketEmbedding(args, vocab_size_raw)
        self.encoder_raw = str2encoder[args.encoder](args)

        # Size modality encoder (with temporal/IAT support)
        self.embedding_size = PacketSizeEmbedding(args, vocab_size_size, vocab_size_temporal)
        self.encoder_size = str2encoder[args.encoder](args)

        # Fusion module
        num_fusion_layers = getattr(args, 'num_fusion_layers', 6)
        use_fusion_gate = getattr(args, 'use_fusion_gate', False)
        self.fusion = MultiModalFusionEncoder(args, num_layers=num_fusion_layers, use_gate=use_fusion_gate)

        # Classification head
        self.simple_classifier = getattr(args, 'simple_classifier', False)
        if self.simple_classifier:
            # Simple: CLS-only, single linear layer (less overfitting)
            self.classifier = nn.Linear(2 * args.hidden_size, labels_num)
        else:
            # Full: CLS + mean pooling, two-layer MLP
            self.classifier = nn.Sequential(
                nn.Linear(4 * args.hidden_size, args.hidden_size),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, labels_num)
            )

    def forward(self, raw_src, packet_ids, directions, size_src, iat_src):
        """
        Args:
            raw_src: [batch, seq_len_raw] - Raw token IDs
            packet_ids: [batch, seq_len_raw] - Packet indices
            directions: [batch, seq_len_raw] - Direction indices
            size_src: [batch, seq_len_size] - Size token IDs
            iat_src: [batch, seq_len_size] - IAT temporal token IDs

        Returns:
            logits: [batch, labels_num]
        """
        # Raw encoder
        raw_emb = self.embedding_raw(raw_src, packet_ids, directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)  # [batch, seq_len, hidden]

        # Size encoder (with IAT temporal information)
        size_emb = self.embedding_size(size_src, iat_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)  # [batch, seq_len, hidden]

        # Fusion
        raw_fused, size_fused = self.fusion(raw_output, size_output, raw_seg, size_seg)

        # Extract fused CLS tokens
        raw_cls = raw_fused[:, 0, :]  # [batch, hidden]
        size_cls = size_fused[:, 0, :]  # [batch, hidden]

        if self.simple_classifier:
            combined = torch.cat([raw_cls, size_cls], dim=-1)  # [batch, 2*hidden]
        else:
            # Mean pooling over non-CLS, non-PAD positions
            raw_mask = raw_seg[:, 1:].unsqueeze(-1).float()  # [batch, seq_len-1, 1]
            raw_mean = (raw_fused[:, 1:, :] * raw_mask).sum(1) / (raw_mask.sum(1) + 1e-9)
            size_mask = size_seg[:, 1:].unsqueeze(-1).float()
            size_mean = (size_fused[:, 1:, :] * size_mask).sum(1) / (size_mask.sum(1) + 1e-9)
            combined = torch.cat([raw_cls, size_cls, raw_mean, size_mean], dim=-1)  # [batch, 4*hidden]

        logits = self.classifier(combined)

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

    # Only classifier weights should be missing (they are randomly initialized)
    classifier_missing = [k for k in missing if k.startswith('classifier')]
    other_missing = [k for k in missing if not k.startswith('classifier')]

    # Count excluded parameters by category
    excluded_count = len(state_dict) - len(filtered_state)

    print(f"  Checkpoint total keys: {len(state_dict)}")
    print(f"  Excluded keys (momentum/ITC/target/queues): {excluded_count}")
    print(f"  Filtered keys (should be loaded): {len(filtered_state)}")
    print(f"  Missing keys: {len(missing)} (classifier: {len(classifier_missing)}, other: {len(other_missing)})")
    print(f"  Unexpected keys: {len(unexpected)}")

    if len(other_missing) == 0 and len(unexpected) == 0:
        # Count loaded parameters by module
        from collections import defaultdict
        loaded_by_module = defaultdict(int)
        for k in filtered_state.keys():
            module = k.split('.')[0]
            loaded_by_module[module] += 1

        print(f"  ✓ All encoder/fusion parameters loaded successfully!")
        print(f"    Loaded modules:")
        for module in ['embedding_raw', 'encoder_raw', 'embedding_size', 'encoder_size', 'fusion']:
            if module in loaded_by_module:
                print(f"      - {module}: {loaded_by_module[module]} params")

        # Verify temporal_embedding was loaded
        if 'embedding_size.temporal_embedding.weight' in filtered_state:
            print(f"    ✓ temporal_embedding loaded (IAT support enabled)")

        print(f"  Classifier randomly initialized ({len(classifier_missing)} params)")
    else:
        print(f"  ✗ Load incomplete:")
        if other_missing:
            print(f"    Missing non-classifier keys ({len(other_missing)}): {other_missing[:10]}...")

            # CRITICAL: Check for temporal_embedding in Size encoder
            if 'embedding_size.temporal_embedding.weight' in other_missing:
                print(f"    ⚠️  CRITICAL: embedding_size.temporal_embedding.weight is missing!")
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


def batch_loader(batch_size, dataset, shuffle=False):
    """Generate batches from dataset"""
    if shuffle:
        random.shuffle(dataset)

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = dataset[i * batch_size: (i + 1) * batch_size]

        raw_src = torch.LongTensor([s['raw_src'] for s in batch])
        packet_ids = torch.LongTensor([s['packet_ids'] for s in batch])
        directions = torch.LongTensor([s['directions'] for s in batch])
        size_src = torch.LongTensor([s['size_src'] for s in batch])
        iat_src = torch.LongTensor([s['iat_src'] for s in batch])
        tgt = torch.LongTensor([s['label'] for s in batch])

        yield raw_src, packet_ids, directions, size_src, iat_src, tgt


def train_epoch(args, model, optimizer, scheduler, train_data, criterion, ema):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    step = 0

    for raw_src, packet_ids, directions, size_src, iat_src, tgt in batch_loader(args.batch_size, train_data, shuffle=True):
        raw_src = raw_src.to(args.device)
        packet_ids = packet_ids.to(args.device)
        directions = directions.to(args.device)
        size_src = size_src.to(args.device)
        iat_src = iat_src.to(args.device)
        tgt = tgt.to(args.device)

        model.zero_grad()

        # Get logits from model
        logits = model(raw_src, packet_ids, directions, size_src, iat_src)

        # Compute loss
        loss = criterion(logits, tgt)

        # Handle DataParallel case
        if loss.dim() > 0:
            loss = torch.mean(loss)

        loss.backward()

        # Gradient clipping
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


def evaluate(args, model, eval_data, print_confusion=False):
    """Evaluate model on dataset"""
    model.eval()

    y_true, y_pred = [], []
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    with torch.no_grad():
        for raw_src, packet_ids, directions, size_src, iat_src, tgt in batch_loader(args.batch_size, eval_data):
            raw_src = raw_src.to(args.device)
            packet_ids = packet_ids.to(args.device)
            directions = directions.to(args.device)
            size_src = size_src.to(args.device)
            iat_src = iat_src.to(args.device)
            tgt = tgt.to(args.device)

            logits = model(raw_src, packet_ids, directions, size_src, iat_src)
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

        # Fusion parameters (middle LR)
        fusion_params = []
        fusion_params_no_decay = []
        for name, param in actual_model.named_parameters():
            if 'fusion' in name:
                if any(nd in name for nd in no_decay):
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

    else:
        # Standard optimizer: same LR for all parameters
        param_optimizer = list(model.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

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
    """Freeze all parameters except classifier head"""
    actual_model = model.module if hasattr(model, 'module') else model
    for name, param in actual_model.named_parameters():
        if not name.startswith('classifier'):
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

    # Model options
    parser.add_argument("--num_fusion_layers", type=int, default=6,
                        help="Number of fusion layers (should match pretrained model)")
    parser.add_argument("--use_fusion_gate", action="store_true",
                        help="Enable learnable gate mechanism in fusion layers. "
                             "MUST match the pretrained model's configuration! "
                             "If pretrained model was trained with --use_fusion_gate, "
                             "this flag MUST be set during fine-tuning to load gate weights.")

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

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data) if test_data else 0}")

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

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )

    print(f"  Loss: CrossEntropy")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Class weights: {'Enabled (' + args.class_weight_method + ')' if args.use_class_weight else 'Disabled'}")
    print(f"  Gradient clipping: {args.max_grad_norm if args.max_grad_norm > 0 else 'Disabled'}")

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

            avg_loss = train_epoch(args, model, p1_optimizer, p1_scheduler, train_data, criterion, ema_dummy)
            print(f"Training loss: {avg_loss:.4f}")

            print("Validation:")
            f1, _ = evaluate(args, model, dev_data)

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

            avg_loss = train_epoch(args, model, optimizer, scheduler, train_data, criterion, ema)
            print(f"Training loss: {avg_loss:.4f}")

            # Evaluate with both original and EMA weights
            print("Validation:")
            print("  [Original weights]")
            f1_orig, _ = evaluate(args, model, dev_data)

            ema.apply_shadow()
            print("  [EMA weights]")
            f1_ema, _ = evaluate(args, model, dev_data)
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
            avg_loss = train_epoch(args, model, optimizer, scheduler, train_data, criterion, ema)
            print(f"Training loss: {avg_loss:.4f}")

            # Initialize EMA after first epoch
            if epoch == 1:
                ema.init_shadow()
                print("  EMA initialized with epoch 1 weights")

            # Evaluate on dev set
            print("\nValidation:")
            print("  [Original weights]")
            f1_orig, _ = evaluate(args, model, dev_data)

            # EMA evaluation (only after initialization)
            use_ema_for_save = False
            if ema.initialized:
                ema.apply_shadow()
                print("  [EMA weights]")
                f1_ema, _ = evaluate(args, model, dev_data)
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

            if patience_counter >= args.earlystop:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    if test_data:
        print("\n" + "=" * 50)
        print("Test Set Evaluation")
        print("=" * 50)

        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(args.output_model_path, map_location=args.device))
        else:
            model.load_state_dict(torch.load(args.output_model_path, map_location=args.device))

        evaluate(args, model, test_data, print_confusion=True)

    print("\nTraining complete!")
    print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
