"""
Flow Prediction Training Script

Train a model to predict flow_bytes and flow_duration from first 8 packets.

Architecture:
    - WindowFeatureExtractor: Pretrained encoders + fusion (frozen or fine-tuned)
    - FlowPredictionHead: MLP-based regression (trained from scratch)

Usage:
    python fine-tuning/run_flow_predictor.py \
        --train_path datasets/flow_processed/train.pkl \
        --dev_path datasets/flow_processed/val.pkl \
        --test_path datasets/flow_processed/test.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --pretrained_model_path models/multimodal_stage2.bin \
        --output_model_path models/flow_predictor.bin \
        --config_path models/bert/base_config.json \
        --epochs_num 50 \
        --batch_size 64
"""

import os
import sys
sys.path.append(os.getcwd())

import random
import argparse
import torch
import torch.nn as nn
import pickle
import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats

from flow_model import (
    FlowPredictionModel, FlowPredictionLoss,
    load_pretrained_for_flow, freeze_encoder, count_parameters
)

from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.utils import *
from uer.opts import finetune_opts


class EMA:
    """Exponential Moving Average for smoother model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.initialized = False

    def init_shadow(self):
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
        if not self.initialized:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        if not self.initialized:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def load_dataset(path):
    """Load dataset from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_batch(samples, device, seq_length_raw, seq_length_size):
    """
    Prepare a batch of samples for model input.

    Args:
        samples: list of sample dicts
        device: torch device
        seq_length_raw: sequence length for raw modality
        seq_length_size: sequence length for size modality

    Returns:
        batch_dict: dict with input tensors
        target: [batch, 2] target tensor (flow_bytes_log, flow_duration_log)
    """
    batch_size = len(samples)

    raw_src = torch.zeros(batch_size, seq_length_raw, dtype=torch.long)
    packet_ids = torch.zeros(batch_size, seq_length_raw, dtype=torch.long)
    directions = torch.zeros(batch_size, seq_length_raw, dtype=torch.long)
    size_src = torch.zeros(batch_size, seq_length_size, dtype=torch.long)
    iat_src = torch.zeros(batch_size, seq_length_size, dtype=torch.long)
    target = torch.zeros(batch_size, 2, dtype=torch.float)

    for b_idx, sample in enumerate(samples):
        raw_len = min(len(sample['raw_src']), seq_length_raw)
        raw_src[b_idx, :raw_len] = torch.tensor(sample['raw_src'][:raw_len])
        packet_ids[b_idx, :raw_len] = torch.tensor(sample['packet_ids'][:raw_len])
        directions[b_idx, :raw_len] = torch.tensor(sample['directions'][:raw_len])

        size_len = min(len(sample['size_src']), seq_length_size)
        size_src[b_idx, :size_len] = torch.tensor(sample['size_src'][:size_len])
        iat_src[b_idx, :size_len] = torch.tensor(sample['iat_src'][:size_len])

        # Targets: log-transformed values
        target[b_idx, 0] = sample['flow_bytes_log']
        target[b_idx, 1] = sample['flow_duration_log']

    batch_dict = {
        'raw_src': raw_src.to(device),
        'packet_ids': packet_ids.to(device),
        'directions': directions.to(device),
        'size_src': size_src.to(device),
        'iat_src': iat_src.to(device)
    }
    target = target.to(device)

    return batch_dict, target


def batch_loader(batch_size, dataset, device, seq_length_raw, seq_length_size, shuffle=False):
    """Generate batches from dataset"""
    if shuffle:
        random.shuffle(dataset)

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_samples = dataset[i * batch_size: (i + 1) * batch_size]
        batch_dict, target = prepare_batch(
            batch_samples, device, seq_length_raw, seq_length_size
        )
        yield batch_dict, target


def train_epoch(args, model, optimizer, scheduler, train_data, criterion, ema):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = defaultdict(float)
    step = 0

    for batch_dict, target in batch_loader(
            args.batch_size, train_data, args.device,
            args.seq_length_raw, args.seq_length_size, shuffle=True):

        model.zero_grad()
        pred = model(batch_dict)
        loss, loss_dict = criterion(pred, target)

        if loss.dim() > 0:
            loss = torch.mean(loss)

        loss.backward()

        if args.max_grad_norm > 0:
            actual_model = model.module if hasattr(model, 'module') else model
            torch.nn.utils.clip_grad_norm_(actual_model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        ema.update()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] += v
        step += 1

        if step % args.report_steps == 0:
            avg_loss = total_loss / step
            loss_str = f"bytes: {loss_components['bytes']/step:.4f}, duration: {loss_components['duration']/step:.4f}"
            print(f"  Step {step}, Loss: {avg_loss:.4f} ({loss_str})")

    return total_loss / step, {k: v / step for k, v in loss_components.items()}


def evaluate(args, model, eval_data, criterion, prefix="Eval"):
    """Evaluate model on dataset"""
    model.eval()

    total_loss = 0.0
    loss_components = defaultdict(float)
    all_preds = []
    all_targets = []
    step = 0

    with torch.no_grad():
        for batch_dict, target in batch_loader(
                args.batch_size, eval_data, args.device,
                args.seq_length_raw, args.seq_length_size, shuffle=False):

            pred = model(batch_dict)
            loss, loss_dict = criterion(pred, target)

            if loss.dim() > 0:
                loss = torch.mean(loss)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] += v
            step += 1

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_flow_metrics(all_preds, all_targets)
    avg_loss = total_loss / step

    print(f"\n{prefix} Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Flow Bytes (log space):")
    print(f"    MSE: {metrics['bytes_mse']:.4f}, MAE: {metrics['bytes_mae']:.4f}, "
          f"RMSE: {metrics['bytes_rmse']:.4f}, R2: {metrics['bytes_r2']:.4f}")
    print(f"  Flow Bytes (original space):")
    print(f"    MAPE: {metrics['bytes_mape']:.2%}, MedianAPE: {metrics['bytes_median_ape']:.2%}, "
          f"Spearman: {metrics['bytes_spearman']:.4f}")
    print(f"  Flow Duration (log space):")
    print(f"    MSE: {metrics['duration_mse']:.4f}, MAE: {metrics['duration_mae']:.4f}, "
          f"RMSE: {metrics['duration_rmse']:.4f}, R2: {metrics['duration_r2']:.4f}")
    print(f"  Flow Duration (original space):")
    print(f"    MAPE: {metrics['duration_mape']:.2%}, MedianAPE: {metrics['duration_median_ape']:.2%}, "
          f"Spearman: {metrics['duration_spearman']:.4f}")

    return avg_loss, metrics


def compute_flow_metrics(pred, target):
    """
    Compute comprehensive evaluation metrics for flow prediction.

    Args:
        pred: [batch, 2] predicted values (log_bytes, log_duration)
        target: [batch, 2] target values (log_bytes, log_duration)

    Returns:
        dict with metrics for both flow_bytes and flow_duration
    """
    metrics = {}

    # === Flow Bytes Metrics ===
    pred_bytes_log = pred[:, 0]
    target_bytes_log = target[:, 0]

    # Log-space metrics
    metrics['bytes_mse'] = torch.mean((pred_bytes_log - target_bytes_log) ** 2).item()
    metrics['bytes_mae'] = torch.mean(torch.abs(pred_bytes_log - target_bytes_log)).item()
    metrics['bytes_rmse'] = torch.sqrt(torch.tensor(metrics['bytes_mse'])).item()

    # R2 score
    ss_res = torch.sum((target_bytes_log - pred_bytes_log) ** 2)
    ss_tot = torch.sum((target_bytes_log - torch.mean(target_bytes_log)) ** 2)
    metrics['bytes_r2'] = (1 - ss_res / (ss_tot + 1e-6)).item()

    # Original-space metrics (convert from log10)
    pred_bytes = 10 ** pred_bytes_log - 1
    target_bytes = 10 ** target_bytes_log - 1

    # MAPE (Mean Absolute Percentage Error)
    ape = torch.abs(pred_bytes - target_bytes) / (target_bytes + 1e-6)
    metrics['bytes_mape'] = torch.mean(ape).item()
    metrics['bytes_median_ape'] = torch.median(ape).item()

    # Spearman correlation
    pred_bytes_np = pred_bytes.numpy()
    target_bytes_np = target_bytes.numpy()
    spearman_corr, _ = scipy_stats.spearmanr(pred_bytes_np, target_bytes_np)
    metrics['bytes_spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0

    # === Flow Duration Metrics ===
    pred_duration_log = pred[:, 1]
    target_duration_log = target[:, 1]

    # Log-space metrics
    metrics['duration_mse'] = torch.mean((pred_duration_log - target_duration_log) ** 2).item()
    metrics['duration_mae'] = torch.mean(torch.abs(pred_duration_log - target_duration_log)).item()
    metrics['duration_rmse'] = torch.sqrt(torch.tensor(metrics['duration_mse'])).item()

    # R2 score
    ss_res = torch.sum((target_duration_log - pred_duration_log) ** 2)
    ss_tot = torch.sum((target_duration_log - torch.mean(target_duration_log)) ** 2)
    metrics['duration_r2'] = (1 - ss_res / (ss_tot + 1e-6)).item()

    # Original-space metrics (convert from log10)
    # Note: flow_duration_log = log10(duration + epsilon), so 10^log - epsilon
    epsilon = 1e-6
    pred_duration = 10 ** pred_duration_log - epsilon
    target_duration = 10 ** target_duration_log - epsilon

    # MAPE
    ape = torch.abs(pred_duration - target_duration) / (target_duration + 1e-6)
    metrics['duration_mape'] = torch.mean(ape).item()
    metrics['duration_median_ape'] = torch.median(ape).item()

    # Spearman correlation
    pred_duration_np = pred_duration.numpy()
    target_duration_np = target_duration.numpy()
    spearman_corr, _ = scipy_stats.spearmanr(pred_duration_np, target_duration_np)
    metrics['duration_spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0

    return metrics


def build_optimizer(args, model):
    """Build optimizer and scheduler with optional LLRD"""
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    actual_model = model.module if hasattr(model, 'module') else model

    if args.use_llrd:
        encoder_lr = args.learning_rate * args.llrd_encoder_ratio
        predictor_lr = args.learning_rate

        optimizer_grouped_parameters = []

        # Feature extractor (encoder + fusion)
        encoder_params = []
        encoder_params_no_decay = []
        for name, param in actual_model.feature_extractor.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in no_decay):
                    encoder_params_no_decay.append(param)
                else:
                    encoder_params.append(param)

        optimizer_grouped_parameters.append({
            'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 0.01
        })
        optimizer_grouped_parameters.append({
            'params': encoder_params_no_decay, 'lr': encoder_lr, 'weight_decay': 0.0
        })

        # Predictor (MLP head)
        predictor_params = []
        predictor_params_no_decay = []
        for name, param in actual_model.predictor.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in no_decay):
                    predictor_params_no_decay.append(param)
                else:
                    predictor_params.append(param)

        optimizer_grouped_parameters.append({
            'params': predictor_params, 'lr': predictor_lr, 'weight_decay': 0.01
        })
        optimizer_grouped_parameters.append({
            'params': predictor_params_no_decay, 'lr': predictor_lr, 'weight_decay': 0.0
        })

        print(f"  LLRD: encoder_lr={encoder_lr:.2e}, predictor_lr={predictor_lr:.2e}")
    else:
        param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)

    # Vocabulary paths
    parser.add_argument('--vocab_path_raw', type=str, required=True)
    parser.add_argument('--vocab_path_size', type=str, required=True)
    parser.add_argument('--vocab_path_temporal', type=str, required=True)

    # Sequence lengths
    parser.add_argument('--seq_length_raw', type=int, default=512,
                        help='Sequence length for Raw modality (8 packets)')
    parser.add_argument('--seq_length_size', type=int, default=10,
                        help='Sequence length for Size modality (8 packets + CLS + SEP)')

    # Model architecture
    parser.add_argument('--mlp_hidden', type=int, default=256,
                        help='Hidden dimension for MLP predictor')
    parser.add_argument('--num_fusion_layers', type=int, default=6)
    parser.add_argument('--use_itgca', action='store_true')

    # Encoder freezing
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='Number of epochs to freeze encoder (0 = no freezing)')

    # Loss weights
    parser.add_argument('--bytes_weight', type=float, default=1.0,
                        help='Weight for flow_bytes loss')
    parser.add_argument('--duration_weight', type=float, default=1.0,
                        help='Weight for flow_duration loss')
    parser.add_argument('--use_huber', action='store_true',
                        help='Use Huber loss instead of MSE')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss')

    # LLRD (Layer-wise Learning Rate Decay)
    parser.add_argument('--use_llrd', action='store_true')
    parser.add_argument('--llrd_encoder_ratio', type=float, default=0.1)

    # Training
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # GPU
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpu_ranks', default=[], nargs='+', type=int)

    args = parser.parse_args()

    if args.config_path:
        args = load_hyperparam(args)

    args.max_seq_length = max(args.seq_length_raw, args.seq_length_size)
    set_seed(args.seed)

    print("=" * 60)
    print("Flow Prediction Training")
    print("=" * 60)

    # Load vocabularies
    print("\nLoading vocabularies...")
    vocab_raw = Vocab()
    vocab_raw.load(args.vocab_path_raw)
    vocab_size = Vocab()
    vocab_size.load(args.vocab_path_size)
    vocab_temporal = Vocab()
    vocab_temporal.load(args.vocab_path_temporal)

    print(f"  Raw vocab size: {len(vocab_raw)}")
    print(f"  Size vocab size: {len(vocab_size)}")
    print(f"  Temporal vocab size: {len(vocab_temporal)}")

    # Load datasets
    print("\nLoading datasets...")
    train_data = load_dataset(args.train_path)
    dev_data = load_dataset(args.dev_path)
    test_data = load_dataset(args.test_path) if args.test_path else None

    print(f"  Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data) if test_data else 0}")

    # Print dataset statistics
    if len(train_data) > 0:
        bytes_log_values = [s['flow_bytes_log'] for s in train_data]
        duration_log_values = [s['flow_duration_log'] for s in train_data]
        print(f"\nTraining data statistics:")
        print(f"  flow_bytes_log: mean={np.mean(bytes_log_values):.4f}, "
              f"std={np.std(bytes_log_values):.4f}")
        print(f"  flow_duration_log: mean={np.mean(duration_log_values):.4f}, "
              f"std={np.std(duration_log_values):.4f}")

    # Build model
    print("\nBuilding model...")
    model = FlowPredictionModel(
        args,
        vocab_size_raw=len(vocab_raw),
        vocab_size_size=len(vocab_size),
        vocab_size_temporal=len(vocab_temporal),
        mlp_hidden=args.mlp_hidden,
        num_outputs=2,
        dropout=args.dropout
    )

    # Load pretrained weights
    if args.pretrained_model_path:
        load_pretrained_for_flow(model, args.pretrained_model_path)

    # Freeze encoder if specified
    if args.freeze_encoder or args.freeze_epochs > 0:
        freeze_encoder(model)

    # GPU setup
    ranks_num = len(args.gpu_ranks)
    if args.world_size > 1 and ranks_num > 1:
        assert torch.cuda.is_available()
        primary_gpu = args.gpu_ranks[0]
        args.device = torch.device(f"cuda:{primary_gpu}")
        model = model.to(args.device)
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ranks)
        print(f"Using DataParallel on GPUs: {args.gpu_ranks}")
    elif ranks_num == 1:
        assert torch.cuda.is_available()
        gpu_id = args.gpu_ranks[0]
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

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Build optimizer and scheduler
    args.train_steps = int(len(train_data) * args.epochs_num / args.batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)

    # Loss function
    criterion = FlowPredictionLoss(
        bytes_weight=args.bytes_weight,
        duration_weight=args.duration_weight,
        use_huber=args.use_huber,
        huber_delta=args.huber_delta
    )
    loss_type = "Huber" if args.use_huber else "MSE"
    print(f"  Loss function: {loss_type} (bytes_weight={args.bytes_weight}, duration_weight={args.duration_weight})")

    # EMA
    ema = EMA(model, decay=0.999)
    print(f"  EMA: Enabled (decay=0.999, init after epoch 1)")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs_num + 1):
        print(f"\n{'='*20} Epoch {epoch}/{args.epochs_num} {'='*20}")

        # Unfreeze encoder after freeze_epochs
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print("Unfreezing encoder...")
            actual_model = model.module if hasattr(model, 'module') else model
            for param in actual_model.feature_extractor.parameters():
                param.requires_grad = True

            optimizer, scheduler = build_optimizer(args, model)
            ema = EMA(model, decay=0.999)

            trainable_params = count_parameters(model, trainable_only=True)
            print(f"  Trainable parameters: {trainable_params:,}")

        # Train
        train_loss, _ = train_epoch(args, model, optimizer, scheduler, train_data, criterion, ema)
        print(f"\nTrain Loss: {train_loss:.4f}")

        # Initialize EMA after first epoch
        if epoch == 1:
            ema.init_shadow()
            print("  EMA initialized")

        # Evaluate with original weights
        print("\nValidation [Original weights]:")
        dev_loss_orig, metrics_orig = evaluate(args, model, dev_data, criterion, prefix="Dev")

        # Evaluate with EMA weights
        use_ema = False
        if ema.initialized:
            ema.apply_shadow()
            print("\nValidation [EMA weights]:")
            dev_loss_ema, metrics_ema = evaluate(args, model, dev_data, criterion, prefix="Dev")
            ema.restore()

            if dev_loss_ema <= dev_loss_orig:
                dev_loss = dev_loss_ema
                use_ema = True
                print(f"  -> EMA wins ({dev_loss_ema:.4f} <= {dev_loss_orig:.4f})")
            else:
                dev_loss = dev_loss_orig
                print(f"  -> Original wins ({dev_loss_orig:.4f} < {dev_loss_ema:.4f})")
        else:
            dev_loss = dev_loss_orig

        # Check for improvement
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            patience_counter = 0

            if use_ema:
                ema.apply_shadow()
                save_model(model, args.output_model_path)
                ema.restore()
            else:
                save_model(model, args.output_model_path)
            print(f"  New best model saved! Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.early_stopping}")

        if patience_counter >= args.early_stopping:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    print(f"\nLoading best model from epoch {best_epoch}...")
    actual_model = model.module if hasattr(model, 'module') else model
    actual_model.load_state_dict(torch.load(args.output_model_path, map_location=args.device))

    print("\nDev Set:")
    evaluate(args, model, dev_data, criterion, prefix="Dev")

    if test_data:
        print("\nTest Set:")
        evaluate(args, model, test_data, criterion, prefix="Test")

    print("\nTraining complete!")
    print(f"Best model saved to: {args.output_model_path}")


if __name__ == '__main__':
    main()
