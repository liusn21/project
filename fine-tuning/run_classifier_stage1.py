"""
Stage 1 Multi-Modal Classifier

Uses two separate pretrained encoders (Raw + Size) without fusion.
Classification is done by concatenating CLS tokens from both modalities.

Usage:
    python fine-tuning/run_classifier_stage1.py \
        --train_path datasets/processed/train.pkl \
        --dev_path datasets/processed/val.pkl \
        --test_path datasets/processed/test.pkl \
        --label2id_path datasets/processed/label2id.pkl \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --pretrained_raw_path models/raw_encoder.bin \
        --pretrained_size_path models/size_encoder.bin \
        --output_model_path models/classifier_stage1.bin \
        --config_path models/bert/base_config.json \
        --epochs_num 10 \
        --batch_size 32
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
from uer.encoders import str2encoder
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.optimizers import str2optimizer, str2scheduler
from uer.model_saver import save_model
from uer.utils.constants import PAD_ID


class Stage1Classifier(nn.Module):
    """
    Stage 1 Multi-Modal Classifier

    Architecture:
        - Raw encoder (pretrained): embedding + transformer
        - Size encoder (pretrained): embedding + transformer
        - Modality combination -> Classification head

    Supports three modality modes:
        - 'raw': Only use Raw modality
        - 'size': Only use Size modality
        - 'both': Concatenate Raw and Size (default)
    """

    def __init__(self, args, vocab_size_raw, vocab_size_size, labels_num):
        super(Stage1Classifier, self).__init__()

        self.hidden_size = args.hidden_size
        self.labels_num = labels_num
        self.modality = getattr(args, 'modality', 'both')  # raw, size, both

        # Raw modality encoder
        if self.modality in ['raw', 'both']:
            self.embedding_raw = RawPacketEmbedding(args, vocab_size_raw)
            self.encoder_raw = str2encoder[args.encoder](args)

        # Size modality encoder
        if self.modality in ['size', 'both']:
            self.embedding_size = PacketSizeEmbedding(args, vocab_size_size)
            self.encoder_size = str2encoder[args.encoder](args)

        # Classification head
        if self.modality == 'both':
            classifier_input_size = 2 * args.hidden_size
        else:
            classifier_input_size = args.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, args.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, labels_num)
        )

    def forward(self, raw_src, packet_ids, directions, size_src, tgt=None):
        """
        Args:
            raw_src: [batch, seq_len_raw] - Raw token IDs
            packet_ids: [batch, seq_len_raw] - Packet indices
            directions: [batch, seq_len_raw] - Direction indices
            size_src: [batch, seq_len_size] - Size token IDs
            tgt: [batch] - Labels (optional)

        Returns:
            loss, logits if tgt is provided, else None, logits
        """
        features = []

        # Raw encoder
        if self.modality in ['raw', 'both']:
            raw_emb = self.embedding_raw(raw_src, packet_ids, directions)
            raw_seg = (raw_src != PAD_ID).long()
            raw_output = self.encoder_raw(raw_emb, raw_seg)  # [batch, seq_len, hidden]
            raw_cls = raw_output[:, 0, :]  # [batch, hidden]
            features.append(raw_cls)

        # Size encoder
        if self.modality in ['size', 'both']:
            size_emb = self.embedding_size(size_src)
            size_seg = (size_src != PAD_ID).long()
            size_output = self.encoder_size(size_emb, size_seg)  # [batch, seq_len, hidden]
            size_cls = size_output[:, 0, :]  # [batch, hidden]
            features.append(size_cls)

        # Combine features
        if len(features) == 2:
            combined = torch.cat(features, dim=-1)  # [batch, 2*hidden]
        else:
            combined = features[0]  # [batch, hidden]

        # Classification
        logits = self.classifier(combined)  # [batch, labels_num]

        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt)
            return loss.unsqueeze(0), logits  # 保持维度，避免 DataParallel 警告
        else:
            return None, logits


def load_pretrained_encoder(model, pretrained_path, encoder_type='raw'):
    """
    Load pretrained encoder weights

    Args:
        model: Stage1Classifier model
        pretrained_path: Path to pretrained checkpoint
        encoder_type: 'raw' or 'size'
    """
    if pretrained_path is None:
        return

    print(f"Loading pretrained {encoder_type} encoder from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')

    # Extract embedding and encoder weights
    prefix_emb = 'embedding.'
    prefix_enc = 'encoder.'
    embedding_state = {k[len(prefix_emb):]: v
                       for k, v in state_dict.items() if k.startswith(prefix_emb)}
    encoder_state = {k[len(prefix_enc):]: v
                     for k, v in state_dict.items() if k.startswith(prefix_enc)}

    if encoder_type == 'raw':
        emb_result = model.embedding_raw.load_state_dict(embedding_state, strict=False)
        enc_result = model.encoder_raw.load_state_dict(encoder_state, strict=False)
    else:
        emb_result = model.embedding_size.load_state_dict(embedding_state, strict=False)
        enc_result = model.encoder_size.load_state_dict(encoder_state, strict=False)

    # 打印详细的加载信息
    print(f"  Checkpoint keys: {len(state_dict)}")
    print(f"  Embedding params to load: {len(embedding_state)}")
    print(f"  Encoder params to load: {len(encoder_state)}")

    # 检查是否全部加载成功
    emb_success = len(emb_result.missing_keys) == 0 and len(emb_result.unexpected_keys) == 0
    enc_success = len(enc_result.missing_keys) == 0 and len(enc_result.unexpected_keys) == 0

    if emb_success and enc_success:
        print(f"  ✓ All parameters loaded successfully!")
    else:
        print(f"  ✗ Load incomplete:")
        if emb_result.missing_keys:
            print(f"    Missing embedding keys ({len(emb_result.missing_keys)}): {emb_result.missing_keys}")
        if emb_result.unexpected_keys:
            print(f"    Unexpected embedding keys ({len(emb_result.unexpected_keys)}): {emb_result.unexpected_keys}")
        if enc_result.missing_keys:
            print(f"    Missing encoder keys ({len(enc_result.missing_keys)}): {enc_result.missing_keys[:5]}...")
        if enc_result.unexpected_keys:
            print(f"    Unexpected encoder keys ({len(enc_result.unexpected_keys)}): {enc_result.unexpected_keys[:5]}...")


def load_dataset(path):
    """Load dataset from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)


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
        tgt = torch.LongTensor([s['label'] for s in batch])

        yield raw_src, packet_ids, directions, size_src, tgt


def train_epoch(args, model, optimizer, scheduler, train_data):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    step = 0

    for raw_src, packet_ids, directions, size_src, tgt in batch_loader(args.batch_size, train_data, shuffle=True):
        raw_src = raw_src.to(args.device)
        packet_ids = packet_ids.to(args.device)
        directions = directions.to(args.device)
        size_src = size_src.to(args.device)
        tgt = tgt.to(args.device)

        model.zero_grad()
        loss, _ = model(raw_src, packet_ids, directions, size_src, tgt)

        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)

        loss.backward()
        optimizer.step()
        scheduler.step()

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
        for raw_src, packet_ids, directions, size_src, tgt in batch_loader(args.batch_size, eval_data):
            raw_src = raw_src.to(args.device)
            packet_ids = packet_ids.to(args.device)
            directions = directions.to(args.device)
            size_src = size_src.to(args.device)
            tgt = tgt.to(args.device)

            _, logits = model(raw_src, packet_ids, directions, size_src, None)
            pred = torch.argmax(logits, dim=-1)

            for p, g in zip(pred.cpu().tolist(), tgt.cpu().tolist()):
                confusion[g, p] += 1  # [ground_truth, predicted] - 标准惯例
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
            # confusion[g, p]: 行=ground_truth, 列=predicted
            # Precision = TP / (TP + FP) = 预测为i的中有多少是对的
            # Recall = TP / (TP + FN) = 真实为i的中有多少被预测对了
            p = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)  # 列和
            r = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)  # 行和
            f1 = 2 * p * r / (p + r + eps)
            print(f"  Label {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    return macro_f1, confusion


def build_optimizer(args, model):
    """Build optimizer and scheduler"""
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
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

    # Path options
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training data (pickle)")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path to validation data (pickle)")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Path to test data (pickle)")
    parser.add_argument("--label2id_path", type=str, required=True,
                        help="Path to label2id mapping (pickle)")
    parser.add_argument("--vocab_path_raw", type=str, required=True,
                        help="Path to raw modality vocabulary")
    parser.add_argument("--vocab_path_size", type=str, required=True,
                        help="Path to size modality vocabulary")
    parser.add_argument("--pretrained_raw_path", type=str, default=None,
                        help="Path to pretrained raw encoder checkpoint")
    parser.add_argument("--pretrained_size_path", type=str, default=None,
                        help="Path to pretrained size encoder checkpoint")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path to save the best model")
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json",
                        help="Path to model config")

    # Training options
    parser.add_argument("--epochs_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_steps", type=int, default=100)
    parser.add_argument("--earlystop", type=int, default=5)

    # Model options
    parser.add_argument("--encoder", type=str, default="transformer")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--modality", choices=["raw", "size", "both"], default="both",
                        help="Modality mode: 'raw' (only Raw), 'size' (only Size), 'both' (concat both)")

    # Sequence lengths
    parser.add_argument("--seq_length_raw", type=int, default=512)
    parser.add_argument("--seq_length_size", type=int, default=256)

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
    print(f"  Modality mode: {args.modality}")
    model = Stage1Classifier(args, len(vocab_raw), len(vocab_size), args.labels_num)

    # Load pretrained encoders based on modality
    if args.modality in ['raw', 'both']:
        load_pretrained_encoder(model, args.pretrained_raw_path, 'raw')
    if args.modality in ['size', 'both']:
        load_pretrained_encoder(model, args.pretrained_size_path, 'size')

    # Move to device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    model = model.to(args.device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Build optimizer
    args.train_steps = int(len(train_data) * args.epochs_num / args.batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)

    # Training loop
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0  # 连续没有改进的 epoch 数

    for epoch in range(1, args.epochs_num + 1):
        print(f"\nEpoch {epoch}/{args.epochs_num}")
        print("-" * 30)

        # Train
        avg_loss = train_epoch(args, model, optimizer, scheduler, train_data)
        print(f"Training loss: {avg_loss:.4f}")

        # Evaluate on dev set
        print("\nValidation:")
        f1, _ = evaluate(args, model, dev_data)

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            patience_counter = 0
            save_model(model, args.output_model_path)
            print(f"New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.earlystop}")

        # Early stopping
        if patience_counter >= args.earlystop:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.earlystop} epochs)")
            break

    # Final evaluation on test set
    if test_data:
        print("\n" + "=" * 50)
        print("Test Set Evaluation")
        print("=" * 50)

        # Load best model
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))

        evaluate(args, model, test_data, print_confusion=True)

    print("\nTraining complete!")
    print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
