"""
Flow Prediction Model

Predicts flow_bytes (total bytes) and flow_duration from first 8 packets.

Architecture:
    - WindowFeatureExtractor: Reuses pretrained encoders + fusion for the input window
    - FlowPredictionHead: MLP-based regression head

Usage:
    model = FlowPredictionModel(args, vocab_size_raw, vocab_size_size, vocab_size_temporal)
    pred = model(batch)  # [batch, 2] (log_flow_bytes, log_flow_duration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from uer.layers import RawPacketEmbedding, PacketSizeEmbedding
from uer.layers.multimodal_fusion import MultiModalFusionEncoder
from uer.encoders import str2encoder
from uer.utils.constants import PAD_ID


class WindowFeatureExtractor(nn.Module):
    """
    Extract fused features from a single window.

    Reuses pretrained components:
        - embedding_raw, encoder_raw: Raw packet encoder
        - embedding_size, encoder_size: Size+IAT encoder
        - fusion: 6-layer bidirectional cross-attention

    Output: Concatenated fused CLS tokens [batch, hidden_size * 2]

    This is identical to qos_model.py WindowFeatureExtractor.
    """

    def __init__(self, args, vocab_size_raw, vocab_size_size, vocab_size_temporal):
        super(WindowFeatureExtractor, self).__init__()

        self.hidden_size = args.hidden_size

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

    def forward(self, raw_src, packet_ids, directions, size_src, iat_src):
        """
        Extract fused features from a single window.

        Args:
            raw_src: [batch, seq_len_raw] - Raw token IDs
            packet_ids: [batch, seq_len_raw] - Packet indices
            directions: [batch, seq_len_raw] - Direction indices
            size_src: [batch, seq_len_size] - Size token IDs
            iat_src: [batch, seq_len_size] - IAT temporal token IDs

        Returns:
            fused_cls: [batch, hidden_size * 2] - Concatenated CLS tokens
        """
        # Raw encoder
        raw_emb = self.embedding_raw(raw_src, packet_ids, directions)
        raw_seg = (raw_src != PAD_ID).long()
        raw_output = self.encoder_raw(raw_emb, raw_seg)

        # Size encoder
        size_emb = self.embedding_size(size_src, iat_src)
        size_seg = (size_src != PAD_ID).long()
        size_output = self.encoder_size(size_emb, size_seg)

        # Fusion
        raw_fused, size_fused, _ = self.fusion(raw_output, size_output, raw_seg, size_seg)

        # Extract and concat CLS tokens
        raw_cls = raw_fused[:, 0, :]  # [batch, hidden]
        size_cls = size_fused[:, 0, :]  # [batch, hidden]

        fused_cls = torch.cat([raw_cls, size_cls], dim=-1)  # [batch, hidden * 2]
        return fused_cls


class FlowPredictionHead(nn.Module):
    """
    MLP head for flow prediction (regression).

    Takes fused CLS features and predicts flow_bytes_log and flow_duration_log.
    """

    def __init__(self, input_dim, hidden_dim=256, num_outputs=2, dropout=0.1):
        """
        Args:
            input_dim: dimension of input features (hidden_size * 2)
            hidden_dim: MLP hidden dimension
            num_outputs: number of outputs (2: flow_bytes_log, flow_duration_log)
            dropout: dropout rate
        """
        super(FlowPredictionHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_outputs)
        )

    def forward(self, fused_cls):
        """
        Predict flow metrics from fused features.

        Args:
            fused_cls: [batch, input_dim] - Fused CLS tokens

        Returns:
            pred: [batch, num_outputs] - Predictions (log_bytes, log_duration)
        """
        return self.mlp(fused_cls)


class FlowPredictionModel(nn.Module):
    """
    Complete Flow Prediction Model.

    Combines:
        - WindowFeatureExtractor: Extract fused features from first 8 packets
        - FlowPredictionHead: MLP-based regression prediction
    """

    def __init__(self, args, vocab_size_raw, vocab_size_size, vocab_size_temporal,
                 mlp_hidden=256, num_outputs=2, dropout=0.1):
        """
        Args:
            args: model configuration
            vocab_size_*: vocabulary sizes
            mlp_hidden: MLP hidden dimension
            num_outputs: number of prediction targets (2: bytes, duration)
            dropout: dropout rate
        """
        super(FlowPredictionModel, self).__init__()

        self.hidden_size = args.hidden_size

        # Window feature extractor (pretrained)
        self.feature_extractor = WindowFeatureExtractor(
            args, vocab_size_raw, vocab_size_size, vocab_size_temporal
        )

        # Flow prediction head (newly trained)
        self.predictor = FlowPredictionHead(
            input_dim=args.hidden_size * 2,
            hidden_dim=mlp_hidden,
            num_outputs=num_outputs,
            dropout=dropout
        )

    def forward(self, batch):
        """
        Forward pass for flow prediction.

        Args:
            batch: dict with keys:
                - raw_src: [batch, seq_len_raw]
                - packet_ids: [batch, seq_len_raw]
                - directions: [batch, seq_len_raw]
                - size_src: [batch, seq_len_size]
                - iat_src: [batch, seq_len_size]

        Returns:
            pred: [batch, 2] - Predicted (log_flow_bytes, log_flow_duration)
        """
        # Extract fused features
        fused_cls = self.feature_extractor(
            batch['raw_src'],
            batch['packet_ids'],
            batch['directions'],
            batch['size_src'],
            batch['iat_src']
        )

        # Predict
        pred = self.predictor(fused_cls)

        return pred

    def forward_with_features(self, batch):
        """
        Forward pass returning both predictions and intermediate features.

        Useful for analysis and visualization.
        """
        fused_cls = self.feature_extractor(
            batch['raw_src'],
            batch['packet_ids'],
            batch['directions'],
            batch['size_src'],
            batch['iat_src']
        )
        pred = self.predictor(fused_cls)

        return pred, fused_cls


class FlowPredictionLoss(nn.Module):
    """
    Loss function for flow prediction.

    Supports MSE or Huber loss for both metrics with configurable weights.
    """

    def __init__(self, bytes_weight=1.0, duration_weight=1.0, use_huber=False, huber_delta=1.0):
        """
        Args:
            bytes_weight: weight for flow_bytes_log loss
            duration_weight: weight for flow_duration_log loss
            use_huber: whether to use Huber loss instead of MSE
            huber_delta: delta parameter for Huber loss
        """
        super(FlowPredictionLoss, self).__init__()

        self.bytes_weight = bytes_weight
        self.duration_weight = duration_weight

        if use_huber:
            self.loss_fn = nn.HuberLoss(delta=huber_delta, reduction='mean')
        else:
            self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        """
        Compute flow prediction loss.

        Args:
            pred: [batch, 2] - Predicted (log_bytes, log_duration)
            target: [batch, 2] - Target (log_bytes, log_duration)

        Returns:
            total_loss: scalar tensor
            loss_dict: dict with individual losses
        """
        # Split predictions and targets
        pred_bytes = pred[:, 0]
        pred_duration = pred[:, 1]
        target_bytes = target[:, 0]
        target_duration = target[:, 1]

        # Compute individual losses
        loss_bytes = self.loss_fn(pred_bytes, target_bytes)
        loss_duration = self.loss_fn(pred_duration, target_duration)

        # Weighted total
        total_loss = self.bytes_weight * loss_bytes + self.duration_weight * loss_duration

        loss_dict = {
            'bytes': loss_bytes.item(),
            'duration': loss_duration.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


def load_pretrained_for_flow(model, pretrained_path):
    """
    Load pretrained Stage 2 multimodal model weights for flow prediction.

    Following the same logic as run_classifier_stage2.py:load_pretrained_model.

    Only loads encoder and fusion weights, excludes:
        - Momentum encoders (*_m)
        - ITC projections
        - Target layers
        - Queues

    Args:
        model: FlowPredictionModel
        pretrained_path: path to pretrained checkpoint
    """
    if pretrained_path is None:
        print("No pretrained model specified, using random initialization")
        return

    print(f"Loading pretrained model from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')

    # Filter out pretraining-specific components
    exclude_prefixes = [
        'embedding_raw_m', 'encoder_raw_m',
        'embedding_size_m', 'encoder_size_m',
        'itc_proj_raw_m', 'itc_proj_size_m',
        'target', 'raw_queue', 'size_queue', 'queue_ptr'
    ]

    # Map pretrained keys to feature_extractor keys
    filtered_state = {}
    for k, v in state_dict.items():
        if any(k.startswith(prefix) for prefix in exclude_prefixes):
            continue
        # Map to feature_extractor namespace
        new_key = f"feature_extractor.{k}"
        filtered_state[new_key] = v

    # Load into model
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    # Categorize missing keys
    predictor_missing = [k for k in missing if k.startswith('predictor')]
    encoder_missing = [k for k in missing if k.startswith('feature_extractor')]

    # Count excluded parameters
    excluded_count = len(state_dict) - len([k for k in state_dict.keys()
                                            if not any(k.startswith(p) for p in exclude_prefixes)])

    print(f"  Checkpoint total keys: {len(state_dict)}")
    print(f"  Excluded keys (momentum/ITC/target/queues): {excluded_count}")
    print(f"  Filtered keys (should be loaded): {len(filtered_state)}")
    print(f"  Missing keys: {len(missing)} (predictor: {len(predictor_missing)}, encoder: {len(encoder_missing)})")
    print(f"  Unexpected keys: {len(unexpected)}")

    if len(encoder_missing) == 0 and len(unexpected) == 0:
        # Count loaded parameters by module
        from collections import defaultdict
        loaded_by_module = defaultdict(int)
        for k in filtered_state.keys():
            parts = k.split('.')
            if len(parts) >= 2:
                module = parts[1]
                loaded_by_module[module] += 1

        print(f"  All encoder/fusion parameters loaded successfully!")
        print(f"    Loaded modules:")
        for module in ['embedding_raw', 'encoder_raw', 'embedding_size', 'encoder_size', 'fusion']:
            if module in loaded_by_module:
                print(f"      - {module}: {loaded_by_module[module]} params")

        # Verify temporal_embedding was loaded
        if 'feature_extractor.embedding_size.temporal_embedding.weight' in filtered_state:
            print(f"    temporal_embedding loaded (IAT support enabled)")
        else:
            print(f"    temporal_embedding NOT found in checkpoint!")

        print(f"  Predictor randomly initialized ({len(predictor_missing)} params)")
    else:
        print(f"  Load incomplete:")
        if encoder_missing:
            print(f"    Missing encoder keys ({len(encoder_missing)}): {encoder_missing[:5]}...")

            if 'feature_extractor.embedding_size.temporal_embedding.weight' in encoder_missing:
                print(f"    CRITICAL: temporal_embedding.weight is missing!")
                print(f"    The pretrained model was trained WITHOUT IAT temporal information.")
                print(f"    Please use a Stage 2 model trained with IAT.")

        if unexpected:
            print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")


def count_parameters(model, trainable_only=True):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_encoder(model):
    """Freeze feature extractor (encoder + fusion) weights"""
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"Froze encoder. Trainable: {trainable:,} / {total:,} params")


def unfreeze_encoder(model, lr_scale=0.1):
    """
    Unfreeze encoder with lower learning rate.

    Returns param groups for optimizer.
    """
    for param in model.feature_extractor.parameters():
        param.requires_grad = True

    encoder_params = list(model.feature_extractor.parameters())
    predictor_params = list(model.predictor.parameters())

    trainable = count_parameters(model, trainable_only=True)
    print(f"Unfroze encoder. Trainable: {trainable:,} params")

    return [
        {'params': encoder_params, 'lr_scale': lr_scale},
        {'params': predictor_params, 'lr_scale': 1.0}
    ]
