import torch
import torch.nn as nn
from uer.utils.constants import PAD_ID


class PacketSizeModel(nn.Module):
    """
    Model for Packet Size pretraining with Temporal Information

    Architecture:
        - embedding: PacketSizeEmbedding (token + temporal + position)
        - encoder: Transformer encoder
        - target: DualMlmTarget (Size MLM + Temporal MLM)
    """
    def __init__(self, args, embedding, encoder, target):
        super(PacketSizeModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target


    def forward(self, src, iat_tokens, tgt_mlm_size, tgt_mlm_temporal):
        """
        Args:
            src: [batch, seq_len] - size token IDs (direction encoded)
            iat_tokens: [batch, seq_len] - IAT temporal token IDs (0-999 + special)
            tgt_mlm_size: [batch, seq_len] - size MLM targets
            tgt_mlm_temporal: [batch, seq_len] - temporal MLM targets

        Returns:
            loss_info: Tuple of:
                - loss_size: Size MLM loss (CrossEntropy)
                - correct_size: Number of correct size predictions
                - denominator_size: Number of masked size positions
                - loss_temporal: Temporal MLM loss (soft-label KL divergence)
                - correct_temporal_exact: Number of exact temporal matches
                - correct_temporal_range: Number of predictions within ±sigma range
                - denominator_temporal: Number of masked temporal positions
        """
        # Generate embeddings (token + temporal + position)
        emb = self.embedding(src, iat_tokens)

        # Generate segment IDs for attention mask
        # seg = 0 for PAD positions, seg = 1 for valid positions
        # The encoder uses (seg > 0) to create attention mask
        seg = (src != PAD_ID).long()  # 0 for PAD, 1 for non-PAD
        output = self.encoder(emb, seg)

        # Compute dual MLM loss (size + temporal)
        loss_info = self.target(output, tgt_mlm_size, tgt_mlm_temporal)

        return loss_info
