import torch
import torch.nn as nn
from uer.utils.constants import PAD_ID


class PacketSizeModel(nn.Module):
    """
    Model for Packet Size pretraining

    Architecture:
        - embedding: PacketSizeEmbeddingV2 (token + position)
        - encoder: Transformer encoder
        - target: MlmTarget (Masked Language Modeling)
    """
    def __init__(self, args, embedding, encoder, target):
        super(PacketSizeModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        # Tie weights between token embedding and output layer if specified
        if args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.token_embedding.weight


    def forward(self, src, tgt):
        """
        Args:
            src: [batch, seq_len] - token IDs (direction encoded in size tokens)
            tgt: [batch, seq_len] - MLM targets

        Returns:
            loss_info: (loss, correct, denominator)
        """
        # Generate embeddings
        emb = self.embedding(src)

        # Generate segment IDs for attention mask
        # seg = 0 for PAD positions, seg = 1 for valid positions
        # The encoder uses (seg > 0) to create attention mask
        seg = (src != PAD_ID).long()  # 0 for PAD, 1 for non-PAD
        output = self.encoder(emb, seg)

        # Compute MLM loss
        loss_info = self.target(output, tgt)

        return loss_info
