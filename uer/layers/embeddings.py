import torch
import math
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm

class RawPacketEmbedding(nn.Module):
    """
    NEW Embedding for Raw Packet modality

    Components:
    - Token embedding: byte tokens (vocab_size from vocab file)
    - Position embedding: sequence position
    - Packet embedding: which packet (0-7 for up to 8 packets, 8 for special tokens/padding)
    - Direction embedding: packet direction (0=downlink/-1, 1=neutral/padding, 2=uplink/1)

    Formula: emb = token_emb + position_emb + packet_emb + direction_emb

    Note: Protocol embedding is removed as per requirements
    """
    def __init__(self, args, vocab_size):
        super(RawPacketEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length

        # Token embedding (from vocab.txt)
        self.token_embedding = nn.Embedding(vocab_size, args.emb_size)

        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)

        # Packet embedding: 9 indices (0-7 for packets, 8 for special tokens/padding)
        self.packet_embedding = nn.Embedding(9, args.emb_size)

        # Direction embedding: 3 indices (0=downlink/-1, 1=neutral/padding, 2=uplink/1)
        self.direction_embedding = nn.Embedding(3, args.emb_size)

        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, packet_ids, directions):
        """
        Args:
            src: [batch, seq_len] - token IDs
            packet_ids: [batch, seq_len] - packet indices (0-7 for packets, 8 for special/padding)
            directions: [batch, seq_len] - direction indices (0=downlink, 1=neutral, 2=uplink)

        Returns:
            emb: [batch, seq_len, emb_size]
        """
        batch_size, seq_len = src.size()

        # Token embedding
        token_emb = self.token_embedding(src)

        # Position embedding
        pos_emb = self.position_embedding(
            torch.arange(0, seq_len, device=src.device, dtype=torch.long)
            .unsqueeze(0).repeat(batch_size, 1)
        )

        # Packet embedding
        packet_emb = self.packet_embedding(packet_ids)

        # Direction embedding
        direction_emb = self.direction_embedding(directions)

        # Combine all embeddings
        emb = token_emb + pos_emb + packet_emb + direction_emb

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        return emb


class PacketSizeEmbedding(nn.Module):
    """
    NEW Embedding for Packet Size modality with Temporal Information

    Components:
    - Token embedding: size tokens (vocab_size from vocab file)
    - Temporal embedding: IAT tokens (vocab_size_temporal=1005 for 0-999 + special tokens)
    - Position embedding: sequence position

    Formula: emb = token_emb + temporal_emb + position_emb

    Note:
    - Direction is already encoded in size tokens (size * direction + 1500)
    - IAT provides temporal dynamics information
    - Protocol embedding is removed as per requirements
    """
    def __init__(self, args, vocab_size, vocab_size_temporal):
        super(PacketSizeEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length

        # Token embedding (size tokens with direction encoded)
        self.token_embedding = nn.Embedding(vocab_size, args.emb_size)

        # Temporal embedding (IAT tokens: 0-999 + special tokens)
        self.temporal_embedding = nn.Embedding(vocab_size_temporal, args.emb_size)

        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)

        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, iat_tokens):
        """
        Args:
            src: [batch, seq_len] - size token IDs (direction encoded)
            iat_tokens: [batch, seq_len] - IAT temporal token IDs (0-999 + special)

        Returns:
            emb: [batch, seq_len, emb_size]
        """
        batch_size, seq_len = src.size()

        # Token embedding
        token_emb = self.token_embedding(src)

        # Temporal embedding
        temporal_emb = self.temporal_embedding(iat_tokens)

        # Position embedding
        pos_emb = self.position_embedding(
            torch.arange(0, seq_len, device=src.device, dtype=torch.long)
            .unsqueeze(0).repeat(batch_size, 1)
        )

        # Combine embeddings: token + temporal + position
        emb = token_emb + temporal_emb + pos_emb

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        return emb
