import torch
import math
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm

class WordPosSegEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(WordPosSegEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb

class RawPacketEmbedding(nn.Module):
    """
    NEW Embedding for Raw Packet modality

    Components:
    - Token embedding: bigram tokens (vocab_size from vocab file)
    - Position embedding: sequence position
    - Packet embedding: which packet (0-7 for up to 8 packets, 8 for special tokens/padding)
    - Direction embedding: packet direction (0=downlink/-1, 1=neutral/padding, 2=uplink/1)

    Formula: emb = token_emb + position_emb + packet_emb + direction_emb

    Note: Protocol embedding is removed as per requirements
    """
    def __init__(self, args, vocab_size):
        super(RawPacketEmbeddingV2, self).__init__()
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
    Embedding for Packet Size modality (Stage 1)

    Size tokens already encode direction: size_token = size * direction + 1500
    So we don't need separate direction embedding.

    Components:
    - Size token embedding: vocab_size from vocab file (typically 3006)
    - Position embedding: sequence position
    - Protocol embedding: TCP/UDP

    Formula: emb = size_emb + position_emb + protocol_emb
    """
    def __init__(self, args, vocab_size):
        super(PacketSizeEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length

        # Size token embedding (direction already encoded)
        self.size_embedding = nn.Embedding(vocab_size, args.emb_size)

        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)

        # Protocol embedding: 0=TCP, 1=UDP
        self.protocol_embedding = nn.Embedding(2, args.emb_size)

        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, protocol=None):
        """
        Args:
            src: [batch, seq_len] - size token IDs (direction encoded)
            protocol: [batch] - protocol type (0=TCP, 1=UDP)

        Returns:
            emb: [batch, seq_len, emb_size]
        """
        batch_size, seq_len = src.size()

        # Size token embedding
        size_emb = self.size_embedding(src)

        # Position embedding
        pos_emb = self.position_embedding(
            torch.arange(0, seq_len, device=src.device, dtype=torch.long)
            .unsqueeze(0).repeat(batch_size, 1)
        )

        # Combine: size + position
        emb = size_emb + pos_emb

        # Protocol embedding (optional, broadcast across sequence)
        if protocol is not None:
            protocol_emb = self.protocol_embedding(protocol).unsqueeze(1)
            protocol_emb = protocol_emb.expand(-1, seq_len, -1)
            emb = emb + protocol_emb

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        return emb
