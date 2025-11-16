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


class MultiModalEmbedding(nn.Module):
    """
    Multi-modal embedding for traffic analysis (PTU-inspired):
    - Packet tokens (raw hex bigrams): vocab_size = 65536
    - Temporal tokens (IAT): vocab_size = 1000
    - Size+Direction tokens: vocab_size = 3001

    Each token type has its own embedding layer.
    Additional type embedding distinguishes the three modalities.
    """
    def __init__(self, args):
        super(MultiModalEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length

        # Three separate embedding layers for different modalities
        self.packet_embedding = nn.Embedding(65536, args.emb_size)  # 0x0000-0xFFFF
        self.temporal_embedding = nn.Embedding(1000, args.emb_size)  # [0, 999]
        self.size_embedding = nn.Embedding(3001, args.emb_size)      # [0, 3000]

        # Type embedding to distinguish modalities (0=packet, 1=temporal, 2=size)
        self.type_embedding = nn.Embedding(3, args.emb_size)

        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)

        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, token_types, positions=None):
        """
        Args:
            src: [batch_size, seq_len] - token IDs
            token_types: [batch_size, seq_len] - token types (0/1/2)
            positions: [batch_size, seq_len] - optional position indices

        Returns:
            emb: [batch_size, seq_len, emb_size]
        """
        batch_size, seq_len = src.size()

        # Route tokens to appropriate embedding based on type
        # Create masks for each token type
        mask_packet = (token_types == 0)    # packet tokens
        mask_temporal = (token_types == 1)  # temporal tokens
        mask_size = (token_types == 2)      # size tokens

        # Initialize token embeddings
        token_emb = torch.zeros(batch_size, seq_len, self.packet_embedding.embedding_dim,
                                device=src.device, dtype=torch.float)

        # Apply embeddings based on token type
        token_emb[mask_packet] = self.packet_embedding(src[mask_packet])
        token_emb[mask_temporal] = self.temporal_embedding(src[mask_temporal])
        token_emb[mask_size] = self.size_embedding(src[mask_size])

        # Type embedding
        type_emb = self.type_embedding(token_types)

        # Position embedding
        if positions is None:
            positions = torch.arange(0, seq_len, device=src.device, dtype=torch.long)
            positions = positions.unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings: token + type + position
        emb = token_emb + type_emb + pos_emb

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        return emb
