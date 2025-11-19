from uer.layers.embeddings import WordPosSegEmbedding, RawPacketEmbedding, PacketSizeEmbedding

str2embedding = {
    "word_pos_seg": WordPosSegEmbedding,
    "raw_packet": RawPacketEmbedding,
    "packet_size": PacketSizeEmbedding
}

__all__ = ["WordPosSegEmbedding", "RawPacketEmbedding", "PacketSizeEmbedding", "str2embedding"]