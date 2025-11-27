import torch
from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.models.model import Model
from uer.models.raw_packet_model import RawPacketModel
from uer.models.packet_size_model import PacketSizeModel
from uer.models.multimodal_model import MultiModalModel
from uer.model_loader import load_model


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """

    # For multimodal target, use MultiModalModel with two encoders
    if args.target == "multimodal":
        # Build Raw Packet encoder (to be loaded from pretrained)
        embedding_raw = RawPacketEmbedding(args, len(args.vocab_raw))
        encoder_raw = str2encoder[args.encoder](args)

        # Build Packet Size encoder (to be loaded from pretrained)
        embedding_size = PacketSizeEmbedding(args, len(args.vocab_size))
        encoder_size = str2encoder[args.encoder](args)

        # Build fusion module
        from uer.layers.multimodal_fusion import GatedMultiModalFusion
        fusion = GatedMultiModalFusion(
            hidden_size=args.hidden_size,
            num_attention_heads=args.heads_num,
            dropout=args.dropout
        )

        # Build target module
        target = str2target[args.target](
            args,
            hidden_size=args.hidden_size,
            vocab_size_raw=len(args.vocab_raw),
            vocab_size_size=len(args.vocab_size)
        )

        # Build model
        model = MultiModalModel(
            args,
            embedding_raw, encoder_raw,
            embedding_size, encoder_size,
            fusion, target
        )

        # Load pretrained encoders (if specified)
        if hasattr(args, 'pretrained_raw_path') and args.pretrained_raw_path:
            print(f"Loading pretrained Raw encoder from {args.pretrained_raw_path}")
            # Load state dict
            raw_state = torch.load(args.pretrained_raw_path, map_location="cpu")

            # Load embedding and encoder weights
            model.embedding_raw.load_state_dict(
                {k.replace('embedding.', ''): v for k, v in raw_state.items() if k.startswith('embedding.')},
                strict=False
            )
            model.encoder_raw.load_state_dict(
                {k.replace('encoder.', ''): v for k, v in raw_state.items() if k.startswith('encoder.')},
                strict=False
            )

        if hasattr(args, 'pretrained_size_path') and args.pretrained_size_path:
            print(f"Loading pretrained Size encoder from {args.pretrained_size_path}")
            # Load state dict
            size_state = torch.load(args.pretrained_size_path, map_location="cpu")

            # Load embedding and encoder weights
            model.embedding_size.load_state_dict(
                {k.replace('embedding.', ''): v for k, v in size_state.items() if k.startswith('embedding.')},
                strict=False
            )
            model.encoder_size.load_state_dict(
                {k.replace('encoder.', ''): v for k, v in size_state.items() if k.startswith('encoder.')},
                strict=False
            )

    # For raw_packet target, use RawPacketEmbedding and RawPacketModel
    elif args.target == "raw_packet":
        embedding = RawPacketEmbedding(args, len(args.vocab))
        encoder = str2encoder[args.encoder](args)
        target = str2target[args.target](args, len(args.vocab))
        model = RawPacketModel(args, embedding, encoder, target)
    # For packet_size target, use PacketSizeEmbedding and PacketSizeModel
    elif args.target == "packet_size":
        embedding = PacketSizeEmbedding(args, len(args.vocab))
        encoder = str2encoder[args.encoder](args)
        target = str2target[args.target](args, len(args.vocab))
        model = PacketSizeModel(args, embedding, encoder, target)
    else:
        embedding = str2embedding[args.embedding](args, len(args.vocab))
        encoder = str2encoder[args.encoder](args)
        target = str2target[args.target](args, len(args.vocab))
        model = Model(args, embedding, encoder, target)

    return model
