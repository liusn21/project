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

        # Build fusion module (使用原框架的MultiHeadedAttention)
        from uer.layers.multimodal_fusion import GatedMultiModalFusion

        # Calculate attention_head_size
        attention_head_size = args.hidden_size // args.heads_num

        # Get gate_temperature from args (default 0.5)
        gate_temperature = getattr(args, 'gate_temperature', 0.5)

        fusion = GatedMultiModalFusion(
            hidden_size=args.hidden_size,
            num_attention_heads=args.heads_num,
            attention_head_size=attention_head_size,
            dropout=args.dropout,
            gate_temperature=gate_temperature
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

        # Phase-specific loading
        is_phase1 = getattr(args, 'phase1', False)
        is_phase2 = getattr(args, 'phase2', False)

        if is_phase1:
            # Phase 1: Load pretrained encoders from Stage 1
            if hasattr(args, 'pretrained_raw_path') and args.pretrained_raw_path:
                print(f"[Phase 1] Loading pretrained Raw encoder from {args.pretrained_raw_path}")
                raw_state = torch.load(args.pretrained_raw_path, map_location="cpu")
                model.embedding_raw.load_state_dict(
                    {k.replace('embedding.', ''): v for k, v in raw_state.items() if k.startswith('embedding.')},
                    strict=False
                )
                model.encoder_raw.load_state_dict(
                    {k.replace('encoder.', ''): v for k, v in raw_state.items() if k.startswith('encoder.')},
                    strict=False
                )
            else:
                print("WARNING: Phase 1 without pretrained_raw_path - encoders will be randomly initialized")

            if hasattr(args, 'pretrained_size_path') and args.pretrained_size_path:
                print(f"[Phase 1] Loading pretrained Size encoder from {args.pretrained_size_path}")
                size_state = torch.load(args.pretrained_size_path, map_location="cpu")
                model.embedding_size.load_state_dict(
                    {k.replace('embedding.', ''): v for k, v in size_state.items() if k.startswith('embedding.')},
                    strict=False
                )
                model.encoder_size.load_state_dict(
                    {k.replace('encoder.', ''): v for k, v in size_state.items() if k.startswith('encoder.')},
                    strict=False
                )
            else:
                print("WARNING: Phase 1 without pretrained_size_path - encoders will be randomly initialized")

            # Freeze encoders for Phase 1
            model._freeze_encoders()
            print("[Phase 1] Encoders frozen")

        elif is_phase2:
            # Phase 2: Full model will be loaded via --pretrained_model_path in trainer.py
            # Encoders should NOT be frozen
            print("[Phase 2] Full parameter training mode")
            print("  Note: Phase 1 checkpoint should be loaded via --pretrained_model_path")

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
