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

    # For multimodal target, use MultiModalModel (ALBEF-style)
    if args.target == "multimodal":
        # Build Raw Packet encoder
        embedding_raw = RawPacketEmbedding(args, len(args.vocab_raw))
        encoder_raw = str2encoder[args.encoder](args)

        # Build Packet Size encoder
        embedding_size = PacketSizeEmbedding(args, len(args.vocab_size))
        encoder_size = str2encoder[args.encoder](args)

        # Build fusion module (6-layer bidirectional cross-attention)
        from uer.layers.multimodal_fusion import MultiModalFusionEncoder

        num_fusion_layers = getattr(args, 'num_fusion_layers', 6)
        fusion = MultiModalFusionEncoder(args, num_layers=num_fusion_layers)

        # Build target module (ITC + ITM + MLM)
        target = str2target[args.target](
            args,
            hidden_size=args.hidden_size,
            vocab_size_raw=len(args.vocab_raw),
            vocab_size_size=len(args.vocab_size)
        )

        # Get queue and momentum parameters
        queue_size = getattr(args, 'queue_size', 4096)
        momentum = getattr(args, 'momentum', 0.995)

        # Build model
        model = MultiModalModel(
            args,
            embedding_raw, encoder_raw,
            embedding_size, encoder_size,
            fusion, target,
            queue_size=queue_size,
            momentum=momentum
        )

        # Load pretrained encoders if specified
        prefix_emb = 'embedding.'
        prefix_enc = 'encoder.'

        if hasattr(args, 'pretrained_raw_path') and args.pretrained_raw_path:
            print(f"Loading pretrained Raw encoder from {args.pretrained_raw_path}")
            raw_state = torch.load(args.pretrained_raw_path, map_location="cpu")
            raw_emb_state = {k[len(prefix_emb):]: v for k, v in raw_state.items() if k.startswith(prefix_emb)}
            raw_enc_state = {k[len(prefix_enc):]: v for k, v in raw_state.items() if k.startswith(prefix_enc)}

            emb_result = model.embedding_raw.load_state_dict(raw_emb_state, strict=False)
            enc_result = model.encoder_raw.load_state_dict(raw_enc_state, strict=False)
            # Also load into momentum encoder
            model.embedding_raw_m.load_state_dict(raw_emb_state, strict=False)
            model.encoder_raw_m.load_state_dict(raw_enc_state, strict=False)

            print(f"  Checkpoint keys: {len(raw_state)}, Embedding: {len(raw_emb_state)}, Encoder: {len(raw_enc_state)}")
            if len(emb_result.missing_keys) == 0 and len(enc_result.missing_keys) == 0:
                print(f"  Raw encoder loaded successfully!")
            else:
                print(f"  Raw encoder load incomplete:")
                if emb_result.missing_keys:
                    print(f"    Missing embedding: {emb_result.missing_keys}")
                if enc_result.missing_keys:
                    print(f"    Missing encoder: {enc_result.missing_keys[:5]}...")

        if hasattr(args, 'pretrained_size_path') and args.pretrained_size_path:
            print(f"Loading pretrained Size encoder from {args.pretrained_size_path}")
            size_state = torch.load(args.pretrained_size_path, map_location="cpu")
            size_emb_state = {k[len(prefix_emb):]: v for k, v in size_state.items() if k.startswith(prefix_emb)}
            size_enc_state = {k[len(prefix_enc):]: v for k, v in size_state.items() if k.startswith(prefix_enc)}

            emb_result = model.embedding_size.load_state_dict(size_emb_state, strict=False)
            enc_result = model.encoder_size.load_state_dict(size_enc_state, strict=False)
            # Also load into momentum encoder
            model.embedding_size_m.load_state_dict(size_emb_state, strict=False)
            model.encoder_size_m.load_state_dict(size_enc_state, strict=False)

            # 打印加载信息
            print(f"  Checkpoint keys: {len(size_state)}, Embedding: {len(size_emb_state)}, Encoder: {len(size_enc_state)}")
            if len(emb_result.missing_keys) == 0 and len(enc_result.missing_keys) == 0:
                print(f"  Size encoder loaded successfully!")
            else:
                print(f"  Size encoder load incomplete:")
                if emb_result.missing_keys:
                    print(f"    Missing embedding: {emb_result.missing_keys}")
                if enc_result.missing_keys:
                    print(f"    Missing encoder: {enc_result.missing_keys[:5]}...")

        print(f"  Fusion layers: {num_fusion_layers}")
        print(f"  Queue size: {queue_size}")
        print(f"  Momentum: {momentum}")
        print(f"  ITC temperature: {getattr(args, 'itc_temperature', 0.07)}")
        print(f"  ITM temperature: {getattr(args, 'itm_temperature', 0.07)}")

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
