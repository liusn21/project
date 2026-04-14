import json
from argparse import Namespace


def load_hyperparam(args):
    with open(args.config_path, mode="r", encoding="utf-8") as f:
        param = json.load(f)

    args_dict = vars(args)
    args_dict.update(param)
    args = Namespace(**args_dict)

    return args


# Fields that must be shared across the two multimodal encoders because they
# determine the dimensions that fusion cross-attention sees on both sides.
_SHARED_ENCODER_FIELDS = ("hidden_size", "heads_num", "emb_size")


def apply_modality_configs(args):
    """
    Populate args.layers_num_raw / args.layers_num_size from optional
    per-modality config files (args.config_path_raw / args.config_path_size).

    If a modality config is not given, that modality falls back to args.layers_num.
    Other fields in the modality config are checked for consistency with the
    main config: shared dims (hidden_size / heads_num / emb_size) must match,
    because fusion cross-attention requires identical embedding geometry.

    Only layers_num is actually applied; everything else is validated-only.
    """
    shared = {k: getattr(args, k, None) for k in _SHARED_ENCODER_FIELDS}
    base_layers = getattr(args, "layers_num", None)

    def _resolve(path, label):
        if not path:
            return base_layers
        with open(path, mode="r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k in _SHARED_ENCODER_FIELDS:
            if k in cfg and shared.get(k) is not None and cfg[k] != shared[k]:
                raise ValueError(
                    f"{label} config {path}: field '{k}'={cfg[k]} disagrees "
                    f"with main config '{k}'={shared[k]}. Shared encoder "
                    f"dims must match across modalities for fusion "
                    f"cross-attention."
                )
        return cfg.get("layers_num", base_layers)

    args.layers_num_raw = _resolve(getattr(args, "config_path_raw", None), "raw")
    args.layers_num_size = _resolve(getattr(args, "config_path_size", None), "size")
    return args
