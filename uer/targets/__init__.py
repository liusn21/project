from uer.targets.mlm_target import MlmTarget
from uer.targets.multimodal_target import MultiModalTarget

str2target = {
    "mlm": MlmTarget,
    "raw_packet": MlmTarget,  # Raw packet modality uses MLM directly
    "packet_size": MlmTarget,  # Stage 1: Packet size modality uses MLM
    "multimodal": MultiModalTarget  # Stage 2: Multi-modal pretraining
}

__all__ = ["MlmTarget", "MultiModalTarget", "str2target"]
