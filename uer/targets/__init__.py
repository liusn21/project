from uer.targets.mlm_target import MlmTarget
from uer.targets.bertflow_target import BertFlowTarget

str2target = {
    "bertflow": BertFlowTarget,
    "mlm": MlmTarget,
    "raw_packet": MlmTarget,  # Raw packet modality uses MLM directly
    "packet_size": MlmTarget  # Stage 1: Packet size modality uses MLM
}

__all__ = ["MlmTarget", "BertFlowTarget", "str2target"]