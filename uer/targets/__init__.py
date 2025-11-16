from uer.targets.mlm_target import MlmTarget
from uer.targets.bertflow_target import BertFlowTarget

str2target = {"bertflow": BertFlowTarget, "mlm": MlmTarget}

__all__ = ["MlmTarget", "BertFlowTarget", "str2target"]