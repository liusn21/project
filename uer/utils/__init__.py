from uer.utils.tokenizers import BertTokenizer
from uer.utils.data import BertFlowDataset, BertFlowDataLoader
from uer.utils.data import RawPacketDataset, RawPacketDataLoader
from uer.utils.data import PacketSizeDataset, PacketSizeDataLoader
from uer.utils.data import MultiModalDataset, MultiModalDataLoader
from uer.utils.data import mask_seq, truncate_seq_pair
from uer.utils.act_fun import *
from uer.utils.optimizers import *

str2tokenizer = {"bert": BertTokenizer, "bertflow": BertTokenizer, "raw_packet": BertTokenizer, "packet_size": BertTokenizer, "multimodal": BertTokenizer}
str2dataset = {
    "bertflow": BertFlowDataset,
    "raw_packet": RawPacketDataset,
    "packet_size": PacketSizeDataset,
    "multimodal": MultiModalDataset
}
str2dataloader = {
    "bertflow": BertFlowDataLoader,
    "raw_packet": RawPacketDataLoader,
    "packet_size": PacketSizeDataLoader,
    "multimodal": MultiModalDataLoader
}

str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}
str2optimizer = {"adamw": AdamW, "adafactor": Adafactor}
str2scheduler = {"linear": get_linear_schedule_with_warmup,
                "cosine": get_cosine_schedule_with_warmup,
                "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
                "polynomial": get_polynomial_decay_schedule_with_warmup,
                "constant": get_constant_schedule,
                "constant_with_warmup": get_constant_schedule_with_warmup}

__all__ = ["BertTokenizer", "str2tokenizer",
            "BertFlowDataset", "RawPacketDataset", "PacketSizeDataset", "MultiModalDataset", "str2dataset",
            "BertFlowDataLoader", "RawPacketDataLoader", "PacketSizeDataLoader", "MultiModalDataLoader", "str2dataloader",
            "mask_seq", "truncate_seq_pair",
            "gelu", "gelu_fast", "relu", "silu", "linear", "str2act",
            "AdamW", "Adafactor", "str2optimizer",
            "get_linear_schedule_with_warmup", "get_cosine_schedule_with_warmup",
            "get_cosine_with_hard_restarts_schedule_with_warmup",
            "get_polynomial_decay_schedule_with_warmup",
            "get_constant_schedule", "get_constant_schedule_with_warmup", "str2scheduler"]