import torch.nn as nn
from uer.layers.transformer import TransformerLayer


class TransformerEncoder(nn.Module):
    """
    BERT encoder built from a stack of TransformerLayer modules.
    """
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        self.transformer = nn.ModuleList(
            [TransformerLayer(args) for _ in range(self.layers_num)]
        )

    def forward(self, emb, seg, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if self.mask == "fully_visible":
            mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            raise ValueError(f"Unsupported mask type: {self.mask}. Only 'fully_visible' is supported.")

        hidden = emb
        for i in range(self.layers_num):
            hidden, _ = self.transformer[i](hidden, mask)
        return hidden
