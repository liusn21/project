import torch.nn as nn
from uer.utils import *


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer. """
    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]

    def forward(self, x):
        inter = self.act(self.linear_1(x))
        output = self.linear_2(inter)
        return output
