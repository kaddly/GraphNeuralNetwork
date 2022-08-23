import torch
from torch import nn
import torch.nn.functional as F
from .NodeAttention import GATConv
from .SemanticAttention import SemanticAttention


class HANLayer(nn.Module):
    def __init__(self, **kwargs):
        super(HANLayer, self).__init__(**kwargs)
        nn.ModuleList()

    def forward(self):
        pass


class HANModel(nn.Module):
    def __init__(self, **kwargs):
        super(HANModel, self).__init__(**kwargs)

    def forward(self):
        pass
