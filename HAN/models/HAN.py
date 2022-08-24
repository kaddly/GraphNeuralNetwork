import torch
from torch import nn
import torch.nn.functional as F
from .NodeAttention import GATConv
from .SemanticAttention import SemanticAttention


class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout, **kwargs):
        super(HANLayer, self).__init__(**kwargs)
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.add_module(f'meta_path_model{i}',
                                       GATConv(in_size, out_size, layer_num_heads, dropout))

    def forward(self):
        pass


class HANModel(nn.Module):
    def __init__(self, **kwargs):
        super(HANModel, self).__init__(**kwargs)

    def forward(self):
        pass
