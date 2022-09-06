import torch
from torch import nn
from models.GTLayer import GTLayer


class GTN(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_layers, norm, **kwargs):
        super(GTN, self).__init__(**kwargs)
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
