import torch
from torch import nn
import torch.nn.functional as F


class HANLayer(nn.Module):
    def __init__(self, **kwargs):
        super(HANLayer, self).__init__(**kwargs)
        nn.ModuleList()

    def forward(self):
        pass
