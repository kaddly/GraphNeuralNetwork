import torch
from torch import nn
import torch.nn.functional as F


class CTConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CTConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)

    def forward(self, A):  # self.weight:带有channel的conv;
        """
        0) 对weight(conv)进行softmax
        1) 对每个节点在每个edgeType上进行[2, 5, 1, 1]的卷积操作;
        2) 对每个edgeType进行加权求和，加权是通过0)softmax
        """
        # F.softmax(self.weight, dim=1) 对self.weight做softmax:[2, 5, 1, 1]
        # A: [1, 5, 8994, 8994]:带有edgeType的邻接矩阵
        # [1, 5, 8994, 8994]*[2, 5, 1, 1] => [2, 5, 8994, 8994]
        # sum:[2, 8994, 8994]
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A
