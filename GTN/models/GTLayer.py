import torch
from torch import nn
import torch.nn.functional as F
from models.GTConv import GTConv


class GTLayer(nn.Module):
    # 不同edge类型的组合
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # 1x1卷积的channel数量
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)  # W1
            self.conv2 = GTConv(in_channels, out_channels)  # W2
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):  # A:[1,edgeType,N,N]
        if self.first:
            a = self.conv1(A)  # GTConv=>[2, N, N] #Q1
            b = self.conv2(A)  # Q2
            # *** 作了第一次矩阵相乘，得到A1
            H = torch.bmm(a, b)  # torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3;
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),
                 (F.softmax(self.conv2.weight, dim=1)).detach()]  # conv-softmax: 是为了下一次直接使用吗？
        else:
            a = self.conv1(A)  # 第二层只有一个conv1; output:Conv输出归一化edge后的结果
            H = torch.bmm(H_, a)  # H_上一层的输出矩阵A1; 输出这一层后的结果A2;
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H, W  # H = A(1) ... A(l); W = 归一化后的权重矩阵
