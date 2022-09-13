import torch
from torch import nn
import torch.nn.functional as F
from models.GTLayer import GTLayer


def norm(H, add=False):
    H = H.t()  # t
    if not add:
        H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor))  # 建立一个对角阵; 除了自身节点，对应位置相乘。Degree(排除本身)
    else:
        H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
    deg = torch.sum(H, dim=1)  # 按行求和, 即每个节点的degree的和
    deg_inv = deg.pow(-1)  # deg-1 归一化操作
    deg_inv[deg_inv == float('inf')] = 0
    deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.FloatTensor)  # 转换成n*n的矩阵
    H = torch.mm(deg_inv, H)  # 矩阵内积
    H = H.t()
    return H


class GTN_Model(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_layers, is_norm, **kwargs):
        super(GTN_Model, self).__init__(**kwargs)
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = is_norm
        layers = []
        for i in range(num_layers):  # layers多个GTLayer组成的; 多头channels
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))  # 第一个GT层,edge类别构建的矩阵
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))  # GCN
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, X, H):  # 自己写了一个GCN
        X = torch.mm(X, self.weight)  # X-features; self.weight-weight
        H = norm(H, add=True)  # H-第i个channel下邻接矩阵;
        return torch.mm(H.t(), X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i == 0:
                H_ = norm(H[i, :, :]).unsqueeze(0)  # Q1
            else:
                H_ = torch.cat((H_, norm(H[i, :, :]).unsqueeze(0)), dim=0)  # Q2
        return H_

    def forward(self, A, X, target_x):
        A = A.unsqueeze(0).permute(0, 3, 1, 2)  # A.unsqueeze(0)=[1,N,N,edgeType]=>[1,edgeType,N,N]; 卷积输出的channel数量
        Ws = []
        for i in range(self.num_layers):  # 两层GTLayer:{edgeType}
            if i == 0:
                H, W = self.layers[i](A)  # GTN0:两层GTConv; A:edgeType的邻接矩阵; output: H(A(l)), W:归一化的Conv
            else:
                H = self.normalization(H)  # Conv矩阵，D-1*A的操作
                H, W = self.layers[i](A, H)  # 第一层计算完了A(原始矩阵), H(上一次计算后的A(l)); output: A2, W(第二层Conv1)
            Ws.append(W)

        # H,W1 = self.layer1(A)
        # H = self.normalization(H)
        # H,W2 = self.layer2(A, H)
        # H = self.normalization(H)
        # H,W3 = self.layer3(A, H)
        for i in range(self.num_channels):  # conv的channel数量
            if i == 0:
                X_ = F.relu(self.gcn_conv(X, H[i]))  # X-features; H[i]-第i个channel输出的邻接矩阵Al[i]; gcn_conv:Linear
            else:
                X_tmp = F.relu(self.gcn_conv(X, H[i]))
                X_ = torch.cat((X_, X_tmp), dim=1)  # X_拼接之后输出
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        return y, Ws
