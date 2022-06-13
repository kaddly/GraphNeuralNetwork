import torch
from torch import nn


class GCN_Model(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, num_layers, dropout, **kwargs):
        super(GCN_Model, self).__init__(**kwargs)
        self.gcn_blocks = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.gcn_blocks.add_module(f'gcn{i}', Graph_conv_layer(num_features, num_hidden))
            elif i == num_layers - 1:
                self.gcn_blocks.add_module(f'gcn{i}', Graph_conv_layer(num_hidden, num_classes))
            else:
                self.gcn_blocks.add_module(f'gcn{i}', Graph_conv_layer(num_hidden, num_hidden))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X, adj):
        for gcn_block in self.gcn_blocks:
            pass


class Graph_conv_layer(nn.Module):
    def __init__(self, in_features, out_features, is_bias=True, **kwargs):
        super(Graph_conv_layer, self).__init__(**kwargs)
        self.dense = nn.Linear(in_features, out_features, bias=False)
        if is_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, X_input, adj):
        support = self.dense(X_input)
        output = torch.spmm(adj, support)  # 稀疏矩阵的相乘，和mm一样的效果
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
