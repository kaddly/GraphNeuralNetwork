import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, **kwargs):
        super(GraphAttentionLayer, self, ).__init__(**kwargs)
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        N = Wh.size()[0]
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1,
                                                                                            2 * self.out_features)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 之前计算的是一个节点和所有节点的attention，其实需要的是连接的节点的attention系数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime)  # elu-激活函数
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATConv(nn.Module):
    def __init__(self, feat_size, hidden_size, num_class, dropout, alpha, num_heads, **kwargs):
        super(GATConv, self).__init__(**kwargs)
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        for i in range(num_heads):
            self.attentions.add_module(f'AttentionHead{i}',
                                       GraphAttentionLayer(feat_size, hidden_size, dropout=dropout, alpha=alpha, concat=True))
            self.out_att = GraphAttentionLayer(hidden_size * num_heads, num_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        return F.elu(self.out_att(x, adj))  # 第二层的attention layer
