import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphAttentionLayer, SpGraphAttentionLayer


class GATBase(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(GATBase, self).__init__(**kwargs)
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.out_att = None

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        return F.elu(self.out_att(x, adj))  # 第二层的attention layer


class GAT(GATBase):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, **kwargs):
        """Dense version of GAT."""
        super(GAT, self).__init__(dropout, **kwargs)
        for i in range(nheads):
            self.attentions.add_module(f'AttentionHead{i}',
                                       GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True))
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)


class SpGAT(GATBase):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, **kwargs):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__(dropout, **kwargs)
        for i in range(nheads):
            self.attentions.add_module(f'AttentionHead{i}',
                                       SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True))
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
