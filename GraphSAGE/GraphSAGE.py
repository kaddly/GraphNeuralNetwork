import torch
from torch import nn
import torch.nn.functional as F


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size, gcn=False, **kwargs):
        super(SageLayer, self).__init__(**kwargs)

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(
            torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))  # 创建weight

        self.init_params()  # 初始化参数

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)  # concat自己信息和邻居信息
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, aggregator, gcn=False, Unsupervised=True, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn

        self.aggregator = aggregator

        self.blocks = nn.Sequential()
        for index in range(num_layers):
            layer_size = out_size if index != 0 else input_size
            self.blocks.add_module(f'sage{index}', SageLayer(layer_size, out_size, gcn=self.gcn))

        if not Unsupervised:
            self.dense = nn.Linear(out_size, layer_size)

    def forward(self, nodes, feat_data, neighbors):
        neigh_feat = self.aggregator(feat_data, nodes, neighbors)
        for block in self.blocks:
            feat_data = block(feat_data, neigh_feat)
            neigh_feat = self.aggregator(feat_data, nodes, neighbors)
        if self.Unsupervised:
            return feat_data
        else:
            return feat_data, self.dense(feat_data)