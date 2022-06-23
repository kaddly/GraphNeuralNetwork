import torch
from torch import nn
import torch.nn.functional as F
from .graph_utils import Aggregator


class SageLayer(nn.Module):
    def __init__(self, input_size, output_size, gcn=False, **kwargs):
        super(SageLayer, self).__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.gcn = gcn
        self.weight = nn.Linear(self.input_size if self.gcn else 2 * self.input_size, self.output_size, bias=False)

    def forward(self, self_feats, aggregate_feats):
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)  # concat自己信息和邻居信息
        else:
            combined = aggregate_feats
        return F.relu(self.weight(combined))


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_size, out_size, gcn=False, agg_func='MEAN', Unsupervised=True, class_size=None,
                 **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.gcn = gcn
        self.agg_func = agg_func
        self.sage_blocks = nn.Sequential()
        for index in range(0, num_layers):
            layer_size = out_size if index != 0 else input_size
            self.sage_blocks.add_module('sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))
        if not Unsupervised:
            self.dense = nn.Linear(out_size, class_size)

    def forward(self):
        pass
