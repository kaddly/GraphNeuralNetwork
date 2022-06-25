import torch
from torch import nn
import torch.nn.functional as F
from graph_utils import Aggregator


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
        self.Unsupervised = Unsupervised
        if not Unsupervised:
            self.dense = nn.Linear(out_size, class_size)

    def forward(self, center_feats_data, center_nodes_map, center_neigh_feats_data, center_neigh_nodes_map,
                contexts_negatives_feats_data, contexts_negatives_nodes_map, contexts_negatives_neigh_feats_data,
                contexts_negatives_neigh_nodes_map, contexts_negatives_shape):
        # 监督学习
        if contexts_negatives_feats_data is None:
            for i, block in enumerate(self.sage_blocks):
                aggregator_feats_data = Aggregator(center_neigh_feats_data, self.agg_func)
                feats_data = block(center_feats_data, aggregator_feats_data)
                if i != self.num_layers - 1:
                    center_feats_data = torch.embedding(feats_data, center_nodes_map[i][center_nodes_map[i] != -1])
                    center_neigh_feats_data = torch.embedding(feats_data, center_neigh_nodes_map[i][
                                                                          center_neigh_nodes_map[i][:, 0] != -1, :])
            classes = None
            if not self.Unsupervised:
                classes = self.dense(feats_data)
            return feats_data, classes
        else:
            center_feats_data, _ = self(center_feats_data, center_nodes_map, center_neigh_feats_data,
                                        center_neigh_nodes_map, None, None, None, None, None)
            contexts_negatives_feats_data, _ = self(contexts_negatives_feats_data, contexts_negatives_nodes_map,
                                                    contexts_negatives_neigh_feats_data,
                                                    contexts_negatives_neigh_nodes_map, None, None, None, None, None)
            contexts_negatives_feats_data = contexts_negatives_feats_data.reshape(*contexts_negatives_shape, -1)
            return center_feats_data, torch.bmm(center_feats_data.unsqueeze(1), contexts_negatives_feats_data.permute(0, 2, 1))
