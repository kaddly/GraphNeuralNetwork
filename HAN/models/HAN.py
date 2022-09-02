import torch
from torch import nn
from .NodeAttention import GATConv
from .SemanticAttention import SemanticAttention


class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout, **kwargs):
        super(HANLayer, self).__init__(**kwargs)
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.add_module(f'meta_path_model{i}',
                                       GATConv(in_size, out_size, dropout, layer_num_heads))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)

    def forward(self, gs, h):
        semantic_embeddings = []

        for g, gat_layer in zip(gs, self.gat_layers):
            semantic_embeddings.append(gat_layer(h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)


class HANModel(nn.Module):
    def __init__(self, num_mate_paths, in_size, hidden_size, out_size, num_heads, dropout, **kwargs):
        super(HANModel, self).__init__(**kwargs)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_mate_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(num_mate_paths, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size*num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)
