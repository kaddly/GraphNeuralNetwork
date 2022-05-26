import torch
from torch import nn


class LINE(nn.Module):
    def __init__(self, nodes_num, embedding_size, **kwargs):
        super(LINE, self).__init__(**kwargs)
        self.first_emb = nn.Embedding(nodes_num, embedding_size)
        self.second_emb = nn.Embedding(nodes_num, embedding_size)
        self.context_emb = nn.Embedding(nodes_num, embedding_size)

    def forward(self, v_i, v_j):
        first_v_i = self.first_emb(v_i)
        first_v_j = self.first_emb(v_j)
