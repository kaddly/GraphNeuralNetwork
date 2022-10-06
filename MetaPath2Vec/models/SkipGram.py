import torch
from torch import nn
from torch.nn import init


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, **kwargs):
        """
        :param emb_size:词表长度
        :param emb_dimension:词向量长度
        """
        super(SkipGramModel, self).__init__(**kwargs)
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)

    def forward(self, center, contexts_and_negatives):
        v = self.v_embeddings(center)
        u = self.u_embeddings(contexts_and_negatives)
        return torch.bmm(v, u.permute(0, 2, 1))
