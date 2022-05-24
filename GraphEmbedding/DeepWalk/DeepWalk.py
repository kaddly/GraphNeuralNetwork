import torch
from torch import nn
from .data_utils import deepwalk_walk


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class Word2vec:
    def __init__(self, vocab_size, embed_size):
        self.net = nn.Sequential(nn.Embedding(vocab_size, embed_size), nn.Embedding(vocab_size, embed_size))

    def __call__(self, center, context_negative):
        return skip_gram(center, context_negative, self.net[0], self.net[1])


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph
        self.w2v_model = None
        self._embedding = {}

        self.walker
