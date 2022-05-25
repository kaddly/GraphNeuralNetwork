import torch
from torch import nn
from .data_utils import RandomWalker
from .train_eval import train


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
        self.w2v_model = Word2vec
        self._embedding = {}

        self.walker = RandomWalker(graph)
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers,
                                                    verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        pass
