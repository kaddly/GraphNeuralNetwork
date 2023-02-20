import torch
from torch import nn


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class Word2vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, **kwargs):
        super(Word2vec, self).__init__(**kwargs)
        self.V = nn.Embedding(vocab_size, embedding_size)
        self.U = nn.Embedding(vocab_size, embedding_size)

    def forward(self, center, context_negative):
        return skip_gram(center, context_negative, self.V, self.U)


class BiNEModel:
    def __init__(self, user_vocab_size, item_vocab_size, embedding_size):
        self.user_net = Word2vec(user_vocab_size, embedding_size)
        self.item_net = Word2vec(item_vocab_size, embedding_size)

    def user_implicit_relations(self, center, context_negative):
        return self.user_net(center, context_negative)

    def item_implicit_relations(self, center, context_negative):
        return self.item_net(center, context_negative)

    def explicit_relations(self, center, neighbors):
        user_embed = self.user_net.V(center)
        item_embed = self.item_net.V(neighbors)
        pred = torch.bmm(user_embed, item_embed.permute(0, 2, 1))
        return pred
