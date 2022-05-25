import torch
from torch import nn
from data_utils import RandomWalker, load_data_wiki
from train_eval import train


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
        self._embedding = {}
        self.w2v_model = None
        self.walker = RandomWalker(graph)
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers,
                                                    verbose=1)

    def train(self, lr=0.002, embed_size=128, window_size=5, num_epochs=5):
        data_iter, self.vocab = load_data_wiki(self.sentences, batch_size=128, max_window_size=window_size,
                                               num_noise_words=5)
        model = Word2vec(len(self.vocab), embed_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Learning embedding vectors...")
        train(model, data_iter, lr=lr, num_epochs=num_epochs, device=device)
        print("Learning embedding vectors done!")
        self.w2v_model = model

    def get_embeddings(self):
        if self.w2v_model is None:
            print("model not train")
            return {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.net[0][word]

        return self._embeddings
