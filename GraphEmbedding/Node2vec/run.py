import torch
from word2vec import Word2vec
from utils.data_utils import load_flight_data
from train_eval import train, get_similar_tokens

if __name__ == '__main__':
    batch_size, max_window_size, num_noise_words = 32, 5, 5
    lr, num_epochs, device = 0.002, 5, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter, idx2node, node2idx = load_flight_data('../data/flight/brazil-airports.edgelist',
                                                     batch_size=batch_size, num_walks=16, walk_length=8, workers=4,
                                                     max_window_size=max_window_size, num_noise_words=num_noise_words)
    vocab_size, embed_size = len(node2idx), 100
    model = Word2vec(vocab_size, embed_size)
    train(model, data_iter, lr, num_epochs, device)
