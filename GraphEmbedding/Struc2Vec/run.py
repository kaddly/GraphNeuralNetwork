import torch
from word2vec import Word2vec
from utils.data_utils import load_flight_data
from train_eval import train, get_embedding

if __name__ == '__main__':
    batch_size, max_window_size, num_noise_words = 256, 5, 5
    lr, num_epochs, device = 0.002, 100, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter, idx2node, node2idx, G = load_flight_data('../data/flight/brazil-airports.edgelist',
                                                        batch_size=batch_size,
                                                        num_walks=80, walk_length=10, workers=4,
                                                        max_window_size=max_window_size,
                                                        num_noise_words=num_noise_words, opt1_reduce_len=True,
                                                        opt2_reduce_sim_calc=True, opt3_num_layers=None, stay_prob=0.3)
    vocab_size, embed_size = len(node2idx), 128
    model = Word2vec(vocab_size, embed_size)
    train(model, data_iter, lr, num_epochs, device)
    emb = get_embedding(model, G, node2idx)
    print(emb)
