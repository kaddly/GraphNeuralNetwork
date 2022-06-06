import torch
from SDNE import SDNE_model
from data_utils import load_wiki

if __name__ == '__main__':
    batch_size, alpha, beta, gama = 32, 1e-6, 5., 0.3
    lr, num_epochs, device = 0.002, 100, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = [256, 128]
    data_iter, G, idx2node, node2idx = load_wiki('../data/wiki/Wiki_edgelist.txt', batch_size)
    model = SDNE_model(len(idx2node), hidden_size)
