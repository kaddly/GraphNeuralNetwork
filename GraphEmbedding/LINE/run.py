import torch
from LINE import LINE
from data_utils import load_data_wiki
from train_eval import train, get_embedding

if __name__ == '__main__':
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]

    num_noise_words, embedding_size = 5, 128
    lr, batch_size, num_epochs = 0.002, 32, 100
    idx2node, node2idx, data_iter, G = load_data_wiki('../data/wiki/Wiki_edgelist.txt', batch_size, num_noise_words)
    net = LINE(len(idx2node), embedding_size)
    # train(net, data_iter, lr, num_epochs, devices)
    emb = get_embedding(net, G, node2idx)
    print(emb)
