import torch
from GraphSAGE import GraphSAGE
from data_utils import load_pubmed_data
from train_eval import train, test

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_gcn, is_unsupervised = False, False
    num_neighs, num_layers = 10, 2
    batch_size, num_epochs, lr = 64, 50, 0.05
    input_size, hidden_size = 500, 128
    train_iter, val_iter, test_iter = load_pubmed_data('./data/pubmed-data', batch_size, num_layers,
                                                       num_neighs, window_size=5, num_noise_words=5, is_gcn=is_gcn,
                                                       is_unsupervised=is_unsupervised)
    net = GraphSAGE(num_layers, input_size, hidden_size, is_gcn, agg_func="MEAN", Unsupervised=is_unsupervised, class_size=3)
    train(net, train_iter, val_iter, lr, num_epochs, device, is_unsupervised)
    test(net, test_iter, device, is_unsupervised)
