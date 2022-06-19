import torch
from GraphSAGE import GraphSage
from train_eval import train
from data_utils import load_pubmed_data

if __name__ == '__main__':
    lr, num_epochs, batch_size, device = 0.01, 5, 64, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_layers, hidden_size = 2, 500
    sample_neigh, Unsupervised = 10, False
    train_iter, val_iter, test_iter, feat_data = load_pubmed_data('./data/pubmed-data', batch_size, sample_neigh,
                                                                  Unsupervised=Unsupervised)
    net = GraphSage(num_layers, len(feat_data[0]), hidden_size, feat_data, agg_func='MEAN', gcn=False,
                    Unsupervised=Unsupervised, class_nums=3)
    train(net, train_iter, val_iter, lr, num_epochs, device, Unsupervised)
