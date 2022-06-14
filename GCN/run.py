import torch
from data_utils import load_cora
from GCN import GCN_Model
from train_eval import train, test

if __name__ == '__main__':
    lr, num_epochs, device = 0.002, 4000, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_hidden = 128
    adj, features, labels, idx_train, idx_val, idx_test = load_cora()
    model = GCN_Model(features.shape[1], num_hidden=num_hidden, num_classes=labels.max().item() + 1, num_layers=2,
                      dropout=0.5)
    train(model, (adj, features, labels, idx_train, idx_val, idx_test), lr, num_epochs, device)
    test(model, (adj, features, labels, idx_train, idx_val, idx_test), device)
