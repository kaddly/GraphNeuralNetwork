import torch
from data_utils import load_cora
from models import GAT
from train_eval import train, test

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs, lr = 1000, 0.01
    num_hidden, dropout, alpha, nheads = 8, 0.6, 0.2, 8
    adj, features, labels, idx_train, idx_val, idx_test = load_cora()
    model = GAT(features.shape[1], num_hidden, labels.max().item() + 1, dropout, alpha, nheads)
    train(model, (adj, features, labels, idx_train, idx_val, idx_test), lr, num_epochs, device)
    test(model, (adj, features, labels, idx_train, idx_val, idx_test), device)
