import torch
from utils import load_data
from models import HANModel
from train_utils import train, train_batch, test

if __name__ == '__main__':
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    is_batch_train = False
    num_epochs, lr = 100, 0.05
    num_hidden, dropout, nheads = 8, 0.6, 8
    if is_batch_train:
        HGs_adj, train_iter, val_iter, test_iter = load_data(data_set='acm_raw', is_batch=is_batch_train)
    else:
        HGs_adj, features, labels, train_idx, val_idx, test_idx = load_data(data_set='acm_raw', is_batch=is_batch_train)
    net = HANModel(len(HGs_adj), features.shape[-1], num_hidden, out_size=3, num_heads=nheads, dropout=dropout)
    train_batch(net, train_iter, val_iter, lr, num_epochs, devices) if is_batch_train \
        else train(net, [HGs_adj, features, labels, train_idx, val_idx], lr, num_epochs, devices)
