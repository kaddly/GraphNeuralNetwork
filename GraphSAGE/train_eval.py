import torch
from torch import nn
import torch.nn.functional as F
import time
import os
from datetime import timedelta


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.to(dtype=y.dtype) == y
    cmp = cmp.to(dtype=y.dtype)
    return float(cmp.sum()) / len(cmp)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions

    with torch.no_grad():
        acc, loss = [], []
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            acc.append(accuracy(y_hat, y))
            loss.append(F.cross_entropy(y_hat, y))
    return sum(acc) / len(acc), sum(loss) / len(loss)


def train(net, train_iter, val_iter, lr, num_epochs, device, Unsupervised=True):
    net = net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    # 模型参数保存路径
    saved_dir = './saved_dict'
    model_file = 'GraphSAGE'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        if Unsupervised:
            pass
        else:
            for i, batch in enumerate(train_iter):
                optimizer.zero_grad()
                nodes, samp_neighs, val_lens, labels = [data.to(device) for data in batch]
                emb, pre_labels = net(nodes, samp_neighs, val_lens)
                l = loss(pre_labels, labels)
                l.backward()
                optimizer.step()
                if total_batch % 20 == 0:
                    train_acc = accuracy(pre_labels, labels)
                    dev_acc, dev_loss = evaluate_accuracy_gpu(net, val_iter)
                    if dev_loss < dev_best_loss:
                        torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                        dev_best_loss = dev_loss
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
