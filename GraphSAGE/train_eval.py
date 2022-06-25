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


def evaluate_accuracy_gpu(net, data_iter, is_unsupervised, device=None):
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
            if isinstance(X, tuple):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            if is_unsupervised:
                _, y_hat = net(*X)
            else:
                _, y_hat = net(*X, None, None, None, None, None)
            acc.append(accuracy(y_hat, y))
            loss.append(F.cross_entropy(y_hat, y))
    return sum(acc) / len(acc), sum(loss) / len(loss)


def train(net, train_iter, val_iter, lr, num_epochs, device, is_unsupervised=True):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
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
        for i, (X, labels) in enumerate(train_iter):
            optimizer.zero_grad()
            if isinstance(X, tuple):
                # Required for BERT fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            labels = labels.to(device)
            if is_unsupervised:
                emb, pre_labels = net(*X, labels.shape)
            else:
                emb, pre_labels = net(*X, None, None, None, None, None)
            l = loss(pre_labels, labels)
            l.backward()
            optimizer.step()
            if total_batch % 20 == 0:
                train_acc = accuracy(pre_labels, labels)
                dev_acc, dev_loss = evaluate_accuracy_gpu(net, val_iter, is_unsupervised)
                if dev_loss < dev_best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, l.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                net.train()
            total_batch += 1
            if total_batch - last_improve > 2000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
