import torch
from torch import nn
import torch.nn.functional as F
import time
import os
from datetime import timedelta


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


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


def train(net, train_iter, val_iter, lr, num_epochs, device):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_iter), num_epochs)
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
            if isinstance(X, list):
                # Required for BERT fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            labels = labels.to(device)
            y_hat = net(X)
            train_l = loss(y_hat, labels)
            train_l.backward()
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            if total_batch % 20 == 0:
                train_acc = accuracy(y_hat, labels)
                dev_acc, dev_loss = evaluate_accuracy_gpu(net, val_iter)
                if dev_loss < dev_best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.2%},  Train lr: {3:>5.4},  Val Loss: {4:>5.4},  Val Acc: {5:>6.2%},  Time: {6} {7}'
                print(msg.format(total_batch, train_l.item(), train_acc, lr, dev_loss, dev_acc, time_dif, improve))
                net.train()
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(model, data_iter, device=None):
    if not os.path.exists('./saved_dict/GraphSAGE/GraphSAGE.ckpt'):
        print('please train before!')
        return
    model.load_state_dict(torch.load('./saved_dict/GraphSAGE/GraphSAGE.ckpt'), False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        acc, loss = [], []
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            acc.append(accuracy(y_hat, y))
            loss.append(F.cross_entropy(y_hat, y))
    print("Test set results:",
          "loss= {:.4f}".format(sum(loss) / len(loss)),
          "accuracy= {:.4f}".format(sum(acc) / len(acc)))
