import torch
from torch import nn
from torch.nn import functional as F
import time
import os
from datetime import timedelta


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
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


def train(model, data_iter, lr, num_epochs, device):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)
    model.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()

    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数

    # 模型参数保存路径
    saved_dir = './saved_dict'
    model_file = 'GAT'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)
    adj, features, labels, idx_train, idx_val, idx_test = [data.to(device) for data in data_iter]
    lr_scheduler = create_lr_scheduler(optimizer, 1, num_epochs, warmup=True)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        train_loss = loss(output[idx_train], labels[idx_train])
        train_acc = accuracy(output[idx_train], labels[idx_train])
        train_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        if epoch % 20 == 0:
            model.eval()
            output = model(features, adj)
            val_loss = loss(output[idx_val], labels[idx_val])
            if dev_best_loss > val_loss:
                torch.save(model.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                dev_best_loss = val_loss
                improve = '*'
                last_improve = epoch
            else:
                improve = ''
            time_dif = timedelta(seconds=int(round(time.time() - start_time)))
            msg = 'Epoch [{0}/{1}]:  train_loss: {2:>5.3f},  train_acc: {3:>6.2%},  train_lr: {4:>5.3f},  val_loss: {5:>5.3f}, val_acc: {6:>6.2%}, Time: {7} {8}'
            print(msg.format(epoch + 1, num_epochs, train_loss.item(), train_acc, lr, val_loss.item(),
                             accuracy(output[idx_val], labels[idx_val]), time_dif,
                             improve))
        if epoch - last_improve > 1000:
            print("No optimization for a long time, auto-stopping...")
            break


def test(model, data_iter, device):
    adj, features, labels, idx_train, idx_val, idx_test = [data.to(device) for data in data_iter]
    if not os.path.exists('./saved_dict/GAT/GAT.ckpt'):
        print('please train before!')
        return
    model.load_state_dict(torch.load('./saved_dict/GAT/GAT.ckpt'), False)
    model.to(device)
    model.eval()
    with torch.no_grad:
        output = model(features, adj)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test))
