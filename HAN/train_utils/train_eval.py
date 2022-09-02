import time
import os
import torch
from torch import nn
import torch.nn.functional as F
from datetime import timedelta
from .scale_utils import accuracy
from .optimizer_utils import create_lr_scheduler
from .distributed_utils import Accumulator


def evaluate_accuracy_gpu(net, val_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    with torch.no_grad():
        val_acc, val_loss = [], []
        for batch in val_iter:
            HG_adj, features, labels = [
                x.to(device) if not isinstance(x, (list, tuple)) else [adj.to(device) for adj in x] for x in
                batch]
            y_hat = net(HG_adj, features)
            val_loss.append(F.cross_entropy(y_hat, labels))
            val_acc.append(accuracy(y_hat, labels))
    return sum(val_acc) / len(val_acc), sum(val_loss) / len(val_loss)


def train_batch(net, train_iter, val_iter, lr, num_epochs, devices, is_current_train=True):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    # 模型参数保存路径
    saved_dir = './saved_dict'
    model_file = 'HAN_batch'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if is_current_train and os.path.exists('./saved_dict/HAN_batch/HAN_batch.ckpt'):
        net.load_state_dict(torch.load('./saved_dict/HAN_batch/HAN_batch.ckpt'), False)
    else:
        net.apply(init_weights)

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_iter), num_epochs)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    metric = Accumulator(3)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            HG_adj, features, labels = [
                x.to(devices[0]) if not isinstance(x, (list, tuple)) else [adj.to(devices[0]) for adj in x] for x in
                batch]
            y_hat = net(HG_adj, features)
            train_loss = loss(y_hat, labels)
            train_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                metric.add(train_loss.sum(), accuracy(y_hat, labels), labels.shape[0])
            if total_batch % 20 == 0:
                lr_current = optimizer.param_groups[0]["lr"]
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
                print(
                    msg.format(total_batch, metric[0] / metric[2], metric[1] / (total_batch + 1), lr_current, dev_loss,
                               dev_acc, time_dif, improve))
                net.train()
            total_batch += 1
            if total_batch - last_improve > 5000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def train(net, data_iter, lr, num_epochs, devices, is_current_train=True):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    # 模型参数保存路径
    saved_dir = './saved_dict'
    model_file = 'HAN'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if is_current_train and os.path.exists('./saved_dict/HAN/HAN.ckpt'):
        net.load_state_dict(torch.load('./saved_dict/HAN/HAN.ckpt'), False)
    else:
        net.apply(init_weights)

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    lr_scheduler = create_lr_scheduler(optimizer, 1, num_epochs)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    metric = Accumulator(2)
    HGs_adj, features, labels, train_idx, val_idx = [
        x.to(devices[0]) if not isinstance(x, (list, tuple)) else [adj.to(devices[0]) for adj in x] for x in data_iter]
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        output = net(features, HGs_adj)
        train_loss = loss(output[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        with torch.no_grad():
            metric.add(train_loss.sum(), accuracy(output, labels))
        if epoch % 20 == 0:
            net.eval()
            lr_current = optimizer.param_groups[0]["lr"]
            output = net(features, HGs_adj)
            val_loss = loss(output[val_idx], labels[val_idx])
            if dev_best_loss > val_loss:
                torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                dev_best_loss = val_loss
                improve = '*'
                last_improve = epoch
            else:
                improve = ''
            time_dif = timedelta(seconds=int(round(time.time() - start_time)))
            msg = 'Epoch [{0}/{1}]:  train_loss: {2:>5.3f},  train_acc: {3:>6.2%},  train_lr: {4:>5.3f},  val_loss: {5:>5.3f}, val_acc: {6:>6.2%}, Time: {7} {8}'
            print(msg.format(epoch + 1, num_epochs, metric[0] / epoch, metric[1] / epoch, lr_current, val_loss.item(),
                             accuracy(output[val_idx], labels[val_idx]), time_dif, improve))
        if epoch - last_improve > 5000:
            print("No optimization for a long time, auto-stopping...")
            break


def test():
    pass
