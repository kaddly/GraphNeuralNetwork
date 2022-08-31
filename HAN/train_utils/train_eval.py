import time
import os
import torch
from torch import nn
import torch.nn.functional as F
from datetime import timedelta
from .scale_utils import accuracy
from .optimizer_utils import create_lr_scheduler
from .distributed_utils import Accumulator


def evaluate_accuracy_gpu(net, val_iter):
    acc, loss = [], []
    return acc, loss


def train_batch(net, train_iter, val_iter, lr, num_epochs, devices, is_current_train=True):
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


def train():
    pass


def test():
    pass
