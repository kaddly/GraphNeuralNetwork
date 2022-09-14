import time
import os
from datetime import timedelta
import torch
from torch import nn
import torch.nn.functional as F
from .scale_utils import accuracy, recall, f_beta_score, precision
from .optimizer_utils import create_lr_scheduler
from .distributed_utils import Accumulator


def train(net, data_iter, args):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    # 模型参数保存路径
    parameter_path = os.path.join(args.model_dict_path, args.model)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if args.is_current_train and os.path.exists(os.path.join(parameter_path, 'HAN.ckpt')):
        net.load_state_dict(torch.load(os.path.join(parameter_path, 'HAN.ckpt')), False)
    else:
        net.apply(init_weights)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    if args.adaptive_lr:
        optimizer = torch.optim.Adam([{'params': net.weight},
                                      {'params': net.linear1.parameters()},
                                      {'params': net.linear2.parameters()},
                                      {"params": net.layers.parameters(), "lr": 0.5}
                                      ], lr=0.005, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler_lr:
        lr_scheduler = create_lr_scheduler(optimizer, 1, args.epoch)

    loss = nn.CrossEntropyLoss()
    start_time = time.time()
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    metric = Accumulator(3)
    A, node_feature, labels, train_idx, val_idx = [x.to(device) for x in data_iter]
    num_classes = torch.max(labels).item() + 1
    for epoch in range(args.epoch):
        net.train()
        optimizer.zero_grad()
        y_hat, ws = net(A, node_feature, train_idx)
        train_loss = loss(y_hat, labels[train_idx])
        train_loss.backward()
        optimizer.step()
        if args.scheduler_lr:
            lr_scheduler.step()
        with torch.no_grad():
            metric.add(train_loss.sum(), accuracy(y_hat, labels[train_idx]),
                       f_beta_score(y_hat, labels[train_idx], num_classes))
        if epoch % args.print_freq == 0:
            net.eval()
            lr_current = optimizer.param_groups[0]["lr"]
            output = net(A, node_feature, val_idx)
            val_loss = loss(output, labels[val_idx])
            if dev_best_loss > val_loss:
                torch.save(net.state_dict(), os.path.join(parameter_path, args.model + '.ckpt'))
                dev_best_loss = val_loss
                improve = '*'
                last_improve = epoch
            else:
                improve = ''
            time_dif = timedelta(seconds=int(round(time.time() - start_time)))
            msg = 'Epoch [{0}/{1}]:  train_loss: {2:>5.3f},  train_acc: {3:>6.2%}, train_f1_score: {4:>5.3f},  train_lr: {5:>5.3f},  val_loss: {6:>5.3f}, val_acc: {7:>6.2%}, val_f1_score: {8:>5.3f}, Time: {9} {10}'
            print(msg.format(epoch + 1, args.epoch, metric[0] / epoch, metric[1] / epoch, metric[2] / epoch, lr_current,
                             val_loss.item(), accuracy(output, labels[val_idx]),
                             f_beta_score(output, labels[val_idx], num_classes), time_dif, improve))
        if epoch - last_improve > 5000:
            print("No optimization for a long time, auto-stopping...")
            break


def test(net, args):
    pass
