import time
import os
from datetime import timedelta
import torch
from torch import nn
import torch.nn.functional as F
from .optimizer_utils import create_lr_scheduler
from .scale_utils import accuracy, f_beta_score, recall
from .distributed_utils import Accumulator


def train(net, loss, train_iter, val_iter, args):
    # 模型参数保存路径
    parameter_path = os.path.join(args.model_dict_path, args.model)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if args.is_current_train and os.path.exists(os.path.join(parameter_path, args.model + '.ckpt')):
        net.load_state_dict(torch.load(os.path.join(parameter_path, args.model + '.ckpt')), False)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    loss = loss.to(device)
    optimizer = torch.optim.SGD([{"params": net.parameters()}, {"params": loss.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler_lr:
        lr_scheduler = create_lr_scheduler(optimizer, args.num_batch, args.num_epoch)

    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch
    best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    metric = Accumulator(2)

    for epoch in range(args.num_epoch):
        metric.reset()
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epoch))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            center, context, types, neigh = [data.to(device) for data in batch]
            emb = net(center, types, neigh)
            train_loss = loss(center, emb, context)
            train_loss.backward()
            optimizer.step()
            if args.scheduler_lr:
                lr_scheduler.step()
            with torch.no_grad():
                metric.add(train_loss.item(), i)
            if total_batch % args.print_freq == 0:
                net.eval()
                lr_current = optimizer.param_groups[0]["lr"]
                if train_loss < best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, args.model + '.ckpt'))
                    best_loss = train_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train lr: {2:>5.4},  val loss: {3:>5.4},  val Acc: {4:>6.2%},  val recall: {5:6.2%},  val f1 score: {6:6.2%},  Time: {7} {8}'
                print(msg.format(total_batch, metric[0] / metric[1], lr_current, time_dif, improve))
                net.train()



