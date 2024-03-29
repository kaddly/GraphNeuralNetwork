import time
import os
from datetime import timedelta
import torch
from torch import nn
import torch.nn.functional as F
from train_utils.optimizer_utils import create_lr_scheduler
from train_utils.distributed_utils import Accumulator
from train_utils.scale_utils import f_beta_score, accuracy, recall


class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super(SigmoidBCELoss, self).__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


def train(net, data_iter, args):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    # 模型参数保存路径
    parameter_path = os.path.join(args.model_dict_path, args.model)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if args.is_current_train and os.path.exists(os.path.join(parameter_path, args.model + '.ckpt')):
        net.load_state_dict(torch.load(os.path.join(parameter_path, args.model + '.ckpt')), False)
    else:
        net.apply(init_weights)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler_lr:
        lr_scheduler = create_lr_scheduler(optimizer, args.num_batch, args.num_epoch)
    loss = SigmoidBCELoss()
    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch
    best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    metric = Accumulator(5)
    for epoch in range(args.num_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epoch))
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = net(center, context_negative)
            train_loss = (
                    loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1]).sum()

            train_loss.backward()
            optimizer.step()
            if args.scheduler_lr:
                lr_scheduler.step()
            with torch.no_grad():
                metric.add(train_loss, accuracy(pred.reshape(-1), label.reshape(-1), 2, mask.reshape(-1)),
                           recall(pred.reshape(-1), label.reshape(-1), 2, mask.reshape(-1)).mean(),
                           f_beta_score(pred.reshape(-1), label.reshape(-1), 2, mask.reshape(-1)).mean(), label.shape[0])
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
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.2%},  Train  recall: {3:6.2%},  Train f1 score: {4:6.2%},  Train lr: {5:>5.4},  Time: {6} {7}'
                print(
                    msg.format(total_batch, metric[0] / metric[4], metric[1] / (total_batch + 1),
                               metric[2] / (total_batch + 1), metric[3] / (total_batch + 1), lr_current, time_dif,
                               improve))
                net.train()
            total_batch += 1
            if total_batch - last_improve > args.Max_auto_stop_epoch:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
