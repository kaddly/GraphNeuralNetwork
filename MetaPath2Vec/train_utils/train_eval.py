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


def train_batch(net, data_iter, args):
    # 模型参数保存路径
    parameter_path = os.path.join(args.model_dict_path, args.model)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if args.is_current_train and os.path.exists(os.path.join(parameter_path, args.model + '.ckpt')):
        net.load_state_dict(torch.load(os.path.join(parameter_path, args.model + '.ckpt')), False)
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
        lr_scheduler = create_lr_scheduler(optimizer, 1, args.num_epoch)
    loss = SigmoidBCELoss()
    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    metric = Accumulator(3)
    for epoch in range(args.num_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = net(center, context_negative)
            train_loss = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1])
            train_loss.sum().backward()
            optimizer.step()
            if args.scheduler_lr:
                lr_scheduler.step()
            with torch.no_grad():
                metric.add(train_loss.sum(), accuracy(pred, label), label.shape[0])
            if total_batch % 20 == 0:
                lr_current = optimizer.param_groups[0]["lr"]
