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
    metric = Accumulator(5)
