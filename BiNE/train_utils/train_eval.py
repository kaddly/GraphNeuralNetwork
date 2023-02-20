import os
import logging
from tqdm import tqdm
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import BiNEModel
from utils import load_data, setup_logging, ContextsNegativesGenerator
from .loss_utils import SigmoidBCELoss
from .optimizer_utils import create_lr_scheduler
from .scale_utils import f_beta_score, accuracy


def evaluateTestSet(model, device):
    pass


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_iter, test_iter, user_contexts, user_negatives, item_contexts, item_negatives, user_vocab, item_vocab = load_data(
        args)
    data_tool = ContextsNegativesGenerator(user_contexts, user_negatives, item_contexts, item_negatives)
    model = BiNEModel(len(user_vocab), len(item_vocab), args.embedding_size)
    loss = SigmoidBCELoss()
    optimizer = torch.optim.AdamW([{"params": model.user_net}, {"params": model.item_net}], lr=args.lr,
                                  weight_decay=args.weight_decay)

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    if args.scheduler_lr:
        lr_scheduler = create_lr_scheduler(optimizer, 1, args.num_epoch)
    start_time = time.time()
    total_num = 0
    best_loss = float('inf')
    last_improve = 0
    flag = False
    for epoch in range(args.num_epoch):
        logging.info(f"Starting epoch {epoch}:")
        pBar = tqdm(train_iter)
        for u, i, w in pBar:
            optimizer.zero_grad()
            u, i, w, u_cn, u_m, u_l, i_cn, i_m, i_l = [data.to(device) for data in
                                                       data_tool.get_contexts_negatives_masks_labels(u, i, w)]
            pred = model.explicit_relations(u, i)
            u_hat = model.user_implicit_relations(u, u_cn)
            i_hat = model.item_implicit_relations(i, i_cn)
            o1 = F.binary_cross_entropy_with_logits(pred, w)
            o2 = (loss(u_hat.float(), u_l.float(), u_m) / u_m.sum(axis=1) * u_m.shape[1])
            o3 = (loss(i_hat.float(), i_l.float(), i_m) / i_m.sum(axis=1) * i_m.shape[1])
            train_loss = args.alpha * o1 + args.beta * o2.sum() + args.gamma * o3.sum()
            train_loss.backward()
            optimizer.step()
            if args.scheduler_lr:
                lr_scheduler.step()
            pBar.set_postfix(train_loss=train_loss.item(), lr=optimizer.param_groups[0]["lr"])
            logger.add_scalar("train_loss", train_loss.item(), global_step=total_num)

            if total_num % args.print_freq == 0:
                pass
            total_num += 1
            if total_num - last_improve > args.Max_auto_stop_epoch:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
