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


def evaluateTestSet(model: BiNEModel, data_iter, device=None):
    if not device:
        device = next(iter(model.user_net)).device
    with torch.no_grad():
        users, items, labels = [data.to(device) for data in data_iter]
        pred = model.explicit_relations(users.reshape(-1, 1), items.reshape(-1, 1))
    return accuracy(pred, labels, 2), F.binary_cross_entropy_with_logits(pred.reshape(labels.shape),
                                                                         labels).mean(), f_beta_score(pred,
                                                                                                      labels, 2).mean()


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_iter, test_iter, user_contexts, user_negatives, item_contexts, item_negatives, user_vocab, item_vocab = load_data(
        args)
    data_tool = ContextsNegativesGenerator(user_contexts, user_negatives, item_contexts, item_negatives)
    model = BiNEModel(len(user_vocab), len(item_vocab), args.embedding_size)
    model.user_net.to(device)
    model.item_net.to(device)
    loss = SigmoidBCELoss()
    optimizer = torch.optim.AdamW([{"params": model.user_net.parameters()}, {"params": model.item_net.parameters()}],
                                  lr=args.lr,
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
            o1 = F.binary_cross_entropy_with_logits(pred.reshape(w.shape), w)
            o2 = (loss(u_hat.reshape(u_l.shape).float(), u_l.float(), u_m) / u_m.sum(axis=1) * u_m.shape[1])
            o3 = (loss(i_hat.reshape(i_l.shape).float(), i_l.float(), i_m) / i_m.sum(axis=1) * i_m.shape[1])
            train_loss = args.alpha * o1.mean() + args.beta * o2.sum() + args.gamma * o3.sum()
            train_loss.backward()
            optimizer.step()
            if args.scheduler_lr:
                lr_scheduler.step()
            with torch.no_grad():
                u_f1 = f_beta_score(u_hat, u_l, 2, u_m)
                i_f1 = f_beta_score(i_hat, i_l, 2, i_m)
                pBar.set_postfix({'train loss': '{0:>5.4f}'.format(train_loss.item()),
                                  'user f1 score': '{0:>6.2%}'.format(u_f1.mean().item()),
                                  'item f1 score': '{0:>6.2%}'.format(i_f1.mean().item()),
                                  'lr': '{0:>5.4}'.format(optimizer.param_groups[0]["lr"])})
            logger.add_scalar("train loss", train_loss.item(), global_step=total_num)
            logger.add_scalar('user f1 scare', u_f1.mean().item(), global_step=total_num)
            logger.add_scalar('item f1 scare', i_f1.mean().item(), global_step=total_num)

            if total_num % args.print_freq == 0:
                model.user_net.eval()
                model.item_net.eval()
                lr_current = optimizer.param_groups[0]["lr"]
                test_acc, test_loss, test_f1 = evaluateTestSet(model, test_iter, device)
                if test_loss < best_loss:
                    torch.save(model.user_net.state_dict(), os.path.join("models", args.run_name, f"userNet.pt"))
                    torch.save(model.item_net.state_dict(), os.path.join("models", args.run_name, f"itemNet.pt"))
                    best_loss = test_loss
                    improve = '*'
                    last_improve = total_num
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0},  Train Loss: {1:>5.4},  Train lr: {2:>5.4},  val loss: {3:>5.4},  val Acc: {4:>6.2%},  val f1 score: {5:6.2%},  Time: {6} {7}'
                logging.info(
                    msg.format(total_num, train_loss, lr_current, test_loss, test_acc, test_f1, time_dif, improve))
                model.user_net.train()
                model.item_net.train()
            total_num += 1
            if total_num - last_improve > args.Max_auto_stop_epoch:
                # 验证集loss超过1000batch没下降，结束训练
                logging.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
