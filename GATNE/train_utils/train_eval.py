import time
import os
import numpy as np
from datetime import timedelta
import torch
from torch import nn
import torch.nn.functional as F
from .optimizer_utils import create_lr_scheduler
from .scale_utils import accuracy, f_beta_score, recall
from .distributed_utils import Accumulator


def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    y_true = torch.Tensor(true_list)  # true label
    y_scores = torch.Tensor(prediction_list)  # predict proba
    return F.cross_entropy(y_scores, y_true), accuracy(y_scores, y_true, 2), f_beta_score(y_scores, y_true, 2), recall(y_scores, y_true, 2)


class ValScale:
    def __init__(self, num_nodes, edge_types, edge_type_count, neighbors, vocab):
        self.num_nodes = num_nodes
        self.edge_types = edge_types
        self.edge_type_count = edge_type_count
        self.neighbors = neighbors
        self.vocab = vocab

    def get_model(self, model, device):
        final_model = dict(zip(self.edge_types, [dict() for _ in range(self.edge_type_count)]))  # 每个类别下节点的embedding;
        for i in range(self.num_nodes):
            train_inputs = torch.tensor([i for _ in range(self.edge_type_count)]).to(device)  # 节点的多个类别，求每个类别下的embedding
            train_types = torch.tensor(list(range(self.edge_type_count))).to(device)
            node_neigh = torch.tensor(  # 节点在每个类别下的neighbors
                [self.neighbors[i] for _ in range(self.edge_type_count)]
            ).to(device)
            node_emb = model(train_inputs, train_types,
                             node_neigh)  # [node1, node1]; [type1, type2]; [node1_neigh, node1_neigh]
            for j in range(self.edge_type_count):  # 每个节点在各个类别下的embedding
                final_model[self.edge_types[j]][self.vocab.to_tokens[i]] = (
                    node_emb[j].cpu().detach().numpy()
                )
        return final_model

    def val_eval(self, model, device, valid_true_data_by_edge, valid_false_data_by_edge, args):
        final_model = self.get_model(model, device)
        valid_loss, valid_auc, valid_f1, valid_rcl = [], [], [], []
        for i in range(self.edge_type_count):
            if args.eval_type == "all" or self.edge_types[i] in args.eval_type.split(","):
                tmp_loss, tmp_auc, tmp_f1, tmp_rcl = evaluate(
                    final_model[self.edge_types[i]],
                    valid_true_data_by_edge[self.edge_types[i]],
                    valid_false_data_by_edge[self.edge_types[i]],
                )
                valid_loss.append(tmp_loss)
                valid_auc.append(tmp_auc)
                valid_f1.append(tmp_f1)
                valid_rcl.append(tmp_rcl)
        return np.mean(valid_loss), np.mean(valid_auc), np.mean(valid_f1), np.mean(valid_rcl)


def train(net, loss, train_iter, val_scale: ValScale, val_iter, args):
    # 模型参数保存路径
    parameter_path = os.path.join(args.model_dict_path, args.model)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if args.is_current_train and os.path.exists(os.path.join(parameter_path, args.model + '.ckpt')):
        net.load_state_dict(torch.load(os.path.join(parameter_path, args.model + '.ckpt')), False)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    loss = loss.to(device)
    optimizer = torch.optim.SGD([{"params": net.parameters()}, {"params": loss.parameters()}], lr=args.lr,
                                weight_decay=args.weight_decay)
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
                valid_loss, valid_auc, valid_f1, valid_rcl = val_scale.val_eval(net, device, *val_iter, args)
                if valid_loss < best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, args.model + '.ckpt'))
                    best_loss = valid_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train lr: {2:>5.4},  val loss: {3:>5.4},  val Acc: {4:>6.2%},  val recall: {5:6.2%},  val f1 score: {6:6.2%},  Time: {7} {8}'
                print(msg.format(total_batch, metric[0] / metric[1], lr_current, time_dif, improve))
                net.train()
