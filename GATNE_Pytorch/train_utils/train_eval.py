import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F
from train_utils.loss_utils import SigmoidBCELoss
from train_utils.optimizer_utils import create_lr_scheduler
from train_utils.scale_utils import accuracy, f_beta_score, recall
from train_utils.distributed_utils import Accumulator


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
    return F.binary_cross_entropy_with_logits(y_scores, y_true), accuracy(y_scores, y_true, 2), f_beta_score(y_scores,
                                                                                                             y_true,
                                                                                                             2).mean(), recall(
        y_scores, y_true, 2).mean()


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
            node_emb = model.encoder(train_inputs, train_types,
                                     node_neigh)  # [node1, node1]; [type1, type2]; [node1_neigh, node1_neigh]
            for j in range(self.edge_type_count):  # 每个节点在各个类别下的embedding
                final_model[self.edge_types[j]][self.vocab.to_tokens(i)] = (
                    node_emb[j].cpu().detach().numpy()
                )
        return final_model

    def val_eval(self, model, device, valid_true_data_by_edge, valid_false_data_by_edge, args):
        final_model = self.get_model(model, device)
        valid_loss, valid_acc, valid_f1, valid_rcl = [], [], [], []
        for i in range(self.edge_type_count):
            if args.eval_type == "all" or self.edge_types[i] in args.eval_type.split(","):
                tmp_loss, tmp_auc, tmp_f1, tmp_rcl = evaluate(
                    final_model[self.edge_types[i]],
                    valid_true_data_by_edge[self.edge_types[i]],
                    valid_false_data_by_edge[self.edge_types[i]],
                )
                valid_loss.append(tmp_loss.item())
                valid_acc.append(tmp_auc.item())
                valid_f1.append(tmp_f1.item())
                valid_rcl.append(tmp_rcl.item())
        return np.mean(valid_loss), np.mean(valid_acc), np.mean(valid_f1), np.mean(valid_rcl)


def train(net, train_iter, val_scale: ValScale, val_iter, args):
    # 模型参数保存路径
    parameter_path = os.path.join(args.model_dict_path, args.model)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    if args.is_current_train and os.path.exists(os.path.join(parameter_path, args.model + '.ckpt')):
        net.load_state_dict(torch.load(os.path.join(parameter_path, args.model + '.ckpt')), False)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    loss = SigmoidBCELoss()
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            centers, type_ids, neighbors, contexts_negatives, masks, labels = [data.to(device) for data in batch]
            pred = net(centers, type_ids, neighbors, contexts_negatives)
            train_loss = (loss(pred.float(), labels.float(), masks) / masks.sum(axis=1) * masks.shape[1])
            train_loss.sum().backward()
            optimizer.step()
            if args.scheduler_lr:
                lr_scheduler.step()
            with torch.no_grad():
                metric.add(train_loss.sum().item(), i + 1)
            if total_batch % args.print_freq == 0:
                net.eval()
                lr_current = optimizer.param_groups[0]["lr"]
                valid_loss, valid_acc, valid_f1, valid_rcl = val_scale.val_eval(net, device, *val_iter, args)
                if valid_loss < best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, args.model + '.ckpt'))
                    best_loss = valid_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0},  Train Loss: {1:>5.4},  Train lr: {2:>5.4},  val loss: {3:>5.4},  val Acc: {4:>6.2%},  val recall: {5:6.2%},  val f1 score: {6:6.2%},  Time: {7} {8}'
                print(msg.format(total_batch, metric[0] / metric[1], lr_current, valid_loss, valid_acc, valid_rcl,
                                 valid_f1, time_dif, improve))
                net.train()
            total_batch += 1
            if total_batch - last_improve > args.Max_auto_stop_epoch:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(net, data_iter, val_scale: ValScale, args):
    if not os.path.exists(os.path.join(os.path.abspath('.'), 'saved_dict', args.model)):
        print('please train before!')
        return
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    net.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'saved_dict', args.model, args.model + '.ckpt')), False)
    net.eval()
    test_loss, test_acc, test_f1, test_rcl = val_scale.val_eval(net, device, *data_iter, args)
    print("Test set results:",
          "loss= {:>5.3f}".format(test_loss),
          "accuracy= {:>6.2%}".format(test_acc),
          "f1 score= {:>6.2%}".format(test_f1),
          "recall= {:>6.2%}".format(test_rcl))
