import os
import torch
from torch.utils.data import DataLoader, Dataset


def read_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    train_data_path = os.path.join(data_dir, dataset, 'train.txt')
    val_data_path = os.path.join(data_dir, dataset, 'valid.txt')
    test_data_path = os.path.join(data_dir, dataset, 'test.txt')
    edge_data_by_type = dict()  # 每个type对应到的相连接节点
    all_nodes = list()  # 所有节点的

    with open(train_data_path, 'r') as f:
        for line in f:
            words = line[:-1].split(" ")
            if words[0] not in edge_data_by_type:  # edge type涉及到的节点
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))  # nodes去重
    print('Total training nodes: ' + str(len(all_nodes)))

    def process_val_data(f_index):
        true_edge_data_by_type = dict()  # true样本
        false_edge_data_by_type = dict()  # false样本
        for sentence in f_index:
            tokens = sentence[:-1].split(' ')
            x, y = tokens[0], tokens[1]
            if int(tokens[3]) == 1:  # true对应到的节点
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[tokens[0]] = list()  # true对应到的type相连接节点
                true_edge_data_by_type[tokens[0]].append((x, y))
            else:
                if tokens[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[tokens[0]] = list()
                false_edge_data_by_type[tokens[0]].append((x, y))
        return true_edge_data_by_type, false_edge_data_by_type

    with open(val_data_path, 'r') as f:
        val_true_edge_data_by_type, val_false_edge_data_by_type = process_val_data(f)

    with open(test_data_path, 'r') as f:
        test_true_edge_data_by_type, test_false_edge_data_by_type = process_val_data(f)

    return edge_data_by_type, val_true_edge_data_by_type, val_false_edge_data_by_type, test_true_edge_data_by_type, test_false_edge_data_by_type


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def load_data():
    pass
