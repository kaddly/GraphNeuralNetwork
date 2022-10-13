import math
import os
import random
import pickle
import torch
from torch.utils.data import DataLoader, Dataset


def read_train_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    data_path = os.path.join(data_dir, dataset, 'train.txt')
    edge_data_by_type = dict()  # 每个type对应到的相连接节点
    all_nodes = list()  # 所有节点的集合
    print("We are loading data from:" + dataset + '/train.txt')
    with open(data_path, 'r') as f:
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
    return edge_data_by_type  # 每个type连接的点边情况


def read_test_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon', file_name='test.txt'):
    data_path = os.path.join(data_dir, dataset, file_name)
    true_edge_data_by_type = dict()  # true样本
    false_edge_data_by_type = dict()  # false样本
    all_nodes = list()
    print("We are loading data from:" + dataset + "./" + file_name)
    with open(data_path, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:  # true对应到的节点
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()  # true对应到的type相连接节点
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type


def read_feature(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    data_path = os.path.join(data_dir, dataset, 'feature.txt')
    feature_dict = {}
    print("We are loading data from:" + dataset + '/feature.txt')
    with open(data_path, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dict[items[0]] = items[1:]
    return feature_dict


def read_node_type(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    data_path = os.path.join(data_dir, dataset, 'node_type.txt')
    node_type = {}
    print('We are loading node type from:' + dataset + '/node_type.txt')
    with open(data_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def load_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    training_data_by_type = read_train_data(data_dir, dataset)
    valid_true_data_by_edge, valid_false_data_by_edge = read_test_data(data_dir, dataset, file_name='valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = read_test_data(data_dir, dataset, file_name='test.txt')

