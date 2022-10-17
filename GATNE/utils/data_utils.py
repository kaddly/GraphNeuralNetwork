import math
import os
import random
import pickle
import tqdm
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from utils.sample_utils import RWGraph


def read_train_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    data_path = os.path.join(data_dir, dataset, 'train.txt')
    edge_data_by_type = dict()  # 每个type对应到的相连接节点
    all_nodes = list()  # 所有节点的集合
    print("We are loading data from:" + data_path)
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
    print("We are loading data from:" + data_path)
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
    print("We are loading data from:" + data_path)
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
    print('We are loading node type from:' + data_path)
    with open(data_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def get_G_from_edges(edges):
    edge_dict = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)
        edge_dict[v].add(u)
    return edge_dict  # 每个节点和它相连接的节点


def generate_walks(network_data, num_walks, walk_length, schema, data_dir=os.path.join(os.path.abspath('.'), 'data'),
                   dataset='amazon', num_workers=2):
    if schema is not None:
        node_type = read_node_type(data_dir, dataset)
    else:
        node_type = None

    all_walks = []  # 所有游走的list
    for layer_id, layer_name in enumerate(network_data):
        tmp_data = network_data[layer_name]  # 每个type对应到的点边信息
        # start to do the random walk on a layer
        # get_G_from_edges(tmp_data): 每个节点对应到相连接的点
        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type, num_workers)  # RandomWalk Graph
        print('Generating random walks for layer', layer_id)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)  # 生成随机游走的序列; 每个节点游走次数; 游走长度;

        all_walks.append(layer_walks)

        print('Finish generating the walks')

    return all_walks


def generator_pairs(all_walks, vocab, window_size):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        print('Generator training pairs for layer', layer_id)
        for walk in tqdm(walks):
            for i in range(len(walk)):  # 每个单词循环
                for j in range(1, skip_window + 1):  # 向前向后的窗口长度
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))  # 向前窗口涉及到的单词
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))  # 向后窗口涉及到的单词
    return pairs  # 所有单词上线文的索引, type


def load_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    training_data_by_type = read_train_data(data_dir, dataset)
    valid_true_data_by_edge, valid_false_data_by_edge = read_test_data(data_dir, dataset, file_name='valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = read_test_data(data_dir, dataset, file_name='test.txt')
