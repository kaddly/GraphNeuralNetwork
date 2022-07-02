import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import defaultdict
import random
from sample_utils import multihop_sampling


def read_pubmed_data(data_dir, dataset='pubmed'):
    data_dir = os.path.join(data_dir, dataset)
    pubmed_cite_file = os.path.join(data_dir, 'Pubmed-Diabetes.DIRECTED.cites.tab')
    pubmed_content_file = os.path.join(data_dir, 'Pubmed-Diabetes.NODE.paper.tab')

    feat_data = []
    labels = []  # label sequence of node
    node_map = {}  # map node to Node_ID

    with open(pubmed_content_file, 'r', encoding='UTF-8') as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels.append(int(info[1].split("=")[1]) - 1)
            tmp_list = [0] * (len(feat_map) - 2)
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                tmp_list[feat_map[word_info[0]]] = float(word_info[1])
            feat_data.append(tmp_list)

    adj_lists = defaultdict(set)
    with open(pubmed_cite_file, 'r', encoding='UTF-8') as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    assert len(feat_data) == len(adj_lists) == len(labels)
    return feat_data, labels, adj_lists


def train_test_split(node_nums, test_split=0.6, val_split=0.3):
    test_size = int(node_nums * test_split)
    val_size = int(node_nums * val_split)
    train_size = node_nums - (test_size + val_size)
    return train_size, val_size, test_size


class collate_fn:
    def __init__(self, adj_lists, feat_data, num_neighbor_list):
        self.adj_lists = adj_lists
        self.feat_data = torch.Tensor(feat_data)
        self.num_neighbor_list = num_neighbor_list

    def __call__(self, data):
        pass


class Pubmed_dataset(Dataset):
    def __init__(self, nodes, labels, **kwargs):
        super(Pubmed_dataset, self).__init__(**kwargs)
        self.nodes = nodes
        self.labels = labels

    def __getitem__(self, item):
        return self.nodes[item], self.labels

    def __len__(self):
        return len(self.nodes)


def load_pubmed_data(data_dir, batch_size, val_split, test_split):
    feat_data, labels, adj_lists = read_pubmed_data(data_dir)
    all_nodes = list(range(len(feat_data)))
    train_size, val_size, test_size = train_test_split(len(all_nodes), val_split=val_split, test_split=test_split)

    train_dataset = Pubmed_dataset(all_nodes[:train_size], labels[:train_size])
    val_dataset = Pubmed_dataset(all_nodes[train_size:train_size + val_size], labels[train_size:train_size+val_size])
    test_dataset = Pubmed_dataset(all_nodes[-test_size:], labels[-test_size:])

    train_iter = DataLoader(train_dataset, batch_size)
    val_iter = DataLoader(val_dataset, batch_size)
    test_iter = DataLoader(test_dataset, batch_size)

    return train_iter, val_iter, test_iter, feat_data, labels, adj_lists
