import os
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, DataLoader


def read_data(data_dir):
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
    with open(pubmed_cite_file) as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    assert len(feat_data) == len(labels) == len(adj_lists)
    return feat_data, labels, adj_lists


def train_test_split(node_nums, test_split=0.3, val_split=0.6):
    test_size = int(node_nums * test_split)
    val_size = int(node_nums * val_split)
    train_size = node_nums - (test_size + val_size)
    return train_size, val_size, test_size


def sample_neigh(adj_lists, sample_neigh_num=10):
    nodes, sample_nodes = [], []
    for node, neigh_nodes in adj_lists.items():
        nodes.append(node)
        if sample_neigh_num < len(neigh_nodes):
            # 有无放回抽样
            neigh_nodes = random.sample(neigh_nodes, sample_neigh_num)
        else:
            # 有放回抽样
            neigh_nodes = random.choices(neigh_nodes, sample_neigh_num)
        sample_nodes.append(neigh_nodes)
    return nodes, sample_nodes


class pubmed_dataset(Dataset):
    def __init__(self, nodes, samp_neighs, labels):
        assert len(nodes) == len(samp_neighs) == len(labels)
        print('load data:'+str(len(nodes)))
        self.nodes = torch.tensor(nodes)
        self.samp_neighs = torch.tensor(samp_neighs)
        self.labels = torch.tensor(labels)

    def __getitem__(self, item):
        return (self.nodes[item], self.samp_neighs[item]), self.labels[item]

    def __len__(self):
        return len(self.labels)


def load_pubmed_data(data_dir, batch_size, sample_neigh_num, Unsupervised=True):
    feat_data, labels, adj_lists = read_data(data_dir)
    train_size, val_size, test_size = train_test_split(len(adj_lists))
    nodes, sample_nodes = sample_neigh(adj_lists, sample_neigh_num)
    if Unsupervised:
        pass
    else:
        train_dataset = pubmed_dataset(nodes[:train_size], sample_nodes[:train_size], labels[:train_size])
        val_dataset = pubmed_dataset(nodes[train_size:train_size + val_size],
                                     sample_nodes[train_size:train_size + val_size],
                                     labels[train_size:train_size + val_size])
        test_dataset = pubmed_dataset(nodes[-test_size:], sample_nodes[-test_size:], labels[-test_size:])

        train_iter = DataLoader(train_dataset, batch_size)
        val_iter = DataLoader(val_dataset, batch_size)
        test_iter = DataLoader(test_dataset, batch_size)
    return train_iter, val_iter, test_iter, torch.Tensor(feat_data)
