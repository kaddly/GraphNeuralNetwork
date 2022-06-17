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
    rand_indices = list(range(node_nums))
    random.shuffle(rand_indices)

    test_size = int(node_nums * test_split)
    val_size = int(node_nums * val_split)
    train_size = node_nums - (test_size + val_size)

    test_indexs = rand_indices[:test_size]
    val_indexs = rand_indices[test_size:(test_size + val_size)]
    train_indexs = rand_indices[(test_size + val_size):]

    return train_indexs, val_indexs, test_indexs


class pubmed_dataset(Dataset):
    def __init__(self, feat_data, labels, adj_lists):
        self.feat_data = torch.Tensor(feat_data)
        self.labels = torch.Tensor(labels)

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.labels)


def load_pubmed_data(data_dir, batch_size):
    feat_data, labels, adj_lists = read_data(data_dir)
    print()
