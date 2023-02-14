import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils.graph_utils import Vocab, HeteroGraph


def read_data(data_set, file_name):
    users, items, edge_weights = set(), set(), {}
    file_path = os.path.join(os.path.abspath('.'), 'data', data_set, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            user, item, edge_weight = line.strip().split()
            if edge_weights.get(user) is None:
                edge_weights[user] = {}
            edge_weights[user][item] = float(edge_weight)
            users.add(user)
            items.add(item)
    return users, items, edge_weights


def generator_hidden_relations():
    pass


class BipartiteDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def load_data(args):
    users, items, edge_weights = read_data(args.data_set, args.file_name)
