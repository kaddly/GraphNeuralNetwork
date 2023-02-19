import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils.graph_utils import BipartiteGraph


def read_data(data_set, file_name):
    users_list, items_list, weights_list = [], [], []
    file_path = os.path.join(os.path.abspath('.'), 'data', data_set, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            user, item, edge_weight = line.strip().split()
            users_list.append(user)
            items_list.append(item)
            weights_list.append(float(edge_weight))
    return [users_list, items_list], weights_list


def generator_explicit_relations():
    pass


def generator_implicit_relations(meta_path, BG: BipartiteGraph, vocab, maxT, minT, percentage, hits_dict):
    assert meta_path[0] == meta_path[-1]
    adj = BG.meta_path_adj(meta_path)


class BipartiteDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def load_data(args):
    relation_list, weights_list = read_data(args.data_set, args.file_name)
    BG = BipartiteGraph(relation_list, edge_types=['U', 'I'], meta_path=args.meta_path, edge_frames=weights_list,
                        is_digraph=args.is_digraph)
    user_vocab, item_vocab = BG.get_vocab
