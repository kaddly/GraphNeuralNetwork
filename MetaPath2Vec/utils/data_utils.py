import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.graph_utils import procession_graph, HeteroGraph
from utils.sample_utils import RandomWalker


def read_JData(data_dir=os.path.join('../', 'data'), sample_num=10000):
    edge_f = pd.read_csv(os.path.join(data_dir, 'data_action.csv'))
    user_features = pd.read_csv(os.path.join(data_dir, 'user_features.csv'))
    nodes_features = pd.read_csv(os.path.join(data_dir, 'item_features.csv'))
    edge_f = edge_f.sample(sample_num)
    user_features = user_features[user_features['node_id'].isin(list(edge_f['user_id']))]
    nodes_features = nodes_features[nodes_features['node_id'].isin(list(edge_f['sku_id']))]
    idx_to_users, user_to_idx, idx_to_items, item_to_idx = procession_graph(edge_f)
    user_item_src = [user_to_idx.get(user_id) for user_id in edge_f['user_id']]
    user_item_dst = [item_to_idx.get(item_id) for item_id in edge_f['sku_id']]
    HG = HeteroGraph([user_item_src, user_item_dst], edge_types=['user', 'item'], meta_path=['user', 'item', 'user'])
    return HG, user_features, nodes_features, idx_to_users, user_to_idx, idx_to_items, item_to_idx


def load_JData(batch_size=128):
    HG, user_features, nodes_features, idx_to_users, user_to_idx, idx_to_items, item_to_idx = read_JData()
    generator = RandomWalker(HG)
    contexts = generator.simulate_walks(num_walks=10, meta_path=['user', 'item', 'user', 'item', 'user'], workers=2)


load_JData()
