import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.graph_utils import procession_graph, HeteroGraph


def read_JData(data_dir=os.path.join('../', 'data'), sample_num=10000):
    edge_f = pd.read_csv(os.path.join(data_dir, 'data_action.csv'))
    user_features = pd.read_csv(os.path.join(data_dir, 'user_features.csv'))
    nodes_features = pd.read_csv(os.path.join(data_dir, 'item_features.csv'))
    edge_f = edge_f.sample(sample_num)
    idx_to_users, user_to_idx, idx_to_items, item_to_idx = procession_graph(edge_f)
    user_item_src = [user_to_idx.get(user_id) for user_id in edge_f['user_id']]
    user_item_dst = [item_to_idx.get(item_id) for item_id in edge_f['sku_id']]
    HG = HeteroGraph([user_item_src, user_item_dst], edge_types=['user', 'item'], meta_path=['user', 'item', 'user'])


read_JData()
