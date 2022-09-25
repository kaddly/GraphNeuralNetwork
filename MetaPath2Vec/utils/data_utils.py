import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .graph_utils import procession_graph, HeteroGraph


def read_JData(data_dir=os.path.join('./', 'data')):
    edge_f = pd.read_csv(os.path.join(data_dir, 'data_action.csv'))
    user_features = pd.read_csv(os.path.join(data_dir, 'user_features.csv'))
    nodes_features = pd.read_csv(os.path.join(data_dir, 'item_features.csv'))
    idx_to_users, user_to_idx, idx_to_items, item_to_idx = procession_graph(edge_f)
    user_item_src = []
    user_item_dst = []
    for row in edge_f.iterrows():
        user_item_src.append(user_to_idx.get(row['user_id']))
        user_item_dst.append(item_to_idx.get(row['sku_id']))
    HeteroGraph()
