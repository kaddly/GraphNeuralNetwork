import collections
import pandas as pd


def procession_graph(edges: pd.DataFrame, sample_num=10000):
    edges = edges.sample(sample_num)
    users = set()
    items = set()
    for row in edges.iterrows():
        users.add(row['user_id'])
        items.add(row['sku_id'])
    user_to_idx = {x: i for i, x in enumerate(users)}  # user编号
    item_to_idx = {x: i for i, x in enumerate(items)}  # item编号
    return list(users), user_to_idx, list(items), item_to_idx


class HeteroGraph:
    def __init__(self, gidx=[], ntypes=['_N'], etypes=['_E'], node_frames=None, edge_frames=None):
        pass
