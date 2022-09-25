import collections
import pandas as pd
import networkx as nx


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


class HeteroGraph(object):
    def __init__(self, gidx=[], ntypes=['_N'], etypes=['_E'], node_frames=None, edge_frames=None):
        assert len(gidx) == len(ntypes)
        self.graph_idx = gidx
        self.node_types = ntypes
        self.edge_types = etypes
        self.node_features = node_frames
        self.edge_features = edge_frames

    def _relation_to_adj(self):
        pass

    def __repr__(self):
        pass
