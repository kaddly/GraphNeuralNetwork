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

    def _node_map_index(self):
        node_type_index = {}
        node_types = self.node_types if isinstance(self.node_types[0], str) else [node for node_type in self.node_types for node in node_type]
        relations = self.graph_idx if isinstance(self.node_types[0], str) else [relation for relations in self.graph_idx for relation in relations]
        for i, node_type in node_types:
            src_set = set(relations[i])

    def relation_to_adj(self):
        HG_adj = {}
        if not isinstance(self.node_types[0], str):
            for r, nodes in zip(self.graph_idx, self.node_types):
                self._single_relation_to_adj(r, nodes)
        else:
            self._single_relation_to_adj(self.graph_idx, self.node_types)

    def _single_relation_to_adj(self, relation):
        src_len = set(relation[0])
        dst_len = set(relation[1])
        relation_matrix = [[0 for _ in range(len(src_len))] for _ in range(len(dst_len))]
        for (x_index, y_index) in relation:
            relation_matrix[x_index][y_index] = 1
        return relation_matrix

    def __repr__(self):
        pass
