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
    def __init__(self, graph_idx=[], edge_types=['_U', '_I'], meta_path=[['_E', '_A', '_E']], node_frames=None, edge_frames=None):
        assert len(graph_idx) == len(edge_types)
        self.graph_idx = graph_idx
        self.edge_types = edge_types
        self.meta_path = meta_path
        self.node_features = node_frames
        self.edge_features = edge_frames
        self.node_index_map = self._node_map_index()

    def _node_map_index(self):
        node_type_index = {}
        node_types = self.edge_types if isinstance(self.edge_types[0], str) else [node for node_type in self.edge_types
                                                                                  for node in node_type]
        relations = self.graph_idx if isinstance(self.edge_types[0], str) else [relation for relations in self.graph_idx
                                                                                for relation in relations]
        for i, node_type in node_types:
            if node_type not in node_type_index:
                node_type_index[node_type] = set(relations[i])
            else:
                if len(set(relations[i])) > len(node_type_index[node_type]):
                    node_type_index[node_type] = set(relations[i])
        return node_type_index

    def relation_to_adj(self):
        HG_adj = {}
        if not isinstance(self.edge_types[0], str):
            for r, edges in zip(self.graph_idx, self.edge_types):
                adj = self._single_relation_to_adj(r, edges)
                HG_adj[edges[0]+'->'+edges[1]] = adj
        else:
            adj = self._single_relation_to_adj(self.graph_idx, self.edge_types)
            HG_adj[self.edge_types[0]+'->'+self.edge_types[1]] = adj

    def _single_relation_to_adj(self, relation, edge_types):
        src = self.node_index_map[edge_types[0]]
        dst = self.node_index_map[edge_types[1]]
        relation_matrix = [[0 for _ in range(len(src))] for _ in range(len(dst))]
        for (x_index, y_index) in relation:
            relation_matrix[x_index][y_index] = 1
        return relation_matrix

    def __repr__(self):
        ret = ('Graph(num_nodes={node},\n'
               '      num_edges={edge},\n'
               '      metagraph={meta})')
        nnode_dict = {self.ntypes[i]: self._graph.number_of_nodes(i)
                      for i in range(len(self.ntypes))}
        nedge_dict = {self.canonical_etypes[i]: self._graph.number_of_edges(i)
                      for i in range(len(self.etypes))}
        meta = str(self.metagraph().edges(keys=True))
        return ret.format(node=nnode_dict, edge=nedge_dict, meta=meta)
