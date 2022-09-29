from scipy.sparse import csr_matrix
import pandas as pd


def procession_graph(edges: pd.DataFrame):
    users = set(edges['user_id'])
    items = set(edges['sku_id'])
    user_to_idx = {x: i for i, x in enumerate(users)}  # user编号
    item_to_idx = {x: i for i, x in enumerate(items)}  # item编号
    return list(users), user_to_idx, list(items), item_to_idx


class HeteroGraph(object):
    def __init__(self, graph_idx=[], edge_types=['_U', '_I'], meta_path=['_U', '_I', '_U'], is_digraph=False,
                 node_frames=None, edge_frames=None):
        assert len(graph_idx) == len(edge_types)
        self.graph_idx = graph_idx
        self.edge_types = edge_types
        self.meta_path = meta_path
        self.is_digraph = is_digraph
        self.node_features = node_frames
        self.edge_features = edge_frames
        self.node_index_map = self._node_map_index()
        self.HG_adj = self.relation_to_adj()

    def _node_map_index(self):
        node_type_index = {}
        node_types = self.edge_types if isinstance(self.edge_types[0], str) else [node for node_type in self.edge_types
                                                                                  for node in node_type]
        relations = self.graph_idx if isinstance(self.edge_types[0], str) else [relation for relations in self.graph_idx
                                                                                for relation in relations]
        for i, node_type in enumerate(node_types):
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
                HG_adj[edges[0] + '->' + edges[1]] = adj
                if not self.is_digraph:
                    HG_adj[edges[1] + '->' + edges[0]] = adj.T
        else:
            adj = self._single_relation_to_adj(self.graph_idx, self.edge_types)
            HG_adj[self.edge_types[0] + '->' + self.edge_types[1]] = adj
            if not self.is_digraph:
                HG_adj[self.edge_types[1] + '->' + self.edge_types[0]] = adj.T
        return HG_adj

    def _single_relation_to_adj(self, relation, edge_types):
        src = self.node_index_map[edge_types[0]]
        dst = self.node_index_map[edge_types[1]]
        data = [1 for _ in range(len(relation[0]))]
        return csr_matrix((data, relation), shape=(len(src), len(dst)))

    @property
    def meta_path_adj(self, meta_path=None):
        self.meta_path = meta_path if meta_path is not None else self.meta_path
        meta_path_adj = None
        if isinstance(self.meta_path[0], str):
            for i in range(len(self.meta_path) - 1):
                if meta_path_adj is None:
                    meta_path_adj = self.HG_adj[self.meta_path[i] + '->' + self.meta_path[i + 1]]
                else:
                    meta_path_adj = meta_path_adj * self.HG_adj[self.meta_path[i] + '->' + self.meta_path[i + 1]]
                    mask = meta_path_adj > 0
                    meta_path_adj[mask] = 1
            return meta_path_adj
        else:
            meta_paths_adj = []
            for mp in self.meta_path:
                for i in range(len(mp) - 1):
                    if meta_path_adj is None:
                        meta_path_adj = self.HG_adj[mp[i] + '->' + mp[i + 1]]
                    else:
                        meta_path_adj = meta_path_adj * self.HG_adj[mp[i] + '->' + mp[i + 1]]
                        mask = meta_path_adj > 0
                        meta_path_adj[mask] = 1
                meta_paths_adj.append(meta_path_adj)
                meta_path_adj = None
            return meta_paths_adj

    def __repr__(self):
        ret = ('Graph(num_nodes={node},\n'
               '      num_edges={edge},\n'
               '      metagraph={meta})')
        nnode_dict = {node: len(index) for node, index in self.node_index_map.items()}
        nedge_dict = {edge: adj.getnnz() for edge, adj in self.HG_adj.items()}
        meta = "->".join(self.meta_path) if isinstance(self.meta_path[0], str) else ",".join(
            ["->".join(mp) for mp in self.meta_path])
        return ret.format(node=nnode_dict, edge=nedge_dict, meta=meta)
