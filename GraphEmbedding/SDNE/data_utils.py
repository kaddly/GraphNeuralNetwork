import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import networkx as nx
from graph_utils import preprocess_nxgraph


def read_wiki(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])


def creat_A_L(graph, node2idx):
    node_size = graph.number_of_nodes()
    A_data = []
    A_row_index = []
    A_col_index = []

    for edge in graph.edges():
        v1, v2 = edge
        edge_weight = graph[v1][v2].get('weight', 1)

        A_data.append(edge_weight)
        A_row_index.append(node2idx[v1])
        A_col_index.append(node2idx[v2])

    A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size))
    A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                       shape=(node_size, node_size))
    D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
    L = D - A_
    return A, L


def batchify(data):
    idx = [i for _, _, i in data]
    A, L = [], []
    for a, l, _ in data:
        A.append(a.tolist())
        L.append(l[idx].tolist())
    return torch.Tensor(A), torch.Tensor(L)


def load_wiki(data_dir, batch_size):
    G = read_wiki(data_dir)
    print(G)
    idx2node, node2idx = preprocess_nxgraph(G)
    A, L = creat_A_L(G, node2idx)
    A, L = A.toarray(), L.toarray()
    idxs = list(range(len(idx2node)))

    class My_dataset(Dataset):
        def __init__(self, A, L, idxs):
            assert len(A) == len(L) == len(idxs)
            self.A = A
            self.L = L
            self.idxs = idxs

        def __getitem__(self, item):
            return self.A[item], self.L[item], self.idxs[item]

        def __len__(self):
            return len(A)

    dataset = My_dataset(A, L, idxs)
    if batch_size > len(idx2node):
        data_iter = DataLoader(dataset, batch_size=len(idx2node), shuffle=True, collate_fn=batchify)
    else:
        data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=batchify)
    return data_iter, G, idx2node, node2idx
