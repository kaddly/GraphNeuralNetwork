import os
import random
import torch
import networkx as nx
from graph_utils import preprocess_nxgraph


def read_wiki(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.Graph())


def gen_sample_data(G, power):
    idx2node, node2idx = preprocess_nxgraph(G)
    node_size = G.number_of_nodes()
    node_degree = torch.zeros(node_size)
    for edge in G.edges():
        node_degree[node2idx[edge[0]]] += G[edge[0]][edge[1]].get('weight', 1.0)
    total_sum = torch.sum(torch.Tensor([torch.pow(node_degree[i], power) for i in range(node_size)]))
    norm_prob = torch.Tensor([torch.pow(node_degree[j], power) / total_sum for j in range(node_size)])
    print(norm_prob)


gen_sample_data(read_wiki("../data/wiki/Wiki_edgelist.txt"), 0.75)
