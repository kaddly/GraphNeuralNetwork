import os
import random
import torch
import networkx as nx
from graph_utils import preprocess_nxgraph, RandomGenerator


def read_wiki(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.Graph())


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""

    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def gen_sample_data(G, power):
    idx2node, node2idx = preprocess_nxgraph(G)
    node_size = G.number_of_nodes()
    node_degree = torch.zeros(node_size)
    for edge in G.edges():
        node_degree[node2idx[edge[0]]] += G[edge[0]][edge[1]].get('weight', 1.0)
    total_sum = torch.sum(torch.Tensor([torch.pow(node_degree[i], power) for i in range(node_size)]))
    norm_prob = torch.Tensor([torch.pow(node_degree[j], power) / total_sum for j in range(node_size)])
    generator = RandomGenerator(norm_prob)



gen_sample_data(read_wiki("../data/wiki/Wiki_edgelist.txt"), 0.75)
