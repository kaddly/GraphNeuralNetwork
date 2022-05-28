import numpy as np
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from graph_utils import preprocess_nxgraph


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


def get_sampling_weights(G, node2idx, power):
    node_size = G.number_of_nodes()
    node_degree = np.zeros(node_size)
    # 求度
    for edge in G.edges():
        node_degree[node2idx[edge[0]]] += G[edge[0]][edge[1]].get('weight', 1.0)
    total_sum = sum([math.pow(node_degree[i], power) for i in range(node_size)])
    norm_prob = [float(math.pow(node_degree[j], power)) / total_sum for j in range(node_size)]
    return norm_prob


class Wiki_dataset(Dataset):
    def __init__(self, G):
        self.G = G
        self.idx2node, self.node2idx = preprocess_nxgraph(G)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


print(sum(get_sampling_weights(read_wiki("../data/wiki/Wiki_edgelist.txt"), 0.75)))
