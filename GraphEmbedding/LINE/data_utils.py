import numpy as np
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from graph_utils import preprocess_nxgraph


def read_wiki(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.Graph(), )


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""

    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(0, len(sampling_weights)))
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


def get_centers_and_contexts(G):
    nodes, contexts, labels = [], [], []
    for node in G.nodes():
        nodes.append(int(node))
        contexts.append(list(map(int, G.neighbors(node))))
        labels.append([G[node][node_nb].get('weight', 1.0) for node_nb in G.neighbors(node)])
    return nodes, contexts, labels


def get_sampling_weights(G, node2idx, power):
    node_size = G.number_of_nodes()
    node_degree = np.zeros(node_size)
    # 求度
    for edge in G.edges():
        node_degree[node2idx[edge[0]]] += G[edge[0]][edge[1]].get('weight', 1.0)
    total_sum = sum([math.pow(node_degree[i], power) for i in range(node_size)])
    norm_prob = [float(math.pow(node_degree[j], power)) / total_sum for j in range(node_size)]
    return norm_prob


def get_negative(G, all_contexts, idx2node, node2idx, K):
    sampling_weights = get_sampling_weights(G, node2idx, 0.75)
    all_negatives = []
    generator = RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下⽂词
            if idx2node[neg] not in contexts:
                negatives.append(idx2node[neg])
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    """返回带有负采样的跳元模型的⼩批量样本"""
    max_len = max(len(c) + len(n) for _, c, _, n, _ in data)
    centers, contexts_negatives, masks, labels, weights = [], [], [], [], []
    for center, context, label, negative, weight in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [label + [0.0] * (max_len - len(context))]
        weights += [weight]
    return (torch.tensor(centers).reshape((-1, 1))
            , torch.tensor(contexts_negatives)
            , torch.tensor(masks)
            , torch.tensor(labels)
            , torch.tensor(weights))


class Wiki_dataset(Dataset):
    def __init__(self, centers, contexts, labels, negatives, pd):
        assert len(centers) == len(contexts) == len(negatives) == len(labels) == len(pd)
        self.centers = centers
        self.contexts = contexts
        self.labels = labels
        self.negatives = negatives
        self.weight = [pd[str(center)] for center in centers]
        print("load data:"+str(len(centers)))

    def __getitem__(self, item):
        return self.centers[item], self.contexts[item], self.labels[item], self.negatives[item], self.weight[item]

    def __len__(self):
        return len(self.centers)


def load_data_wiki(data_dir, batch_size, num_noise_words):
    G = read_wiki(data_dir)
    idx2node, node2idx = preprocess_nxgraph(G)
    nodes, all_contexts, all_labels = get_centers_and_contexts(G)
    all_negatives = get_negative(G, all_contexts, idx2node, node2idx, num_noise_words)
    dataset = Wiki_dataset(nodes, all_contexts, all_labels, all_negatives, nx.pagerank(G))
    data_iter = DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify)
    return idx2node, node2idx, data_iter
