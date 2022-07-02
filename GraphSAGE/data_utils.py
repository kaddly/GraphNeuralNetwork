import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import defaultdict
import random


def read_pubmed_data(data_dir):
    pubmed_cite_file = os.path.join(data_dir, 'Pubmed-Diabetes.DIRECTED.cites.tab')
    pubmed_content_file = os.path.join(data_dir, 'Pubmed-Diabetes.NODE.paper.tab')

    feat_data = []
    labels = []  # label sequence of node
    node_map = {}  # map node to Node_ID

    with open(pubmed_content_file, 'r', encoding='UTF-8') as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels.append(int(info[1].split("=")[1]) - 1)
            tmp_list = [0] * (len(feat_map) - 2)
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                tmp_list[feat_map[word_info[0]]] = float(word_info[1])
            feat_data.append(tmp_list)

    adj_lists = defaultdict(set)
    with open(pubmed_cite_file, 'r', encoding='UTF-8') as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    assert len(feat_data) == len(adj_lists) == len(labels)
    return feat_data, labels, adj_lists


def train_test_split(node_nums, test_split=0.3, val_split=0.2):
    test_size = int(node_nums * test_split)
    val_size = int(node_nums * val_split)
    train_size = node_nums - (test_size + val_size)
    return train_size, val_size, test_size


def get_context_nodes(nodes, adj_lists, window_size):
    all_contexts = []
    for node in nodes:
        contexts = list(adj_lists[node])
        if len(contexts) > window_size:
            all_contexts.append(contexts[:window_size])
        else:
            all_contexts.append(random.choices(contexts, k=window_size))
    return all_contexts


def get_negative_nodes(all_contexts, all_nodes, K):
    all_negatives = []
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = random.choice(all_nodes)
            if neg not in contexts and neg not in negatives:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def adj_nodes_pad(X, pad_len, value=0):
    """将层邻居矩阵pad到同一维度"""
    neighs_len = len(X[0])
    adj_nodes_len = len(X)
    pad_data = [value] * neighs_len
    X += (pad_len - adj_nodes_len) * [pad_data]
    return X


def get_layer_adj_nodes(nodes, adj_lists, num_layers, num_neighs, is_gcn):
    layer_neigh_nodes = defaultdict(list)
    layer_nodes_map = defaultdict(dict)
    layer_nodes = set()
    layer_center_nodes = [nodes]
    for i in range(num_layers):
        for idx, node in enumerate(nodes):
            layer_nodes_map[i][node] = idx
            neighs = adj_lists[node]
            if len(neighs) > num_neighs:
                sample_neighs = random.sample(list(neighs), k=num_neighs)
            else:
                sample_neighs = random.choices(list(neighs), k=num_neighs)
            if is_gcn:
                sample_neighs.append(node)
            else:
                layer_nodes.add(node)
            layer_neigh_nodes[i].append(sample_neighs)
            layer_nodes = layer_nodes.union(set(sample_neighs))
        nodes = layer_nodes
        layer_center_nodes.append(list(nodes))
        layer_nodes = set()
    center_nodes, neigh_nodes_map = [], []
    for i in reversed(range(num_layers)):
        if i == num_layers - 1:
            neigh_nodes_map.append(layer_neigh_nodes[i])
            pad_len = len(layer_neigh_nodes[i])
            center_nodes.append(layer_center_nodes[i])
        else:
            nodes_map = layer_nodes_map[i + 1]
            sample_neigh_nodes = layer_neigh_nodes[i]
            sample_neigh_nodes = list(map(lambda x: list(map(lambda i: nodes_map[i], x)), sample_neigh_nodes))
            neigh_nodes_map.append(adj_nodes_pad(sample_neigh_nodes, pad_len, -1))
            sample_center_nodes = list(map(lambda x: nodes_map[x], layer_center_nodes[i]))
            center_nodes.append(sample_center_nodes+[-1 for _ in range(pad_len-len(sample_center_nodes))])
    return neigh_nodes_map, center_nodes


class collate_fn:
    def __init__(self, adj_lists, feat_data, num_layers, num_neighs, is_gcn, is_unsupervised):
        self.adj_lists = adj_lists
        self.feat_data = torch.Tensor(feat_data)
        self.num_layers = num_layers
        self.num_neighs = num_neighs
        self.is_gcn = is_gcn
        self.is_unsupervised = is_unsupervised

    def __call__(self, data):
        if self.is_unsupervised:
            center_nodes, contexts_negatives, batch_labels = [], [], []
            for node, contexts, negatives in data:
                center_nodes.append(node)
                contexts_negatives.extend(contexts + negatives)
                batch_labels.append([1] * len(contexts) + [0] * len(negatives))
            center_neigh_nodes_map, center_nodes_map = get_layer_adj_nodes(center_nodes, self.adj_lists,
                                                                           self.num_layers, self.num_neighs,
                                                                           self.is_gcn)
            center_neigh_nodes_map = torch.tensor(center_neigh_nodes_map)
            center_nodes_map = torch.tensor(center_nodes_map)

            contexts_negatives_neigh_nodes_map, contexts_negatives_nodes_map = get_layer_adj_nodes(
                contexts_negatives, self.adj_lists, self.num_layers, self.num_neighs, self.is_gcn)
            contexts_negatives_neigh_nodes_map = torch.tensor(contexts_negatives_neigh_nodes_map)
            contexts_negatives_nodes_map = torch.tensor(contexts_negatives_nodes_map)

            return (torch.embedding(self.feat_data, center_nodes_map[0]), center_nodes_map[1:], \
                    torch.embedding(self.feat_data, center_neigh_nodes_map[0]), center_neigh_nodes_map[1:], \
                    torch.embedding(self.feat_data, contexts_negatives_nodes_map[0]), contexts_negatives_nodes_map[1:], \
                    torch.embedding(self.feat_data,
                                    contexts_negatives_neigh_nodes_map[0]),
                    contexts_negatives_neigh_nodes_map[1:]), torch.tensor(batch_labels)

        else:
            data = list(map(list, zip(*data)))
            batch_labels = torch.tensor(data[1])
            neigh_nodes_map, nodes_map = get_layer_adj_nodes(data[0], self.adj_lists, self.num_layers, self.num_neighs,
                                                             self.is_gcn)
            neigh_nodes_map = torch.tensor(neigh_nodes_map)
            nodes_map = torch.tensor(nodes_map)
            return (torch.embedding(self.feat_data, nodes_map[0]), nodes_map[1:], \
                   torch.embedding(self.feat_data, neigh_nodes_map[0]), neigh_nodes_map[1:]), batch_labels


def load_pubmed_data(data_dir, batch_size, num_layers, num_neighs, window_size=5, num_noise_words=5, is_gcn=False,
                     is_unsupervised=True):
    feat_data, labels, adj_lists = read_pubmed_data(data_dir)
    nodes = list(range(len(adj_lists)))
    train_size, val_size, test_size = train_test_split(len(adj_lists))
    batchify = collate_fn(adj_lists, feat_data, num_layers, num_neighs, is_gcn, is_unsupervised)
    if is_unsupervised:
        train_contexts = get_context_nodes(nodes[:train_size], adj_lists, window_size)
        val_contexts = get_context_nodes(nodes[train_size:train_size + val_size], adj_lists, window_size)
        test_contexts = get_context_nodes(nodes[-test_size:], adj_lists, window_size)

        train_negatives = get_negative_nodes(train_contexts, nodes, num_noise_words)
        val_negatives = get_negative_nodes(val_contexts, nodes, num_noise_words)
        test_negatives = get_negative_nodes(test_contexts, nodes, num_noise_words)

        class pubmed_dataset(Dataset):
            def __init__(self, nodes, contexts, negatives):
                assert len(nodes) == len(contexts) == len(negatives)
                self.nodes = nodes
                self.contexts = contexts
                self.negatives = negatives

            def __getitem__(self, item):
                return self.nodes[item], self.contexts[item], self.negatives[item]

            def __len__(self):
                return len(self.nodes)

        train_dataset = pubmed_dataset(nodes[:train_size], train_contexts, train_negatives)
        val_dataset = pubmed_dataset(nodes[train_size:train_size + val_size], val_contexts, val_negatives)
        test_dataset = pubmed_dataset(nodes[-test_size:], test_contexts, test_negatives)

        train_iter = DataLoader(train_dataset, batch_size, collate_fn=batchify)
        val_iter = DataLoader(val_dataset, batch_size, collate_fn=batchify)
        test_iter = DataLoader(test_dataset, batch_size, collate_fn=batchify)

    else:
        class pubmed_dataset(Dataset):
            def __init__(self, nodes, labels):
                assert len(nodes) == len(labels)
                self.labels = labels
                self.nodes = nodes

            def __getitem__(self, item):
                return self.nodes[item], self.labels[item]

            def __len__(self):
                return len(self.nodes)

        train_dataset = pubmed_dataset(nodes[:train_size], labels[:train_size])
        val_dataset = pubmed_dataset(nodes[train_size:train_size + val_size], labels[train_size:train_size + val_size])
        test_dataset = pubmed_dataset(nodes[-test_size:], labels[-test_size:])

        train_iter = DataLoader(train_dataset, batch_size, collate_fn=batchify)
        val_iter = DataLoader(val_dataset, batch_size, collate_fn=batchify)
        test_iter = DataLoader(test_dataset, batch_size, collate_fn=batchify)
    return train_iter, val_iter, test_iter
