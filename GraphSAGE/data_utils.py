import os
from collections import defaultdict
import random
from torch.utils.data import Dataset, DataLoader


def read_data(data_dir):
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
    with open(pubmed_cite_file) as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    assert len(feat_data) == len(labels) == len(adj_lists)
    return feat_data, labels, adj_lists


def train_test_split(node_nums, test_split=0.3, val_split=0.6):
    test_size = int(node_nums * test_split)
    val_size = int(node_nums * val_split)
    train_size = node_nums - (test_size + val_size)
    nodes = list(range(node_nums))
    random.shuffle(nodes)
    return nodes[:train_size], nodes[train_size:train_size + val_size], nodes[-test_size:]


def get_context(nodes, adj_lists, max_window_size):
    all_contexts = []
    for node in nodes:
        contexts = list(adj_lists[node])
        window_size = random.randint(1, max_window_size)
        if len(contexts) > window_size:
            all_contexts.append(contexts[:window_size])
        else:
            all_contexts.append(contexts)
    return all_contexts


def get_negative(all_contexts, node_nums, K):
    all_nodes = list(range(node_nums))
    all_negatives = []
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = random.choice(all_nodes)
            if neg not in contexts and neg not in negatives:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    """返回带有负采样的跳元模型的⼩批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * len(context)+[len(negative)]*len(negative) + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (centers, contexts_negatives, masks, labels)


def load_pubmed_data(data_dir, batch_size, Unsupervised=True, max_window_size=5, num_noise_words=5):
    feat_data, labels, adj_lists = read_data(data_dir)
    train_nodes, val_nodes, test_nodes = train_test_split(len(adj_lists))
    if Unsupervised:
        train_contexts, val_contexts, test_contexts = [get_context(nodes, adj_lists, max_window_size) for nodes in
                                                       [train_nodes, val_nodes, test_nodes]]
        train_negatives, val_negatives, test_negatives = [get_negative(nodes, len(adj_lists), num_noise_words) for nodes
                                                          in [train_contexts, val_contexts, test_contexts]]

        class PubmedDataset(Dataset):
            def __init__(self, centers, contexts, negatives):
                assert len(centers) == len(contexts) == len(negatives)
                self.centers = centers
                self.contexts = contexts
                self.negatives = negatives

            def __getitem__(self, item):
                return (self.centers[item], self.negatives[item], self.contexts[item])

            def __len__(self):
                return len(self.centers)

        train_dataset = PubmedDataset(train_nodes, train_contexts, train_negatives)
        val_dataset = PubmedDataset(val_nodes, val_contexts, val_negatives)
        test_dataset = PubmedDataset(test_nodes, test_contexts, test_negatives)
        train_iter = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=batchify)
        val_iter = DataLoader(val_dataset, batch_size, shuffle=True, collate_fn=batchify)
        test_iter = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=batchify)
    else:
        class PubmedDataset(Dataset):
            def __init__(self, nodes):
                self.nodes = nodes

            def __getitem__(self, item):
                return self.nodes[item]

            def __len__(self):
                return len(self.nodes)

        train_dataset = PubmedDataset(train_nodes)
        val_dataset = PubmedDataset(val_nodes)
        test_dataset = PubmedDataset(test_nodes)
        train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
        val_iter = DataLoader(val_dataset, batch_size, shuffle=True)
        test_iter = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_iter, val_iter, test_iter, feat_data, labels, adj_lists
