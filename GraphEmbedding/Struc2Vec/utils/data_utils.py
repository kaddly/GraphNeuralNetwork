import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from .graph_utils import preprocess_nxgraph, preprocess_struct, count_corpus
from .sample_utils import RandomWalker, RandomGenerator


def read_flight(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])


def plot_graph(G):
    print(G)
    nx.draw(G, node_size=10, font_size=2, font_color="blue", font_weight="bold")
    plt.show()


def subsample(sentences):
    """下采样高频词"""
    # 排除未知词元'<UNK>'
    sentences = [[token for token in line] for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return (random.randint(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences], counter)


def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词与上下文单词"""
    centers, context = [], []
    for line in corpus:
        # 形成“中心词-上下文词”对，每个句子至少需要两个单词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))

            # 从上下文词中排除中心词
            indices.remove(i)
            context.append([line[idx] for idx in indices])
    return centers, context


def get_negative(all_contexts, idx2node, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[idx2node[i]] ** 0.75 for i in range(0, len(idx2node))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下⽂词
            if neg not in contexts:
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
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1))
            , torch.tensor(contexts_negatives)
            , torch.tensor(masks)
            , torch.tensor(labels))


def load_flight_data(data_dir, batch_size, num_walks, walk_length, workers, max_window_size, num_noise_words,
                     opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                     opt3_num_layers=None):
    G = read_flight(data_dir)
    idx2node, node2idx = preprocess_nxgraph(G)
    layer_adj, layers_accept, layers_alias, gamma = preprocess_struct(G, idx2node, node2idx, opt1_reduce_len,
                                                                      opt2_reduce_sim_calc, opt3_num_layers)
    walker = RandomWalker(idx2node, layer_adj, layers_alias, layers_accept, gamma)
    all_contexts = walker.simulate_walks(num_walks, walk_length, workers=workers)
    print('load contexts:' + str(len(all_contexts)))
    subsampled, counter = subsample(all_contexts)
    print('load subsampled contexts:' + str(len(subsampled)))
    corpus = [[node2idx[token] for token in line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    print('load all_centers:' + str(len(all_centers)))
    all_negatives = get_negative(all_contexts, idx2node, counter, num_noise_words)
    print('load all_negatives:' + str(len(all_negatives)))

    class FlightDataset(Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, item):
            return (self.centers[item], self.negatives[item], self.contexts[item])

        def __len__(self):
            return len(self.centers)

    dataset = FlightDataset(all_centers, all_contexts, all_negatives)
    data_iter = DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify)
    return data_iter, idx2node, node2idx, G
