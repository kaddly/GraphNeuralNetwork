import torch
from torch.utils.data import DataLoader, Dataset
import math
from .token_utils import Vocab, count_corpus
from joblib import Parallel, delayed
import itertools
import random
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(data_dir):
    G = nx.read_edgelist(data_dir, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    nx.draw(G, node_size=10, font_size=10, font_color="blue", font_weight="bold")
    plt.show()


# plot_graph('../data/wiki/Wiki_edgelist.txt')

def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


class RandomWalker:
    def __init__(self, G):
        self.G = G

    def deepwalk_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
        return walks


def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元'<UNK>'
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return (random.randint(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences], counter)


# sentences = read_ptb()
# vocab = Vocab(sentences, min_freq=10)
# subsampled, counter = subsample(sentences, vocab)
# show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence', 'count', sentences, subsampled)
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


# generator = RandomGenerator([2, 3, 4])
# print([generator.draw() for _ in range(10)])

def get_negative(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
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


def load_data_wiki(sentences, batch_size, max_window_size, num_noise_words):
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negative(all_contexts, vocab, counter, num_noise_words)

    class WikiDataset(Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, item):
            return (self.centers[item], self.negatives[item], self.contexts[item])

        def __len__(self):
            return len(self.centers)

    dataset = WikiDataset(all_centers, all_contexts, all_negatives)
    data_iter = DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify)
    return data_iter, vocab
