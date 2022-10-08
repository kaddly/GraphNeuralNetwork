import os
import math
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from utils.graph_utils import count_corpus, Vocab
from utils.sample_utils import RandomGenerator


def read_meta_paths(data_dir=os.path.join('./', 'data')):
    if not os.path.exists(os.path.join(data_dir, 'output_path.txt')):
        raise FileNotFoundError("please generate meta_paths!")
    contexts = []
    with open(os.path.join(data_dir, 'output_path.txt'), 'r') as f:
        for meta_path in f.readlines():
            if len(meta_path) == 0:
                continue
            contexts.append(meta_path.split(','))
    with open(os.path.join(data_dir, 'HG.pkl'), 'rb') as f:
        HG, idx_to_users, idx_to_items, meta_path = pickle.load(f)
    return contexts, HG, idx_to_users, idx_to_items, meta_path


def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元'<UNK>'
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return random.randint(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens)

    return [[token for token in line if keep(token)] for line in sentences], counter


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


class JDataset(Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, item):
        return self.centers[item], self.contexts[item], self.negatives[item]

    def __len__(self):
        return len(self.centers)


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


def load_JData(batch_size=128, max_window_size=4, num_noise_words=4, is_meta_path_ultra=False):
    sentences, HG, idx_to_users, idx_to_items, meta_path = read_meta_paths()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negative(all_contexts, vocab, counter, num_noise_words)
    dataset = JDataset(all_centers, all_contexts, all_negatives)
    data_iter = DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify)
    return data_iter, vocab
