import os
import pandas as pd
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils.graph_utils import count_corpus
from utils.sample_utils import RandomGenerator


def read_meta_paths(data_dir=os.path.join('../', 'data'), sample_num=10000):
    return None


def subsample(sentences):
    """下采样高频词"""
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


def get_negative(all_contexts, idx2node, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[i] ** 0.75 for i in range(0, len(idx2node))]
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


def load_JData(batch_size=128, max_window_size=2, num_noise_words=2):
    all_contexts = read_meta_paths()
    print(f'load contexts:{len(all_contexts)}')
    subsampled, counter = subsample(all_contexts)
    print(f'load subsampled contexts:{len(subsampled)}')
    all_centers, all_contexts = get_centers_and_contexts(subsampled, max_window_size)
    print(f'load all_centers:{len(all_contexts)}')
    all_negatives = get_negative(all_contexts, counter, num_noise_words)
    print(f'load all_negatives:{len(all_negatives)}')


load_JData()
