import os
import pandas as pd
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils.graph_utils import procession_graph, HeteroGraph, count_corpus
from utils.sample_utils import RandomWalker, RandomGenerator


def read_JData(data_dir=os.path.join('../', 'data'), sample_num=10000):
    edge_f = pd.read_csv(os.path.join(data_dir, 'data_action.csv'))
    user_features = pd.read_csv(os.path.join(data_dir, 'user_features.csv'))
    nodes_features = pd.read_csv(os.path.join(data_dir, 'item_features.csv'))
    edge_f = edge_f.sample(sample_num)
    user_features = user_features[user_features['node_id'].isin(list(edge_f['user_id']))]
    nodes_features = nodes_features[nodes_features['node_id'].isin(list(edge_f['sku_id']))]
    idx_to_users, user_to_idx, idx_to_items, item_to_idx = procession_graph(edge_f)
    user_item_src = [user_to_idx.get(user_id) for user_id in edge_f['user_id']]
    user_item_dst = [item_to_idx.get(item_id) for item_id in edge_f['sku_id']]
    HG = HeteroGraph([user_item_src, user_item_dst], edge_types=['user', 'item'], meta_path=['user', 'item', 'user'])
    return HG, user_features, nodes_features, idx_to_users, user_to_idx, idx_to_items, item_to_idx


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


def load_JData(batch_size=128, max_window_size=5, num_noise_words=2):
    HG, user_features, nodes_features, idx_to_users, user_to_idx, idx_to_items, item_to_idx = read_JData()
    generator = RandomWalker(HG)
    all_contexts = generator.simulate_walks(num_walks=10, meta_path=['user', 'item', 'user', 'item', 'user'], workers=2)
    print('load contexts:'+str(len(all_contexts)))
    subsampled, counter = subsample(all_contexts)
    print('load subsampled contexts:' + str(len(subsampled)))
    corpus = [[node2idx[token] for token in line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    print('load all_centers:'+str(len(all_centers)))
    all_negatives = get_negative(all_contexts, idx2node, counter, num_noise_words)
    print('load all_negatives:' + str(len(all_negatives)))


load_JData()
