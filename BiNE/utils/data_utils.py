import os
import random
import math
import torch
from torch.utils.data import DataLoader, Dataset
from utils.graph_utils import BipartiteGraph
from utils.sample_utils import RandomWalker, RandomGenerator


def read_data(data_set, file_name):
    users_list, items_list, weights_list = [], [], []
    file_path = os.path.join(os.path.abspath('.'), 'data', data_set, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            user, item, edge_weight = line.strip().split()
            users_list.append(user)
            items_list.append(item)
            weights_list.append(float(edge_weight))
    return [users_list, items_list], weights_list


def generator_walks(meta_path, BG: BipartiteGraph, vocab, maxT, minT, percentage, hits_dict):
    assert meta_path[0] == meta_path[-1]
    adj = BG.meta_path_adj(meta_path)
    RW = RandomWalker(adj, vocab, maxT, minT, percentage, hits_dict)
    return RW.homogeneous_graph_random_walks()


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


def get_negative(all_contexts, vocab, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [vocab.token_counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
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


def generator_explicit_relations():
    pass


def generator_implicit_relations(corpus, vocab, max_window_size, K):
    centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negative(all_contexts, vocab, K)
    return centers, all_contexts, all_negatives


class BipartiteDataset(Dataset):
    def __init__(self, user_centers, user_contexts, user_negatives, item_centers, item_contexts, item_negatives):
        assert len(user_centers) == len(user_contexts)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def load_data(args):
    relation_list, weights_list = read_data(args.data_set, args.file_name)
    BG = BipartiteGraph(relation_list, edge_types=['U', 'I'], meta_path=args.meta_path, edge_frames=weights_list,
                        is_digraph=args.is_digraph)
    user_vocab, item_vocab = BG.get_vocab
    u_hits_dict, i_hits_dict = BG.calculate_centrality()
    user_corpus = generator_walks(meta_path=['U', 'I', 'U'], BG=BG, vocab=user_vocab, maxT=args.maxT, minT=args.minT,
                                  percentage=args.percentage, hits_dict=u_hits_dict)
    item_corpus = generator_walks(meta_path=['I', 'U', 'I'], BG=BG, vocab=item_vocab, maxT=args.maxT, minT=args.minT,
                                  percentage=args.percentage, hits_dict=i_hits_dict)
    user_centers, user_contexts, user_negatives = generator_implicit_relations(user_corpus, user_vocab,
                                                                               args.max_window_size, args.K)
    item_centers, item_contexts, item_negatives = generator_implicit_relations(item_corpus, item_vocab,
                                                                               args.max_window_size, args.K)
