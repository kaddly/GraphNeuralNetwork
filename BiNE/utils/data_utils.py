import os
import random
import torch
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
    context = {}
    for line in corpus:
        # 形成“中心词-上下文词”对，每个句子至少需要两个单词
        if len(line) < 2:
            continue
        for i in range(len(line)):
            if i not in context:
                context[i] = []
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            context[i].append([line[idx] for idx in indices])
    return context


def get_negative(contexts, vocab, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [vocab.token_counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives, generator = {}, RandomGenerator(sampling_weights)
    for center, context in contexts:
        if center not in all_negatives:
            all_negatives[center] = []
        negatives = []
        while len(negatives) < len(context[len(all_negatives[center])]) * K:
            neg = generator.draw()
            # 噪声词不能是上下⽂词
            if neg not in context[len(all_negatives[center])]:
                negatives.append(neg)
        all_negatives[center].append(negatives)
    return all_negatives


def generator_implicit_relations(corpus, vocab, max_window_size, K):
    all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negative(all_contexts, vocab, K)
    return all_contexts, all_negatives


def readTestDataset(data_set, file_name):
    [users, items], labels = read_data(data_set, file_name)
    return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


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
    user_contexts, user_negatives = generator_implicit_relations(user_corpus, user_vocab, args.max_window_size, args.K)
    item_contexts, item_negatives = generator_implicit_relations(item_corpus, item_vocab, args.max_window_size, args.K)
    testDataset = readTestDataset(args.data_set, args.test_file_name)
    return (user_vocab[relation_list[0]], item_vocab[relation_list[1]],
            weights_list), testDataset, user_contexts, user_negatives, item_contexts, item_negatives, user_vocab, item_vocab
