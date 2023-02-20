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


def readTestDataset(data_set, file_name, user_vocab, item_vocab):
    [users, items], labels = read_data(data_set, file_name)
    return torch.tensor(user_vocab[users]), torch.tensor(item_vocab[items]), torch.tensor(labels)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def ContextsNegativesProcess(data):
    max_len = max(len(c) + len(n) for c, n in data)
    contexts_negatives, masks, labels = [], [], []
    for context, negative in data:
        cur_len = len(context) + len(negative)
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return contexts_negatives, masks, labels


class ContextsNegativesGenerator:
    def __init__(self, user_contexts, user_negatives, item_contexts, item_negatives):
        self.user_contexts = user_contexts
        self.user_negatives = user_negatives
        self.item_contexts = item_contexts
        self.item_negatives = item_negatives

    def get_contexts_negatives_masks_labels(self, user_center, item_center, weights):
        assert len(self.user_contexts.get(user_center)) == len(self.user_negatives.get(user_center)) and len(
            self.item_contexts.get(item_center)) == len(self.item_negatives.get(item_center))
        user_contexts_negatives, user_masks, user_labels = ContextsNegativesProcess(
            (self.user_contexts.get(user_center), self.user_negatives.get(user_center)))
        item_contexts_negatives, item_masks, item_labels = ContextsNegativesProcess(
            (self.item_contexts.get(item_center), self.item_negatives.get(item_center)))
        return torch.tensor(user_center), torch.tensor(item_center), torch.Tensor(weights),\
               torch.tensor(user_contexts_negatives), torch.tensor(user_masks), torch.tensor(user_labels),\
               torch.tensor(item_contexts_negatives), torch.tensor(item_masks), torch.tensor(item_labels)


def load_data(args):
    relation_list, weights_list = read_data(args.data_set, args.train_file_name)
    BG = BipartiteGraph(relation_list, edge_types=['U', 'I'], meta_path=['U', 'I', 'U'], edge_frames=weights_list,
                        is_digraph=args.is_digraph)
    user_vocab, item_vocab = BG.get_vocab
    u_hits_dict, i_hits_dict = BG.calculate_centrality()
    user_corpus = generator_walks(meta_path=['U', 'I', 'U'], BG=BG, vocab=user_vocab, maxT=args.maxT, minT=args.minT,
                                  percentage=args.percentage, hits_dict=u_hits_dict)
    item_corpus = generator_walks(meta_path=['I', 'U', 'I'], BG=BG, vocab=item_vocab, maxT=args.maxT, minT=args.minT,
                                  percentage=args.percentage, hits_dict=i_hits_dict)
    user_contexts, user_negatives = generator_implicit_relations(user_corpus, user_vocab, args.max_window_size, args.K)
    item_contexts, item_negatives = generator_implicit_relations(item_corpus, item_vocab, args.max_window_size, args.K)
    testDataset = readTestDataset(args.data_set, args.test_file_name, user_vocab, item_vocab)
    return (user_vocab[relation_list[0]], item_vocab[relation_list[1]],
            weights_list), testDataset, user_contexts, user_negatives, item_contexts, item_negatives, user_vocab, item_vocab
