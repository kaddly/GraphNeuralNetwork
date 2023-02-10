import os
import random
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from utils.graph_utils import get_G_from_edges, Vocab
from utils.sample_utils import RWGraph, RandomGenerator


def read_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    train_data_path = os.path.join(data_dir, dataset, 'train.txt')
    val_data_path = os.path.join(data_dir, dataset, 'valid.txt')
    test_data_path = os.path.join(data_dir, dataset, 'test.txt')
    edge_data_by_type = dict()  # 每个type对应到的相连接节点
    all_nodes = list()  # 所有节点的

    with open(train_data_path, 'r') as f:
        for line in f:
            words = line[:-1].split(" ")
            if words[0] not in edge_data_by_type:  # edge type涉及到的节点
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))  # nodes去重
    print('Total training nodes: ' + str(len(all_nodes)))

    def process_val_data(f_index):
        true_edge_data_by_type = dict()  # true样本
        false_edge_data_by_type = dict()  # false样本
        for sentence in f_index:
            tokens = sentence[:-1].split(' ')
            x, y = tokens[1], tokens[2]
            if int(tokens[3]) == 1:  # true对应到的节点
                if tokens[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[tokens[0]] = list()  # true对应到的type相连接节点
                true_edge_data_by_type[tokens[0]].append((x, y))
            else:
                if tokens[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[tokens[0]] = list()
                false_edge_data_by_type[tokens[0]].append((x, y))
        return true_edge_data_by_type, false_edge_data_by_type

    with open(val_data_path, 'r') as f:
        val_true_edge_data_by_type, val_false_edge_data_by_type = process_val_data(f)

    with open(test_data_path, 'r') as f:
        test_true_edge_data_by_type, test_false_edge_data_by_type = process_val_data(f)

    return edge_data_by_type, val_true_edge_data_by_type, val_false_edge_data_by_type, test_true_edge_data_by_type, test_false_edge_data_by_type


def read_features(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    data_path = os.path.join(data_dir, dataset, 'feature.txt')
    feature_dict = {}
    print("We are loading data from:" + data_path)
    with open(data_path, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dict[items[0]] = items[1:]
    return feature_dict


def read_node_types(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    data_path = os.path.join(data_dir, dataset, 'node_type.txt')
    node_type = {}
    print('We are loading node type from:' + data_path)
    with open(data_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def generate_walks(network_data, num_walks, walk_length, schema, data_dir=os.path.join(os.path.abspath('.'), 'data'),
                   dataset='amazon', num_workers=2):
    if schema is not None:
        node_type = read_node_types(data_dir, dataset)
    else:
        node_type = None

    all_walks = []  # 所有游走的list
    for layer_id, layer_name in enumerate(network_data):
        tmp_data = network_data[layer_name]  # 每个type对应到的点边信息
        # start to do the random walk on a layer
        # get_G_from_edges(tmp_data): 每个节点对应到相连接的点
        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type, num_workers)  # RandomWalk Graph
        print('Generating random walks for layer', layer_name)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)  # 生成随机游走的序列; 每个节点游走次数; 游走长度;

        all_walks.append(layer_walks)

        print('Finish generating the walks')

    return all_walks


def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元'<UNK>'
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]

    counter = vocab.token_counter
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return random.randint(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens)

    return [[token for token in line if keep(token)] for line in sentences]


def get_centers_and_contexts(all_corpus, max_window_size):
    """返回跳元模型中的中心词与上下文单词"""
    centers, context = [], []
    for layer_id, corpus in enumerate(all_corpus):
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
                context.append(([line[idx] for idx in indices], layer_id))
    return centers, context


def get_negative(all_contexts, vocab, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [vocab.token_counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts, _ in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下⽂词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


class MulEdgeDataset(Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, item):
        return self.centers[item], self.contexts[item], self.negatives[item]

    def __len__(self):
        return len(self.centers)


class Collate_fn:
    def __init__(self, neighbors):
        self.neighbors = neighbors

    def __call__(self, data):
        max_len = max(len(c) + len(n) for _, (c, _), n in data)
        centers, type_ids, neighbors, contexts_negatives, masks, labels = [], [], [], [], [], []
        for center, (context, type_id), negative in data:
            cur_len = len(context) + len(negative)
            centers += [center]
            type_ids += [type_id]
            neighbors.append(self.neighbors[center])
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
        return (torch.tensor(centers)
                , torch.tensor(type_ids)
                , torch.tensor(neighbors)
                , torch.tensor(contexts_negatives)
                , torch.tensor(masks)
                , torch.tensor(labels))


def generator_neighbor(network_data, vocab, num_nodes, edge_types, neighbor_samples):
    edge_type_count = len(edge_types)
    neighbors = [[[] for _ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        print("Generator neighbors for later", r)
        g = network_data[edge_types[r]]  # 每个type涉及到的节点
        for (x, y) in tqdm(g):
            ix = vocab[x]  # x对应到的索引
            iy = vocab[y]  # y对应到的索引
            neighbors[ix][r].append(iy)  # 邻居信息
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:  # 节点在这个类别下，如果没有节点和它连接，邻居就是该节点本身
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:  # 如果邻居节点数量小于采样邻居数量，进行重采样
                neighbors[i][r].extend(
                    list(random.choices(neighbors[i][r], k=neighbor_samples - len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:  # 如果邻居节点数量大于采样邻居数量，进行邻居大小数量的采样
                neighbors[i][r] = list(random.sample(neighbors[i][r], k=neighbor_samples))
    return neighbors  # 每个节点的邻居采样


def load_data(args):
    train_edge_data_by_type, val_true_edge_data_by_type, val_false_edge_data_by_type, test_true_edge_data_by_type, test_false_edge_data_by_type = read_data(
        data_dir=args.data_dir, dataset=args.dataset)
    train_walks = generate_walks(train_edge_data_by_type, num_walks=args.num_walks, walk_length=args.walk_length, schema=args.schema,
                                 data_dir=args.data_dir, dataset=args.dataset, num_workers=args.num_workers)
    vocab = Vocab([layer_walks for layer_walks in train_walks], min_freq=4)
    train_neighbors = generator_neighbor(train_edge_data_by_type, vocab, len(vocab),
                                         list(train_edge_data_by_type.keys()), args.neighbor_samples)
    train_walks = [subsample(layer_walks, vocab) for layer_walks in train_walks]
    train_walks = [[vocab[line] for line in layer_walks] for layer_walks in train_walks]
    train_centers, train_contexts = get_centers_and_contexts(train_walks, max_window_size=args.max_window_size)
    train_negatives = get_negative(train_contexts, vocab, args.K)
    train_dataset = MulEdgeDataset(train_centers, train_contexts, train_negatives)
    collate_fn = Collate_fn(train_neighbors)
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    return train_iter, train_edge_data_by_type, train_neighbors, val_true_edge_data_by_type, val_false_edge_data_by_type, test_true_edge_data_by_type, test_false_edge_data_by_type, vocab
