import numpy as np
from joblib import Parallel, delayed
import random
import itertools
import math


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


def create_alias_table(Prob_val):
    """
    :param Prob_val:传入概率列表
    :return:返回一个accept 概率数组 和 alias的标号数组
    """
    L = len(Prob_val)
    # 初始化两个数组
    accept_prob = np.zeros(L)  # 存的是概率
    alias_index = np.zeros(L, dtype=np.int)  # 存的是下标/序号

    # 大的队列用于存储面积大于1的节点标号，小的队列用于存储面积小于1的节点标号
    small_queue = []
    large_queue = []

    # 把Prob_val list中的值分配到大小队列中
    for index, prob in enumerate(Prob_val):
        accept_prob[index] = L * prob

        if accept_prob[index] < 1.0:
            small_queue.append(index)
        else:
            large_queue.append(index)

    # 1.每次从两个队列中各取一个，让大的去补充小的，然后小的出small队列
    # 2.在看大的减去补给小的之后剩下的值，如果大于1，继续放到large队列；如果恰好等于1，也出队列；如果小于1加入small队列中
    while small_queue and large_queue:
        small_index = small_queue.pop()
        large_index = large_queue.pop()
        # 因为alias_index中存的：另一个事件的标号，那现在用大的概率补充小的概率，标号就要变成大的事件的标号了
        alias_index[small_index] = large_index
        # 补充的原则是：大的概率要把小的概率补满（补到概率为1），然后就是剩下的
        accept_prob[large_index] = accept_prob[large_index] + accept_prob[small_index] - 1.0
        # 判断补完后，剩下值的大小
        if accept_prob[large_index] < 1.0:
            small_queue.append(large_index)
        else:
            large_queue.append(large_index)
    return accept_prob, alias_index


def alias_sample(accept_prob, alias_index):
    N = len(accept_prob)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept_prob[i]:
        return i
    else:
        return alias_index[i]


class AliasGenerator:
    def __init__(self, sampling_weights):
        self.accept_prob, self.alias_index = create_alias_table(sampling_weights)

    def draw(self):
        return alias_sample(self.accept_prob, self.alias_index)


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""

    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(0, len(sampling_weights)))
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


class RandomWalker:
    def __init__(self, idx2node, layers_adj, layers_alias, layers_accept, gamma):
        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.layers_adj = layers_adj
        self.layers_alias = layers_alias
        self.layers_accept = layers_accept
        self.gamma = gamma

    def simulate_walks(self, num_walks, walk_length, stay_prob=0.3, workers=1, verbose=0):
        nodes = self.idx

        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(nodes, num, walk_length, stay_prob, self.layers_adj, self.layers_accept,
                                          self.layers_alias, self.gamma) for num in partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias,
                                                    v, walk_length, gamma, stay_prob))
        return walks

    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        initialLayer = 0
        layer = initialLayer

        path = [self.idx2node[v]]

        while len(path) < walk_length:
            r = random.random()
            if r < stay_prob:  # same layer
                v = chooseNeighbor(v, graphs, layers_alias,
                                   layers_accept, layer)
                path.append(self.idx2node[v])
            else:  # different layer
                r = random.random()
                try:
                    x = math.log(gamma[layer][v] + math.e)
                    p_moveup = (x / (x + 1))
                except:
                    print(layer, v)
                    raise ValueError()

                if r > p_moveup:
                    if layer > initialLayer:
                        layer = layer - 1
                else:
                    if (layer + 1) in graphs and v in graphs[layer + 1]:
                        layer = layer + 1

        return path


def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):
    v_list = graphs[layer][v]

    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    v = v_list[idx]

    return v
