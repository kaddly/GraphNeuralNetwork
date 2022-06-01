import numpy as np
from joblib import Parallel, delayed
import random
import itertools


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

    # 扔第一个骰子，产生第一个1~N的随机数,决定落在哪一列
    random_num1 = int(np.floor(np.random.rand() * N))
    # 扔第二个骰子，产生0~1之间的随机数，判断与accept_prob[random_num1]的大小
    random_num2 = np.random.rand()

    # 如果小于Prab[i]，则采样i，如果大于Prab[i]，则采样Alias[i]
    if random_num2 < accept_prob[random_num1]:
        return random_num1
    else:
        alias_index[random_num1]


class AliasGenerator:
    def __init__(self, sampling_weights):
        self.accept_prob, self.alias_index = create_alias_table(sampling_weights)

    def draw(self):
        return alias_sample(self.accept_prob, self.alias_index)


class RandomWalker:
    def __init__(self, G, p=1, q=1):
        self.G = G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            if not G.is_directed():
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_edges = alias_edges
        self.alias_nodes = alias_nodes
