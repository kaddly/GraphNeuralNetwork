import random
from joblib import Parallel, delayed
import itertools


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


class RandomWalker:
    def __init__(self, HG, user_to_idx=None):
        self.HG = HG
        self.user_to_idx = user_to_idx

    def meta_path_walk(self, start_node, meta_path):
        HG = self.HG
        walk = [start_node]
        candidate = start_node
        for i in range(len(meta_path) - 1):
            meta_path_adj = HG.HG_adj[meta_path[i] + '->' + meta_path[i + 1]]
            candidates = meta_path_adj[candidate].nonzero()[1]
            candidate = random.choice(candidates)
            walk.append(candidate)
        return walk

    def simulate_walks(self, num_walks, meta_path, workers=1, verbose=0):
        HG = self.HG
        nodes = list(HG.node_index_map[meta_path[0]])
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(nodes, num, meta_path) for num in
            partition_num(num_walks, workers))
        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, meta_path):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.meta_path_walk(meta_path=meta_path, start_node=v))
        return walks


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
