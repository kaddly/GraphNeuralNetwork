import random
import math
from tqdm import tqdm
from joblib import Parallel, delayed
from utils.graph_utils import Vocab


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
    def __init__(self, meta_path_adj, vocab: Vocab, maxT, minT, percentage, hits_dict, num_workers=2):
        self.meta_path_adj = meta_path_adj
        self.vocab = vocab
        self.maxT = maxT
        self.minT = minT
        self.rand = random.Random()
        self.p = percentage
        self.hits_dict = hits_dict
        self.num_workers = num_workers

    def node_list(self, nodes):
        for node in nodes:
            num_walks = max(int(math.ceil(self.maxT*self.hits_dict[node])), self.minT)
            for _ in range(num_walks):
                yield node

    def homogeneous_graph_random_walks(self, verbose=0):
        nodes = self.vocab.token_to_idx.keys()
        all_walks = Parallel(n_jobs=self.num_workers, verbose=verbose)(
            delayed(self._simulate_walks)(node) for node in tqdm(self.node_list(nodes)))

        return all_walks

    def _simulate_walks(self, start):
        pass


