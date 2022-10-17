import random
from joblib import Parallel, delayed
import tqdm
import itertools


class RWGraph:
    def __init__(self, nx_G, node_type_arr=None, num_workers=16):
        self.G = nx_G
        self.node_type = node_type_arr
        self.num_workers = num_workers

    def node_list(self, nodes, num_walks):
        for loop in range(num_walks):  # 循环num_walks次数
            for node in nodes:
                yield node

    def simulate_walks(self, num_walks, walk_length, schema=None, workers=1, verbose=0):
        all_walks = []
        nodes = list(self.G.keys())  # 节点顶点数量
        random.shuffle(nodes)

        if schema is None:
            results = Parallel(n_jobs=workers, verbose=verbose)(
                delayed(self.walk)(walk_length, node, '') for node in tqdm(self.node_list(nodes, num_walks)))
            all_walks = list(itertools.chain(*results))
        else:
            schema_list = schema.split(',')
            for schema_iter in schema_list:
                results = Parallel(n_jobs=workers, verbose=verbose)(
                    delayed(self.walk)(walk_length, node, schema_iter) for node in tqdm(self.node_list(nodes, num_walks)) if
                    schema_iter.split('-')[0] == self.node_type[node])
                walks = list(itertools.chain(*results))
                all_walks.extend(walks)

        return all_walks

    def walk(self, args):
        walk_length, start, schema = args
        # Simulate a random walk starting from start node.
        rand = random.Random()

        if schema:
            schema_items = schema.split('-')
            assert schema_items[0] == schema_items[-1]  # metapath前后一致; A-B-A

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]  # 当前节点
            candidates = []
            for node in self.G[cur]:  # 和cur节点相连接的节点; 候选节点
                if schema == '' or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return [str(node) for node in walk]


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
