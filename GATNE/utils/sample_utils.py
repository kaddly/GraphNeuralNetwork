import random
from joblib import Parallel, delayed
import multiprocessing
import tqdm


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


def walk(args):
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
        for node in G[cur]:  # 和cur节点相连接的节点; 候选节点
            if schema == '' or node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                candidates.append(node)
        if candidates:
            walk.append(rand.choice(candidates))
        else:
            break
    return [str(node) for node in walk]


def initializer(init_G, init_node_type):
    global G
    G = init_G
    global node_type
    node_type = init_node_type


class RWGraph:
    def __init__(self, nx_G, node_type_arr=None, num_workers=16):
        self.G = nx_G
        self.node_type = node_type_arr
        self.num_workers = num_workers

    def node_list(self, nodes, num_walks):
        for loop in range(num_walks):  # 循环num_walks次数
            for node in nodes:
                yield node

    def simulate_walks(self, num_walks, walk_length, schema=None):
        all_walks = []
        nodes = list(self.G.keys())  # 节点顶点数量
        random.shuffle(nodes)

        if schema is None:
            with multiprocessing.Pool(self.num_workers, initializer=initializer,
                                      initargs=(self.G, self.node_type)) as pool:
                all_walks = list(
                    pool.imap(walk, ((walk_length, node, '') for node in tqdm(self.node_list(nodes, num_walks))),
                              chunksize=256))
        else:
            schema_list = schema.split(',')
            for schema_iter in schema_list:
                with multiprocessing.Pool(self.num_workers, initializer=initializer,
                                          initargs=(self.G, self.node_type)) as pool:
                    walks = list(pool.imap(walk, ((walk_length, node, schema_iter) for node in
                                                  tqdm(self.node_list(nodes, num_walks)) if
                                                  schema_iter.split('-')[0] == self.node_type[node]), chunksize=512))
                all_walks.extend(walks)

        return all_walks
