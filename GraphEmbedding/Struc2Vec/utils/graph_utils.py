import collections
from collections import deque
from .fastdtw import fastdtw


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1d或者2d列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def _get_order_degree_list_node(graph, idx2node, node2idx, root, opt1_reduce_len, max_num_layers):
    if max_num_layers is None:
        max_num_layers = float('inf')
    ordered_degree_sequence_dict = {}
    visited = [False] * len(graph.nodes())
    queue = deque()
    level = 0
    queue.append(root)
    visited[root] = True

    while len(queue) > 0 and level <= max_num_layers:
        count = len(queue)
        if opt1_reduce_len:
            degree_list = {}
        else:
            degree_list = []
        while count > 0:
            top = queue.popleft()
            node = idx2node[top]
            degree = len(graph[node])
            if opt1_reduce_len:
                degree_list[degree] = degree_list.get(degree, 0) + 1
            else:
                degree_list.append(degree)
            for nei in graph[node]:
                nei_idx = node2idx[nei]
                if not visited[nei_idx]:
                    visited[nei_idx] = True
                    queue.append(nei_idx)
            count -= 1
        if opt1_reduce_len:
            ordered_degree_list = [(degree, freq) for degree, freq in degree_list.items()]
            ordered_degree_list.sort(key=lambda x: x[0])
        else:
            ordered_degree_list = sorted(degree_list)
        ordered_degree_sequence_dict[level] = ordered_degree_list
        level += 1
    return ordered_degree_sequence_dict


def _compute_ordered_degree_list(graph, idx2node, node2idx, opt1_reduce_len=True, max_num_layers=None):
    degreeList = {}
    vertices = list(range(len(idx2node)))
    for v in vertices:
        degreeList[v] = _get_order_degree_list_node(graph, idx2node, node2idx, v, opt1_reduce_len=opt1_reduce_len, max_num_layers=max_num_layers)
    return degreeList


def compute_dtw_dist(part_list, degreeList, dist_func):
    dtw_dist = {}
    for v1, nbs in part_list:
        lists_v1 = degreeList[v1]  # lists_v1 :orderd degree list of v1
        for v2 in nbs:
            lists_v2 = degreeList[v2]  # lists_v1 :orderd degree list of v2
            max_layer = min(len(lists_v1), len(lists_v2))  # valid layer
            dtw_dist[v1, v2] = {}
            for layer in range(0, max_layer):
                dist, path = fastdtw(
                    lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                dtw_dist[v1, v2][layer] = dist
    return dtw_dist


def _compute_structural_distance(self, max_num_layers, workers=1, verbose=0):
    pass


def cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return (m / mi) - 1


def cost_min(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * min(a[1], b[1])


def cost_max(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])
