import collections
from collections import ChainMap, deque
from .fastdtw import fastdtw
from .sample_utils import create_alias_table
import numpy as np
import math
from joblib import Parallel, delayed


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


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
        degreeList[v] = _get_order_degree_list_node(graph, idx2node, node2idx, v, opt1_reduce_len=opt1_reduce_len,
                                                    max_num_layers=max_num_layers)
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
                dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                dtw_dist[v1, v2][layer] = dist
    return dtw_dist


def _compute_structural_distance(graph, idx2node, node2idx, opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                                 max_num_layers=None, workers=1,
                                 verbose=0):
    if opt1_reduce_len:
        dist_func = cost_max
    else:
        dist_func = cost
    degreeList = _compute_ordered_degree_list(graph, idx2node, node2idx, opt1_reduce_len, max_num_layers)
    if opt2_reduce_sim_calc:
        degrees = _create_vectors(graph, idx2node)
        degreeListsSelected = {}
        vertices = {}
        n_nodes = len(idx2node)
        for v in range(n_nodes):  # c:list of vertex
            nbs = get_vertices(
                v, len(graph[idx2node[v]]), degrees, n_nodes)
            vertices[v] = nbs  # store nbs
            degreeListsSelected[v] = degreeList[v]  # store dist
            for n in nbs:
                # store dist of nbs
                degreeListsSelected[n] = degreeList[n]
    else:
        vertices = {}
        for v in degreeList:
            vertices[v] = [vd for vd in degreeList.keys() if vd > v]
    results = Parallel(n_jobs=workers, verbose=verbose, )(
        delayed(compute_dtw_dist)(part_list, degreeList, dist_func) for part_list in partition_dict(vertices, workers))
    dtw_dist = dict(ChainMap(*results))
    structural_dist = convert_dtw_struc_dist(dtw_dist)

    return structural_dist


def _create_vectors(G, idx2node):
    degrees = {}  # sotre v list of degree
    degrees_sorted = set()  # store degree
    for v in range(len(idx2node)):
        degree = len(G[idx2node[v]])
        degrees_sorted.add(degree)
        if degree not in degrees:
            degrees[degree] = {}
            degrees[degree]['vertices'] = []
        degrees[degree]['vertices'].append(v)
    degrees_sorted = np.array(list(degrees_sorted), dtype='int')
    degrees_sorted = np.sort(degrees_sorted)

    l = len(degrees_sorted)
    for index, degree in enumerate(degrees_sorted):
        if index > 0:
            degrees[degree]['before'] = degrees_sorted[index - 1]
        if index < (l - 1):
            degrees[degree]['after'] = degrees_sorted[index + 1]

    return degrees


def _get_transition_probs(layers_adj, layers_distances):
    layers_alias = {}
    layers_accept = {}

    for layer in layers_adj:

        neighbors = layers_adj[layer]
        layer_distances = layers_distances[layer]
        node_alias_dict = {}
        node_accept_dict = {}
        norm_weights = {}

        for v, neighbors in neighbors.items():
            e_list = []
            sum_w = 0.0

            for n in neighbors:
                if (v, n) in layer_distances:
                    wd = layer_distances[v, n]
                else:
                    wd = layer_distances[n, v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            norm_weights[v] = e_list
            accept, alias = create_alias_table(e_list)
            node_alias_dict[v] = alias
            node_accept_dict[v] = accept

        layers_alias[layer] = node_alias_dict
        layers_accept[layer] = node_accept_dict

    return layers_accept, layers_alias, norm_weights


def prepare_biased_walk(norm_weights):
    sum_weights = {}
    sum_edges = {}
    average_weight = {}
    gamma = {}
    layer = 0
    while norm_weights:
        probs = norm_weights
        for v, list_weights in probs.items():
            sum_weights.setdefault(layer, 0)
            sum_edges.setdefault(layer, 0)
            sum_weights[layer] += sum(list_weights)
            sum_edges[layer] += len(list_weights)

        average_weight[layer] = sum_weights[layer] / sum_edges[layer]

        gamma.setdefault(layer, {})

        for v, list_weights in probs.items():
            num_neighbours = 0
            for w in list_weights:
                if w > average_weight[layer]:
                    num_neighbours += 1
            gamma[layer][v] = num_neighbours

        layer += 1
    return average_weight, gamma


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


def convert_dtw_struc_dist(distances, startLayer=1):
    """

    :param distances: dict of dict
    :param startLayer:
    :return:
    """
    for vertices, layers in distances.items():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers), startLayer)
        for layer in range(0, startLayer):
            keys_layers.pop(0)

        for layer in keys_layers:
            layers[layer] += layers[layer - 1]
    return distances


def get_vertices(v, degree_v, degrees, n_nodes):
    a_vertices_selected = 2 * math.log(n_nodes, 2)
    vertices = []
    try:
        c_v = 0

        for v2 in degrees[degree_v]['vertices']:
            if v != v2:
                vertices.append(v2)  # same degree
                c_v += 1
                if c_v > a_vertices_selected:
                    raise StopIteration

        if 'before' not in degrees[degree_v]:
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if 'after' not in degrees[degree_v]:
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if degree_b == -1 and degree_a == -1:
            raise StopIteration  # not anymore v
        degree_now = verifyDegrees(degree_v, degree_a, degree_b)
        # nearest valid degree
        while True:
            for v2 in degrees[degree_now]['vertices']:
                if v != v2:
                    vertices.append(v2)
                    c_v += 1
                    if c_v > a_vertices_selected:
                        raise StopIteration

            if degree_now == degree_b:
                if 'before' not in degrees[degree_b]:
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if 'after' not in degrees[degree_a]:
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']

            if degree_b == -1 and degree_a == -1:
                raise StopIteration

            degree_now = verifyDegrees(degree_v, degree_a, degree_b)

    except StopIteration:
        return list(vertices)


def _get_layer_rep(pair_distances):
    layer_distances = {}
    layer_adj = {}
    for v_pair, layer_dist in pair_distances.items():
        for layer, distance in layer_dist.items():
            vx = v_pair[0]
            vy = v_pair[1]

            layer_distances.setdefault(layer, {})
            layer_distances[layer][vx, vy] = distance

            layer_adj.setdefault(layer, {})
            layer_adj[layer].setdefault(vx, [])
            layer_adj[layer].setdefault(vy, [])
            layer_adj[layer][vx].append(vy)
            layer_adj[layer][vy].append(vx)

    return layer_adj, layer_distances


def verifyDegrees(degree_v_root, degree_a, degree_b):
    if degree_b == -1:
        degree_now = degree_a
    elif degree_a == -1:
        degree_now = degree_b
    elif abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now


def preprocess_struc2vec(graph, idx2node, node2idx, opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                         opt3_num_layers=None):
    pair_distances = _compute_structural_distance(graph, idx2node, node2idx, opt1_reduce_len, opt2_reduce_sim_calc,
                                                  opt3_num_layers)
    layer_adj, layer_distances = _get_layer_rep(pair_distances)
    layers_accept, layers_alias, norm_weights = _get_transition_probs(layers_adj=layer_adj,
                                                                      layer_distances=layer_distances)
    average_weight, gamma = prepare_biased_walk(norm_weights)

    return layers_accept, layers_alias, average_weight, gamma
