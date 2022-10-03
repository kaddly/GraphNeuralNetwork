import os
import pandas as pd
from utils.graph_utils import procession_graph, HeteroGraph
from utils.sample_utils import RandomWalker


def read_JData(data_dir=os.path.join('./', 'data'), sample_num=10000):
    edge_f = pd.read_csv(os.path.join(data_dir, 'data_action.csv'))
    user_features = pd.read_csv(os.path.join(data_dir, 'user_features.csv'))
    nodes_features = pd.read_csv(os.path.join(data_dir, 'item_features.csv'))
    edge_f = edge_f.sample(sample_num)
    user_features = user_features[user_features['node_id'].isin(list(edge_f['user_id']))]
    nodes_features = nodes_features[nodes_features['node_id'].isin(list(edge_f['sku_id']))]
    idx_to_users, user_to_idx, idx_to_items, item_to_idx = procession_graph(edge_f)
    user_item_src = [user_to_idx.get(user_id) for user_id in edge_f['user_id']]
    user_item_dst = [item_to_idx.get(item_id) for item_id in edge_f['sku_id']]
    HG = HeteroGraph([user_item_src, user_item_dst], edge_types=['user', 'item'], meta_path=['user', 'item', 'user'])
    return HG, user_features, nodes_features, idx_to_users, user_to_idx, idx_to_items, item_to_idx


def parse_trace(trace, user_index_id_map, item_index_id_map):
    s = []
    for index in range(len(trace)):
        if index % 2 == 0:
            s.append(user_index_id_map[trace[index]])
        else:
            s.append(item_index_id_map[trace[index]])
    return ','.join(s)


def generate_meta_paths(meta_path=['user', 'item', 'user', 'item', 'user']):
    HG, user_features, nodes_features, idx_to_users, user_to_idx, idx_to_items, item_to_idx = read_JData()
    generator = RandomWalker(HG)
    trs = generator.simulate_walks(num_walks=10, meta_path=meta_path, workers=2)
    f = open("./data/output_path.txt", "w")
    for tr in trs:
        res = parse_trace(tr, idx_to_users, idx_to_items)
        f.write(res+'\n')
    f.close()
