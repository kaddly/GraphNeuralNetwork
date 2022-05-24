import torch
from torch.utils.data import DataLoader, Dataset
import os
from joblib import Parallel, delayed
import random
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(data_dir):
    G = nx.read_edgelist(data_dir, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    nx.draw(G, node_size=10, font_size=10, font_color="blue", font_weight="bold")
    plt.show()


# plot_graph('../data/wiki/Wiki_edgelist.txt')

def deepwalk_walk(walk_length, start_node, G):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk
