import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from .graph_utils import preprocess_nxgraph, preprocess_struc2vec


def read_flight(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])


def plot_graph(G):
    print(G)
    nx.draw(G, node_size=10, font_size=2, font_color="blue", font_weight="bold")
    plt.show()


G = read_flight('../../data/flight/brazil-airports.edgelist')
