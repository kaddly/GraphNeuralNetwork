import torch
from torch.utils.data import DataLoader, Dataset
import networkx as nx
from sample_utils import RandomWalker
from graph_utils import preprocess_nxgraph
import random


def read_flight(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])


print(read_flight('../../data/flight/brazil-airports.edgelist'))
