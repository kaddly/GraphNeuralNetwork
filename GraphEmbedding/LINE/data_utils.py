import os
import torch
import networkx as nx


def read_wiki(data_dir):
    return nx.read_edgelist(data_dir, create_using=nx.DiGraph())



