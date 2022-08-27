import os
import random
import pickle
import errno
import scipy
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def load_data(data_dir='../data/ACM.mat'):
    matHG = sio.loadmat(data_dir)
    keys = matHG.keys()
    print(keys)


class Heterograph:
    def __init__(self, HGraphs: dict):
        self.HGraphs = HGraphs

    def get_mate_path_graph(self, mate_path):
        return self.HGraphs[mate_path]*self.HGraphs[mate_path].T


def HG_meta_path():
    pass

load_data()
