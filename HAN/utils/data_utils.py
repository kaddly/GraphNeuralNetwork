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


def read_data(data_dir='../data/ACM.mat'):
    """
    异构数据处理
    :param data_dir:
    :return:
    """
    matHG = sio.loadmat(data_dir)
    p_vs_l = matHG['PvsL']  # paper-field?
    p_vs_a = matHG['PvsA']  # paper-author
    p_vs_t = matHG['PvsT']  # paper-term, bag of words
    p_vs_c = matHG['PvsC']  # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    '''
    array([[array(['KDD'], dtype='<U3')],
       [array(['SIGMOD'], dtype='<U6')],
       [array(['WWW'], dtype='<U3')],
       [array(['SIGIR'], dtype='<U5')],
       [array(['CIKM'], dtype='<U4')],
       [array(['SODA'], dtype='<U4')],
       [array(['STOC'], dtype='<U4')],
       [array(['SOSP'], dtype='<U4')],
       [array(['SPAA'], dtype='<U4')],
       [array(['SIGCOMM'], dtype='<U7')],
       [array(['MobiCOMM'], dtype='<U8')],
       [array(['ICML'], dtype='<U4')],
       [array(['COLT'], dtype='<U4')],
       [array(['VLDB'], dtype='<U4')]], dtype=object)
    '''
    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]
    feature = p_vs_t.toarray()
    return


class HeteroGraph:
    def __init__(self, HGraphs: dict):
        self.HGraphs = HGraphs

    def __getitem__(self, item):
        return self.get_mate_path_graph(self.HGraphs.keys()[item])

    def get_mate_path_graph(self, mate_path):
        mate_path_adj = self.HGraphs[mate_path] * self.HGraphs[mate_path].T
        mask = mate_path_adj > 0
        mate_path_adj[mask] = 1
        return mate_path_adj


def load_data(batch_size):
    read_data()
