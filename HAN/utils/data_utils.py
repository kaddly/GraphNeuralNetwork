import scipy.io as sio
import numpy as np
import torch


def read_acm(data_dir='../data/ACM3025.mat'):
    matHG = sio.loadmat(data_dir)
    return [matHG['PAP'], matHG['PLP']], matHG['feature'], matHG['label'], \
           matHG['train_idx'], matHG['val_idx'], matHG['test_idx']


def read_acm_row(data_dir='../data/ACM.mat'):
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

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected))
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    features = p_vs_t.toarray()

    HG = HeteroGraph({'p_vs_l': p_vs_l, 'p_vs_a': p_vs_a})

    # 每个类别均匀划分训练集、验证集、测试集
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    return HG, features, labels, train_idx, val_idx, test_idx


class HeteroGraph:
    def __init__(self, HGraphs: dict):
        self.HGraphs = HGraphs

    def __getitem__(self, item):
        return self.get_mate_path_graph(self.HGraphs.keys()[item])

    def __len__(self):
        return len(self.HGraphs)

    def get_mate_path_graph(self, mate_path):
        mate_path_adj = self.HGraphs[mate_path] * self.HGraphs[mate_path].T
        mask = mate_path_adj > 0
        mate_path_adj[mask] = 1
        return mate_path_adj.toarray()


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_data(data_set='acm_raw'):
    if data_set == 'acm_raw':
        HGs, features, labels, train_idx, val_idx, test_idx = read_acm_row()
    elif data_set == 'acm':
        HGs, features, labels, train_idx, val_idx, test_idx = read_acm()
    else:
        raise ValueError('unsupported dataset!')
    num_nodes = labels.shape[0]
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    HGs_adj = [torch.tensor(hg) for hg in HGs]
    features = torch.Tensor(features)
    labels = torch.LongTensor(labels)

    return HGs_adj, features, labels, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask
