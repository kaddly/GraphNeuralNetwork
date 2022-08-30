import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader


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

    def __iter__(self):
        for key in self.HGraphs.keys():
            yield self.get_mate_path_graph(key)

    def __len__(self):
        return len(self.HGraphs)

    def get_mate_path_graph(self, mate_path):
        mate_path_adj = self.HGraphs[mate_path] * self.HGraphs[mate_path].T
        mask = mate_path_adj > 0
        mate_path_adj[mask] = 1
        return mate_path_adj.toarray()


class collect_f:
    def __init__(self, HGs, features, labels):
        self.HGs_adj = [torch.tensor(hg) for hg in HGs]
        self.features = torch.Tensor(features)
        self.labels = torch.LongTensor(labels)

    def __call__(self, data):
        idx = list(data)
        HGs_adj = [HG_adj[idx:idx] for HG_adj in self.HGs_adj]
        return HGs_adj, self.features[idx], self.labels[idx]


def load_data(data_set='acm_raw', is_batch=False, batch_size=32):
    if data_set == 'acm_raw':
        HGs, features, labels, train_idx, val_idx, test_idx = read_acm_row()
    elif data_set == 'acm':
        HGs, features, labels, train_idx, val_idx, test_idx = read_acm()
    else:
        raise ValueError('unsupported dataset!')
    if is_batch:
        batchify = collect_f(HGs, features, labels)
        train_iter = DataLoader(data_set=train_idx, batch_size=batch_size, shuffle=True, collate_fn=batchify)
        val_iter = DataLoader(data_set=val_idx, batch_size=batch_size, collate_fn=batchify)
        test_iter = DataLoader(data_set=test_idx, batch_size=batch_size, collate_fn=batchify)

        return train_iter, val_iter, test_iter

    else:
        HGs_adj = [torch.tensor(hg) for hg in HGs]
        features = torch.Tensor(features)
        labels = torch.LongTensor(labels)
        return HGs_adj, features, labels, train_idx, val_idx, test_idx
