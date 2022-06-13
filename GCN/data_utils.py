import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def read_cora(data_dir='./data/cora/', dataset="cora"):
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(data_dir, dataset), dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("{}{}.cites".format(data_dir, dataset), type=np.int32)
    return idx_features_labels, edges_unordered


def preprocess_node_map(idx_features_labels):
    idx2node = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 节点
    node2idx = {j: i for i, j in enumerate(idx2node)}  # 构建节点的索引字典
    return idx2node, node2idx


def preprocess_data(idx_features_labels, edges_unordered, node2idx):
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 取特征feature
    labels = encode_onehot(idx_features_labels[:, -1])  # one-hot label
    edges = np.array(list(map(node2idx.get, edges_unordered.flatten())), dtype=np.int32).reshape(
        edges_unordered.shape)  # 将之前的转换成字典编号后的边
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)  # 构建边的邻接矩阵
    # build symmetric adjacency matrix，计算转置矩阵。将有向图转成无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return features, labels, adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mx.sum(1))  # 矩阵行求和
    r_inv = np.power(row_sum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.  # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx


def load_cora(data_dir, dataset):
    idx_features_labels, edges_unordered = read_cora(data_dir, dataset)
    idx2node, node2idx = preprocess_node_map(idx_features_labels)
    features, labels, adj = preprocess_data(idx_features_labels, edges_unordered, node2idx)
    features = normalize(features)   # 对特征做了归一化的操作
    adj = normalize(adj + sp.eye(adj.shape[0]))   # 对A+I归一化
