import numpy as np
import torch
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
    edges_unordered = np.genfromtxt("{}{}.cites".format(data_dir, dataset), dtype=np.int32)
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
