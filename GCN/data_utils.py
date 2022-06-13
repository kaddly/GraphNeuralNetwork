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


def normalize_features(mx):
    '''Row-normalize sparse matrix'''
    # 矩阵行求和
    rowsum = np.array(mx.sum(1))
    # 求和的-1次方
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    # 如果是inf，转换成0
    r_inv[np.isinf(r_inv)] = 0
    # 构建对角形矩阵
    r_mat_inv = sp.diags(r_inv)
    # 构造D-I*A, 非对称方式, 简化方式
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_cora(data_dir='./data/cora/', dataset="cora"):
    idx_features_labels, edges_unordered = read_cora(data_dir, dataset)
    idx2node, node2idx = preprocess_node_map(idx_features_labels)
    features, labels, adj = preprocess_data(idx_features_labels, edges_unordered, node2idx)
    features = normalize_features(features)  # 对特征做了归一化的操作
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # 对A+I归一化
    # 训练，验证，测试的样本
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    features = torch.Tensor(features.toarray())
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test
