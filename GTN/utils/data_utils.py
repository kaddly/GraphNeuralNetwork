import os
import pickle
import scipy.io as sio
from scipy.sparse import csr_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def read_acm(data_dir='../data/ACM.mat'):
    """
    读取mat数据
    :param data_dir:
    :return:
    """
    return sio.loadmat(data_dir)


def process_paper_target(mat_file):
    """
    构建论文idx与对应的标签
    :param mat_file:
    :return:
    """
    paper_conf = mat_file['PvsC'].nonzero()[1]
    # DataBase
    paper_db_idx = np.where(np.isin(paper_conf, [1, 13]))[0]
    paper_db_idx = np.sort(np.random.choice(paper_db_idx, 994, replace=False))
    # Data Mining
    paper_dm_idx = np.where(np.isin(paper_conf, [0]))[0]
    # Wireless Communication
    paper_wc_idx = np.where(np.isin(paper_conf, [9, 10]))[0]
    paper_idx = np.sort(list(paper_db_idx) + list(paper_dm_idx) + list(paper_wc_idx))
    # 处理labels
    paper_target = []
    for idx in paper_idx:
        if idx in paper_db_idx:
            paper_target.append(0)
        elif idx in paper_wc_idx:
            paper_target.append(1)
        else:
            paper_target.append(2)
    paper_target = np.array(paper_target)
    return paper_idx, paper_target


def process_edge_feature(mat_file, paper_idx):
    """
    处理邻接矩阵以及特征
    :param mat_file:
    :param paper_idx:
    :return:
    """
    # 处理作者
    authors = mat_file['PvsA'][paper_idx].nonzero()[1]
    author_dic = {}
    re_authors = []
    for author in authors:
        if author not in author_dic:
            author_dic[author] = len(author_dic) + len(paper_idx)
        re_authors.append(author_dic[author])
    re_authors = np.array(re_authors)

    # 领域
    subjects = mat_file['PvsL'][paper_idx].nonzero()[1]
    subject_dic = {}
    re_subjects = []
    for subject in subjects:
        if subject not in subject_dic:
            subject_dic[subject] = len(subject_dic) + len(paper_idx) + len(author_dic)
        re_subjects.append(subject_dic[subject])
    re_subjects = np.array(re_subjects)

    node_num = len(paper_idx) + len(author_dic) + len(subject_dic)
    papers = mat_file['PvsA'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_pa = csr_matrix((data, (papers, re_authors)), shape=(node_num, node_num))
    A_ap = A_pa.transpose()

    papers = mat_file['PvsL'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_ps = csr_matrix((data, (papers, re_subjects)), shape=(node_num, node_num))
    A_sp = A_ps.transpose()

    terms = mat_file['TvsP'].transpose()[paper_idx].nonzero()[1]
    term_dic = {}
    re_terms = []
    for term in terms:
        if term not in term_dic:
            term_dic[term] = len(term_dic) + len(paper_idx) + len(author_dic) + len(subject_dic)
        re_terms.append(term_dic[term])
    re_terms = np.array(re_terms)
    # tmp
    tmp_num_node = node_num + len(term_dic)

    papers = mat_file['PvsA'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_pa_tmp = csr_matrix((data, (papers, re_authors)), shape=(tmp_num_node, tmp_num_node))

    papers = mat_file['PvsL'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_ps_tmp = csr_matrix((data, (papers, re_subjects)), shape=(tmp_num_node, tmp_num_node))

    papers = mat_file['PvsT'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_pt_tmp = csr_matrix((data, (papers, re_terms)), shape=(tmp_num_node, tmp_num_node))

    paper_feat = np.array(A_pt_tmp[:len(paper_idx), -len(term_dic):].toarray() > 0, dtype=np.int32)
    author_feat = np.array(A_pa_tmp.transpose().dot(A_pt_tmp)[len(paper_idx):len(paper_idx) + len(author_dic),
                           -len(term_dic):].toarray() > 0, dtype=np.int32)
    subject_feat = np.array(A_ps_tmp.transpose().dot(A_pt_tmp)[
                            len(paper_idx) + len(author_dic):len(paper_idx) + len(author_dic) + len(subject_dic),
                            -len(term_dic):].toarray() > 0, dtype=np.int32)
    node_feature = np.concatenate((paper_feat, author_feat, subject_feat))
    return [A_pa, A_ap, A_ps, A_sp], node_feature


def train_test_split(paper_target):
    # Train, Valid
    train_valid_DB = list(np.random.choice(np.where(paper_target == 0)[0], 300, replace=False))
    train_valid_WC = list(np.random.choice(np.where(paper_target == 1)[0], 300, replace=False))
    train_valid_DM = list(np.random.choice(np.where(paper_target == 2)[0], 300, replace=False))

    train_idx = np.array(train_valid_DB[:200] + train_valid_WC[:200] + train_valid_DM[:200])
    valid_idx = np.array(train_valid_DB[200:] + train_valid_WC[200:] + train_valid_DM[200:])
    test_idx = np.array(list((set(np.arange(paper_target.shape[0])) - set(train_idx)) - set(valid_idx)))
    return train_idx, valid_idx, test_idx


def load_acm(data_root='../data'):
    train_process_path = os.path.join(data_root, 'train_process')
    if os.path.exists(os.path.join(train_process_path, 'train.pkl')):
        with open(os.path.join(train_process_path, 'train.pkl'), 'rb') as f:
            paper_idx, paper_target, edges, node_feature = pickle.load(f)
    else:
        mat_file = read_acm(os.path.join(data_root, 'ACM.mat'))
        paper_idx, paper_target = process_paper_target(mat_file)
        edges, node_feature = process_edge_feature(mat_file, paper_idx)
        with open(os.path.join(train_process_path, 'train.pkl'), 'wb') as f:
            pickle.dump((paper_idx, paper_target, edges, node_feature), f)

    train_idx, valid_idx, test_idx = train_test_split(paper_target)
    num_nodes = edges[0].shape[0]
    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)  # 添加末尾一位
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)  # 添加一个单位对角阵
    return A, torch.from_numpy(node_feature).type(torch.FloatTensor), torch.from_numpy(paper_target).type(
        torch.LongTensor), torch.tensor(train_idx), torch.tensor(valid_idx), torch.tensor(test_idx)
