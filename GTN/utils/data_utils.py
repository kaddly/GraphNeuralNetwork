import os
import pickle
import scipy.io as sio
from scipy.sparse import csr_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def read_acm(data_dir='../data/acm.mat'):
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


def process_edge(mat_file, paper_idx):
    """
    处理邻接矩阵
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

    return [A_pa, A_ap, A_ps, A_sp]


def load_acm():
    mat_file = read_acm()
    paper_idx, paper_target = process_paper_target(mat_file)
    edges = process_edge(mat_file, paper_idx)
