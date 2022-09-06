import os
import pickle
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


def read_acm(data_dir='../data/acm.mat'):
    mat_file = sio.loadmat(data_dir)
    p_vs_l = mat_file['PvsL']  # paper-field?
    p_vs_a = mat_file['PvsA']  # paper-author
    p_vs_t = mat_file['PvsT']  # paper-term, bag of words
    p_vs_c = mat_file['PvsC']  # paper-conference, labels come from that
