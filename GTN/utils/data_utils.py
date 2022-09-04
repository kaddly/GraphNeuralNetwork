import os
import pickle
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


def read_acm(data_dir='./data/acm.mat'):
    data = sio.loadmat(data_dir)
