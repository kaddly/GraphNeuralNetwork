import os
import torch
from torch.utils.data import DataLoader, Dataset


def read_data(data_dir=os.path.join(os.path.abspath('.'), 'data'), dataset='amazon'):
    train_data_path = os.path.join(data_dir, dataset, 'train.txt')
    val_data_path = os.path.join(data_dir, dataset, 'valid.txt')
    test_data_path = os.path.join(data_dir, dataset, 'test.txt')
    edge_data_by_type = dict()  # 每个type对应到的相连接节点
    all_nodes = list()  # 所有节点的


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def load_data():
    pass
