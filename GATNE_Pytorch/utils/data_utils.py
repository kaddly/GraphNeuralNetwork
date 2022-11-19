import os
import torch
from torch.utils.data import DataLoader, Dataset


def read_data(data_dir=os.path.join(os.path.abspath('.'), 'data')):
    pass


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def load_data():
    pass
