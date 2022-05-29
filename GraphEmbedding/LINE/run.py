import torch
from LINE import LINE
from data_utils import load_data_wiki
from train_eval import train

if __name__ == '__main__':
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
