import torch
from utils import load_data
from models import HANModel
from train_utils import train, train_batch, test


if __name__ == '__main__':
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
