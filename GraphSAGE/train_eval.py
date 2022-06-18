import torch
from torch import nn
import time
import os
from datetime import timedelta


def train(net, train_iter, val_iter, lr, num_epochs, device):
    net = net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    # 模型参数保存路径
    saved_dir = './saved_dict'
    model_file = 'GraphSAGE'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
