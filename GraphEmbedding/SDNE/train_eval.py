import torch
from torch import nn
import time
import os
from datetime import timedelta


class loss_first(nn.Module):
    def __init__(self, alpha):
        super(loss_first, self).__init__()
        self.alpha = alpha

    def forward(self, y_hat, y):
        l1_loss = self.alpha * 2 * torch.trace(torch.mm(torch.mm(y_hat.T, y), y_hat)) / y.shape[0]
        return l1_loss


class loss_second(nn.Module):
    def __init__(self, beta):
        super(loss_second, self).__init__()
        self.beta = beta

    def forward(self, y_hat, y):
        b_ = torch.ones(y.shape).cuda()
        b_[y != 0] = self.beta
        l2_loss = torch.square((y_hat - y) * b_).sum(dim=-1)
        return l2_loss.mean()


def train(net, data_iter, lr, num_epochs, device, alpha, beta, wd):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=wd)
    l_1st = loss_first(alpha)
    l_2nd = loss_second(beta)
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    # 模型参数保存路径
    saved_dir = '../saved_dict'
    model_file = 'SDNE'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            AM, LM = [data.to(device) for data in batch]
            AM_hat, hidden_hat = net(AM)
            l_local = l_1st(hidden_hat, LM)
            l_total = l_2nd(AM_hat, AM)
            l = l_local + l_total
            l.backward()
            optimizer.step()
            if total_batch % 20 == 0:
                if l < dev_best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                    dev_best_loss = l
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  loss: {1:>5.3f},  total_loss: {2:>5.3f},  local_loss: {3:>5.3f},  Time: {4} {5}'
                print(msg.format(total_batch, l, l_total, l_local, time_dif, improve))
            total_batch += 1
            if total_batch - last_improve > 2000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def get_embedding(net, G, node2idx):
    emb = {}
    if not os.path.exists('../saved_dict/Struc2Vec/SDNE.ckpt'):
        print('please train before!')
        return
    net.load_state_dict(torch.load('../saved_dict/Struc2Vec/Struc2Vec.ckpt'), False)
    net.eval()

