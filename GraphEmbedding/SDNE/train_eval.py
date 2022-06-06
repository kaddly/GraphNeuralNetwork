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
        l1_loss = self.alpha * 2 * torch.trace(torch.mm(torch.mm(y_hat.T, y), y_hat)) / (y.shape[0].type(torch.float))
        return l1_loss


class loss_second(nn.Module):
    def __init__(self, beta):
        super(loss_second, self).__init__()
        self.beta = beta

    def forward(self, y_hat, y):
        b_ = torch.ones(y.shape)
        b_[y != 0] = self.beta
        l2_loss = torch.square((y_hat - y) * b_).sum(dim=-1)
        return l2_loss.mean()


def train():
    pass
