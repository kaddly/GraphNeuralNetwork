import torch
from torch import nn
import time
import os
from datetime import timedelta


class mix_loss(nn.Module):
    def __init__(self, beta, alpha):
        self.beta = beta
        self.alpha = alpha

    def forward(self, y_hat, y):
        b_ = torch.ones(y.shape)
        b_[y != 0] = self.beta
        l2_loss = torch.square((y-y_hat)*b_).sum()
        l1_loss = self.alpha * 2 * torch.trace(torch.mm(torch.mm(y.T, y_hat), y))
        return l1_loss.sum()+l2_loss.sum()




