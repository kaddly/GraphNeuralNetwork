import math
import torch
from torch import nn
from torch.nn import functional as F


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size, **kwargs):
        super(NSLoss, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = nn.Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weight = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, y_hat, emb, label):
        n = y_hat.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(emb, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weight, self.num_sampled*n, replacement=True
        ).resize(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, emb.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n
