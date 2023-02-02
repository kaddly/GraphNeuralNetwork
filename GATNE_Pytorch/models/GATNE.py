import math
import torch
from torch import nn
import torch.nn.functional as F


class GATNEModel(nn.Module):
    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_type_count, attention_size, features, **kwargs):
        """
        初试化函数
        :param num_nodes:节点数量
        :param embedding_size: baseEmbedding嵌入的维度
        :param embedding_u_size: edgeEmbedding嵌入的维度
        :param edge_type_count: 边的类型数
        :param attention_size: 注意力层的维度
        :param features: 节点的属性特征
        :param kwargs:其他参数
        """
        super(GATNEModel, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.dim_a = attention_size

        self.features = None
        if features is not None:
            self.features = features
            feature_dim = self.features.shape[-1]
            self.embed_trans = nn.Parameter(torch.FloatTensor(feature_dim, embedding_size))

    def forward(self):
        pass
