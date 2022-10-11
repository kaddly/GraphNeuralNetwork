import math
import torch
from torch import nn
import torch.nn.functional as F


class GATNEModel(nn.Module):
    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features, **kwargs):
        super(GATNEModel, self).__init__(**kwargs)
        self.num_nodes = num_nodes  # 节点数量
        self.embedding_size = embedding_size  # 每个节点输出的embedding_size
        self.embedding_u_size = embedding_u_size  # 节点作为邻居初始化size
        self.edge_type_count = edge_type_count  # 类别数量
        self.dim_a = dim_a  # 中间隐层特征数量

        self.features = None
        if features is not None:  # GATNE-I
            self.features = features
            feature_dim = self.features.shape[-1]
            self.embed_trans = nn.Parameter(
                torch.FloatTensor(feature_dim, embedding_size))  # [142, 200]; bi-base embedding
            self.u_embed_trans = nn.Parameter(
                torch.FloatTensor(edge_type_count, feature_dim, embedding_u_size))  # [2, 142, 10]; 初始化ui
        else:  # 初始化 base embedding GATNE-T
            self.node_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, embedding_size))  # [511, 200]
            self.node_type_embeddings = nn.Parameter(  # 初始化 edge embedding
                torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size)
            )  # [511, 2, 10]
        self.trans_weights = nn.Parameter(  # [2, 10, 200]; 定义Mr矩阵
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        self.trans_weights_s1 = nn.Parameter(  # [2, 10, 20]  计算attention使用
            torch.FloatTensor(edge_type_count, embedding_u_size, dim_a)
        )
        self.trans_weights_s2 = nn.Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))  # [2, 20, 1]

        self.reset_parameters()

    def reset_parameters(self):
        if self.features is not None:
            self.embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
            self.u_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        else:
            self.node_embeddings.data.uniform_(-1.0, 1.0)
            self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, train_inputs, train_types, node_neigh):
        if self.features is None:
            node_embed = self.node_embeddings[train_inputs]  # 每个节点对应的mebedding
            node_embed_neighbors = self.node_type_embeddings[node_neigh]  # 每个节点对应的neighbors
        else:  # self.features:节点特征; self.embed_trans
            node_embed = torch.mm(self.features[train_inputs], self.embed_trans)  # [64, 200]
            node_embed_neighbors = torch.einsum('bijk,akm->bijam', self.features[node_neigh], self.u_embed_trans)  # 生成ui; [64, 2, 10, 142]*[2, 142, 10];
        node_embed_tmp = torch.cat(  # [64, 2, 10, 10]; 聚合每个类别周围邻居信息
            [
                node_embed_neighbors[:, i, :, i, :].unsqueeze(1)  # [64, 1, 10, 10]
                for i in range(self.edge_type_count)
            ],
            dim=1,
        )
        node_type_embed = torch.sum(node_embed_tmp, dim=2)  # Ui; 对邻居信息求和; [64, 2, 10]

        trans_w = self.trans_weights[train_types]  # [64, 10, 200]
        trans_w_s1 = self.trans_weights_s1[train_types]  # [64, 10, 20]
        trans_w_s2 = self.trans_weights_s2[train_types]  # [64, 20, 1]

        attention = F.softmax(  # [64, 1, 2]
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)  # [64, 1, 2] * [64, 2, 10] 对node_type_embed做attention求和
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)  # [64, 200] + [64, 1, 10] * [64, 10, 200] => [64, 200]

        last_node_embed = F.normalize(node_embed, dim=1)  # dim=1, L2-norm; (last_node_embed*last_node_embed).sum(axis=1)

        return last_node_embed
