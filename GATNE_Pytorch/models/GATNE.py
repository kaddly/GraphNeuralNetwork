import torch
from torch import nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_type_count, attention_size, features,
                 agg_func='SUM', **kwargs):
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
        super(GraphEncoder, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = attention_size
        self.agg_func = agg_func

        self.features = None
        if features is not None:
            self.features = features
            self.feature_dim = self.features.shape[-1]
            self.embed_trans = nn.Parameter(torch.FloatTensor(self.feature_dim, embedding_size))
            self.u_embed_trans = nn.Parameter(torch.FloatTensor(edge_type_count, self.feature_dim, embedding_u_size))
        else:
            self.node_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, embedding_size))
            self.node_type_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, edge_type_count, embedding_size))
        # 定义MR矩阵
        self.trans_weights = nn.Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        # 计算attention矩阵
        self.trans_weights_s1 = nn.Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, attention_size))
        self.trans_weights_s2 = nn.Parameter(torch.FloatTensor(edge_type_count, attention_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.features is not None:
            nn.init.xavier_uniform_(self.embed_trans.data)
            nn.init.xavier_uniform_(self.u_embed_trans.data)
        else:
            nn.init.uniform_(self.node_embeddings.data)
            nn.init.uniform_(self.node_type_embeddings.data)
        nn.init.xavier_uniform_(self.trans_weights.data)
        nn.init.xavier_uniform_(self.trans_weights_s1.data)
        nn.init.xavier_uniform_(self.trans_weights_s2.data)

    def forward(self, inputs, node_types, node_neigh):
        if self.features is None:
            node_embed = self.node_embeddings[inputs]
            # [64,2,10,32]
            node_embed_neighbors = torch.cat(
                [self.node_type_embeddings[:, i, :][node_neigh[:, i, :]].unsqueeze(1) for i in
                 range(self.edge_type_count)], dim=1)
        else:
            node_embed = torch.mm(self.features[inputs], self.embed_trans)
            # [2,64*10,142]*[2,142,32]->[2,64*10,32]->[64,2,10,32]
            node_embed_neighbors = torch.bmm(
                self.features[node_neigh].permute(1, 0, 2, 3).reshape(self.edge_type_count, -1, self.feature_dim),
                self.u_embed_trans).reshape(self.edge_type_count, -1, node_neigh.shape[-1],
                                            self.embedding_u_size).permute(1, 0, 2, 3)
        # Ui; 对邻居信息求和; [64, 2, 32]
        if self.agg_func == 'SUM':
            node_type_embed = torch.sum(node_embed_neighbors, dim=2)
        elif self.agg_func == 'MEAN':
            node_type_embed = torch.mean(node_embed_neighbors, dim=2)
        else:
            raise ValueError("please choice else aggregator!")

        # [64, 32, 256]
        trans_w = self.trans_weights[node_types]
        # [64, 32, 16]
        trans_w_s1 = self.trans_weights_s1[node_types]
        # [64, 16, 1]
        trans_w_s2 = self.trans_weights_s2[node_types]
        # [64, 1, 2]
        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)

        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed


class GraphDecoder(nn.Module):
    def __init__(self, num_nodes, embedding_size, **kwargs):
        super(GraphDecoder, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.weights = nn.Parameter(torch.FloatTensor(num_nodes, embedding_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)

    def forward(self, embed, contest_negative):
        pre = torch.bmm(embed, self.weights[contest_negative].permute(0, 2, 1))
        return pre


class GATNEModel(nn.Module):
    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_type_count, attention_size, features,
                 **kwargs):
        super(GATNEModel, self).__init__(**kwargs)
        self.encoder = GraphEncoder(num_nodes, embedding_size, embedding_u_size, edge_type_count, attention_size,
                                    features, **kwargs)
        self.decoder = GraphDecoder(num_nodes, embedding_size)

    def forward(self, inputs, node_types, node_neigh, context_negative):
        embed = self.encoder(inputs, node_types, node_neigh)
        return self.decoder(embed, context_negative)
