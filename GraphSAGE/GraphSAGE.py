import torch
from torch import nn
import torch.nn.functional as F
import random


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size, gcn=False, **kwargs):
        super(SageLayer, self).__init__(**kwargs)

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(
            torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))  # 创建weight

        self.init_params()  # 初始化参数

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)  # concat自己信息和邻居信息
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN',
                 **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        self.raw_features = raw_features  # 点的特征
        self.adj_lists = adj_lists  # 边的连接

        self.blocks = nn.Sequential()
        for index in range(num_layers):
            layer_size = out_size if index != 0 else input_size
            self.blocks.add_module(f'sage{index}', SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings.    《minbatch 过程，涉及到的所有节点》
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]  # 第一次放入的节点，batch节点
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):  # 每层的Sage
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes)  # 获得neighbors。 聚合自己和邻居节点，点的dict，涉及到的所有节点
            nodes_batch_layers.insert(0, (
                lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))  # 聚合自己和邻居节点，点的dict，涉及到的所有节点
        # insert,0 从最外层开始聚合
        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]  # 聚合自己和周围的节点
            pre_neighs = nodes_batch_layers[index - 1]  # 这层节点的上层邻居的所有信息。聚合自己和邻居节点，点的dict，涉及到的所有节点
            # self.dc.logger.info('aggregate_feats.') aggrefate_feats=>输出GraphSAGE聚合后的信息
            aggregate_feats = self.aggregate(nb, pre_hidden_embs,
                                             pre_neighs)  # 聚合函数。nb-这一层的节点， pre_hidden_embs-feature，pre_neighs-上一层节点
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)  # 第一层的batch节点，没有进行转换
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)  # 进入SageLayer。weight*concat(node,neighbors)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]  # 记录将上一层的节点编号。
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]  # self.adj_lists边矩阵，获取节点的邻居
        if not num_sample is None:  # 对邻居节点进行采样，如果大于邻居数据，则进行采样
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
                           in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # 聚合本身节点和邻居节点
        _unique_nodes_list = list(set.union(*samp_neighs))  # 这个batch涉及到的所有节点
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))  # 字典编号
        return samp_neighs, unique_nodes, _unique_nodes_list  # 聚合自己和邻居节点，点的dict，涉及到的所有节点

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs  # 聚合自己和邻居节点，涉及到的所有节点，点的dict

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]  # 都是True，因为上文中，将nodes加入到neighs中了
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]  # 在把中心节点去掉
        # self.dc.logger.info('2')
        if len(pre_hidden_embs) == len(unique_nodes):  # 如果涉及到所有节点，保留原矩阵。如果不涉及所有节点，保留部分矩阵。
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')  将对应到的边，构建邻接矩阵
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))  # 本层节点数量，涉及到上层节点数量
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  # 构建邻接矩阵
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1  # 加上上两个步骤，都是构建邻接矩阵;
        # self.dc.logger.info('4')
        # mask - 邻接矩阵
        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)  # 按行求和，保持和输入一个维度
            mask = mask.div(num_neigh).to(embed_matrix.device)  # 归一化操作
            aggregate_feats = mask.mm(embed_matrix)  # 矩阵相乘，相当于聚合周围邻接信息求和

        elif self.agg_func == 'MAX':
            # print(mask)
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            # self.dc.logger.info('5')
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        # self.dc.logger.info('6')

        return aggregate_feats
