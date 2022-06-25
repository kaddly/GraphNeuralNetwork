import torch


def Aggregator(neigh_feat, agg_func='MEAN'):
    if agg_func == 'MEAN':
        return torch.mean(neigh_feat, dim=1)
    elif agg_func == 'MAX':
        return torch.argmax(neigh_feat, dim=1)
    else:
        print('请选择合适的聚合函数')
        raise
