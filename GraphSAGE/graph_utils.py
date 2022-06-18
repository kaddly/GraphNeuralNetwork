import torch


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def Aggregator(feat_data, neigh_feat, val_lens, agg_func='MEAN', gcn=False):

    if agg_func == 'MEAN':
        neigh_feat = sequence_mask(neigh_feat, val_lens)
        if gcn:
            neigh_feat = torch.cat([feat_data.unsqueeze(1), neigh_feat], dim=1)
            val_lens = val_lens+1
        return torch.sum(neigh_feat, dim=1) / val_lens.repeat(1, neigh_feat.shape[3])

    elif agg_func == 'MAX':
        neigh_feat = sequence_mask(neigh_feat, val_lens, value=-1e6)
        if gcn:
            neigh_feat = torch.cat([feat_data.unsqueeze(1), neigh_feat], dim=1)
        return torch.argmax(neigh_feat, dim=1)

    else:
        print('请选择合适的聚合函数')
        raise
