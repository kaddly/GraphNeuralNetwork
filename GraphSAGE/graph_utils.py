import torch


def Aggregator(feat_data, neigh_feat, agg_func='MEAN', gcn=False):
    if not gcn:
        neigh_feat = torch.cat([feat_data.unsqueeze(1), neigh_feat], dim=1)

    if agg_func == 'MEAN':
        return torch.mean(neigh_feat, dim=1)

    elif agg_func == 'MAX':
        return torch.argmax(neigh_feat, dim=1)
