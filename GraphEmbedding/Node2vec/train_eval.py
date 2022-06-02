import torch
from torch import nn
import time
import os
from datetime import timedelta


class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super(SigmoidBCELoss, self).__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


def train(model, data_iter, lr, num_epochs, device):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    model.net.apply(init_weights)
    net = model.net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss = SigmoidBCELoss()
    start_time = time.time()

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    # 模型参数保存路径
    saved_dir = '../saved_dict'
    model_file = 'Node2Vec'
    parameter_path = os.path.join(saved_dir, model_file)
    if not os.path.exists(parameter_path):
        os.mkdir(parameter_path)

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = model(center, context_negative)
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[
                1])
            l.sum().backward()
            optimizer.step()
            if total_batch % 20 == 0:
                with torch.no_grad():
                    l_sum = l.sum()
                    l_nums = l.numel()
                if l_sum < dev_best_loss:
                    torch.save(net.state_dict(), os.path.join(parameter_path, model_file + '.ckpt'))
                    dev_best_loss = l_sum
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:>6},  total_loss: {1:>5.3f},  Time: {2} {3}'
                print(msg.format(total_batch, l_sum / l_nums, time_dif, improve))
            total_batch += 1
            if total_batch - last_improve > 2000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def get_similar_tokens(query_token, k, embed, vocab):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输⼊词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
