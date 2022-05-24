import torch
from torch import nn
import time


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

    for epoch in range(num_epochs):
        tik = time.time()
        num_batches = len(data_iter)
        loss = SigmoidBCELoss()
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = model(center, context_negative)
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[
                1])
            l.sum().backward()
            optimizer.step()
            with torch.no_grad():
                l_sum = l.sum()
                l_nums = l.numel()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f"step:{epoch + (i + 1) / num_batches}>>loss:{l_sum / l_nums:.3f}")
    print(f'loss {l_sum / l_nums:.3f}, {l_nums / (time.time() - tik):.1f} 'f'tokens/sec on {str(device)}')


def get_similar_tokens(query_token, k, embed, vocab):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输⼊词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
