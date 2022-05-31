import numpy as np


def create_alias_table(Prob_val):
    """
    :param Prob_val:传入概率列表
    :return:返回一个accept 概率数组 和 alias的标号数组
    """
    L = len(Prob_val)
    # 初始化两个数组
    accept_prob = np.zeros(L)  # 存的是概率
    alias_index = np.zeros(L, dtype=np.int)  # 存的是下标/序号

    # 大的队列用于存储面积大于1的节点标号，小的队列用于存储面积小于1的节点标号
    small_queue = []
    large_queue = []

    # 把Prob_val list中的值分配到大小队列中
    for index, prob in enumerate(Prob_val):
        accept_prob[index] = L * prob

        if accept_prob[index] < 1.0:
            small_queue.append(index)
        else:
            large_queue.append(index)

    # 1.每次从两个队列中各取一个，让大的去补充小的，然后小的出small队列
    # 2.在看大的减去补给小的之后剩下的值，如果大于1，继续放到large队列；如果恰好等于1，也出队列；如果小于1加入small队列中
    while small_queue and large_queue:
        small_index = small_queue.pop()
        large_index = large_queue.pop()
        # 因为alias_index中存的：另一个事件的标号，那现在用大的概率补充小的概率，标号就要变成大的事件的标号了
        alias_index[small_index] = large_index
        # 补充的原则是：大的概率要把小的概率补满（补到概率为1），然后就是剩下的
        accept_prob[large_index] = accept_prob[large_index] + accept_prob[small_index] - 1.0
        # 判断补完后，剩下值的大小
        if accept_prob[large_index] < 1.0:
            small_queue.append(large_index)
        else:
            large_queue.append(large_index)
    return accept_prob, alias_index


def alias_sample(accept_prob, alias_index):
    N = len(accept_prob)

    # 扔第一个骰子，产生第一个1~N的随机数,决定落在哪一列
    random_num1 = int(np.floor(np.random.rand() * N))
    # 扔第二个骰子，产生0~1之间的随机数，判断与accept_prob[random_num1]的大小
    random_num2 = np.random.rand()

    # 如果小于Prab[i]，则采样i，如果大于Prab[i]，则采样Alias[i]
    if random_num2 < accept_prob[random_num1]:
        return random_num1
    else:
        alias_index[random_num1]


class AliasGenerator:
    def __init__(self, sampling_weights):
        self.accept_prob, self.alias_index = create_alias_table(sampling_weights)

    def draw(self):
        return alias_smaple(self.accept_prob, self.alias_index)
