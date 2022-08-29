import random
import collections


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""

    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(0, len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


class Poisson:
    def __init__(self, rate, time=1):
        self.rate = rate
        self.time = time

        self.EXP_NUM = 100000  # 实验次数
        self.NUM_LEVEL = 2  # 数量级

    def generator(self, prob):
        """仿真结果生成器"""
        while True:
            if random.random() < prob:
                yield 1
            else:
                yield 0

    def perform_exp(self, rate, time):
        """
        进行一次实验
        每次实验中，时间分片的数量比rate高两个数量级
        :param rate:单位时间内发生的频率
        :param time:观测事件发生次数的时间范围
        :return:
        """
        level = len(str(rate))
        shard_num = 10 ** (level + self.NUM_LEVEL)  # 计算时间分片的数量

        gen = self.generator(rate / shard_num)

        cnt = 0
        for _ in range(time * shard_num):
            cnt += next(gen)

        return cnt

    def perform_exps(self, exp_num, rate, time):
        """多次实验，得到分布"""
        lst = []
        for _ in range(exp_num):
            lst.append(self.perform_exp(rate, time))

        return sorted(collections.Counter(lst).items(), key=lambda e: e[0])

    @property
    def sample_weights(self):
        sorted_list = self.perform_exps(self.EXP_NUM, self.rate, self.time)
        s = sum([e[1] for e in sorted_list])
        return [e[1] / s for e in sorted_list]

    @staticmethod
    def calculator(rate, t, k):
        """用于计算泊松函数的概率 P(k|t,lambda)
           rate: lambda
           t: t
           k: k
        """
        import math
        return (rate * t) ** k / math.factorial(k) * math.exp(-rate * t)
