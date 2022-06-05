import numpy as np
from collections import defaultdict


def fastdtw(x, y, radius=1, dist=lambda a, b: np.sum(np.abs(a - b))):
    # 参数：节点x、y的时间序列；radius;距离函数

    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, dist=dist)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
    window = __expand_window(path, len(x), len(y), radius)
    return dtw(x, y, window, dist=dist)


def dtw(x, y, window=None, dist=lambda a, b: np.sum(np.abs(a - b))):
    # 参数：节点x、y的时间序列；搜索范围；距离函数
    len_x, len_y = len(x), len(y)  # 时间序列的长度
    # 搜所范围的确定
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)

    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)  # (距离，x时间点来源，y时间点来源)

    # 若从左上角向右下角寻找最短路径过去的话
    for i, j in window:
        dt = dist(x[i - 1], y[j - 1])  # 计算当前时刻组合的左上角时刻组合的‘距离’
        D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                      (D[i, j - 1][0] + dt, i, j - 1),
                      (D[i - 1, j - 1][0] + dt, i - 1, j - 1), key=lambda a: a[0])  # 移动法则

    # 路径回溯，从终点坐标(len_x-1,len_y-1)开始
    path = []  # 存放路径坐标的列表

    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i - 1, j - 1))  # 首先将终点或者当前坐标加入path
        i, j = D[i, j][1], D[i, j][2]
    # 自己写的，注意查看windows范围
    # i,j=len_x-1,len_y-1
    # while not(i==j==0):
    # path.append((i,j))
    # i,j=D[i,j][1],D[i,j][2]
    path.reverse()
    return D[len_x, len_y][0], path


def __expand_window(path, len_x, len_y, radius):
    # 路径加粗
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius + 1)
                     for b in range(-radius, radius + 1)):
            path_.add((a, b))
    # 根据加粗的路径得到限制移动窗口
    # 数轴扩大2倍，原先的一个小方格括为4个之后的坐标集合：（1个坐标：4个坐标）
    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j
    return window


def __reduce_by_half(x):
    """
    input list, make each two element together by half of sum of them
    :param x:
    :return:
    """
    x_reduce = []
    lens = len(x)
    for i in range(0, lens, 2):
        if (i + 1) >= lens:
            half = x[i]
        else:
            if isinstance(x[i], tuple):
                d_list = (x[i][0] + x[i + 1][0]) / 2
                d_freq = (x[i][1] + x[i + 1][1]) / 2
                half = (d_list, d_freq)
            else:
                half = (x[i] + x[i + 1]) / 2
        x_reduce.append(half)
    return x_reduce
