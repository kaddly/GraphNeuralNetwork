import random


def sampling(src_nodes, sample_num, neighbor_table):
    """
    根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    """
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        if len(neighbor_table[sid]) < sample_num:
            res = random.choices(neighbor_table[sid], k=sample_num)
        else:
            res = random.sample(neighbor_table[sid], k=sample_num)
        results.extend(res)
    return results


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样

    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result
