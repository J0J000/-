'''
NearestNeighborDistanceMetric类
最近邻距离度量类
'''

# vim: expandtab:ts=4:sw=4
import numpy as np

#计算欧式距离
def _pdist(a, b):
    '''
    计算成对的平方距离
    a：NxM维，代表N个对象，每个对象有M个数值作为embedding进行比较
    b：LxM维，代表L个对象，每个对象有M个数值作为embedding进行比较
    返回NxL矩阵，例如dist[i][j]表示a[i]和b[j]之间的平方和距离
    '''
    a, b = np.asarray(a), np.asarray(b) #拷贝一份数据
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1) #求每个embedding的平方和
    # sum(N) + sum(L) - 2 x [NxM]x[MxL] = [NxL]
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

#计算余弦距离（=1-余弦相似度）
def _cosine_distance(a, b, data_is_normalized=False):
    #需要将余弦相似度转化为类似于欧式距离的余弦距离
    if not data_is_normalized:
        #np.linalg.norm用于求向量的范式，默认为L2范式，等同于求向量的欧式距离
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

#最近邻距离度量类，对于每个目标，返回一个最近的距离
class NearestNeighborDistanceMetric(object):

    def __init__(self, metric, matching_threshold, budget=None):
    #默认matching_threshold=0.2 budge=100

        # 使用最近邻欧式距离
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        # 使用最近邻余弦距离
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        #在级联匹配的函数中调用
        self.matching_threshold = matching_threshold
        self.budget = budget #用于控制feature大小
        self.samples = {} #samples是一个字典{id -> feature list}

    def partial_fit(self, features, targets, active_targets):
    #用于部分拟合，用新的数据更新测量记录
    #在特征集更新模块部分（tracker.update())中调用

        for feature, target in zip(features, targets):
            #对应目标下添加新的feature，更新feature集
            #目标id：feature list
            self.samples.setdefault(target, []).append(feature)

            #控制每个类最大目标数，超过budge时之间忽略
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

        # 筛选激活的目标
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
    #用于比较feature和targets之间的距离，返回一个代价矩阵（cost matrix）
    #调用：在匹配阶段，将distance封装为门控矩阵（gated_metric）
    #      外观信息（ReID得到的深度特征）+运动信息（马氏距离用于度量两个分布相似度）

        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
