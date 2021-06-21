'''
Detection类
保存检测框，包含坐标信息、bbox的置信度和embedding
提供不同bbox位置格式的转换方法

tlwh：左上角坐标+宽高
tlbr：左上角坐标+右下角坐标
xyah：中心坐标+宽高比+高
'''

# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
