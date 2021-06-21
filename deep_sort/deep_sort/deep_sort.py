'''
DeepSort类
核心类，调用其他模块(ReID、Track、Tracker)

对外接口：
self.deepsort = DeepSort(args.deepsort_checkpoint) #实例化
outputs = self.deepsort.update(bbox_xcycwh,cls_conf,im) #通过接收目标检测结果进行更新
'''

import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    #定义了某些阈值，加载特征提取器的网络
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence #yolov5中检测结果的置信度阈值，筛选置信度<0.3的detection
        self.nms_max_overlap = nms_max_overlap #非极大抑制阈值，默认为1.0表示不进行抑制

        # 提取图片的embedding（外观特征），返回一个batch图片对应的特征
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist #用于级联匹配部分，若大于该阈值则忽略
        nn_budget = 100 #每个类别的最大样本数，若超过该值则删除旧值

        # 计算距离，返回最近的样本，提供了欧氏距离和余弦距离两种度量方式（即第一个参数可选填“cosine”或“euclidean”）
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    '''
    DeepSort的update流程：

    1、根据传入参数（bbox_xywh, conf, img）是，使用ReID模型提取对应bbox的表观特征
    2、构造detections列表，其中就是detection类，在此处限制了bbox的最小置信度
    3、使用NMS（默认nms_thres=1，即不使用）
    4、Tracker类进行一次预测，然后将detection传入，进行更新
    5、最后将Tracker中保存的轨迹中状态属于confirmed态的轨迹返回
    '''
    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]

        #从原图中crop bbox对应图片，并计算得到embedding
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        #筛选<min_confidence的目标，构造一个由detection对象构成的列表detections
        #detection是一个存储图中一个bbox结果
        #需要：bbox（tlwh格式）、对应置信度、对应embedding
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        #使用非极大抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        #tracker给出一个预测结果，然后将detection传入，进行卡尔曼滤波
        self.tracker.predict()
        self.tracker.update(detections)

        #存储结果，实现可视化
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


