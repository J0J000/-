from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2

ID_txt_path='ID_inf.txt'

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

#检测框标签
def plot_bboxes(image, bboxes, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  #字体粗细
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  #框
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        '''
        lb=coco_id_name_map[int(cls_id)]
        with open(save_txt_path, 'a') as f:
            f.write("%s %d %d %d %d %d\n" % (lb, pos_id,x1, y1, x2, y2))
        '''

    return image


def update_tracker(target_detector, image):

        new_faces = []
        _, bboxes = target_detector.detect(image)

        bbox_xywh = []
        confs = []
        bboxes2draw = []
        face_bboxes = []
        if len(bboxes):

            #将detections格式归一化
            for x1, y1, x2, y2, _, conf in bboxes:
                
                obj = [
                    int((x1+x2)/2), int((y1+y2)/2),
                    x2-x1, y2-y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            #将detections送入跟踪器（deepsort）
            outputs = deepsort.update(xywhs, confss, image)

            #输出结果
            if len(outputs) != 0:
                for j, output in enumerate(outputs):
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2]
                    bbox_h = output[3]
                    identity = output[-1]
                    with open(ID_txt_path, 'a') as f:
                        #f.write("%d %d %d %d %d\n" % (identity,bbox_left, bbox_top, bbox_w, bbox_h))
                        f.write("%d\n"%identity)

            for value in list(outputs):
                x1,y1,x2,y2,track_id = value
                bboxes2draw.append(
                    (x1, y1, x2, y2, '', track_id)
                )

        image = plot_bboxes(image, bboxes2draw)

        return image, new_faces, face_bboxes
