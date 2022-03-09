# 面向视频内容的物品检测定位和识别技术研究与实现


[https://github.com/J0J000/-](https://github.com/J0J000/-)

## 项目目标：

可以在视频中准确地检测和定位包，车，手机等物品，并可以对物品进行准确识别。

## 项目介绍：
采用基于深度神经网络的技术路线。

检测方面，采用YOLOv5网络结构；网络模型的训练使用MS COCO数据集，其中包含91种常见物体，328,000张图像和2,500,000个标签。

跟踪方面，提取目标运动信息和表观信息，通过相似度计算和数据关联获取目标运动轨迹。

## 程序运行：
### 1.运行环境：

`GPU + CUDA11.0 + cuDNN8.0 + Pytorch1.7.1`

### 2.依赖配置：

`$ pip install -r requirements.txt`

### 3.目标类别设置：

修改`MOT\Detector.py`文件，第63行代码`if not lbl in ['person']:`

可实现的目标类型包括：

`coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}`

### 4.目标文件设置：

修改`MOT\demo.py`文件，第7~9行代码

`video_path='example_input.MP4'`

`ID_txt_path='example_ID_inf.txt'`

`Duration_txt_path='example_Duration_inf.txt'`

### 5.程序运行：

`$ python demo.py`

### 6.结果展示：

![](https://github.com/J0J000/-/blob/main/output/1/det_frames/frames10.jpg)
![](https://github.com/J0J000/Multiple-Objects-Tracking/blob/main/output/4/c00ca5d7f86fd00082b20cea847e93f.png)
![](https://github.com/J0J000/Multiple-Objects-Tracking/blob/main/output/4/3c1e063ab920a4164dff17d249141fd.png)

