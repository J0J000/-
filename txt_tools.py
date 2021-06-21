'''
输出文本信息的相关函数
'''

import cv2
import glob
import os
from datetime import datetime

def video_to_frames(path):
    """
    输入：path(视频文件的路径)
    """
    # VideoCapture视频读取类
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        #cv2.imwrite("C:\\Users\\15228\\Desktop\\test\\images\\frames%d.jpg" % (i), frame)
    print("Video length=",'{:.3f}'.format(videoCapture.get(0)/1000),"s")
    return

def txt_sec(in_path, out_path):
    """
    输出各目标ID，出现至消失的时间段
    """
    time = 0
    dict_f = {}
    dict_l = {}

    with open(in_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line[0] == 'S':
                if float(line[2:]) == 0: continue
                time = float(line[2:])
            else:
                if line in dict_f:
                    dict_l[line] = time
                else:
                    dict_f[line] = time
    f.close()
    with open(out_path, "w") as f:
        f.write('Video length: %.3fs\nMax ID: %d\n\n' % (time, len(dict_f)))
        f.write('{:<10}{:<25}{:<25}{:<25}*in second\n'.format('ID', 'Frist', 'Last', 'Duration'))
        for i in range(len(dict_f) + 1):
            if not str(i) in dict_f: continue
            f.write('%-10d %-22.3f %-22.3f %-22.3f\n' % (
            i, dict_f[str(i)], dict_l[str(i)], dict_l[str(i)] - dict_f[str(i)]))
    return