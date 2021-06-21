from Detector import Detector
import imutils
import cv2
from datetime import datetime
from txt_tools import video_to_frames,txt_sec

video_path='2.MP4'
ID_txt_path='ID_inf.txt'
Duration_txt_path='Duration_inf.txt'

def main():

    # 调用检测接口
    func_status = {}
    func_status['headpose'] = None
    
    name = 'MOT_result'

    # 创建检测器
    det = Detector()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))

    #print('fps:', fps)
    t = int(1000/fps)

    #size = None
    videoWriter = None

    d1 = datetime.now()
    j=0
    while True:

        # try:
        _, im = cap.read()

        with open(ID_txt_path, 'a') as f:
            #f.write("frame %d\n" %j) #以帧为单位
            f.write("S "+'{:.3f}'.format(cap.get(0)/1000)+"\n") #以秒为单位
        j=j+1

        if im is None:
            break
        
        result = det.feedCap(im, func_status) #调用检测接口，im为BGR图像，返回的results是字典
        result = result['frame'] #返回可视化后的图像
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))


        videoWriter.write(result)
        '''
        cv2.imshow(name, result)
        cv2.waitKey(t)
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        '''
        # except Exception as e:
        #     print(e)
        #     break

    d2 = datetime.now()
    print("Detection & Tracking time cost = ", (d2 - d1))

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    pp1=datetime.now()
    video_to_frames(video_path)
    pp2=datetime.now()
    print("Pre-processing time cost = ", (pp2 - pp1))

    t1 = datetime.now()
    main()
    p1 = datetime.now()
    txt_sec(ID_txt_path,Duration_txt_path)
    p2 = datetime.now()
    t2 = datetime.now()
    print("Txt Processing time cost = ", (p2 - p1))
    print("Whole time cost = ", (t2 - t1))
