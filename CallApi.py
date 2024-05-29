import detect_api
import cv2
import torch
import time


def streaming_detect(rtsp_path:str):
    #ip_camera_url = 'http://admin:admin@192.168.242.98:8081/video'
    cap = cv2.VideoCapture(rtsp_path)
    a = detect_api.DetectAPI(weights='yolov5s.pt',device='0')
    with torch.no_grad():
        while cap.isOpened():
            st_time = time.time()
            rec, img = cap.read()
            if rec:
                result, names = a.detect([img])
                img = result[0][0]  # 每一帧图片的处理结果图片
                # 每一帧图像的识别结果（可包含多个物体）
                for cls, (x1, y1, x2, y2), conf in result[0][1]:
                    print(names[cls], x1, y1, x2, y2, conf)  # 识别物体种类、左上角x坐标、左上角y轴坐标、右下角x轴坐标、右下角y轴坐标，置信度
                    '''
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
                    cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))'''
                print()  # 将每一帧的结果输出分开
                height,width = img.shape[:2]
                res = cv2.resize(img,(width//2,height//2),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）
                cv2.imshow("video", res)

            if cv2.waitKey(1) == ord('q'):
                break
            print("eltime: {}".format(time.time()-st_time))
    print("quit")

def video_detect(video_path:str):
    is_skip = False
    a = detect_api.DetectAPI(weights='best.pt',device='0')
    cap=cv2.VideoCapture(video_path)
    with torch.no_grad():
        while cap.isOpened():
            rec, img = cap.read()
            if img is None:
                cap = cv2.VideoCapture(video_path)
                rec, img = cap.read()
            if is_skip:
                is_skip = False
                continue
            else:
                is_skip = True
            st_time = time.time()
            rec, img = cap.read()
            if rec:
                result, names = a.detect([img])
                img = result[0][0]  # 每一帧图片的处理结果图片
                # 每一帧图像的识别结果（可包含多个物体）
                for cls, (x1, y1, x2, y2), conf in result[0][1]:
                    print(names[cls], x1, y1, x2, y2, conf)  # 识别物体种类、左上角x坐标、左上角y轴坐标、右下角x轴坐标、右下角y轴坐标，置信度
                    '''
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
                    cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))'''
                print()  # 将每一帧的结果输出分开
                
                height,width = img.shape[:2]
                res = cv2.resize(img,(width//2,height//2),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）
                cv2.imshow("video", res)

            if cv2.waitKey(1) == ord('q'):
                break
            print("eltime: {}".format(time.time()-st_time))
    print("quit")

def picture_detect(img_path:str):
    a = detect_api.DetectAPI(weights='yolov5s.pt',device='0')
    img = cv2.imread(img_path)
    with torch.no_grad():
        st_time = time.time()
        result, names = a.detect([img])
        img = result[0][0]  # 每一帧图片的处理结果图片
        # 每一帧图像的识别结果（可包含多个物体）
        print(len(result[0]))
        for cls, (x1, y1, x2, y2), conf in result[0][1]:
            print(names[cls], x1, y1, x2, y2, conf)  # 识别物体种类、左上角x坐标、左上角y轴坐标、右下角x轴坐标、右下角y轴坐标，置信度
            '''
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))'''
        print()  # 将每一帧的结果输出分开
        height,width = img.shape[:2]
        res = cv2.resize(img,(width//2,height//2),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）
        cv2.imshow("video", res)
        print("eltime: {}".format(time.time()-st_time))
        cv2.waitKey(0)
    print("quit")

if __name__ == '__main__':
    #rtsp_path = r"rtsp://admin:13860368866xzc@10.70.37.10/Streaming/Channels/1"
    img_path = r"D:\Code\Python\BodyCheck\img\1.jpg"
    video_detect(r'C:\Users\01477483\HM Web\RecordFiles\2024-05-28\10.70.37.10_01_20240528130632509_8.mp4')