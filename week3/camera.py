# （1）获取相机/视频
# （2）抽帧
# （3）边缘检测、保存图像
# （4）关闭相机/视频

import cv2
import os
import shutil
import numpy as np

camera_id = 0
cap = cv2.VideoCapture(camera_id)

# vedio_path = "./a.mp4"
# cap = cv2.VideoCapture(vedio_path)

if not cap.isOpened():
    print("can not open camera")
    exit()

save_dir = f"./camera{camera_id}/"
# 
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)


frame_id = 0
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Canny边缘检测
    edge = cv2.Canny(frame, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)
    
    edge = np.tile(edge[...,None], (1,1,3))
    show_img = np.concatenate([frame,edge], axis=0)
    cv2.imshow(str(camera_id), show_img)
    frame_id += 1
    save_name = save_dir + str(frame_id) + ".jpg"
    cv2.imwrite(save_name, show_img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
