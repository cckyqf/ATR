# 两帧差分
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
# if os.path.exists(save_dir):
#     # shutil.rmtree(save_dir)
#     pass
# os.mkdir(save_dir)


# 开运算，去除连通的小区域
# cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(5,5))

threshold = 10
ret, frame_old = cap.read()
frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
frame_old = frame_old.astype(np.float32)
frame_id = 1

while True:

    ret, frame = cap.read()
    if not ret:
        print("can not get frame !")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype(np.float32)

    # 差分
    mask = np.abs(frame - frame_old)

    # 阈值处理
    mask = mask > threshold
    mask = (mask.astype(np.float32)*255).astype(np.uint8)

    # 形态学处理
    mask = cv2.morphologyEx(mask,op=cv2.MORPH_OPEN,kernel=kernel,iterations=1)

    frame_old = frame

    # frame_id += 1
    # save_name = save_dir + str(frame_id) + ".jpg"
    # cv2.imwrite(save_name, mask)

    cv2.imshow(str(camera_id), mask)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
