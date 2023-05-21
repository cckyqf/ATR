# 两帧差分
import cv2
import os
import numpy as np

imgs_dir = "./images/DATA/"
# 把目录下的文件存为一个列表
imgs_ls = os.listdir(imgs_dir)


# 根据帧号排序
imgs_id = [int(os.path.splitext(img_name)[0]) for img_name in imgs_ls]
imgs_id = sorted(imgs_id)
imgs_ls = [str(img_id) + ".jpg" for img_id in imgs_id]

# print(imgs_ls)

# 总帧数
num_frames = len(imgs_ls)
# print(imgs_ls)

kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(5,5))

threshold = 10
frame_old = cv2.imread(imgs_dir + imgs_ls[0])
frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
frame_old = frame_old.astype(np.float32)
frame_id = 1

for i in range(1,num_frames):

    img_path = imgs_dir + imgs_ls[i]
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype(np.float32)


    # 差分
    mask = np.abs(frame - frame_old)

    # 阈值处理
    mask = mask > threshold

    # 形态学处理
    mask = (mask.astype(np.float32)*255).astype(np.uint8)
    mask = cv2.morphologyEx(mask,op=cv2.MORPH_OPEN,kernel=kernel,iterations=1)

    frame_old = frame


    cv2.imshow("images", mask)
    # 等待时间，ms
    if cv2.waitKey(100) == ord('q'):
        break

cv2.destroyAllWindows()
