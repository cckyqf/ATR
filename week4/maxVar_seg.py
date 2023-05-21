# 最大类间方差分割
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./images/3.PNG"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


threshold = 10
# retval, dst =	cv.threshold(src, thresh, maxval, type[, dst])
# thresh:阈值
# maxval: 处理结果的最大值
# type : 阈值处理的类型
#    cv2.THRESH_BINARY	maxval	0
#    cv2.THRESH_BINARY_INV	0	maxval
#    cv2.THRESH_TRUNC	thresh	当前灰度值

#    cv2.THRESH_OTSU
#    cv2.THRESH_TRIANGLE
threshold, mask = cv2.threshold(img, thresh=threshold, maxval=255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU) 


fig, ax = plt.subplots(1,2)
ax = ax.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].axis("off")
ax[0].set_title("image")

ax[1].axis("off")
ax[1].set_title(f"OTSU thr:{threshold}")
ax[1].imshow(mask, cmap='gray')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.show()

