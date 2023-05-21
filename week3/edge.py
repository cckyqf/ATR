# 几种简单的边缘提取方法
import cv2
import matplotlib.pyplot as plt
import numpy as np
from general import extract_edge, fuse_edge_x_y, thr_process

threshold = None

img_path = "./images/lena.bmp"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 简单梯度算子
kernel_h = np.array([[-1,1]], dtype=np.float32)
kernel_v = np.array([[-1],
                     [1]], dtype=np.float32)

edge_1 = extract_edge(img, kernel_h, kernel_v, threshold=threshold)


# Roberts梯度算子
kernel_h = np.array([
    [1, 0],
    [0, -1],
], dtype=np.float32)
kernel_v = np.array([
    [0, 1],
    [-1, 0],
], dtype=np.float32)
edge_roberts = extract_edge(img, kernel_h, kernel_v, threshold=threshold)


# Prewitt梯度算子
kernel_h = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.float32)
kernel_v = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1],
], dtype=np.float32)
edge_prewitt = extract_edge(img, kernel_h, kernel_v, threshold=threshold)


# Sobel梯度算子
kernel_h = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)
kernel_v = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
], dtype=np.float32)
edge_sobel = extract_edge(img, kernel_h, kernel_v, threshold=threshold)


# # # 沿着x/y两个方向的梯度
# edge_sobel = cv2.Sobel(img, cv2.CV_32F, 1, 1)
# # # 沿x方向的边缘检测
# # sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
# # # 沿y方向的边缘检测
# # sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
# # edge_sobel = fuse_edge_x_y(sobelx, sobely)
# edge_sobel = thr_process(edge_sobel, threshold=threshold)



fig, ax = plt.subplots(2,2)
ax = ax.ravel()

ax[0].imshow(edge_1, cmap='gray')
ax[0].axis("off")
ax[0].set_title("简单梯度算子")

ax[1].imshow(edge_roberts, cmap='gray')
ax[1].axis("off")
ax[1].set_title("roberts")

ax[2].imshow(edge_prewitt, cmap='gray')
ax[2].axis("off")
ax[2].set_title("prewitt")

ax[3].imshow(edge_sobel, cmap='gray')
ax[3].axis("off")
ax[3].set_title("Sobel")


#解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei'] # 设置字体
# plt.rcParams['axes.unicode_minus'] = False # 使坐标轴刻度表签正常显示正负号
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.show()
