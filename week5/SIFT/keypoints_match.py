# 特征点匹配

import cv2
import matplotlib.pyplot as plt
import numpy as np

from general import get_SIFT
from general import BFMatch, BFMatchKnn, FlannMatchKnn
from general import compute_homography

print(cv2.__version__)

# 待检测图像
img_path = "./images/lenna_wrap.png"
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测关键点，并计算特征描述子
keypoints, descriptors = get_SIFT(img_gray)

img_mask_path = "./images/lenna.png"
img_mask = cv2.imread(img_mask_path)
img_mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
keypoints_mask, descriptors_mask = get_SIFT(img_mask_gray)


topK = None

# 暴力匹配法
matches = BFMatch(descriptors, descriptors_mask, topK=topK)
img_show_bf = cv2.drawMatches(img, keypoints, img_mask, keypoints_mask, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# KNN暴力匹配法
matches = BFMatchKnn(descriptors, descriptors_mask, topK=topK)
img_show_KNN = cv2.drawMatchesKnn(img, keypoints, img_mask, keypoints_mask, [matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

matches = FlannMatchKnn(descriptors, descriptors_mask, topK=topK)
img_show_FLANN = cv2.drawMatchesKnn(img, keypoints, img_mask, keypoints_mask, [matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# 从查询图像到模板图像的单应矩阵H
H, matches = compute_homography(keypoints, keypoints_mask, matches)
imgOut = cv2.warpPerspective(img, H, (img_mask.shape[1], img_mask.shape[0]), flags=cv2.INTER_LINEAR)


fig, ax = plt.subplots(1, 4)
ax = ax.ravel()

ax[0].imshow(cv2.cvtColor(img_show_bf, cv2.COLOR_BGR2RGB))
ax[0].axis("off")
ax[0].set_title("暴力匹配法")
ax[1].imshow(cv2.cvtColor(img_show_KNN, cv2.COLOR_BGR2RGB))
ax[1].axis("off")
ax[1].set_title("暴力匹配法-KNN")
ax[2].imshow(cv2.cvtColor(img_show_FLANN, cv2.COLOR_BGR2RGB))
ax[2].axis("off")
ax[2].set_title("FLANN匹配法")

ax[3].imshow(cv2.cvtColor(imgOut, cv2.COLOR_BGR2RGB))
ax[3].axis("off")
ax[3].set_title("H变换")


#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.show()
