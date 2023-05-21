# 特征点匹配

import cv2
import matplotlib.pyplot as plt
import numpy as np

from general import get_SIFT
from general import BFMatch
from general import compute_homography

print(cv2.__version__)


# 待检测图像
img_path = "./images/bg.jpg"
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测关键点，并计算特征描述子
keypoints, descriptors = get_SIFT(img_gray)

img_mask_path = "./images/tank.jpg"
img_mask = cv2.imread(img_mask_path)
img_mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
keypoints_mask, descriptors_mask = get_SIFT(img_mask_gray)


topK = 50

# 暴力匹配法
matches = BFMatch(descriptors, descriptors_mask, topK=topK)
# img_show = cv2.drawMatches(img, keypoints, img_mask, keypoints_mask, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 从查询图像到模板图像的单应矩阵H
H, matches = compute_homography(keypoints, keypoints_mask, matches)
img_show = cv2.drawMatches(img, keypoints, img_mask, keypoints_mask, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# H矩阵取逆矩阵，从模板图像变换到查询图像
H = np.mat(H).I.A

# 画出模板图像在查询图像中的位置
h, w = img_mask.shape[:2]
pts = np.float32([[0,0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
# 进行仿射变换
box = cv2.perspectiveTransform(pts, H)


# 绘制查找到的区域
img_results = cv2.polylines(img_show, [np.int32(box)], isClosed=True, color=(0,255,0), thickness=3)

# cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow('result', img_results)
cv2.waitKey(0)
