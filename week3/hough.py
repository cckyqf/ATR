# 几种简单的边缘提取方法
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./images/6_5.tif"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny
edge = cv2.Canny(img, threshold1=50, threshold2=200, apertureSize=3, L2gradient=True)



# 显示标准霍夫变换检测到的直线
img_result_standard = img.copy()
img_result_standard = np.tile(img_result_standard[...,None], (1,1,3))

# 显示概率霍夫变换检测到的线段
img_result_P = img_result_standard.copy()


# 标准霍夫变换，无法检测线段，只能检测直线
# rho，以像素为单位的距离精度。
# theta，以弧度为单位的角度精度。
# threshold累加平面的阈值參数，大于阈值threshold的线段才能被检测到。
lines = cv2.HoughLines(edge, rho=1, theta=np.pi/180, threshold=250)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    point_1 = (x1,y1)
    point_2 = (x2,y2)
    # 画出直线
    cv2.line(img_result_standard, point_1, point_2, (0,255,0),1)


# 概率霍夫变换
# rho，以像素为单位的距离精度。
# theta，以弧度为单位的角度精度。
# threshold累加平面的阈值參数，大于阈值threshold的线段才能被检测到。
# minLineLength，默认值0，表示最短线段的长度。比这个设定參数短的线段就不能被显现出来。
# maxLineGap，默认值0，将同一行点与点之间连接起来的最大的距离。
lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=50, minLineLength=10, maxLineGap=10)
for line in lines:

    x1,y1,x2,y2 = line[0]

    point_1 = (x1,y1)
    point_2 = (x2,y2)
    # 画出检测到的线段
    cv2.line(img_result_P, point_1, point_2, (0,255,0),1)
    # 画出线段的两个端点
    cv2.circle(img_result_P, point_1, radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(img_result_P, point_2, radius=2, color=(0, 0, 255), thickness=-1)


fig, ax = plt.subplots(1,4)
ax = ax.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].axis("off")
ax[0].set_title("原始图像")

ax[1].imshow(edge, cmap='gray')
ax[1].axis("off")
ax[1].set_title("边缘提取")

ax[2].imshow(img_result_standard)
ax[2].axis("off")
ax[2].set_title("标准霍夫变换")

ax[3].imshow(img_result_P)
ax[3].axis("off")
ax[3].set_title("概率霍夫变换")

#解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.show()