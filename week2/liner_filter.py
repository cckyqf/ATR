# 线性滤波器
import cv2
import matplotlib.pyplot as plt
import numpy as np
from add_noise import gauss_noise, salt_pepper_noise

img_path = "./images/lena.bmp"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 加噪声
img_noise = gauss_noise(img,sigma=0.05)

# 定义了滤波器
h = np.array([
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 1],
], dtype=np.float32)
# 归一化
h /= h.sum() 

# cv2.filter2D（src,ddepth,kernel）
# ddepth是处理结果图像的图像深度，一般使用-1表示与原始图像使用相同的图像深度。
# kernel卷积核
img_filter = cv2.filter2D(img_noise, ddepth=-1, kernel=h)



# 高斯平滑
img_filter_gauss = cv2.GaussianBlur(img_noise, ksize=(3,3), sigmaX=2, sigmaY=2)

fig, ax = plt.subplots(1,4)
ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0].axis("off")
ax[0].set_title("原始图像")

ax[1].imshow(img_noise, cmap='gray', vmin=0, vmax=255)
ax[1].axis("off")
ax[1].set_title("高斯噪声")

ax[2].imshow(img_filter, cmap='gray', vmin=0, vmax=255)
ax[2].axis("off")
ax[2].set_title("空间滤波")

ax[3].imshow(img_filter_gauss, cmap='gray', vmin=0, vmax=255)
ax[3].axis("off")
ax[3].set_title("高斯滤波")


# 解决中文显示问题，for windows
plt.rcParams['font.sans-serif']=['SimHei']
# # 解决中文显示问题，for Mac 
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False
plt.show()
