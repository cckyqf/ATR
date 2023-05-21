# 稳健平滑滤波
import cv2
import matplotlib.pyplot as plt
import numpy as np
from add_noise import gauss_noise, salt_pepper_noise


def wenjian_filter(img, ksize=3):
    # 确保核尺寸为奇数
    assert (ksize%2==1)

    new_img = img.copy()
    
    img_h,img_w = new_img.shape

    gap = ksize//2

    mask = np.ones((ksize, ksize), np.bool_)
    mask[gap,gap] = False

    for i in range(gap, img_h-gap, 1):
        for j in range(gap, img_w-gap, 1):

            # 取出邻域内的点
            top = i - gap
            left = j - gap
            bottom = top + ksize
            right = left + ksize
            # # 去掉中心点
            neighborhood = img[top:bottom, left:right][mask]

            value = img[i,j]
            max = neighborhood.max()
            min = neighborhood.min()

            if value < min:
                new_img[i,j] = min
            
            if value > max:
                new_img[i,j] = max
    
    return new_img



img_path = "./images/lena.bmp"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 加噪声
img_noise = salt_pepper_noise(img,amount=0.02)


img_filter = wenjian_filter(img_noise, ksize=7)


fig, ax = plt.subplots(1,3)
ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0].axis("off")
ax[0].set_title("原始图像")

ax[1].imshow(img_noise, cmap='gray', vmin=0, vmax=255)
ax[1].axis("off")
ax[1].set_title("椒盐噪声")

ax[2].imshow(img_filter, cmap='gray', vmin=0, vmax=255)
ax[2].axis("off")
ax[2].set_title("中值滤波")

# 解决中文显示问题，for windows
plt.rcParams['font.sans-serif']=['SimHei']
# # 解决中文显示问题，for Mac 
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False
plt.show()
