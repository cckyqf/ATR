# 指数变换
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./images/01.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度映射
def exp_map(img, gamma, thr_low=0.1, thr_high=0.9):

    new_img = img.copy().astype(np.float32)
    new_img /= 255.0

    mask = new_img < thr_low
    new_img[mask] = 0.0

    mask = new_img > thr_high
    new_img[mask] = 1.0
    # 指数变换
    new_img = np.power(new_img, gamma)

    new_img = (new_img*255.0).astype(np.uint8)
    return new_img


# 灰度变换
gamma = 3
thr_low=0.03 # 低于此阈值，会归为0
thr_high=0.99 # 高于此阈值，会归为255
img_map = exp_map(img, gamma=gamma, thr_low=thr_low, thr_high=thr_high)

fig, ax = plt.subplots(2,2)
ax[0][0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0][0].axis("off")
ax[0][0].set_title("ori")

ax[0][1].imshow(img_map, cmap='gray', vmin=0, vmax=255)
ax[0][1].axis("off")
ax[0][1].set_title("exp_map")

ax[1][0].hist(img.ravel(), bins=256)
ax[1][0].set_title("ori hist")

ax[1][1].hist(img_map.ravel(), bins=256)
ax[1][1].set_title("exp_map hist")


# 画映射曲线
x = np.arange(256)
y = exp_map(x, gamma=gamma, thr_low=thr_low, thr_high=thr_high)
plt.figure()
plt.plot(x,y)
plt.title("映射曲线")

# 解决中文显示问题，for windows
plt.rcParams['font.sans-serif']=['SimHei']
# # 解决中文显示问题，for Mac 
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False
plt.show()
