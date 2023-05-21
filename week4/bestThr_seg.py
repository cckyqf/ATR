# 最佳阈值分割（均值迭代）
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./images/3.PNG"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_temp = img.astype(np.float32)

threshold_old = img_temp.mean()
img1 = img_temp[img_temp > threshold_old]
img2 = img_temp[img_temp < threshold_old]
threshold = (img1.mean() + img2.mean()) / 2.0

# 迭代次数
num_iters = 1
while (threshold != threshold_old):
    threshold_old = threshold
    img1 = img_temp[img_temp > threshold_old]
    img2 = img_temp[img_temp < threshold_old]
    
    threshold = (img1.mean() + img2.mean()) / 2.0

    num_iters += 1


mask = img > threshold

fig, ax = plt.subplots(1,2)
ax = ax.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].axis("off")
ax[0].set_title("image")

ax[1].axis("off")
ax[1].set_title(f"thr:{round(threshold, 1)}, iters:{num_iters}")
ax[1].imshow(mask, cmap='gray')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.show()

