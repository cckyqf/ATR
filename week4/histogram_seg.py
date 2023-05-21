# 直方图阈值分割
import cv2
import matplotlib.pyplot as plt

img_path = "./images/3.PNG"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(1,3)
ax = ax.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].axis("off")
ax[0].set_title("image")

ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("hist")

ax[2].axis("off")
ax[2].set_title("segmentation result")

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.show()


threshold = input("请输入选择的阈值:")

threshold = float(threshold)
mask = img > threshold


fig, ax = plt.subplots(1,3)
ax = ax.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].axis("off")
ax[0].set_title("image")

ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("hist")

ax[2].axis("off")
ax[2].set_title("segmentation result")
ax[2].imshow(mask, cmap='gray')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.show()

