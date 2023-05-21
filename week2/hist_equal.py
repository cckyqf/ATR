# 直方图均衡化
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./images/12.tif"
# img_path = "./images/01.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 均衡化后
equ = cv2.equalizeHist(img)

fig, ax = plt.subplots(2,2)
ax[0][0].imshow(img,cmap='gray',vmin=0, vmax=255)
ax[0][0].axis("off")
ax[0][0].set_title("ori")

ax[0][1].imshow(equ, cmap='gray', vmin=0, vmax=255)
ax[0][1].axis("off")
ax[0][1].set_title("equ")


# cv2.calcHist(images,channels,mask,histSize,ranges)
# images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应用中括号 [] 括来例如[img]
# channels: 同样用中括号括来它会告函数我们统幅图像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0][1][2] 它们分别对应着 BGR。
# mask: 掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
# histSize:BIN 的数目，也使用中括号。
# ranges: 像素值范围常为 [0-256]。
# hist shape (256,1)
hist = cv2.calcHist([img],[0],None,[256],[0,255])
x = range(hist.shape[0])
y = hist.reshape(-1).tolist()
ax[1][0].bar(x, y)
# ax[1][0].hist(img.ravel(), 256)
ax[1][0].set_title("ori hist")

hist_equ = cv2.calcHist([equ],[0],None,[256],[0,255])
x = range(hist_equ.shape[0])
y = hist_equ.reshape(-1).tolist()
ax[1][1].bar(x, y)
# ax[1][1].hist(equ.ravel(), 256)
ax[1][1].set_title("equ hist")


# 画映射曲线
num = np.size(img)
hist = hist / num
map = np.cumsum(hist, axis=0) * 255
plt.figure()
plt.plot(map)
plt.title("映射曲线")

# 解决中文显示问题，for windows
plt.rcParams['font.sans-serif']=['SimHei']
# # 解决中文显示问题，for Mac 
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False
plt.show()
