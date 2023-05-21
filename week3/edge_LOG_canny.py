import cv2
from general import thr_process
import matplotlib.pyplot as plt

threshold = None

img_path = "./images/lena.bmp"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯-拉普拉斯
# 标准差设置为0，则函数内部根据尺寸计算
edge_LOG = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0, sigmaY=0)
edge_LOG = cv2.Laplacian(edge_LOG, ddepth=cv2.CV_32F, ksize=5)
edge_LOG = thr_process(edge_LOG, threshold=threshold)

# Canny
# threshold1：阈值1（最小值）
# threshold2：阈值2（最大值），使用此参数进行明显的边缘检测
# apertureSize：sobel算子（卷积核）大小
# L2gradient ：布尔值。True： 使用更精确的L2范数进行计算梯度强度。False：使用L1范数
edge_canny = cv2.Canny(img, threshold1=200, threshold2=240, apertureSize=3, L2gradient=True)


fig, ax = plt.subplots(1,2)
ax = ax.ravel()

ax[0].imshow(edge_LOG, cmap='gray')
ax[0].axis("off")
ax[0].set_title("LOG")

ax[1].imshow(edge_canny, cmap='gray')
ax[1].axis("off")
ax[1].set_title("Canny")

#解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.show()