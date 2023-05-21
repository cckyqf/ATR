# 人脸检测
import cv2
import matplotlib.pyplot as plt
import numpy as np


def faceDection(img, cr_thr=(145, 165), cb_thr=(145,180), ar_thr=(0.5, 1.5)):

    img = img.astype(np.uint8)

    # 转换到YCbCr颜色空间
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    cr = img_ycrcb[:,:,1]
    cb = img_ycrcb[:,:,2]

    # 阈值分割
    cr_mask = (cr >= cr_thr[0]) & (cr <= cr_thr[1])
    cb_mask = (cb >= cb_thr[0]) & (cb <= cb_thr[1])
    mask = cb_mask & cr_mask
    
    mask = mask.astype(np.float32) * 255.0
    mask = mask.astype(np.uint8)

    # 过滤面积较小的区域（非人脸）
    # 开运算，去除面积过小的区域
    kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(7,7))
    mask=cv2.morphologyEx(src=mask,op=cv2.MORPH_OPEN,kernel=kernel,iterations=1)


    # 膨胀操作，去掉人脸区域中的小空洞
    mask = cv2.dilate(mask, kernel=kernel, iterations=2)
    # # 腐蚀操作
    # img_erode=cv2.erode(src=img,kernel=kernel,iterations=2)


    # 连通域分析
    # num_labels: 连通域数量
    # labels : 连通域的标签
    # stats : 通域的信息,shape:[N, 5] 对应各个轮廓外接矩形的x、y、width、height和面积
    # centroids : 连通域的中心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 去掉背景
    stats = stats[1:]
    boxes = stats.reshape(-1,5)


    # 根据长宽比过滤检测结果
    ar = (boxes[:,2] / (boxes[:,3]+1e-9))
    save_mask = (ar >=ar_thr[0]) & (ar <= ar_thr[1])
    boxes = boxes[save_mask].reshape(-1,5)

    # 分割mask
    mask = mask.astype(np.bool_)

    return mask, boxes


img_path = "./images/76.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# mask, stats = faceDection(img, cr_thr=(133, 173), cb_thr=(67,137))
mask, boxes = faceDection(img, cr_thr=(145, 165), cb_thr=(67,180))

print(boxes)
print(img.shape)
img_show = img.copy()
for box in boxes:
    # 边界框左上角点和右下角点的坐标
    pt1 = box[:2]
    pt2 = box[:2] + box[2:4]
    img_show = cv2.rectangle(img_show, pt1, pt2, color=(0,255,0), thickness=2)

fig, ax = plt.subplots(1,3)
ax = ax.ravel()

ax[0].imshow(img)
ax[0].axis("off")
ax[0].set_title("image")

ax[1].axis("off")
ax[1].set_title("mask")
ax[1].imshow(mask, cmap='gray')

ax[2].axis("off")
ax[2].set_title("检测结果")
ax[2].imshow(img_show, cmap='gray')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.show()

