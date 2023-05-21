import numpy as np
import cv2

def thr_process(edge, threshold=None):

    edge = np.clip(edge, 0.0, 255.0)
    # .astype函数会对超出范围的数值取余
    edge = edge.astype(np.uint8)

    if threshold is not None:
        edge = edge > threshold

    return edge

# 计算梯度幅值
def fuse_edge_x_y(edge_x, edge_y):

    edge_x = edge_x.astype(np.float32)
    edge_y = edge_y.astype(np.float32)

    # 
    edge = np.sqrt(np.power(edge_x,2)+np.power(edge_y,2))
    # edge = np.abs(edge_x) + np.abs(edge_y)

    return edge


def extract_edge(img, kernel_h, kernel_v, threshold=None):
    img_temp = img.astype(np.float32)
    img_edge_h = cv2.filter2D(img_temp, ddepth=cv2.CV_32F, kernel=kernel_h)
    img_edge_v = cv2.filter2D(img_temp, ddepth=cv2.CV_32F, kernel=kernel_v)
    
    # 计算梯度幅值
    edge = fuse_edge_x_y(img_edge_h, img_edge_v)
    # 阈值处理（二值化），确定哪些点是边缘
    edge = thr_process(edge, threshold=threshold)

    return edge