import cv2
import numpy as np


def get_SIFT(img):

    sift = cv2.SIFT_create()

    # 检测关键点，并计算特征描述子
    keypoints, descriptors = sift.detectAndCompute(img, mask=None)      # 第二个参数为mask区域
    
    return keypoints, descriptors


def filter_matches(matches,topK=None):

    # 按照距离从小到大排序
    matches = sorted(matches, key=lambda x:x.distance)

    # 取前topK个匹配的关键点
    if topK is not None and len(matches) > topK:
        matches = matches[:topK]
    
    return matches


def BFMatch(descriptors_query, descriptors_mask, topK=None):

    # Brute-force matcher，暴力匹配方法
    # The result of matches = matcher.match(des1,des2) line is a list of DMatch objects. 
    # This DMatch object has following attributes:
    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors
    # DMatch.queryIdx - Index of the descriptor in query descriptors
    # DMatch.imgIdx - Index of the train image.

    # crossCheck：默认为FALSE。如果设置为TRUE，只有当两组中特征点互相匹配时才算匹配成功。
    # 也就是说A组中x点描述符的最佳匹配点是B组的y点，那么B组的y点的描述符最佳匹配点也要是A组的x点才算匹配成功。
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors_query, descriptors_mask)
    
    
    matches = filter_matches(matches,topK=topK)
    
    return matches


# 使用knnMatch方法为每个查询关键点请求k个最佳匹配的列表。
# 假设：每个查询关键点至多有一个正确的匹配，因此次优的匹配是错误的
# 如果最优匹配和次优匹配的没有拉开足够大的距离，则认为最优匹配不够好，删除这种匹配
# 次优匹配的距离分值乘以一个小于1的值，就可以获得阈值。然后，只有当距离分值小于阈值时，才将最佳匹配视为良好的匹配。
# 这种方法被称为比率检验。
def BFMatchKnn(descriptors_query, descriptors_mask, topK=None, ratio=0.85):

    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_query, descriptors_mask, k=2)
    matches = []
    for m1, m2 in raw_matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance < ratio * m2.distance:
            matches.append(m1)
    
    matches = filter_matches(matches,topK=topK)

    return matches


def FlannMatchKnn(descriptors_query, descriptors_mask, topK=None, ratio=0.85):

    # FLANN是近似最近邻的快速库。它包含一组算法，这些算法针对大型数据集中的快速最近邻搜索和高维特征进行了优化。对于大型数据集，它的运行速度比BFMatcher快。
    # FLANN特征匹配，在进行批量特征匹配时速度更快
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    # 指定索引中的树应递归遍历的次数。较高的值可提供更好的精度，但也需要更多时间。
    search_params = dict(checks=100)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = matcher.knnMatch(descriptors_query, descriptors_mask, k=2)

    matches = []
    for m1, m2 in raw_matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance < ratio * m2.distance:
            matches.append(m1)
    
    matches = filter_matches(matches,topK=topK)

    return matches




# RANSAC方法计算单应矩阵，RANSAC会去掉某些错误的匹配点
def compute_homography(keypoints_query, keypoints_mask, matches):
    
    # 计算单应矩阵，至少需要4对匹配点
    assert len(matches) > 4, "Too few matches."
    
    array_query = np.float32([keypoints_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    array_mask = np.float32([keypoints_mask[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # ransacReprojThreshold取值范围在1到10，最大重投影误差（原图像的点经过变换后点与目标图像上对应点的误差）
    # 超过误差被认为是离线点
    # mask标记出在线点
    H, mask = cv2.findHomography(array_query, array_mask, cv2.RANSAC, ransacReprojThreshold=4)
    
    # 过滤掉错误匹配点
    num = len(matches)
    matches = [matches[i] for i in range(num) if mask[i] == 1]

    return H, matches

