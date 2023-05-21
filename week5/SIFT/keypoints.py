# 特征点检测
import cv2
print(f"OpenCV版本:{cv2.__version__}")

img_path = "./images/lenna.png"
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# SIFT特征提取类
# nfeatures : 保留的特征点个数
# nOctaveLayers : 每个频程的搜索层数，会根据图像分辨率自动计算
# contrastThreshold : 对比度阈值，过滤弱特征点，值越大，检测到的特征点越少
# edgeThreshold : 边缘阈值，过滤边缘点，值越大，检测得到的特征点越多
sift = cv2.SIFT_create(nfeatures=100)

# 检测关键点
keypoints = sift.detect(img_gray, mask=None)
# 计算特征描述子
# 部分无法计算描述子的关键点会被删除，具有多个主方向的关键点会被添加
keypoints, descriptors = sift.compute(img_gray, keypoints)



# 把特征点标记到图片上, 每个关键点有三个信息(x,y,σ,θ)：位置、尺度、方向
# print("数据类型:", type(kp1[i]))
# print("关键点坐标:", kp1[i].pt)
# print("邻域直径:", kp1[i].size)
# print("方向:", kp1[i].angle)
# print("所在的图像金字塔的组:", kp1[i].octave)

# DRAW_MATCHES_FLAGS_DEFAULT：只绘制特征点的坐标点，显示在图像上就是一个个小圆点。
# DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG ：函数不创建输出的图像，而是直接在输出图像变量空间绘制，要求本身输出图像变量就是一个初始化好了的，size与type都是已经初始化好的变量。
# DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS ：单点的特征点不被绘制。
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ：绘制特征点的时候绘制的是一个个带有方向的圆，这种方法同时显示图像的坐标，size和方向，是最能显示特征的一种绘制方式。
img_show = cv2.drawKeypoints(img, keypoints, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("images", img_show)
if cv2.waitKey(0) == ord('q'):
    pass
