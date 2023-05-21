import cv2
import numpy as np
import math

img_path = "./images/lenna.png"

# 读取
img = cv2.imread(img_path)


img_h, img_w = img.shape[:2]
R = np.eye(3)
angle = 60  # add 90deg rotations to small rotations
scale = 0.3
R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(img_h//2, img_w//2), scale=scale)

# Shear
shear_angle = 10
S = np.eye(3)
S[0, 1] = math.tan(shear_angle * math.pi / 180)  # x shear (deg)
S[1, 0] = math.tan(shear_angle * math.pi / 180)  # y shear (deg)


# Translation
T = np.eye(3)
T[0, 2] = -0.2 * img_w  # x translation (pixels)
T[1, 2] = -0.3 * img_h  # y translation (pixels)


M = T @ S @ R  # order of operations (right to left) is IMPORTANT
img = cv2.warpAffine(img, M[:2], dsize=(int(0.6*img_w), int(0.7*img_h)), borderValue=(114, 114, 114))

# 显示
window_name = 'lenna'
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, img)
if cv2.waitKey(0) == ord('q'):  # q to quit
    pass
cv2.destroyAllWindows()

# 保存
save_path = "./images/lenna_wrap.png"
cv2.imwrite(save_path, img)
