# 图像的读取、显示、保存、几何变换
import cv2
import matplotlib.pyplot as plt
import numpy as np


img_path = "./images/lenna.png"

# 读取
img = cv2.imread(img_path)


# 保存
save_path = "./save.jpg"
cv2.imwrite(save_path, img)


# 显示
window_name = 'lenna'
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, img)
if cv2.waitKey(0) == ord('q'):  # q to quit
    pass
cv2.destroyAllWindows()



# BGR -> RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_h, img_w = img.shape[:2]
# 几何变换
R = np.eye(3)
angle = 30  # add 90deg rotations to small rotations
scale = 0.5
# s = 2 ** random.uniform(-scale, scale)
R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(img_h//2, img_w//2), scale=scale)

# Translation
T = np.eye(3)
T[0, 2] = 0.5 * img_w  # x translation (pixels)
T[1, 2] = 0.0 * img_h  # y translation (pixels)


M = T @ R  # order of operations (right to left) is IMPORTANT
img = cv2.warpAffine(img, M[:2], dsize=(img_w, img_h), borderValue=(114, 114, 114))


# 子图显示
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title("lenna")
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cmap='gray')
ax[1].set_title("lenna")
ax[1].axis('off')
# ax[1].set_xlabel('x')
# ax[1].set_ylabel('y')

plt.show()