import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

threshold = 6*1e6
img_path = "./images/shapes-bw1.jpg"
temp_path = "./images/shapeTemplate.jpg"

# threshold = 1*1e6
# img_path = "./images/text1.png"
# temp_path = "./images/a.jpg"

img = cv2.imread(img_path)
img_temp = cv2.imread(temp_path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2GRAY).astype(np.float32)

t1 = time.time()
img_h, img_w = img.shape
temp_h, temp_w = img_temp.shape

mask = np.zeros_like(img, dtype=np.bool_)
for top in range(0, img_h-temp_h+1, 1):
    for left in range(0, img_w-temp_w+1, 1):
        bottom = top + temp_h
        right = left + temp_w
        img_crop = img[top:bottom, left:right]
        distance = img_crop - img_temp
        distance = np.sum(np.power(distance,2))

        if distance < threshold:
            mask[top:bottom, left:right] = 1

img_result = (mask * img).astype(np.uint8)
t2 = time.time()

print(f"消耗时间:{round(t2-t1, 4)}秒")

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img, cmap='gray')
ax[0].set_title("image")
ax[0].axis('off')

ax[1].imshow(img_temp, cmap='gray')
ax[1].set_title("temp")
ax[1].axis('off')

ax[2].imshow(img_result, cmap='gray')
ax[2].set_title("results")
ax[2].axis('off')

plt.show()
plt.close()
