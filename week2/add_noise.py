import cv2
import numpy as np

def gauss_noise(img, mean=0, sigma=0.01):

    img_new = img.copy()
    img_new = img_new.astype(np.float32)

    #根据均值和标准差生成符合高斯分布的噪声
    gauss = np.random.normal(mean,sigma,img_new.shape)
    gauss *= 255
    
    img_noise = img_new + gauss
    img_noise = np.clip(img_noise, 0.0, 255.0)

    img_noise = img_noise.astype(np.uint8)

    return img_noise 

def salt_pepper_noise(img, amount=0.05, salt_ratio=0.5):
    # salt_ratio: 椒盐噪声中，盐噪声的比例
    # amount: 噪声个数
    img_noise = img.copy()
    num_pixs = img_noise.size
    
    #添加salt噪声
    num_salt = np.ceil(amount * num_pixs * salt_ratio)
    # 设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_salt)) for i in img_noise.shape]
    img_noise[coords[0],coords[1]] = 255
    
    # 添加pepper噪声
    num_pepper = np.ceil(amount * num_pixs * (1. - salt_ratio))
    # 设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in img_noise.shape]
    img_noise[coords[0],coords[1]] = 0

    return img_noise 
