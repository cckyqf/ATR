import torch
from torchvision import datasets
import os
from PIL import Image
from tqdm import tqdm

import shutil
def save_imgs(dataset, save_dir, idx_to_class):
    num = len(dataset)
    dataset = iter(dataset)
    dataset = enumerate(dataset)
    dataset = tqdm(dataset, total=num)
    for img_id, (img,cls_id) in dataset:

        classname = idx_to_class[cls_id]

        img_save_dir = save_dir + classname + '/'
        if not os.path.exists(img_save_dir):
            os.mkdir(img_save_dir)

        img_save_path = img_save_dir + f"{img_id}.jpg"

        img.save(img_save_path)

# training_data = datasets.MNIST(root="./", train=True, download=True)
# test_data = datasets.MNIST(root="./", train=False, download=True)

# training_data = datasets.FashionMNIST(root="./", train=True, download=True)
# test_data = datasets.FashionMNIST(root="./", train=False, download=True)

training_data = datasets.CIFAR10(root='./', train=True, download=True)
test_data = datasets.CIFAR10(root='./', train=False, download=True)


class_to_idx = training_data.class_to_idx
idx_to_class = {v:k for k,v in class_to_idx.items()}
# 
# for k,v in idx_to_class.items():
#     # MNIST数据集
#     # idx_to_class[k] = v.split(' ')[0]
#     # FashionMNIST数据集
#     idx_to_class[k] = v.replace(' ', '_', 1).split('/')[0]

print(idx_to_class)    


base_dir = "./CIFAR10/"
train_dir = base_dir + "train/"
test_dir = base_dir + "test/"
for path in (train_dir, test_dir):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


save_imgs(training_data, train_dir, idx_to_class)

save_imgs(test_data, test_dir, idx_to_class)

