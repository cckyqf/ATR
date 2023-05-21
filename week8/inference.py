# 读取模型、图像，输出图像的类别
import torch
import cv2

img_path = "D:/documents/laboratory/20-ATR/experiments/week8_0/data/FashionMNIST/test/Bag/18.jpg"

model_path = "model.pt"

device = torch.device('cpu')
model = torch.load(model_path, map_location=device)['model']

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# shape [H,W]
img = torch.from_numpy(img).float()
img /= 255.0
# shape [B, C, H, W], [1,1, H,W]
img = torch.unsqueeze(img, dim=0)
img = torch.unsqueeze(img, dim=0)

pred = model(img)

cls_id = pred.argmax(1).item()
print(model.classes[cls_id])