from model import CNet
from datasets import data_loader
import torch.nn as nn
import torch
import torch.optim as optim
from test import test

# 训练所在的设备
device = torch.device('cpu')

# 定义模型
img_channel = 1
num_classes = 10
model = CNet(in_channels=img_channel, num_classes=num_classes)
model.to(device)


# 定义数据集
trainpath = "D:/documents/laboratory/20-ATR/experiments/week8_0/data/FashionMNIST/train/"
testpath = "D:/documents/laboratory/20-ATR/experiments/week8_0/data/FashionMNIST/test/"
image_size = 32 
batch_size = 64
train_loader, test_loader = data_loader(
    trainpath, testpath, image_size, batch_size=32)

# ['Ankle_boot', 'Bag', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'T-shirt', 'Trouser']
classes = train_loader.dataset.classes
model.classes = classes

# 定义损失函数
loss_fn = nn.CrossEntropyLoss().to(device)


# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999))


# 开始训练
epochs = 1
for epoch in range(epochs):
    model.train()
    for imgs, labels in train_loader:

        # 图像、标签，放入设备中
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 前向传播
        preds = model(imgs)

        # 计算损失
        loss = loss_fn(preds, labels)
        # 梯度清零
        optimizer.zero_grad()        

        # 反向传播
        loss.backward()

        # 梯度更新
        optimizer.step()

        # 打印损失
        print(f"loss:{loss}")
    
    # 测试
    model.eval()
    loss_test = test(test_loader, model, loss_fn, device)
    print(f"loss_test:{loss_test}")

    # 保存模型
    save_path = "model.pt"
    checkpoint = {
        "model":model,
        "epoch":epoch
    }
    torch.save(checkpoint, save_path)

print("done")