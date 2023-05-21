import torch
from tqdm import tqdm

# 关闭梯度记录
@torch.no_grad()
def test(dataloader, model, loss_fn, device):
    
    num_imgs = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # 设置模型处于eval模式
    model.eval()
    model.to(device)

    loss, accuracy = 0.0, 0.0
    # print("测试2")
    for imgs, labels in tqdm(dataloader, total=num_batches):
        
        imgs = imgs.to(device)
        labels = labels.to(device)

        pred = model(imgs)
        loss += loss_fn(pred, labels).item()
        
        # 统计分类正确的图像个数
        # pred shape : [batch, num_classes]
        accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
    
    # print("测试3")
    # 平均每个batch的损失
    loss /= num_batches
    # 分类准确率
    accuracy /= num_imgs

    

    return loss, accuracy