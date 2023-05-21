import torch

@torch.no_grad()
def test(test_loader, model, loss_fn, device):
    loss_sum = 0
    model.eval()
    for imgs, labels in test_loader:
        
        # 图像、标签，放入设备中
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 前向传播
        preds = model(imgs)

        # 计算损失
        loss = loss_fn(preds, labels)
        loss_sum += loss.item()

    # 平均一个batch上的损失
    loss_sum /= len(test_loader)

    return loss_sum
