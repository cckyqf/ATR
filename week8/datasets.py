# dataloader
from torch.utils.data import DataLoader
# dataset
from torchvision.datasets import ImageFolder
# 数据预处理：归一化、数据增强
from torchvision import transforms

def data_loader(trainpath, testpath, image_size, batch_size=32):
    train_transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)) ,#随机剪裁
                    transforms.Grayscale(num_output_channels=1),#转化为灰度图
                    
                    transforms.RandomHorizontalFlip(),#依照概率水平翻转
                    transforms.ToTensor(),#转化为tensor， 同时除以255
                    ])
    
    val_transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)), 
                    transforms.Grayscale(num_output_channels=1),#转化为灰度图
                    transforms.ToTensor(),#转化为tensor
                    ])
    
    # dataset
    train_dataset = ImageFolder(trainpath, transform=train_transforms)
    test_dataset = ImageFolder(testpath, transform=val_transforms)

    # dataloader
    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True)
    test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle = False)

    return train_loader, test_loader
