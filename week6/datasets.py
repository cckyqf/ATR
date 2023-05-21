
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

def data_loader(trainpath, testpath, image_size, batch_size=32):
    kwopt = {'num_workers': 8, 'pin_memory': True} 
    train_transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)),#随机剪裁
                    transforms.Grayscale(num_output_channels=1),#转化为灰度图
                    transforms.RandomHorizontalFlip(),#依照概率水平翻转
                    transforms.RandomVerticalFlip(),#依照概率垂直翻转
                    transforms.ToTensor(),#转化为tensor， 同时除以255
                    ])
    
    val_transforms = transforms.Compose([
                    transforms.Resize((image_size, image_size)),#随机剪裁
                    transforms.Grayscale(num_output_channels=1),#转化为灰度图
                    transforms.ToTensor(),#转化为tensor
                    ])
    
    train_dataset = ImageFolder(trainpath, transform=train_transforms)
    test_dataset = ImageFolder(testpath, transform=val_transforms)

    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True, drop_last=True, **kwopt)
    test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle = False, drop_last=True, **kwopt)

    return train_loader, test_loader
