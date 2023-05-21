import torch
from torch import nn
import torch.optim as optim

import argparse

# from model import LeNet5 as Model
# from model import miniLeNet5 as Model
from model import CNet as Model


from datasets import data_loader

import random
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from general import increment_path
from pathlib import Path
import os
from test import test
from tqdm import tqdm


def train(writer, device, opt):

    img_channels = 1
    # 创建数据集
    train_loader, test_loader = data_loader(opt.trainpath, opt.testpath, opt.image_size, opt.batch_size)
    # 定义网络
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    model = Model(input_channels=img_channels, num_classes=num_classes).to(device)
    # 定义优化器
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999))

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)
    

    # 网络的前向传播过程写入tensorboard
    example = torch.rand((1, img_channels, opt.image_size,opt.image_size), device=device)
    writer.add_graph(torch.jit.trace(model, example, strict=False), [])

    ## 准备开始训练
    best_accuracy = 0.0
    num_batches = len(train_loader)
    epochs = opt.epochs
    for epoch in range(epochs):
        # 记录整个epoch的损失
        loss_total = 0
        
        # 模型设置在训练状态
        model.train()
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=num_batches, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for iter_idx, (imgs, labels) in pbar:
            
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            pred = model(imgs)
            # 计算损失
            loss = loss_fn(pred, labels)
            # 反向传播，计算梯度            
            loss.backward()
            # 梯度更新
            optimizer.step()

            loss_item = loss.item()
            loss_total += loss_item
            # 打印损失
            if iter_idx % opt.log_interval == 0:
                pbar.set_description('[%d/%d][%d/%d] loss:%.4f' % (epoch,epochs, iter_idx, num_batches, loss_item))

        # print("测试1")
        model.eval()
        # 测试模型的精度
        loss_test, accuracy = test(test_loader, model, loss_fn, device) 

        # 记录模型训练过程中的loss、准确率等情况
        # writer.add_scalar('train/loss', loss_total/num_batches, epoch)        
        # writer.add_scalar('test/loss', loss_test, epoch)
        # writer.add_scalar('metrics/Accuracy',  accuracy, epoch)
        print('Test:[%d] Avg loss:%.4f, Accuracy:%.4f' % (epoch, loss_test, accuracy))


        # 保存训练得到的模型
        checkpoint = {
            'epoch': epoch,
            'model': model,
        }
        torch.save(checkpoint, '%s/weights/last.pt' % (opt.outf))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, '%s/weights/best.pt' % (opt.outf))


    print("Done!")


def parse_opt():

    parser = argparse.ArgumentParser(description='图像分类识别')

    parser.add_argument('--trainpath',default='./data/FashionMNIST/train/')
    parser.add_argument('--testpath',default='./data/FashionMNIST/test/')

    parser.add_argument('--batch-size',type=int,default=64,metavar='N')
    parser.add_argument('--image-size',type=int,default=32,metavar='N')

    parser.add_argument('--epochs',type=int,default=5,metavar='N')

    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--log-interval',type=int,default=20,metavar='N')

    parser.add_argument('--seed', type=int, default=0, metavar='S')

    parser.add_argument('--save_path',default='./test')
    parser.add_argument('--outf',default='./')

    opt = parser.parse_args()

    return opt



def main():
    opt = parse_opt()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    # device = torch.device("cpu")
    # device = torch.device("cuda:0")
    print(f"Using {device} device")


    # 训练中间结果的保存路径
    opt.outf = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    print(f"results saved to {opt.outf}")
    
    if not os.path.exists('%s/model' % (opt.outf)):
        os.makedirs('%s/weights/' % (opt.outf))
    

    # 保存本次训练所设置的参数
    with open(opt.outf + '/opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)    
    
    # 以可视化的方式，记录训练过程中的状态
    writer = SummaryWriter(log_dir=opt.outf)

    
    train(writer, device, opt)


if __name__ == '__main__':
    
    main()