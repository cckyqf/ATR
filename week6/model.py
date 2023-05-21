import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class LeNet5(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)

        # self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x):
        x = self.relu1(self.conv1(x))    # 28x28x6
        x = self.pool1(x)               # 14x14x6
        
        x = self.relu2(self.conv2(x))    # 10x10x16
        x = self.pool2(x)               # 5x5x16

        x = self.flatten(x)             # 400

        x = self.relu3(self.fc1(x))      # 120
        x = self.relu4(self.fc2(x))      # 84
        x = self.fc3(x)                 # 10

        return x

class miniLeNet5(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(120, num_classes)

        # BN层
        # self.bn = nn.BatchNorm2d(100)
        # Dropout层
        # self.droupt = nn.Dropout(p=0.1) # nn.Dropout2d则是把整个通道置0
        # 全局平均池化
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))



    def forward(self, x):
        x = self.relu1(self.conv1(x))    # 28x28x6
        x = self.pool1(x)               # 14x14x6
        
        x = self.relu2(self.conv2(x))    # 10x10x16
        x = self.pool2(x)               # 5x5x16
        
        x = self.relu3(self.conv3(x))      # 1x1x120

        x = self.flatten(x)
        x = self.fc(x)

        return x



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, bias=True, bn=False, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
    


# 任意图像尺寸的输入
class CNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()

        use_bn = True

        self.model = nn.Sequential(
            Conv(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bn=use_bn, act=True),
            Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bn=use_bn, act=True),
            Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bn=use_bn, act=True),
            
            Conv(in_channels=128, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bn=False, act=False),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten()
        )

        # 全局平均池化
        self.globalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()


    def forward(self, x):
        
        return self.model(x)
