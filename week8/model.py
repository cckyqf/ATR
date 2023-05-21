import torch
import torch.nn as nn

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, bias=True, bn=False, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
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
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        use_bn = True
        self.model = nn.Sequential(
            # 2x下采样
            Conv(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bn=use_bn, act=True),
            # 4x下采样
            Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bn=use_bn, act=True),
            # 8x下采样
            Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bn=use_bn, act=True),
            
            Conv(in_channels=128, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bn=False, act=False),
            # 全局平均池化
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten()
        )

    def forward(self, x):
        
        return self.model(x)

