import torch
import torch.nn as nn
from models.common import Conv
import torch.nn.functional as F

class YOLONeck(nn.Module):

    def __init__(self, in_channels=(256, 512)):
        super().__init__()

        # P5
        self.out2 = Conv(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3)
        
        temp_out_channels = 128
        self.temp = nn.Sequential(
            Conv(in_channels=in_channels[1], out_channels=temp_out_channels, kernel_size=1),
            nn.Upsample(size=None, scale_factor=2, mode='nearest')
        )


        self.out1 = Conv(in_channels=in_channels[0]+temp_out_channels, out_channels=in_channels[0], kernel_size=3)


    def forward(self, x):

        out2 = self.out2(x[1])

        out1 = torch.cat((self.temp(x[1]), x[0]), dim=1)
        out1 = self.out1(out1)

        return [out1, out2]



        
class FPN(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), out_channels=256):
        '''
        in_channels: 输入的特征图的通道数列表
        out_channels: 输出特征图的通道数
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        # down-up的顺序
        self.lateral_convs = nn.ModuleList()
        # down-up的顺序
        self.fpn_convs = nn.ModuleList()


        for i in range(self.num_ins):
            # 横向连接将backbone各个层级的特征图的通道数，调整为256
            lateral_conv = nn.Conv2d(self.in_channels[i], self.out_channels, 
                kernel_size=(1,1), stride=(1,1), padding=0, bias=True)
            
            # fpn连接的目的，是对up-down融合后的特征图进一步进行处理
            fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=1, bias=True)
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

    
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # （1）建立横向连接，对齐通道数
        # build laterals
        temporaries = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # （2）建立top-down路径，元素点加
        # build top-down path
        # range(start, stop[, step])，不包括stop
        for i in range(self.num_ins - 1, 0, -1):
            # 这种叠加方法，可以保证最底层的特征图，能够获得上面所有层的特征信息，而不只是相邻的上一层
            temporaries[i - 1] += F.interpolate(
                temporaries[i], scale_factor=2, mode='nearest')
        
        # （3）获得输出，即P3-P5
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](temporaries[i]) for i in range(self.num_ins)
        ]
        
        return outs

        