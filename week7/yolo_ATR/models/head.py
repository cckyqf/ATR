# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import torch
import torch.nn as nn

from utils.general import check_version


class YOLOHead(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, in_channels=(128, 256, 512), inplace=True):  # detection layer
        super().__init__()

        # # 两个检测分支
        # anchors = [
        #     [10,14, 23,27, 37,58],  # P4/16
        #     [81,82, 135,169, 344,319]  # P5/32
        # ]

        # # 三个检测分支
        # anchors = [
        #     [10,13, 16,30, 33,23],  # P3/8
        #     [30,61, 62,45, 59,119],  # P4/16
        #     [116,90, 156,198, 373,326],  # P5/32
        # ]
        
        anchors = [10,13, 16,30, 62,45, 59,119, 156,198, 373,326]


        self.nc = nc  # number of classes
        # 每个anchors 的输出向量的长度
        self.no = nc + 5  # number of outputs per anchor
        
        self.nl = len(in_channels)  # number of detection layers
        self.na = len(anchors) // 2 // self.nl  # 每个检测分支上的anchors的个数
        
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid

        # shape : [num_detect, num_anchors, 2]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in in_channels)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            batch_size, _, grid_h, grid_w = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # shape: [batch_size, channels, H, W], to [batch_size, num_anchors, H, W, channels_per_anchors]
            x[i] = x[i].view(batch_size, self.na, self.no, grid_h, grid_w).permute(0, 1, 3, 4, 2).contiguous()
            
            # 边界框解码
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    
                    # 生成网格的坐标，以及网格anchors
                    self.grid[i], self.anchor_grid[i] = self._make_grid(grid_w, grid_h, i)

                y = x[i].sigmoid()
                # if self.inplace:
                #     y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                
                xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, y[..., 4:]), -1)
                
                z.append(y.view(batch_size, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


