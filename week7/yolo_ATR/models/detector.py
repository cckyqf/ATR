# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from utils.general import LOGGER, check_yaml, print_args
from utils.torch_utils import initialize_weights, model_info, select_device

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

from models.common import *
import math
import torch
import torch.nn as nn
from models.backbone import YOLOBackbone
from models.neck import YOLONeck, FPN
from models.head import YOLOHead


class Model(nn.Module):
    def __init__(self, in_channel=3, num_classes=1):  # model, input channels, number of classes
        super().__init__()

        # Define model        
        
        self.model = nn.Sequential(
            YOLOBackbone(in_channel=in_channel, out_indices=(5,)),
            YOLOHead(nc=num_classes, in_channels=(512,))
        )
        
        # self.model = nn.Sequential(
        #     YOLOBackbone(in_channel=in_channel, out_indices=(4, 5)),
        #     YOLONeck(in_channels=(256, 512)),
        #     YOLOHead(nc=num_classes, in_channels=(256, 512))
        # )

        # self.model = nn.Sequential(
        #     YOLOBackbone(in_channel=in_channel, out_indices=(3, 4, 5)),
        #     FPN(in_channels=(128, 256, 512), out_channels=256),
        #     YOLOHead(nc=num_classes, in_channels=(256, 256, 256))
        # )

        

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, YOLOHead):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, in_channel, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
        

        
        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    
    def forward(self, x):

        return self.model(x)


    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model().to(device)
    model.train()

    print(model)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
