import torch.nn as nn
from models.common import Conv

class YOLOBackbone(nn.Module):

    def __init__(self, in_channel=3, out_indices=(4, 5), frozen_stages=-1):
        super().__init__()

        # 输出的特征层级：C3-C5
        self.out_indices = out_indices

        stage0 = Conv(in_channels=in_channel, out_channels=16, kernel_size=3, stride=1)
        # 2x
        stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        )
        # 4x
        stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        )
        # 8x, C3
        stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        )
        # 16x, C4
        stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        )
        # 32x, C5
        stage5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv(in_channels=256, out_channels=512, kernel_size=3, stride=1),
        )

        self.backbone = nn.Sequential(
            stage0,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
        )

        del stage0, stage1, stage2, stage3, stage4, stage5

        # 冻结模型部分参数
        self.frozen_stages = frozen_stages
        if self.frozen_stages >= 0:
            self._freeze_stages()


    def forward(self, x):
        outs = []
        for i,m in enumerate(self.backbone):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


    # 冻结模型参数，并设置为eval模式，保证BN层不会更新参数，也不会统计batch的个数，也不会更新running_mean和running_var
    def _freeze_stages(self):
        for i,m in enumerate(self.backbone):
            if i <= self.frozen_stages:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
