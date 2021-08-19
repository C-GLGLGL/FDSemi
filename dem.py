import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class DispersiveAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(DispersiveAttention, self).__init__()
        self.channels = channels

        self.atrous_block1 = nn.Conv2d(self.channels, self.channels * 4 // reduction_ratio, 1, 1)
        self.atrous_block2 = nn.Conv2d(self.channels, self.channels * 4 // reduction_ratio, 3, 1, padding=2, dilation=2)
        self.atrous_block5 = nn.Conv2d(self.channels, self.channels * 4 // reduction_ratio, 3, 1, padding=5, dilation=5)

        self.conv_1x1 = nn.Conv2d(self.channels * 4 // reduction_ratio, self.channels, 1, 1)

        self.linear_comb_layer = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(self.channels, self.channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 4, self.channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x1 = self.conv_1x1(x1)
        x2 = self.atrous_block2(x)
        x2 = self.conv_1x1(x2)
        x5 = self.atrous_block5(x)
        x5 = self.conv_1x1(x5)

        max_pool1 = F.max_pool2d(x1, (x1.size(2), x1.size(3)), stride=(x1.size(2), x1.size(3)))
        scale1 = self.linear_comb_layer(max_pool1)

        max_pool2 = F.max_pool2d(x2, (x2.size(2), x2.size(3)), stride=(x2.size(2), x2.size(3)))
        scale2 = self.linear_comb_layer(max_pool2)

        max_pool5 = F.max_pool2d(x5, (x5.size(2), x5.size(3)), stride=(x5.size(2), x5.size(3)))
        scale5 = self.linear_comb_layer(max_pool5)

        x_scale = torch.matmul(scale1, scale2)
        x_scale = torch.matmul(x_scale, scale5)

        scale = x_scale.expand_as(x)
        return scale * x

class DEM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(DEM, self).__init__()
        self.dem = DispersiveAttention(channels, reduction_ratio)

    def forward(self, x):
        dem_out = self.dem(x)
        return dem_out