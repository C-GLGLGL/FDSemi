import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class FocusAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16, global_pool=['avg', 'gc']):
        super(FocusAttention, self).__init__()
        self.channels = channels

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(self.channels, self.channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(self.channels // reduction_ratio, self.channels)
            )

        self.aggregate = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 4 // reduction_ratio, kernel_size=1),
            nn.LayerNorm([self.channels * 4 // reduction_ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels * 4 // reduction_ratio, self.channels, kernel_size=1))

        self.global_pool = global_pool
        # for context modeling
        self.conv_mask = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        f_att_sum = None
        for pool_type in self.global_pool:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                f_att_raw = self.aggregate(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                f_att_raw = self.aggregate(max_pool)
            elif pool_type=='gc':
                batch, channel, height, width = x.size()
                input_x = x
                # [N, C, H * W]
                input_x = input_x.view(batch, channel, height * width)
                # [N, 1, C, H * W]
                input_x = input_x.unsqueeze(1)
                # [N, 1, H, W]
                context_mask = self.conv_mask(x)
                # [N, 1, H * W]
                context_mask = context_mask.view(batch, 1, height * width)
                # [N, 1, H * W]
                context_mask = self.softmax(context_mask)
                # [N, 1, H * W, 1]
                context_mask = context_mask.unsqueeze(-1)
                # context_mask = context_mask.unsqueeze(3)
                # [N, 1, C, 1]
                context = torch.matmul(input_x, context_mask)
                # [N, C, 1, 1]
                context = context.view(batch, channel, 1, 1)
                gc_pool = context
                f_att_raw = self.aggregate(gc_pool)

            if f_att_sum is None:
                f_att_sum = f_att_raw
            else:
                f_att_sum = f_att_sum + f_att_raw

        scale = F.sigmoid(f_att_sum).expand_as(x)
        return scale * x

class FEM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, global_pool=['avg', 'gc']):
        super(FEM, self).__init__()
        self.fem = FocusAttention(channels, reduction_ratio, global_pool)

    def forward(self, x):
        fem_out = self.fem(x)
        return fem_out