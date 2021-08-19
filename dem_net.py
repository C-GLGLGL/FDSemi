import torch.nn as nn
from functools import partial
from torchvision import models
import torch.nn.functional as F
from dem import DEM
import torch

nonlinearity = partial(F.relu, inplace=True)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(ch_in, ch_in, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBlock, self).__init__()
        if use_transpose:
            self.up_op = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1,
                                            output_padding=1)
        else:
            self.up_op = Upsample(scale_factor=2, align_corners=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up_op(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DEM_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, final_deconv=True):
        super(DEM_Net, self).__init__()

        filters = [64, 128, 256, 512]

        self.final_deconv = final_deconv

        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])

        self.cab1 = DEM(filters[0])
        self.cab2 = DEM(filters[1])
        self.cab3 = DEM(filters[2])
        self.cab4 = DEM(filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        if final_deconv:
            self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
            self.finalrelu1 = nonlinearity
            self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0] // 2, kernel_size=3, padding=1)
            self.finalrelu2 = nonlinearity
            self.finalconv3 = nn.Conv2d(filters[0] // 2, n_classes, kernerl_size=3, padding=1)
        else:
            self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.cab1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.cab2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.cab3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.cab4(x4)

        # decoding path + concat path
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        if self.final_deconv:
            d1 = self.finaldeconv1(d2)
            d1 = self.finalrelu1(d1)
            d1 = self.finalconv2(d1)
            d1 = self.finalrelu2(d1)
            d1 = self.finalconv3(d1)
        else:
            d1 = self.Conv_1x1(d2)

        if self.n_classes > 1:
            return F.softmax(d1, dim=1)
        else:
            # return torch.sigmoid(d1)
            return d1


class DEM_Net2(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, use_aspp=True):
        super(DEM_Net2, self).__init__()

        self.use_aspp = use_aspp

        filters = [64, 128, 256, 512]
        # filters = [32, 64, 128, 256]
        resnet = models.resnet34(pretrained=False)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.cab1 = DEM(filters[0])
        self.cab2 = DEM(filters[1])
        self.cab3 = DEM(filters[2])
        self.cab4 = DEM(filters[3])

        self.dblock = DACblock(filters[3])
        self.spp = SPPblock(filters[3])

        if self.use_aspp:
            self.decoder4 = DecoderBlock(516, filters[2])
        else:
            self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.feature_out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0] // 2, kernel_size=3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(filters[0] // 2, n_classes, kernerl_size=3, padding=1)

    def forward(self, x):

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e1 = self.cab1(e1)

        e2 = self.encoder2(e1)
        e2 = self.cab2(e2)

        e3 = self.encoder3(e2)
        e3 = self.cab3(e3)

        e4 = self.encoder4(e3)
        e4 = self.cab4(e4)

        if self.use_aspp:
            e4 = self.dblock(e4)
            e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out