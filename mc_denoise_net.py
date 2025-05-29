import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function, profile
import math

from sympy import false

BASE_CHANNELS = 64


class SingleConv(nn.Module):
    # 单卷积模块
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    # 双卷积模块
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class EfficientECA(nn.Module):
    """ECANet注意力模块，计算量更小"""
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientECA, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SE(nn.Module):
    # 注意力模块
    def __init__(self, channels, reduction=8):
        super(SE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(Up, self).__init__()
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Merge(nn.Module):
    def __init__(self, channels, skips_num=1):
        super(Merge, self).__init__()
        self.skips_num = skips_num
        merge_channels = channels * (skips_num + 1)
        # self.attention = SE(merge_channels, reduction=max(2, merge_channels // 8))
        self.attention = SE(merge_channels, reduction=max(2, merge_channels // 8))
        self.conv = SingleConv(merge_channels, channels, kernel_size=1)

    def forward(self, skips, x):
        if len(skips) != self.skips_num:
            print("Skips error!")

        shape = skips[0].shape[2:]
        for skip in skips:
            if skip.shape[2:] != shape:
                print("Skips error!")

        if x.shape[2:] != shape:
            x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)

        x = torch.cat(skips + [x], dim=1)
        x = self.attention(x)
        x = self.conv(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureFusion, self).__init__()
        self.attention = SE(in_channels)
        self.conv = nn.Sequential(
            SingleConv(in_channels, in_channels, kernel_size=1),
            SingleConv(in_channels, 3, kernel_size=3),
            SingleConv(3, 3, kernel_size=3),
            SingleConv(3, 3, kernel_size=1),
        )

    def forward(self, x):
        x = self.attention(x)
        x = self.conv(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureEncoder, self).__init__()
        ch1, ch2, ch3, ch4, ch5 = BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4, BASE_CHANNELS * 8, BASE_CHANNELS * 16

        self.conv1 = DoubleConv(in_channels, in_channels)
        self.down1 = Down(in_channels, ch1)
        self.conv2 = DoubleConv(ch1, ch1)
        self.down2 = Down(ch1, ch2)
        self.conv3 = DoubleConv(ch2, ch2)
        self.down3 = Down(ch2, ch3)
        self.conv4 = DoubleConv(ch3, ch3)
        # self.down4 = Down(ch3, ch4)
        # self.conv5 = DoubleConv(ch4, ch4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        # x5 = self.conv5(self.down4(x4))
        # return [x1, x2, x3, x4, x5]
        return [x1, x2, x3, x4]


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(ImageEncoder, self).__init__()
        ch1, ch2, ch3, ch4, ch5 = BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4, BASE_CHANNELS * 8, BASE_CHANNELS * 16

        self.conv1 = DoubleConv(in_channels, in_channels)
        self.down1 = Down(in_channels, ch1)
        self.conv2 = DoubleConv(ch1, ch1)
        self.down2 = Down(ch1, ch2)
        self.conv3 = DoubleConv(ch2, ch2)
        self.down3 = Down(ch2, ch3)
        self.conv4 = DoubleConv(ch3, ch3)
        self.down4 = Down(ch3, ch4)
        # self.conv5 = DoubleConv(ch4, ch4)
        # self.down5 = Down(ch4, ch5)
        # self.conv6 = nn.Conv2d(ch5, ch5, 3, padding=1)
        self.conv5 = nn.Conv2d(ch4, ch4, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        # x5 = self.conv5(self.down4(x4))
        # x = self.conv6(self.down5(x5))
        # return [x1, x2, x3, x4, x5], x

        x = self.conv5(self.down4(x4))
        return [x1, x2, x3, x4], x


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        # ch1, ch2, ch3, ch4, ch5 = BASE_CHANNELS * 8, BASE_CHANNELS * 4, BASE_CHANNELS * 2, BASE_CHANNELS, 3
        ch1, ch2, ch3, ch4 = BASE_CHANNELS * 4, BASE_CHANNELS * 2, BASE_CHANNELS, 3

        self.up1 = Up(in_channels, ch1)
        self.merge1 = Merge(ch1, skips_num=2)
        self.up2 = Up(ch1, ch2)
        self.merge2 = Merge(ch2, skips_num=2)
        self.up3 = Up(ch2, ch3)
        self.merge3 = Merge(ch3, skips_num=1)
        self.up4 = Up(ch3, ch4)
        self.merge4 = Merge(ch4, skips_num=1)
        # self.up5 = Up(ch4, ch5)
        # self.merge5 = Merge(ch5, skips_num=1)

    def forward(self, image_skips, features_skips, x):
        # x = self.up1(x)
        # x = self.merge1([image_skips[4], features_skips[4]], x)
        # x = self.up2(x)
        # x = self.merge2([image_skips[3], features_skips[3]], x)
        # x = self.up3(x)
        # x = self.merge3([image_skips[2]], x)
        # x = self.up4(x)
        # x = self.merge4([image_skips[1]], x)
        # x = self.up5(x)
        # x = self.merge5([image_skips[0]], x)

        x = self.up1(x)
        x = self.merge1([image_skips[3], features_skips[3]], x)
        x = self.up2(x)
        x = self.merge2([image_skips[2], features_skips[2]], x)
        x = self.up3(x)
        x = self.merge3([image_skips[1]], x)
        x = self.up4(x)
        x = self.merge4([image_skips[0]], x)
        return x


class MCDenoiseNet(nn.Module):
    # 主体网络
    def __init__(self, image_channels=3, features_channels=6):
        super(MCDenoiseNet, self).__init__()
        self.feature_fusion = FeatureFusion(features_channels)
        self.features_encoder = FeatureEncoder(3)
        self.image_encoder = ImageEncoder(3)
        # self.decoder = Decoder(BASE_CHANNELS * 16)
        self.decoder = Decoder(BASE_CHANNELS * 8)

    def forward(self, image, features):
        # with record_function("[self]feature fusion"):
        #     fused_features = self.feature_fusion(features)
        # with record_function("[self]feature encoder"):
        #     features_skips = self.features_encoder(fused_features)
        # with record_function("[self]image encoder"):
        #     image_skips, latent = self.image_encoder(image)
        # with record_function("[self]decoder"):
        #     x = self.decoder(image_skips, features_skips, latent)

        with record_function("[self]all"):
            fused_features = self.feature_fusion(features)
            features_skips = self.features_encoder(fused_features)
            image_skips, latent = self.image_encoder(image)
            x = self.decoder(image_skips, features_skips, latent)

        return fused_features, x


def init_weights(net):
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    net.apply(init_func)
    return net


if __name__ == '__main__':
    model = MCDenoiseNet(3, 9).cuda()
    for param in model.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    image = torch.randn((1, 3, 1280, 720), dtype=torch.float).cuda()
    features = torch.randn((1, 9, 1280, 720), dtype=torch.float).cuda()

    for _ in range(5):
        model(image, features)
    with profile(with_stack=True) as prof:
        _, output = model(image, features)

    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))