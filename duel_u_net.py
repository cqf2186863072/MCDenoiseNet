import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class SEBlock(nn.Module):
    # 注意力模块
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
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


class SingleConv(nn.Module):
    # 单卷积模块
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(SingleConv, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skips_num=1):
        super(Up, self).__init__()
        self.skips_num = skips_num
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * (skips_num + 1), out_channels)

    def forward(self, skips, x):
        x = self.up(x)

        if len(skips) != self.skips_num:
            print("Skips error!")

        shape = skips[0].shape[2:]
        for skip in skips:
            if skip.shape[2:] != shape:
                print("Skips error!")

        if x.shape[2:] != shape:
            x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        x = torch.cat(skips + [x], dim=1)
        return self.conv(x)


class UpConv(nn.Module):
    # 用作输出的上采样卷积模块
    def __init__(self, in_channels, out_channels=3, skips_num=1):
        super(UpConv, self).__init__()
        self.skips_num = skips_num
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * (skips_num + 1), 64)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, skips, x):
        x = self.up(x)

        if len(skips) != self.skips_num:
            print("Skips error!")

        shape = skips[0].shape[2:]
        for skip in skips:
            if skip.shape[2:] != shape:
                print("Skips error!")

        if x.shape[2:] != shape:
            x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        x = torch.cat(skips + [x], dim=1)

        x = self.conv(x)
        x = self.out_conv(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureFusion, self).__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, 32),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, x):
        return self.conv(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureEncoder, self).__init__()
        self.conv1 = DoubleConv(in_channels, in_channels)
        self.down1 = Down(in_channels, 32)
        self.conv2 = DoubleConv(32, 32)
        self.down2 = Down(32, 64)
        self.conv3 = DoubleConv(64, 64)
        self.down3 = Down(64, 128)
        self.conv4 = DoubleConv(128, 128)
        self.down4 = Down(128, 256)
        self.conv5 = DoubleConv(256, 256)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        x5 = self.conv5(self.down4(x4))
        return [x1, x2, x3, x4, x5]


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(ImageEncoder, self).__init__()
        self.conv1 = DoubleConv(in_channels, in_channels)
        self.down1 = Down(in_channels, 32)
        self.conv2 = DoubleConv(32, 32)
        self.down2 = Down(32, 64)
        self.conv3 = DoubleConv(64, 64)
        self.down3 = Down(64, 128)
        self.conv4 = DoubleConv(128, 128)
        self.down4 = Down(128, 256)
        self.conv5 = DoubleConv(256, 256)
        self.down5 = Down(256, 512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)

        self.se2 = SEBlock(32)
        self.se3 = SEBlock(64)
        self.se4 = SEBlock(128)
        self.se5 = SEBlock(256)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.se2(self.conv2(self.down1(x1)))
        x3 = self.se3(self.conv3(self.down2(x2)))
        x4 = self.se4(self.conv4(self.down3(x3)))
        x5 = self.se5(self.conv5(self.down4(x4)))
        x = self.conv6(self.down5(x5))
        return [x1, x2, x3, x4, x5], x


class Decoder(nn.Module):
    def __init__(self, in_channels=512):
        super(Decoder, self).__init__()
        self.up1 = Up(in_channels, 256, skips_num=2)
        # self.up1 = Up(in_channels, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.up5 = UpConv(32, 3)


    def forward(self, image_skips, features_skips, x):
        x = self.up1([image_skips[4], features_skips[4]], x)
        # x = self.up1([image_skips[4]], x)
        x = self.up2([image_skips[3]], x)
        x = self.up3([image_skips[2]], x)
        x = self.up4([image_skips[1]], x)
        x = self.up5([image_skips[0]], x)

        return x


class DuelUNet(nn.Module):
    # 主体网络
    def __init__(self, image_channels=3, features_channels=3):
        super(DuelUNet, self).__init__()
        self.feature_fusion = FeatureFusion(features_channels)
        self.features_encoder = FeatureEncoder(3)
        self.image_encoder = ImageEncoder(3)
        self.decoder = Decoder(512)

    def forward(self, image, features):
        fused_features = self.feature_fusion(features)
        features_skips = self.features_encoder(fused_features)
        image_skips, latent = self.image_encoder(image)
        x = self.decoder(image_skips, features_skips, latent)
        return x


def init_weights(net):
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


if __name__ == '__main__':
    model = DuelUNet(3, 3)
    # model = init_weights(model)
    # for name, param in model.named_parameters():
    #     if 'weight' in name and 'conv' in name:
    #         print(f"{name} mean: {param.data.mean().item():.4f}, std: {param.data.std().item():.4f}")
    #     elif 'bias' in name:
    #         print(f"{name} value: {param.data.mean().item():.4f}")
    model.eval()

    image = torch.randn(1, 3, 1280, 720)
    features = torch.randn(1, 3, 1280, 720)

    with torch.no_grad():
        output = model(image, features)

    print("Input image shape:", image.shape)
    print("Input feature shape:", features.shape)
    print("Output image shape:", output.shape)