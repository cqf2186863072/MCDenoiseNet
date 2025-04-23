import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


BASE_CHANNELS = 16

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


class SingleConv(nn.Module):
    # 单卷积模块
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=False):
        super(SingleConv, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')]
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
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(Up, self).__init__()
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up(x)


class Merge(nn.Module):
    def __init__(self, channels, skips_num=1):
        super(Merge, self).__init__()
        self.skips_num = skips_num
        merge_channels = channels * (skips_num + 1)
        self.se = SE(merge_channels, reduction=max(2, merge_channels // 8))
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
        x = self.se(x)
        x = self.conv(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureFusion, self).__init__()
        ch = BASE_CHANNELS * 2
        self.conv = nn.Sequential(
            DoubleConv(in_channels, ch),
            nn.Conv2d(ch, 3, 1)
        )

    def forward(self, x):
        return self.conv(x)


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
        self.down4 = Down(ch3, ch4)
        self.conv5 = DoubleConv(ch4, ch4)

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
        ch1, ch2, ch3, ch4, ch5, ch6 = BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4, BASE_CHANNELS * 8, BASE_CHANNELS * 16, BASE_CHANNELS * 32

        self.conv1 = DoubleConv(in_channels, in_channels)
        self.down1 = Down(in_channels, ch1)
        self.conv2 = DoubleConv(ch1, ch1)
        self.down2 = Down(ch1, ch2)
        self.conv3 = DoubleConv(ch2, ch2)
        self.down3 = Down(ch2, ch3)
        self.conv4 = DoubleConv(ch3, ch3)
        self.down4 = Down(ch3, ch4)
        self.conv5 = DoubleConv(ch4, ch4)
        self.down5 = Down(ch4, ch5)
        self.conv6 = nn.Conv2d(ch5, ch5, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        x5 = self.conv5(self.down4(x4))
        x = self.conv6(self.down5(x5))
        return [x1, x2, x3, x4, x5], x


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        ch1, ch2, ch3, ch4, ch5 = BASE_CHANNELS * 8, BASE_CHANNELS * 4, BASE_CHANNELS * 2, BASE_CHANNELS, 3

        self.up1 = Up(in_channels, ch1)
        self.merge1 = Merge(ch1, skips_num=2)
        self.up2 = Up(ch1, ch2)
        self.merge2 = Merge(ch2, skips_num=2)
        self.up3 = Up(ch2, ch3)
        self.merge3 = Merge(ch3, skips_num=1)
        self.up4 = Up(ch3, ch4)
        self.merge4 = Merge(ch4, skips_num=1)
        self.up5 = Up(ch4, ch5)
        self.merge5 = Merge(ch5, skips_num=1)

    def forward(self, image_skips, features_skips, x):
        x = self.up1(x)
        x = self.merge1([image_skips[4], features_skips[4]], x)
        x = self.up2(x)
        x = self.merge2([image_skips[3], features_skips[3]], x)
        x = self.up3(x)
        x = self.merge3([image_skips[2]], x)
        x = self.up4(x)
        x = self.merge4([image_skips[1]], x)
        x = self.up5(x)
        x = self.merge5([image_skips[0]], x)
        return x


class MCDenoiseNet(nn.Module):
    # 主体网络
    def __init__(self, image_channels=3, features_channels=6):
        super(MCDenoiseNet, self).__init__()
        self.feature_fusion = FeatureFusion(features_channels)
        self.features_encoder = FeatureEncoder(3)
        self.image_encoder = ImageEncoder(3)
        self.decoder = Decoder(BASE_CHANNELS * 16)

    def forward(self, image, features):
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
    model = MCDenoiseNet(3, 9)
    model = init_weights(model)
    # for name, param in model.named_parameters():
    #     if 'weight' in name and 'conv' in name:
    #         print(f"{name} mean: {param.data.mean().item():.4f}, std: {param.data.std().item():.4f}")
    #     elif 'bias' in name:
    #         print(f"{name} value: {param.data.mean().item():.4f}")
    model.eval()

    image = torch.randn(1, 3, 1280, 720)
    features = torch.randn(1, 9, 1280, 720)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        _, output = model(image, features)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # 以毫秒为单位
    print(f"Cost time:{elapsed_time / 1000} s")


    print("Input image shape:", image.shape)
    print("Input feature shape:", features.shape)
    print("Output image shape:", output.shape)