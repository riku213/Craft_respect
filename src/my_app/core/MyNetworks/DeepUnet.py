import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DeepUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)
        self.bottleneck = DoubleConv(1024, 2048)

        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec1 = DoubleConv(2048, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec2 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(1024, 512)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec5 = DoubleConv(128, 64) 

        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e5 = self.enc5(F.max_pool2d(e4, 2))
        b = self.bottleneck(F.max_pool2d(e5, 2))
        d1 = self.up1(b)
        # d1のサイズに合わせてe5をクロップ
        e5_cropped = self.crop_tensor(e5, d1)
        d1_cat = torch.cat([e5_cropped, d1], dim=1)
        d1 = self.dec1(d1_cat)

        d2 = self.up2(d1)
        # d2のサイズに合わせてe4をクロップ
        e4_cropped = self.crop_tensor(e4, d2)
        d2_cat = torch.cat([e4_cropped, d2], dim=1)
        d2 = self.dec2(d2_cat)

        d3 = self.up3(d2)
        # d3のサイズに合わせてe3をクロップ
        e3_cropped = self.crop_tensor(e3, d3)
        d3_cat = torch.cat([e3_cropped, d3], dim=1)
        d3 = self.dec3(d3_cat)

        d4 = self.up4(d3)
        # d4のサイズに合わせてe2をクロップ
        e2_cropped = self.crop_tensor(e2, d4)
        d4_cat = torch.cat([e2_cropped, d4], dim=1)
        d4 = self.dec4(d4_cat)

        d5 = self.up5(d4)
        # d5のサイズに合わせてe1をクロップ
        e1_cropped = self.crop_tensor(e1, d5)
        d5_cat = torch.cat([e1_cropped, d5], dim=1)
        d5 = self.dec5(d5_cat)

        return self.out_conv(d5)

    def crop_tensor(self, source, target):
        """
        sourceテンソルをtargetテンソルのサイズに合わせて中央でクロップする
        """
        target_size_h = target.size()[2]
        target_size_w = target.size()[3]
        source_size_h = source.size()[2]
        source_size_w = source.size()[3]
        
        # クロップの開始位置を計算 (中央揃え)
        delta_h = (source_size_h - target_size_h) // 2
        delta_w = (source_size_w - target_size_w) // 2
        
        # スライシングでクロップ
        return source[:, :, delta_h:delta_h + target_size_h, delta_w:delta_w + target_size_w]
