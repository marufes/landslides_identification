import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict
import timm
import numpy as np
import cv2


# ----------------- BFA Module ---------------------
class BFA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BFA, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels + 1, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        gray_image = self.to_grayscale(x)
        edge_image = self.canny_edge_detection(gray_image)
        concatenated_image = torch.cat([x, edge_image], dim=1)
        output = self.conv1x1(concatenated_image)
        return self.relu(output)

    def to_grayscale(self, x):
        gray_image = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        return gray_image.unsqueeze(1)

    def canny_edge_detection(self, gray_image):
        gray_np = gray_image.cpu().detach().numpy().astype(np.uint8)
        edge_np = np.zeros_like(gray_np)
        for i in range(gray_np.shape[0]):
            edge_np[i, 0] = cv2.Canny(gray_np[i, 0], 50, 150)
        edge_tensor = torch.from_numpy(edge_np).float().to(gray_image.device)
        return edge_tensor


# ----------------- Up & OutConv ---------------------
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ----------------- Swin UNet with BFA ---------------------
class BFA_SwinUNet(nn.Module):
    def __init__(self, num_classes):
        super(BFA_SwinUNet, self).__init__()
        # Load pretrained swin transformer
        swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
        self.backbone = swin
        channels = swin.feature_info.channels()  # [128, 256, 512, 1024]

        # BFA applied on last feature map
        self.bfa = BFA(in_channels=channels[-1], out_channels=channels[-1])

        self.up1 = Up(channels[3] + channels[2], channels[2])
        self.up2 = Up(channels[2] + channels[1], channels[1])
        self.up3 = Up(channels[1] + channels[0], channels[0])
        self.up4 = Up(channels[0], 64)

        self.conv = OutConv(64, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)  # List of [B, C, H', W']
        f0, f1, f2, f3 = features  # From shallow to deep
        f3 = self.bfa(f3)          # Apply BFA on deepest feature

        x = self.up1(f3, f2)
        x = self.up2(x, f1)
        x = self.up3(x, f0)
        x = self.up4(x, None)  # no skip connection here

        out = self.conv(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return {"out": out}
