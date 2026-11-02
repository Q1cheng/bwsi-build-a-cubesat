import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layers(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNetSmall(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_channels=32):
        super().__init__()

        self.in_block = DoubleConv(in_channels, base_channels)

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up4 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = Up(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_block(x)   # (B, 32, H, W)
        x2 = self.down1(x1)     # (B, 64, H/2, W/2)
        x3 = self.down2(x2)     # (B, 128, H/4, W/4)
        x4 = self.down3(x3)     # (B, 256, H/8, W/8)
        x5 = self.down4(x4)     # (B, 512, H/16, W/16)

        y4 = self.up4(x5, x4)   # (B, 256, H/8, W/8)
        y3 = self.up3(y4, x3)   # (B, 128, H/4, W/4)
        y2 = self.up2(y3, x2)   # (B, 64, H/2, W/2)
        y1 = self.up1(y2, x1)   # (B, 32, H, W)

        out = self.out_conv(y1) # (B, 3, H, W)
        return out
    
