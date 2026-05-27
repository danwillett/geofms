import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stack.dataset import RadarGaugeDataset


class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PrecipUNet(nn.Module):
    """
    U-Net for precipitation estimation from dual-pol radar + DEM.
    Preserves spatial correspondence via skip connections.
    Outputs a spatial precipitation map at the same resolution as input.
    """

    def __init__(self, in_channels=None, base_filters=64, dropout_rate=0.15):
        super().__init__()
        if in_channels is None:
            in_channels = RadarGaugeDataset.n_input_channels()

        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_channels, f, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(f, f * 2, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(f * 2, f * 4, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 4, f * 8, dropout_rate)

        # Decoder
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, dropout_rate)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, dropout_rate)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, dropout_rate)

        # Output head: single channel precipitation
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

        self.add_bias = False

    def forward(self, x, bias_flag=None):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1).squeeze(1)  # (B, H, W)
        return out


def init_weights(m):
    """Kaiming initialization for conv layers, Xavier for linear."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
