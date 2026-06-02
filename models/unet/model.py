import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Dynamic depth: n_encoder_blocks controls the number of encoder/decoder stages.
    Automatically adapts to input spatial size.

    Recommended settings:
      patch 5x5  -> n_encoder_blocks=1
      patch 9x9  -> n_encoder_blocks=2
      patch 19x19 -> n_encoder_blocks=3
    """

    def __init__(self, in_channels=133, base_filters=64, dropout_rate=0.15, n_encoder_blocks=None):
        super().__init__()
        self.add_bias = False

        # Auto-detect depth if not specified (default 3 for backward compat)
        if n_encoder_blocks is None:
            n_encoder_blocks = 3
        self.n_encoder_blocks = n_encoder_blocks

        f = base_filters

        # Build encoder blocks dynamically
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for i in range(n_encoder_blocks):
            out_ch = f * (2 ** i)
            self.encoders.append(ConvBlock(prev_ch, out_ch, dropout_rate))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = out_ch

        # Bottleneck
        bottleneck_ch = f * (2 ** n_encoder_blocks)
        self.bottleneck = ConvBlock(prev_ch, bottleneck_ch, dropout_rate)

        # Build decoder blocks (reverse order)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = bottleneck_ch
        for i in range(n_encoder_blocks - 1, -1, -1):
            enc_ch = f * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, enc_ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(enc_ch * 2, enc_ch, dropout_rate))
            prev_ch = enc_ch

        # Output head
        self.out_conv = nn.Conv2d(prev_ch, 1, kernel_size=1)

    def forward(self, x, bias_flag=None):
        # Encoder
        enc_features = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            enc_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(enc_features)):
            x = upconv(x)
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))

        out = self.out_conv(x).squeeze(1)  # (B, H, W)
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
