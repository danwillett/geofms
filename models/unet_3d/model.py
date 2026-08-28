import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two 3D conv layers with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PrecipUNet3D(nn.Module):
    """
    3D U-Net for precipitation estimation from a multi-scan radar volume + DEM.

    Input  : (B, C, D, H, W)  — C stacks field×scan volumes + mask + tpos + DEM,
             D is the vertical (z) axis, H/W the horizontal patch.
    Output : (B, H, W)        — a 2D precip map at the gauge plane. The vertical
             dimension is collapsed (mean/max) at the output head so the gauge-pixel
             loss can index the horizontal location directly.

    Pooling/upsampling act on all three (D, H, W) axes symmetrically; odd-size
    mismatches on the way up are repaired with trilinear interpolation to the skip
    tensor's shape, mirroring the 2D PrecipUNet.

    Recommended settings:
      patch 5x5   -> n_encoder_blocks=1
      patch 9x9   -> n_encoder_blocks=2
      patch 19x19 -> n_encoder_blocks=3
    """

    def __init__(self, in_channels=73, base_filters=32, dropout_rate=0.15,
                 n_encoder_blocks=None, z_collapse='mean'):
        super().__init__()
        self.add_bias = False

        if n_encoder_blocks is None:
            n_encoder_blocks = 3
        self.n_encoder_blocks = n_encoder_blocks
        self.z_collapse = z_collapse

        f = base_filters

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for i in range(n_encoder_blocks):
            out_ch = f * (2 ** i)
            self.encoders.append(ConvBlock3D(prev_ch, out_ch, dropout_rate))
            self.pools.append(nn.MaxPool3d(2))
            prev_ch = out_ch

        # Bottleneck
        bottleneck_ch = f * (2 ** n_encoder_blocks)
        self.bottleneck = ConvBlock3D(prev_ch, bottleneck_ch, dropout_rate)

        # Decoder (reverse order)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = bottleneck_ch
        for i in range(n_encoder_blocks - 1, -1, -1):
            enc_ch = f * (2 ** i)
            self.upconvs.append(nn.ConvTranspose3d(prev_ch, enc_ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(enc_ch * 2, enc_ch, dropout_rate))
            prev_ch = enc_ch

        # Output head: collapse Z, then 2D 1x1 conv -> (B, 1, H, W)
        self.out_conv = nn.Conv2d(prev_ch, 1, kernel_size=1)

    def _collapse_z(self, x):
        if self.z_collapse == 'max':
            return x.max(dim=2).values
        return x.mean(dim=2)

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
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))

        # x: (B, C, D, H, W) -> collapse vertical -> (B, C, H, W)
        x = self._collapse_z(x)
        out = self.out_conv(x).squeeze(1)  # (B, H, W)
        return out


def init_weights(m):
    """Kaiming init for conv layers, Xavier for linear."""
    if isinstance(m, (nn.Conv3d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
