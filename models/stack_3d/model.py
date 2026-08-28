import torch
import torch.nn as nn
import torch.nn.functional as F


class RadarEncoder3D(nn.Module):
    """
    3D CNN encoder for multi-scan radar volumes.

    Pools spatially (H, W) more aggressively than vertically (Z) to preserve
    height structure until the decoder head collapses Z.
    """

    def __init__(self, in_channels, latent_dim=512, dropout_rate=0.25, n_blocks=3):
        super().__init__()
        self.n_blocks = n_blocks
        channel_sizes = [64, 128, 256, 384][:n_blocks]

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.dropout_blocks = nn.ModuleList()
        self.pool_after = set()

        prev_ch = in_channels
        for i, out_ch in enumerate(channel_sizes):
            self.conv_blocks.append(nn.Conv3d(prev_ch, out_ch, kernel_size=3, padding=1))
            self.bn_blocks.append(nn.BatchNorm3d(out_ch))
            self.dropout_blocks.append(nn.Dropout3d(p=dropout_rate))
            if i > 0 and i % 2 == 1:
                self.pool_after.add(i)
            prev_ch = out_ch

        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        final_ch = channel_sizes[-1]
        self.fc1 = nn.Linear(final_ch * 4 * 4 * 4, 1024)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, latent_dim)

    @property
    def final_channels(self):
        channel_sizes = [64, 128, 256, 384][:self.n_blocks]
        return channel_sizes[-1]

    def _encode(self, x):
        for i in range(self.n_blocks):
            x = F.relu(self.bn_blocks[i](self.conv_blocks[i](x)))
            x = self.dropout_blocks[i](x)
            if i in self.pool_after:
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return x

    def forward(self, x):
        x = self._encode(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        return self.fc2(x)

    def forward_spatial(self, x):
        return self._encode(x)


class ScalarDecoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.output_size = 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc_out(x).squeeze(-1)


class SpatialConvHead3D(nn.Module):
    """
    3D convolutional head that collapses the vertical dimension to a 2D precip map.
    """

    def __init__(self, in_channels, output_size, dropout_rate=0.1, z_collapse='mean'):
        super().__init__()
        self.output_size = output_size
        self.z_collapse = z_collapse

        mid = max(in_channels // 2, 16)
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            nn.Conv3d(mid, mid // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.head(x)  # (B, 1, D', H', W')
        if self.z_collapse == 'mean':
            x = x.mean(dim=2)
        else:
            x = x.max(dim=2).values
        x = x.squeeze(1)
        if x.shape[-1] != self.output_size:
            x = F.interpolate(
                x.unsqueeze(1),
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        return x


class PrecipitationDecoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_size=9, dropout_rate=0.1):
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim // 4, output_size * output_size)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc_out(x)
        return x.view(-1, self.output_size, self.output_size)


class PrecipitationStack3DModel(nn.Module):
    """3D Stack CNN: Conv3d encoder + spatial head that collapses Z to 2D precip."""

    def __init__(
        self,
        in_channels,
        latent_dim=512,
        add_bias=False,
        dropout_rate=0.25,
        output_size=9,
        scalar_output=False,
        n_encoder_blocks=3,
        spatial_head=True,
        z_collapse='mean',
    ):
        super().__init__()
        self.scalar_output = scalar_output
        self.spatial_head = spatial_head
        self.add_bias = add_bias

        self.radar_encoder = RadarEncoder3D(
            in_channels=in_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            n_blocks=n_encoder_blocks,
        )
        self.bias_embedding = nn.Embedding(num_embeddings=3, embedding_dim=32)

        if spatial_head and not scalar_output:
            self.decoder = SpatialConvHead3D(
                in_channels=self.radar_encoder.final_channels,
                output_size=output_size,
                dropout_rate=0.1,
                z_collapse=z_collapse,
            )
        else:
            decoder_input_dim = latent_dim + 32 if add_bias else latent_dim
            if scalar_output:
                self.decoder = ScalarDecoder(input_dim=decoder_input_dim)
            else:
                self.decoder = PrecipitationDecoder(
                    input_dim=decoder_input_dim,
                    output_size=output_size,
                )

    def forward(self, radar, bias_flag=None):
        if self.spatial_head and not self.scalar_output:
            feat_map = self.radar_encoder.forward_spatial(radar)
            return self.decoder(feat_map)

        emb = self.radar_encoder(radar)
        if self.add_bias and bias_flag is not None:
            bias_idx = (bias_flag + 1).long()
            bias_emb = self.bias_embedding(bias_idx)
            emb = torch.cat([emb, bias_emb], dim=1)
        return self.decoder(emb)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)
