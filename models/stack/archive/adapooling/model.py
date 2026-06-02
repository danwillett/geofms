import torch
import torch.nn as nn
import torch.nn.functional as F


class RadarEncoder(nn.Module):
    """
    CNN encoder for dual-pol radar data.
    Dynamic depth with adaptive pooling to support any spatial patch size.

    Recommended settings:
      patch 5x5  -> n_blocks=2
      patch 9x9  -> n_blocks=3 (default)
      patch 19x19 -> n_blocks=4
    """

    def __init__(self, in_channels=73, latent_dim=512, dropout_rate=0.25, n_blocks=3):
        super().__init__()
        self.n_blocks = n_blocks

        # Channel progression: in -> 192 -> 384 -> 512 -> ...
        channel_sizes = [192, 384, 512, 640, 768][:n_blocks]

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.dropout_blocks = nn.ModuleList()
        self.pool_after = set()

        prev_ch = in_channels
        for i, out_ch in enumerate(channel_sizes):
            self.conv_blocks.append(nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1))
            self.bn_blocks.append(nn.BatchNorm2d(out_ch))
            self.dropout_blocks.append(nn.Dropout2d(p=dropout_rate))
            # Pool after every other block starting from block 1
            if i > 0 and i % 2 == 1:
                self.pool_after.add(i)
            prev_ch = out_ch

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        final_ch = channel_sizes[-1]
        self.fc1 = nn.Linear(final_ch * 4 * 4, 1024)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, latent_dim)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = F.relu(self.bn_blocks[i](self.conv_blocks[i](x)))
            x = self.dropout_blocks[i](x)
            if i in self.pool_after:
                x = F.max_pool2d(x, 2)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        return self.fc2(x)


class ScalarDecoder(nn.Module):
    """MLP decoder producing a single scalar precipitation prediction."""

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


class PrecipitationDecoder(nn.Module):
    """MLP decoder producing a spatial precipitation map from a latent embedding."""

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


class PrecipitationStackModel(nn.Module):
    """
    Complete CNN model for precipitation prediction from dual-pol radar + DEM.
    Dynamic encoder depth via n_encoder_blocks parameter.
    """

    def __init__(self, in_channels=73, latent_dim=512, add_bias=False,
                 dropout_rate=0.25, output_size=9, scalar_output=False,
                 n_encoder_blocks=3):
        super().__init__()
        self.scalar_output = scalar_output

        self.radar_encoder = RadarEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            n_blocks=n_encoder_blocks,
        )

        self.add_bias = add_bias
        self.bias_embedding = nn.Embedding(num_embeddings=3, embedding_dim=32)

        decoder_input_dim = latent_dim + 32 if add_bias else latent_dim

        if scalar_output:
            self.decoder = ScalarDecoder(
                input_dim=decoder_input_dim,
                hidden_dim=512,
                dropout_rate=0.1,
            )
        else:
            self.decoder = PrecipitationDecoder(
                input_dim=decoder_input_dim,
                hidden_dim=1024,
                output_size=output_size,
                dropout_rate=0.1,
            )

    def forward(self, radar, bias_flag=None):
        emb = self.radar_encoder(radar)

        if self.add_bias and bias_flag is not None:
            bias_idx = (bias_flag + 1).long()
            bias_emb = self.bias_embedding(bias_idx)
            emb = torch.cat([emb, bias_emb], dim=1)

        return self.decoder(emb)


def init_weights(m):
    """Xavier/Kaiming initialization to prevent dead ReLUs."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)
