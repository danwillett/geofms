import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stack_10min.dataset import RadarGaugeDataset10min


class RadarEncoder10min(nn.Module):
    """
    CNN encoder for single-scan dual-pol radar patches.
    Input: (B, N_channels, H, W) where N_channels = 5 (4 radar + 1 DEM).
    """

    def __init__(self, in_channels=5, latent_dim=256, dropout_rate=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


class ScalarDecoder10min(nn.Module):
    """MLP decoder producing a single scalar precipitation prediction (mm/10min)."""

    def __init__(self, input_dim=256, hidden_dim=256, dropout_rate=0.1):
        super().__init__()

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


class PrecipModel10min(nn.Module):
    """
    Single-scan CNN model for 10-minute precipitation prediction.
    Outputs a scalar mm/10min value.
    """

    def __init__(self, latent_dim=256, dropout_rate=0.2):
        super().__init__()

        in_channels = RadarGaugeDataset10min.n_input_channels()
        self.encoder = RadarEncoder10min(
            in_channels=in_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )
        self.decoder = ScalarDecoder10min(
            input_dim=latent_dim,
            hidden_dim=latent_dim,
            dropout_rate=0.1,
        )

    def forward(self, radar):
        emb = self.encoder(radar)
        return self.decoder(emb)


def init_weights(m):
    """Xavier/Kaiming initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)
