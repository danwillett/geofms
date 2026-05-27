import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stack.dataset import RadarGaugeDataset


class RadarEncoder(nn.Module):
    """
    CNN encoder for dual-pol radar data.
    3-block CNN operating on 9×9 spatial patches.
    """

    def __init__(self, in_channels=73, latent_dim=512, dropout_rate=0.25):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 9×9 → 4×4
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # 512 * 4 * 4 = 8192
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        return self.fc2(x)


class PrecipitationDecoder(nn.Module):
    """
    MLP decoder producing a 5×5 precipitation map from a latent embedding.
    Uses LayerNorm for stability with small/variable batch sizes.
    """

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
    """

    def __init__(self, latent_dim=512, add_bias=False, dropout_rate=0.25):
        super().__init__()

        in_channels = RadarGaugeDataset.n_input_channels()
        self.radar_encoder = RadarEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )

        self.add_bias = add_bias
        self.bias_embedding = nn.Embedding(num_embeddings=3, embedding_dim=32)

        decoder_input_dim = latent_dim + 32 if add_bias else latent_dim
        self.decoder = PrecipitationDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=1024,
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
