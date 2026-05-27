import pathlib
import torch
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
import pickle
from terratorch.tasks import PixelwiseRegressionTask

class SpatialPrecipitationDecoder(nn.Module):
    """
    Decodes spatial features to a 5×5 precipitation map.
    Compatible with TerraTorch's multi-scale feature lists.
    """
    includes_head = False

    def __init__(self, in_channels=192, target_size=5, output_bias=2.0):
        super().__init__()
        
        self.out_channels = 1
        
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)
        self.final_pool = nn.AdaptiveAvgPool2d((target_size, target_size))

        with torch.no_grad():
            self.conv_out.bias.data.fill_(output_bias)
        
    def forward(self, x):
        """
        Input: List of feature tensors OR single tensor
        Output: (batch, 5, 5) - precipitation map
        """
        # === HANDLE LIST INPUT FROM TERRATORCH ===
        if isinstance(x, list):
            # Take the last feature map (highest-level features)
            x = x[-1]
        
        # If features are (batch, patches, channels), reshape to spatial
        if x.dim() == 3:
            batch_size, num_patches, feat_dim = x.shape
            H = W = int(num_patches ** 0.5)  # e.g., 256 patches → 16×16
            x = x.transpose(1, 2).reshape(batch_size, feat_dim, H, W)
        
        # Now x should be (batch, channels, H, W)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 128, H, W)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, H, W)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 32, H, W)
        
        x = self.conv_out(x)  # (B, 1, H, W)
        x = self.final_pool(x)  # (B, 1, 5, 5)
        
        # DON'T squeeze - TerraTorch expects (B, C, H, W) output
        return x  # (B, 1, 5, 5)

# ── Channel counts ─────────────────────────────────────────────────────────
N_RADAR_CHANNELS = 72   # 5 dual-pol fields × 12 scans × 3 (field, mask, t_pos)
N_DEM_CHANNELS   = 1

def build_task(lr=1e-5, output_bias=3.0) -> PixelwiseRegressionTask:
    """
    Build and return the full PixelwiseRegressionTask with:
      - TerraMind-tiny backbone (frozen except RADAR embedding)
      - SpatialPrecipitationDecoder head (trainable)
      - DEM + RADAR (180-channel dual-pol) modalities
    """
    task = PixelwiseRegressionTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            'backbone':            'terramind_v1_tiny',
            'backbone_pretrained': True,
            'backbone_modalities': ["DEM", {"RADAR": N_RADAR_CHANNELS}],
            'backbone_merge_method': 'concat',
            'decoder': SpatialPrecipitationDecoder(in_channels=384, output_bias=output_bias),
            'rescale': False,
        },
        freeze_backbone=False,
        freeze_decoder=False,
        loss='mse',
        lr=lr,
        lr_overrides={
            'encoder_embeddings.RADAR': 1e-4,  # 10× higher for RADAR embedding
        },
        optimizer='AdamW',          # adds weight decay support
        optimizer_hparams={'weight_decay': 1e-4},
        ignore_index=-9999,
        scheduler='ReduceLROnPlateau',
    )

    # Unfreeze only the RADAR embedding so it adapts to 180-channel input
    backbone = task.model.encoder
    for param in backbone.encoder_embeddings['RADAR'].parameters():
        param.requires_grad = True

    # # Unfreeze the last 2 transformer blocks
    # for param in backbone.encoder[-2:].parameters():
    #     param.requires_grad = True

    return task