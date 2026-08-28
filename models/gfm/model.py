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


class UpscalingPrecipTask(PixelwiseRegressionTask):
    """
    Thin wrapper around PixelwiseRegressionTask that upscales small input
    tiles to 256×256 on the GPU at the start of each forward pass.

    This keeps the DataLoader lightweight (small tensors, fast CPU work)
    while meeting the ViT backbone's fixed 256×256 input requirement.
    """

    @staticmethod
    def _upscale_batch(batch, target=256):
        image = batch['image']
        upscaled = {}
        for key, tensor in image.items():
            if tensor.shape[-1] != target or tensor.shape[-2] != target:
                mode = 'nearest' if key == 'RADAR' else 'bilinear'
                kwargs = {} if mode == 'nearest' else {'align_corners': False}
                upscaled[key] = F.interpolate(
                    tensor.float(), size=(target, target), mode=mode, **kwargs
                )
            else:
                upscaled[key] = tensor
        return {**batch, 'image': upscaled}

    def training_step(self, batch, batch_idx):
        return super().training_step(self._upscale_batch(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(self._upscale_batch(batch), batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return super().predict_step(self._upscale_batch(batch), batch_idx, dataloader_idx)

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
from models.gfm.dataset import compute_radar_channels, DEFAULT_FIELDS

N_DEM_CHANNELS = 1

# Asymmetric losses penalise underprediction (pred < target) more heavily than
# overprediction. Both return element-wise loss (reduction="none") so they can be
# wrapped by TerraTorch's IgnoreIndexLossWrapper, which masks the -9999 pixels.
ASYM_LOSSES = {'asym_mae', 'asym_huber'}


class AsymMAELoss(nn.Module):
    def __init__(self, under_weight=2.0):
        super().__init__()
        self.under_weight = under_weight

    def forward(self, output, target):
        residual = output - target.float()
        w = torch.where(residual < 0,
                        torch.full_like(residual, self.under_weight),
                        torch.ones_like(residual))
        return w * residual.abs()


class AsymHuberLoss(nn.Module):
    def __init__(self, under_weight=2.0, delta=1.0):
        super().__init__()
        self.under_weight = under_weight
        self.delta = delta

    def forward(self, output, target):
        residual = output - target.float()
        abs_r = residual.abs()
        huber = torch.where(abs_r <= self.delta,
                            0.5 * residual ** 2,
                            self.delta * (abs_r - 0.5 * self.delta))
        w = torch.where(residual < 0,
                        torch.full_like(huber, self.under_weight),
                        torch.ones_like(huber))
        return w * huber


def _reinit_dem_embedding(backbone):
    """Randomly re-initialize the DEM encoder embedding.

    Wipes the pretrained DEM patch-projection (proj.weight, a Linear with no
    bias) and modality token (mod_emb), restoring the from-scratch init scheme
    used by ImageEncoderEmbedding (kaiming_uniform proj + N(0, 0.02) mod_emb).
    The sincos pos_emb is a fixed buffer (not modality-specific) so it's left
    alone. Used to isolate the value of the *pretrained* DEM weights: keep the
    pretrained transformer backbone but give it a random DEM input projection.
    """
    # TerraMind keys the DEM embedding by its canonical modality name
    # (e.g. 'untok_dem@224'), not 'DEM'. Resolve it case-insensitively.
    keys = list(backbone.encoder_embeddings.keys())
    dem_keys = [k for k in keys if 'dem' in k.lower()]
    if not dem_keys:
        raise KeyError(f"No DEM embedding found in encoder_embeddings; keys={keys}")
    dem_key = dem_keys[0]
    dem_emb = backbone.encoder_embeddings[dem_key]
    reinit = []
    if hasattr(dem_emb, 'proj') and hasattr(dem_emb.proj, 'reset_parameters'):
        dem_emb.proj.reset_parameters()
        reinit.append('proj')
    if hasattr(dem_emb, 'mod_emb'):
        nn.init.normal_(dem_emb.mod_emb, std=0.02)
        reinit.append('mod_emb')
    print(f"  DEM embedding '{dem_key}' re-initialized from scratch "
          f"(reset: {', '.join(reinit) or 'none'})")


def build_task(lr=1e-5, output_bias=3.0, n_radar_channels=None, output_size=9,
               loss='mse', weight_decay=1e-4, radar_lr=1e-4,
               under_weight=2.0, backbone_pretrained=True,
               dem_init='pretrained') -> UpscalingPrecipTask:
    """
    Build and return the full UpscalingPrecipTask with:
      - TerraMind-tiny backbone (frozen except RADAR embedding)
      - SpatialPrecipitationDecoder head (trainable)
      - DEM + RADAR modalities
      - GPU upscale of small input tiles to 256×256

    Parameters
    ----------
    n_radar_channels : int or None
        Number of RADAR channels. If None, computed from DEFAULT_FIELDS with
        feature masks enabled (252 channels for the 17-field subml set).
        Pass the value from RadarDEMDataModule.n_radar_channels to keep model
        and datamodule in sync.
    output_size : int
        Spatial size of the output mask (must match datamodule's output_size).
        Passed to SpatialPrecipitationDecoder as target_size.
    loss : str
        Loss type. TerraTorch-native: 'mse', 'mae', 'rmse', 'huber'. Custom
        asymmetric: 'asym_mae', 'asym_huber' (penalise underprediction via
        under_weight). Asym losses are injected by overriding the task criterion.
    weight_decay : float
        AdamW weight decay.
    radar_lr : float
        Per-parameter LR override for the RADAR encoder embedding (typically
        higher than the base lr so it adapts to the many-channel radar input).
    under_weight : float
        Multiplier applied to underprediction error for asymmetric losses.
    backbone_pretrained : bool
        If True (default), load TerraMind's pretrained weights. If False, build
        the same tiny architecture with random init — the "scratch" control that
        decouples pretraining from capacity.
    dem_init : str
        'pretrained' (default) keeps TerraMind's pretrained DEM patch-embedding.
        'random' re-initializes only the DEM embedding (proj + mod_emb) while
        leaving the rest of the (pretrained) backbone intact — isolates the
        contribution of the pretrained DEM weights. No-op when
        backbone_pretrained is False (DEM is already random there).
    """
    if n_radar_channels is None:
        n_radar_channels = compute_radar_channels(DEFAULT_FIELDS, use_feature_masks=True)

    if dem_init not in ('pretrained', 'random'):
        raise ValueError(f"dem_init must be 'pretrained' or 'random', got {dem_init!r}")

    # TerraTorch only knows mse/mae/rmse/huber. For asym losses we build with a
    # valid placeholder, then override task.criterion below.
    ignore_index = -9999
    parent_loss = 'mae' if loss in ASYM_LOSSES else loss

    task = UpscalingPrecipTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            'backbone':            'terramind_v1_tiny',
            'backbone_pretrained': backbone_pretrained,
            'backbone_modalities': ["DEM", {"RADAR": n_radar_channels}],
            'backbone_merge_method': 'concat',
            'decoder': SpatialPrecipitationDecoder(
                in_channels=384, target_size=output_size, output_bias=output_bias
            ),
            'rescale': False,
        },
        freeze_backbone=False,
        freeze_decoder=False,
        loss=parent_loss,
        lr=lr,
        lr_overrides={
            'encoder_embeddings.RADAR': radar_lr,
        },
        optimizer='AdamW',
        optimizer_hparams={'weight_decay': weight_decay},
        ignore_index=ignore_index,
        scheduler='ReduceLROnPlateau',
        # Disable TerraTorch's built-in validation sample plotting: it calls
        # val_dataset.plot(), which RadarDEMDataset doesn't implement. We produce
        # our own figures in evaluate.py. (Without this, a default TensorBoard
        # logger re-enables the broken plot path when W&B is off.)
        plot_on_val=0,
    )

    # Inject asymmetric loss, masking the -9999 sparse-target pixels.
    if loss in ASYM_LOSSES:
        from terratorch.tasks.regression_tasks import IgnoreIndexLossWrapper
        base = (AsymMAELoss(under_weight) if loss == 'asym_mae'
                else AsymHuberLoss(under_weight))
        task.criterion = IgnoreIndexLossWrapper(base, ignore_index)
        print(f"  Using custom '{loss}' loss (under_weight={under_weight})")

    # Unfreeze only the RADAR embedding so it adapts to 180-channel input
    backbone = task.model.encoder
    for param in backbone.encoder_embeddings['RADAR'].parameters():
        param.requires_grad = True

    # Optionally wipe the pretrained DEM embedding (random DEM channel control).
    # Only meaningful when the backbone is pretrained; otherwise DEM is already
    # randomly initialized.
    if dem_init == 'random' and backbone_pretrained:
        _reinit_dem_embedding(backbone)

    print(f"  Backbone: terramind_v1_tiny | pretrained={backbone_pretrained} | "
          f"dem_init={dem_init if backbone_pretrained else 'random (scratch)'}")

    # # Unfreeze the last 2 transformer blocks
    # for param in backbone.encoder[-2:].parameters():
    #     param.requires_grad = True

    return task