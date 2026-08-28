"""
models/gfm_untrained/model.py — From-scratch (untrained) multimodal ViT.

Reuses TerraMind's *architecture* (multimodal patch embeddings + transformer
blocks + cross-modal attention) but with `pretrained=False`, so every weight is
randomly initialised. This lets us:
  - feed the native radar resolution (no 256×256 upscaling) by choosing a small
    `patch_size` (1 or 2),
  - treat DEM as just another untrained modality,
  - compare a single combined modality vs. grouped DEM/RADAR/DUALPOL streams.

The model is a self-contained LightningModule (no TerraTorch task wrapper), so
the loss/metrics/optimizer are fully under our control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from terratorch.models.backbones.terramind.model.terramind_vit import TerraMindViT


# ── Decoder ───────────────────────────────────────────────────────────────────

class SpatialPrecipitationDecoder(nn.Module):
    """Decode the ViT token grid to an (output_size × output_size) precip map.

    Accepts either a single (B, N, C) token tensor or TerraTorch-style list of
    per-layer features (uses the last). N must be a perfect square.
    """
    includes_head = False

    def __init__(self, in_channels=128, target_size=18, output_bias=2.0, dropout=0.3):
        super().__init__()
        self.out_channels = 1
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv_out   = nn.Conv2d(32, 1, kernel_size=1)
        self.final_pool = nn.AdaptiveAvgPool2d((target_size, target_size))

        with torch.no_grad():
            self.conv_out.bias.data.fill_(output_bias)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        if x.dim() == 3:
            b, num_patches, feat = x.shape
            hw = int(round(num_patches ** 0.5))
            if hw * hw != num_patches:
                raise ValueError(
                    f"Token count {num_patches} is not a perfect square; the "
                    f"decoder needs a square grid. Check input_size / patch_size."
                )
            x = x.transpose(1, 2).reshape(b, feat, hw, hw)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_out(x)
        x = self.final_pool(x)
        return x  # (B, 1, target_size, target_size)


# ── Losses ────────────────────────────────────────────────────────────────────

ASYM_LOSSES = {'asym_mae', 'asym_huber'}


class AsymMAELoss(nn.Module):
    """Element-wise MAE, penalising underprediction (pred < target) more."""
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
    """Element-wise Huber, penalising underprediction more."""
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


# ── LightningModule ─────────────────────────────────────────────────────────--

class UntrainedGFM(L.LightningModule):
    """From-scratch multimodal ViT for sparse-gauge precipitation regression."""

    def __init__(
        self,
        modality_channels: dict,
        input_size: int = 18,
        patch_size: int = 1,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        modality_drop_rate: float = 0.0,
        output_size: int = 18,
        output_bias: float = 2.0,
        loss: str = 'huber',
        under_weight: float = 2.0,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 150,
        warmup_epochs: int = 10,
        ignore_index: int = -9999,
    ):
        super().__init__()
        self.save_hyperparameters()

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        modalities = [{name: int(ch)} for name, ch in modality_channels.items()]
        self.backbone = TerraMindViT(
            img_size=input_size,
            modalities=modalities,
            merge_method='concat',
            patch_size=patch_size,
            dim=dim,
            encoder_depth=depth,
            num_heads=num_heads,
            modality_drop_rate=modality_drop_rate,
            encoder_norm=True,
            pretrained=False,
        )
        in_channels = dim * len(modality_channels)
        self.decoder = SpatialPrecipitationDecoder(
            in_channels=in_channels, target_size=output_size, output_bias=output_bias)

        self.loss_name    = loss
        self.ignore_index = ignore_index
        self.monitor      = 'val/loss'
        if loss == 'asym_mae':
            self._asym = AsymMAELoss(under_weight)
        elif loss == 'asym_huber':
            self._asym = AsymHuberLoss(under_weight)
        else:
            self._asym = None

        self._val_preds = []
        self._val_targets = []

    # ── core ────────────────────────────────────────────────────────────────
    def forward(self, image: dict):
        feats = self.backbone(image)        # list of (B, N, dim*M)
        return self.decoder(feats)          # (B, 1, out, out)

    def _masked_loss(self, pred, target):
        pred  = pred.squeeze(1)             # (B, out, out)
        valid = target != self.ignore_index
        p = pred[valid]
        t = target[valid].float()
        if p.numel() == 0:
            return pred.sum() * 0.0

        name = self.loss_name
        if name == 'mae':
            return F.l1_loss(p, t)
        if name == 'mse':
            return F.mse_loss(p, t)
        if name == 'rmse':
            return torch.sqrt(F.mse_loss(p, t) + 1e-8)
        if name == 'huber':
            return F.huber_loss(p, t, delta=1.0)
        if self._asym is not None:
            return self._asym(p, t).mean()
        raise ValueError(f"Unknown loss: {name}")

    def _valid_pixels(self, pred, target):
        """Return (pred, target) at the valid (gauge) pixels as 1-D tensors."""
        pred  = pred.squeeze(1)
        valid = target != self.ignore_index
        return pred[valid], target[valid].float()

    # ── steps ─────────────────────────────────────────────────────────────--
    def training_step(self, batch, batch_idx):
        pred = self(batch['image'])
        loss = self._masked_loss(pred, batch['mask'])
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch['image'])
        loss = self._masked_loss(pred, batch['mask'])
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        p, t = self._valid_pixels(pred, batch['mask'])
        self._val_preds.append(p.detach().float().cpu())
        self._val_targets.append(t.detach().float().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds   = torch.cat(self._val_preds).numpy()
        targets = torch.cat(self._val_targets).numpy()
        self._val_preds.clear()
        self._val_targets.clear()

        import numpy as np
        mae = float(np.mean(np.abs(targets - preds)))
        ss_res = float(np.sum((targets - preds) ** 2))
        ss_tot = float(np.sum((targets - targets.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        # Keys mirror the gfm EpochSummary callback (model-space metrics).
        self.log('val/MAE', mae, prog_bar=False)
        self.log('val/R2_Score', r2, prog_bar=True)

    # ── optim ─────────────────────────────────────────────────────────────--
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        warmup_epochs = max(0, int(self.hparams.warmup_epochs))
        max_epochs    = max(1, int(self.hparams.max_epochs))
        cosine_T      = max(1, max_epochs - warmup_epochs)

        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_T)
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_T)

        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}


def build_model(modality_channels: dict, cfg: dict) -> UntrainedGFM:
    """Construct an UntrainedGFM from a config dict + datamodule modality_channels."""
    return UntrainedGFM(
        modality_channels=modality_channels,
        input_size=cfg.get('output_size', 18),
        patch_size=cfg.get('patch_size', 1),
        dim=cfg.get('dim', 128),
        depth=cfg.get('depth', 4),
        num_heads=cfg.get('num_heads', 4),
        modality_drop_rate=cfg.get('modality_drop_rate', 0.0),
        output_size=cfg.get('output_size', 18),
        output_bias=cfg.get('output_bias', 2.0),
        loss=cfg.get('loss_type', 'huber'),
        under_weight=cfg.get('under_weight', 2.0),
        lr=cfg.get('lr', 3e-4),
        weight_decay=cfg.get('weight_decay', 0.05),
        max_epochs=cfg.get('max_epochs', 150),
        warmup_epochs=cfg.get('warmup_epochs', 10),
    )
