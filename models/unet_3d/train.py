"""
train.py — Train the 3D U-Net precipitation model.

Operates on lazy-loaded radar volumes (B, C, D, H, W) from the 3D index pickle
and predicts a 2D precip map, scored at the gauge pixel.

Run from the project root:
    python -m models.unet_3d.train --pickle dataset/outputs/3d/radar_gauge_dataset_3d_9500_temporal.pkl --batch-size 8
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from models.unet_3d.model import PrecipUNet3D, init_weights
from models.stack_3d.dataset import RadarGauge3DDataset, resolve_fields, compute_n_input_channels
from models.stack_3d.train import (
    filter_stations, filter_nan_radar, filter_bad_samples, filter_biased_extremes,
    filter_suspect_station_days, filter_gauge_dumps, create_sampler,
    train_epoch, validate, create_run_dir,
)

# ── DEFAULT CONFIG ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'pickle_path':    'dataset/outputs/3d/radar_gauge_dataset_3d_9500_temporal.pkl',
    'dem_path':       'dem/preserve_dem_10m_utm.tif',
    'checkpoint_dir': 'models/checkpoints/unet_3d_dualpol',
    'lr':             5e-5,
    'weight_decay':   1e-4,
    'max_epochs':     100,
    'batch_size':     8,
    'patience':       20,
    'base_filters':   32,
    'add_bias':       False,
    'loss_type':      'asym_mae',
    'under_weight':   2.0,
    'max_precip':     100.0,
    'log_target':     True,
    'n_encoder_blocks': 3,
    'fields':         'dualpol',
    'z_collapse':     'mean',
    'dropout_rate':   0.15,
    'output_dir':     'evaluation_figures/unet_3d_dualpol',
    'zarr_cache_mb':  1024,
}


# ── LOSS (mirrors models/unet/train.py — asym + log_target capable) ────────────

class GaugePixelLoss(nn.Module):
    """Loss computed only at the gauge pixel location."""

    def __init__(self, max_precip=100.0, loss_type='mae', huber_delta=2.0,
                 log_target=False, under_weight=2.0):
        super().__init__()
        self.max_precip = max_precip
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.log_target = log_target
        self.under_weight = under_weight

    def forward(self, pred_map, target, gauge_pixel):
        batch_size = pred_map.shape[0]

        if pred_map.dim() == 1:
            pred_at_gauge = pred_map
        else:
            batch_idx = torch.arange(batch_size, device=pred_map.device)
            if isinstance(gauge_pixel, (tuple, list)):
                y, x = gauge_pixel
                if isinstance(y, torch.Tensor):
                    pred_at_gauge = pred_map[batch_idx, y.to(pred_map.device), x.to(pred_map.device)]
                else:
                    pred_at_gauge = pred_map[:, y, x]
            elif isinstance(gauge_pixel, torch.Tensor) and gauge_pixel.dim() == 2:
                y = gauge_pixel[:, 0].long().to(pred_map.device)
                x = gauge_pixel[:, 1].long().to(pred_map.device)
                pred_at_gauge = pred_map[batch_idx, y, x]
            else:
                pred_at_gauge = pred_map[:, 2, 2]

        valid = (target >= 0) & (target < (np.log1p(self.max_precip) if self.log_target else self.max_precip))
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device, requires_grad=True)

        pred_v = pred_at_gauge[valid]
        tgt_v = target[valid]

        if self.loss_type == 'mae':
            return torch.abs(pred_v - tgt_v).mean()
        elif self.loss_type == 'mse':
            return ((pred_v - tgt_v) ** 2).mean()
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(pred_v, tgt_v, beta=self.huber_delta)
        elif self.loss_type == 'weighted_mae':
            weights = (1.0 + tgt_v) if self.log_target else (1.0 + tgt_v / 5.0)
            return (weights * torch.abs(pred_v - tgt_v)).mean()
        elif self.loss_type == 'weighted_mae_sq':
            weights = (1.0 + tgt_v ** 2) if self.log_target else (1.0 + (tgt_v / 5.0) ** 2)
            return (weights * torch.abs(pred_v - tgt_v)).mean()
        elif self.loss_type == 'asym_mae':
            err = tgt_v - pred_v
            w = torch.where(err > 0,
                            torch.as_tensor(self.under_weight, device=err.device, dtype=err.dtype),
                            torch.as_tensor(1.0, device=err.device, dtype=err.dtype))
            return (w * err.abs()).mean()
        elif self.loss_type == 'asym_huber':
            err = tgt_v - pred_v
            w = torch.where(err > 0,
                            torch.as_tensor(self.under_weight, device=err.device, dtype=err.dtype),
                            torch.as_tensor(1.0, device=err.device, dtype=err.dtype))
            huber = F.smooth_l1_loss(pred_v, tgt_v, beta=self.huber_delta, reduction='none')
            return (w * huber).mean()
        return torch.abs(pred_v - tgt_v).mean()


# ── CONFIG SAVE ────────────────────────────────────────────────────────────────

def save_config(run_dir, cfg, n_params, train_samples, val_samples, model):
    encoder_channels = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) and 'enc' in name:
            encoder_channels.append(module.out_channels)

    fields = resolve_fields(cfg.get('fields'))
    config_data = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'unet_3d',
        'fields': fields,
        'use_dem': cfg.get('use_dem', True),
        'use_mask': cfg.get('use_mask', True),
        'use_temporal_pos': cfg.get('use_temporal_pos', True),
        'use_feature_masks': cfg.get('use_feature_masks', False),
        'log_target': cfg.get('log_target', True),
        'loss_type': cfg.get('loss_type', 'asym_mae'),
        'under_weight': cfg.get('under_weight', 2.0),
        'lr': cfg.get('lr'),
        'weight_decay': cfg.get('weight_decay'),
        'batch_size': cfg.get('batch_size'),
        'patience': cfg.get('patience'),
        'max_epochs': cfg.get('max_epochs'),
        'base_filters': cfg.get('base_filters', 32),
        'dropout_rate': cfg.get('dropout_rate', 0.15),
        'z_collapse': cfg.get('z_collapse', 'mean'),
        'pickle_path': cfg.get('pickle_path'),
        'dem_path': cfg.get('dem_path'),
        'filter_mode': cfg.get('filter_mode', 'blunt'),
        'no_sampler': cfg.get('no_sampler', False),
        'sampler_type': cfg.get('sampler_type', 'moderate'),
        'exclude_stations': cfg.get('exclude_stations', []),
        'n_parameters': n_params,
        'n_input_channels': compute_n_input_channels(
            fields, cfg.get('use_mask', True), cfg.get('use_temporal_pos', True),
            cfg.get('use_dem', True), cfg.get('use_feature_masks', False)),
        'n_encoder_blocks': cfg.get('n_encoder_blocks', 3),
        'encoder_channels': encoder_channels,
        'output_size': cfg.get('output_size'),
        'train_samples_after_filter': train_samples,
        'val_samples_after_filter': val_samples,
    }
    config_path = run_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"  ✓ Saved config to: {config_path}")
    return config_data


# ── TRAIN ───────────────────────────────────────────────────────────────────────

def train(cfg: dict = None, run_name: str = None):
    cfg = cfg or dict(DEFAULT_CONFIG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_dir = create_run_dir(cfg['checkpoint_dir'], run_name)
    print(f"\n{'='*60}")
    print("  3D U-NET — Precipitation Prediction")
    print(f"{'='*60}")
    print(f"  Device:  {device}")
    print(f"  Dataset: {cfg['pickle_path']}")
    print(f"  Epochs:  {cfg['max_epochs']}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}\n")

    ds_kwargs = dict(
        dem_path=cfg['dem_path'],
        fields=cfg.get('fields'),
        use_dem=cfg.get('use_dem', True),
        use_mask=cfg.get('use_mask', True),
        use_temporal_pos=cfg.get('use_temporal_pos', True),
        log_target=cfg.get('log_target', True),
        use_feature_masks=cfg.get('use_feature_masks', False),
        zarr_cache_mb=cfg.get('zarr_cache_mb', 1024),
    )
    train_ds = RadarGauge3DDataset(cfg['pickle_path'], split='train', augment=True, aug_prob=0.5, **ds_kwargs)
    val_ds = RadarGauge3DDataset(cfg['pickle_path'], split='val', augment=False, **ds_kwargs)

    # Filters (lazy 3D samples use precomputed max_reflectivity_dbz)
    exclude = cfg.get('exclude_stations', [])
    filter_mode = cfg.get('filter_mode', 'blunt')
    print(f"  Filter mode: {filter_mode}")

    train_ds.samples = filter_stations(train_ds.samples, exclude)
    train_ds.samples = filter_nan_radar(train_ds.samples)
    if filter_mode != 'radar':
        train_ds.samples = filter_biased_extremes(train_ds.samples)
        train_ds.samples = filter_bad_samples(train_ds.samples)
    train_ds.samples = filter_suspect_station_days(train_ds.samples)
    train_ds.samples = filter_gauge_dumps(train_ds.samples)

    val_ds.samples = filter_stations(val_ds.samples, exclude)
    val_ds.samples = filter_nan_radar(val_ds.samples)
    if filter_mode != 'radar':
        val_ds.samples = filter_biased_extremes(val_ds.samples)
        val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_suspect_station_days(val_ds.samples)
    val_ds.samples = filter_gauge_dumps(val_ds.samples)

    print(f"\nAfter filtering — Train: {len(train_ds.samples)}, Val: {len(val_ds.samples)}")

    for split_name, ds in [('Train', train_ds), ('Val', val_ds)]:
        targets = [s['hourly_precip_mm'] for s in ds.samples]
        bins = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, float('inf'))]
        counts = [sum(1 for t in targets if lo <= t < hi) for lo, hi in bins]
        labels = ['0-1', '1-5', '5-10', '10-20', '20-40', '40-60', '60+']
        print(f"  {split_name} precip distribution (mm/hr):")
        print(f"    {'  '.join(f'{l}:{c}' for l, c in zip(labels, counts))}")

    if cfg.get('no_sampler', False):
        print("  Using uniform sampling (no weighted sampler)")
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    else:
        sampler = create_sampler(train_ds.samples, sampler_type=cfg.get('sampler_type', 'moderate'))
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Model
    patch_pixels = train_ds.metadata.get('patch_pixels')
    if patch_pixels is None:
        s0 = train_ds.samples[0]
        patch_pixels = s0.get('patch_pixels', s0['x_end'] - s0['x_start'])
    cfg['output_size'] = patch_pixels

    n_encoder_blocks = cfg.get('n_encoder_blocks')
    if n_encoder_blocks is None:
        if patch_pixels <= 5:
            n_encoder_blocks = 1
        elif patch_pixels <= 9:
            n_encoder_blocks = 2
        else:
            n_encoder_blocks = 3
    cfg['n_encoder_blocks'] = n_encoder_blocks

    model = PrecipUNet3D(
        in_channels=train_ds.n_channels,
        base_filters=cfg.get('base_filters', 32),
        dropout_rate=cfg.get('dropout_rate', 0.15),
        n_encoder_blocks=n_encoder_blocks,
        z_collapse=cfg.get('z_collapse', 'mean'),
    ).to(device)
    model.apply(init_weights)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    save_config(run_dir, cfg, n_params, len(train_ds.samples), len(val_ds.samples), model)

    criterion = GaugePixelLoss(max_precip=cfg['max_precip'], loss_type=cfg['loss_type'],
                               log_target=cfg.get('log_target', True),
                               under_weight=cfg.get('under_weight', 2.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt_path = None
    log_target = cfg.get('log_target', True)

    for epoch in range(1, cfg['max_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_r2 = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_R²={val_r2:.3f} | lr={lr_now:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_ckpt_path = str(run_dir / f'best-epoch={epoch:02d}-val_loss={val_loss:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'config': cfg,
                'run_dir': str(run_dir),
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"\n  Early stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'run_dir': str(run_dir),
    }, str(run_dir / 'last.pt'))

    config_path = run_dir / 'config.json'
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config_data['best_epoch'] = int(best_ckpt_path.split('epoch=')[1].split('-')[0]) if best_ckpt_path else None
    config_data['best_val_loss'] = float(best_val_loss)
    config_data['final_epoch'] = epoch
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"\n✓ Training complete.")
    print(f"  Run directory: {run_dir}")
    print(f"  Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path, str(run_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 3D U-Net precipitation model")
    parser.add_argument('--pickle', default=DEFAULT_CONFIG['pickle_path'])
    parser.add_argument('--dem', default=DEFAULT_CONFIG['dem_path'])
    parser.add_argument('--ckpt-dir', default=DEFAULT_CONFIG['checkpoint_dir'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['max_epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'])
    parser.add_argument('--base-filters', type=int, default=DEFAULT_CONFIG['base_filters'])
    parser.add_argument('--loss', choices=['mae', 'mse', 'huber', 'weighted_mae', 'weighted_mae_sq', 'asym_mae', 'asym_huber'],
                        default=DEFAULT_CONFIG['loss_type'])
    parser.add_argument('--under-weight', type=float, default=DEFAULT_CONFIG['under_weight'],
                        help='Underprediction penalty for asym_mae/asym_huber (1.0=symmetric, >1 favors higher predictions)')
    parser.add_argument('--z-collapse', choices=['mean', 'max'], default=DEFAULT_CONFIG['z_collapse'],
                        help='How the vertical dimension is collapsed at the output head')
    parser.add_argument('--no-sampler', action='store_true', help='Disable weighted sampler (use uniform sampling)')
    parser.add_argument('--sampler-type', choices=['light', 'moderate', 'heavy'], default='moderate')
    parser.add_argument('--exclude-stations', nargs='+', default=[], help='Station names to exclude')
    parser.add_argument('--filter-mode', choices=['blunt', 'radar'], default='blunt')
    parser.add_argument('--zarr-cache-mb', type=int, default=DEFAULT_CONFIG['zarr_cache_mb'],
                        help='Bounded zarr LRU chunk-cache size in MB (default: 1024)')
    parser.add_argument('--run-name', default=None, help='Short description suffix for the run folder')
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg['pickle_path'] = args.pickle
    cfg['dem_path'] = args.dem
    cfg['checkpoint_dir'] = args.ckpt_dir
    cfg['lr'] = args.lr
    cfg['max_epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['patience'] = args.patience
    cfg['base_filters'] = args.base_filters
    cfg['loss_type'] = args.loss
    cfg['under_weight'] = args.under_weight
    cfg['z_collapse'] = args.z_collapse
    cfg['no_sampler'] = args.no_sampler
    cfg['sampler_type'] = args.sampler_type
    cfg['exclude_stations'] = args.exclude_stations
    cfg['filter_mode'] = args.filter_mode
    cfg['zarr_cache_mb'] = args.zarr_cache_mb

    train(cfg, run_name=args.run_name)
