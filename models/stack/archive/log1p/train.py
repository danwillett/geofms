"""
train.py — Train the Stack CNN precipitation model.

Run from the project root:
    python -m models.stack.train

Or with custom args:
    python -m models.stack.train --pickle dataset/outputs/radar_gauge_dataset.pkl --epochs 100
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models.stack.model import PrecipitationStackModel, init_weights
from models.stack.dataset import RadarGaugeDataset

# ── DEFAULT CONFIG ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'pickle_path':    'dataset/outputs/radar_gauge_dataset.pkl',
    'dem_path':       'dem/preserve_dem_10m_utm.tif',
    'checkpoint_dir': 'checkpoints/stack_dualpol',
    'lr':             5e-5,
    'weight_decay':   1e-4,
    'max_epochs':     100,
    'batch_size':     32,
    'patience':       20,
    'latent_dim':     512,
    'add_bias':       False,
    'loss_type':      'mae',
    'max_precip':     100.0,
    'scalar_output':  False,
}


# ── LOSS ──────────────────────────────────────────────────────────────────────

class GaugePixelLoss(nn.Module):
    """Loss computed only at the gauge pixel location."""

    def __init__(self, max_precip=100.0, loss_type='mae'):
        super().__init__()
        self.max_precip = max_precip
        self.loss_type = loss_type

    def forward(self, pred_map, target, gauge_pixel):
        batch_size = pred_map.shape[0]

        # Scalar output: pred_map is already (B,)
        if pred_map.dim() == 1:
            pred_at_gauge = pred_map
        else:
            # Spatial map output: extract prediction at gauge pixel
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

        valid = (target >= 0) & (target < np.log1p(self.max_precip))
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device, requires_grad=True)

        pred_v = pred_at_gauge[valid]
        tgt_v = target[valid]

        if self.loss_type == 'mae':
            return torch.abs(pred_v - tgt_v).mean()
        elif self.loss_type == 'mse':
            return ((pred_v - tgt_v) ** 2).mean()
        elif self.loss_type == 'weighted_mae':
            weights = 1.0 + tgt_v
            return (weights * torch.abs(pred_v - tgt_v)).mean()
        elif self.loss_type == 'weighted_mae_sq':
            weights = 1.0 + tgt_v ** 2
            return (weights * torch.abs(pred_v - tgt_v)).mean()
        return torch.abs(pred_v - tgt_v).mean()


# ── SAMPLER ───────────────────────────────────────────────────────────────────

SAMPLER_PRESETS = {
    'light': {
        'dry': 0.9, 'trace': 1.0, 'moderate': 1.3, 'heavy': 1.8, 'extreme': 2.5,
    },
    'moderate': {
        'dry': 0.8, 'trace': 1.0, 'moderate': 1.5, 'heavy': 2.5, 'extreme': 4.0,
    },
    'heavy': {
        'dry': 0.5, 'trace': 0.8, 'moderate': 2.0, 'heavy': 4.0, 'extreme': 6.0,
    },
}


def create_sampler(samples, sampler_type='moderate'):
    preset = SAMPLER_PRESETS.get(sampler_type, SAMPLER_PRESETS['moderate'])
    targets = np.array([s['hourly_precip_mm'] for s in samples])
    weights = np.ones(len(targets))
    weights[targets < 0.1] = preset['dry']
    weights[(targets >= 0.1) & (targets < 2)] = preset['trace']
    weights[(targets >= 2) & (targets < 5)] = preset['moderate']
    weights[(targets >= 5) & (targets < 15)] = preset['heavy']
    weights[(targets >= 15)] = preset['extreme']
    weights = weights / weights.sum() * len(weights)
    print(f"  Using '{sampler_type}' sampler: {preset}")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── FILTERS (same as GFM) ────────────────────────────────────────────────────

def filter_stations(samples, exclude_stations):
    """Remove samples from specific stations."""
    if not exclude_stations:
        return samples
    exclude_set = set(exclude_stations)
    filtered = [s for s in samples if s.get('station_name', '') not in exclude_set]
    removed = len(samples) - len(filtered)
    print(f"✓ filter_stations: removed {removed} (excluded: {', '.join(exclude_stations)})")
    return filtered


def filter_nan_radar(samples):
    """Remove samples where reflectivity is entirely NaN (no valid radar data)."""
    filtered = []
    for s in samples:
        max_dbz = np.nanmax(s['radar_patch'][:, 0, :, :])
        if np.isnan(max_dbz):
            continue
        filtered.append(s)
    removed = len(samples) - len(filtered)
    print(f"✓ filter_nan_radar: removed {removed} (all-NaN reflectivity)")
    return filtered


def filter_bad_samples(samples):
    filtered = []
    for s in samples:
        target = s['hourly_precip_mm']
        max_dbz = np.nanmax(s['radar_patch'][:, 0, :, :])
        if np.isnan(max_dbz):
            continue
        if target > 40.0:
            continue
        if target > 5.0 and max_dbz < 20.0:
            continue
        if max_dbz > 50.0 and target < 2.0:
            continue
        if max_dbz > 60.0:
            continue
        filtered.append(s)
    removed = len(samples) - len(filtered)
    print(f"✓ filter_bad_samples: removed {removed}")
    return filtered


def filter_biased_extremes(samples):
    OVER = {'Dangermond_Bunker Hill', 'Dangermond_Cistern', 'Dangermond_Cojo HQ',
            'Dangermond_Jalachichi', 'Dangermond_Repeator'}
    UNDER = {'Dangermond_Cojo Gate', 'Dangermond_Sutter'}
    filtered = []
    for s in samples:
        name = s['station_name']
        target = s['hourly_precip_mm']
        max_dbz = np.nanmax(s['radar_patch'][:, 0, :, :])
        if np.isnan(max_dbz):
            filtered.append(s)
            continue
        if name in OVER:
            if target > 25.0:
                continue
            if max_dbz > 30.0 and target < 0.3:
                continue
        if name in UNDER:
            if max_dbz > 30.0 and target < 0.3:
                continue
        filtered.append(s)
    removed = len(samples) - len(filtered)
    print(f"✓ filter_biased_extremes: removed {removed}")
    return filtered


def filter_suspect_station_days(samples):
    """Remove samples from station-days where the gauge read zero but others had rain."""
    daily_totals = {}
    for sample in samples:
        station = sample.get('station_name', 'Unknown')
        hour_str = str(sample.get('hour_start', ''))
        date = hour_str[:10]
        precip = sample['hourly_precip_mm']
        key = (station, date)
        if key not in daily_totals:
            daily_totals[key] = 0
        daily_totals[key] += precip

    date_totals = {}
    for (station, date), total in daily_totals.items():
        if date not in date_totals:
            date_totals[date] = []
        date_totals[date].append((station, total))

    suspect_station_days = set()
    for date, station_data in date_totals.items():
        for station, total in station_data:
            if total == 0:
                others = [t for s, t in station_data if s != station]
                others_with_rain = sum(1 for t in others if t > 2.0)
                if len(others) >= 5 and others_with_rain >= 9 and np.mean(others) > 15:
                    suspect_station_days.add((station, date))

    print(f"  Identified {len(suspect_station_days)} suspect station-days")

    filtered = []
    removed_count = 0
    for sample in samples:
        station = sample.get('station_name', 'Unknown')
        hour_str = str(sample.get('hour_start', ''))
        date = hour_str[:10]
        if (station, date) in suspect_station_days:
            removed_count += 1
            continue
        filtered.append(sample)

    print(f"✓ filter_suspect_station_days: removed {removed_count}")
    return filtered


def filter_gauge_dumps(samples, dump_ratio_threshold=0.95, precip_threshold=5.0):
    """
    Remove samples where most of the hourly rainfall came from a single 10-min bin
    AND other stations don't corroborate heavy rain at the same time.
    
    Requires 'dump_ratio' field in samples (added during pickle creation).
    """
    from collections import defaultdict

    has_field = any('dump_ratio' in s for s in samples)
    if not has_field:
        print(f"✓ filter_gauge_dumps: skipped (no 'dump_ratio' field — regenerate pickle to enable)")
        return samples

    by_hour = defaultdict(list)
    for s in samples:
        by_hour[s['hour_start']].append(s)

    filtered = []
    removed_count = 0

    for s in samples:
        if 'dump_ratio' not in s:
            filtered.append(s)
            continue

        target = s['hourly_precip_mm']
        dump_ratio = s['dump_ratio']

        if target < precip_threshold or dump_ratio < dump_ratio_threshold:
            filtered.append(s)
            continue

        same_hour = by_hour[s['hour_start']]
        other_precip = [x['hourly_precip_mm'] for x in same_hour
                        if x['station_id'] != s['station_id']]

        corroboration_threshold = target * 0.3
        others_confirm = any(p >= corroboration_threshold for p in other_precip)

        if others_confirm:
            filtered.append(s)
        else:
            removed_count += 1

    print(f"✓ filter_gauge_dumps: removed {removed_count} "
          f"(precip>={precip_threshold}mm, dump_ratio>={dump_ratio_threshold}, isolated)")
    return filtered


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        radar = batch['radar'].to(device)
        target = batch['target'].to(device)
        gauge_pixel = batch['gauge_pixel']
        bias_flag = batch.get('bias_flag')

        optimizer.zero_grad()

        if model.add_bias and bias_flag is not None:
            pred = model(radar, bias_flag.to(device))
        else:
            pred = model(radar)

        loss = criterion(pred, target, gauge_pixel)
        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            radar = batch['radar'].to(device)
            target = batch['target'].to(device)
            gauge_pixel = batch['gauge_pixel']
            bias_flag = batch.get('bias_flag')

            if model.add_bias and bias_flag is not None:
                pred_map = model(radar, bias_flag.to(device))
            else:
                pred_map = model(radar)

            loss = criterion(pred_map, target, gauge_pixel)
            total_loss += loss.item()
            n_batches += 1

            # Collect predictions at gauge pixel for metrics
            if pred_map.dim() == 1:
                pred_at_gauge = pred_map
            else:
                batch_idx = torch.arange(pred_map.shape[0], device=pred_map.device)
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

            all_preds.append(pred_at_gauge.cpu())
            all_targets.append(target.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # R² in log space
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return total_loss / max(n_batches, 1), r2


# ── MAIN ──────────────────────────────────────────────────────────────────────

def create_run_dir(base_dir, run_name=None):
    """Create a timestamped run directory for experiment tracking."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    folder_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = Path(base_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir, cfg, n_params, train_samples, val_samples, model):
    """Save experiment configuration to config.json in the run directory."""
    from models.stack.model import RadarEncoder, PrecipitationDecoder

    encoder = model.radar_encoder
    encoder_channels = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv' in name:
            encoder_channels.append(module.out_channels)

    config_data = {
        'timestamp': datetime.now().isoformat(),
        'loss_type': cfg.get('loss_type', 'mae'),
        'lr': cfg.get('lr'),
        'weight_decay': cfg.get('weight_decay'),
        'batch_size': cfg.get('batch_size'),
        'patience': cfg.get('patience'),
        'max_epochs': cfg.get('max_epochs'),
        'latent_dim': cfg.get('latent_dim'),
        'add_bias': cfg.get('add_bias'),
        'pickle_path': cfg.get('pickle_path'),
        'dem_path': cfg.get('dem_path'),
        'n_parameters': n_params,
        'n_input_channels': RadarGaugeDataset.n_input_channels(),
        'fields_used': RadarGaugeDataset.FIELDS,
        'encoder_channels': encoder_channels,
        'decoder_hidden_dim': model.decoder.fc1.out_features,
        'dropout_rate': cfg.get('dropout_rate', 0.25),
        'input_size': f'{model.decoder.output_size}x{model.decoder.output_size}',
        'output_size': model.decoder.output_size,
        'train_samples_after_filter': train_samples,
        'val_samples_after_filter': val_samples,
    }

    config_path = run_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"  ✓ Saved config to: {config_path}")
    return config_data


def train(cfg: dict = None, run_name: str = None):
    cfg = cfg or dict(DEFAULT_CONFIG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_dir = create_run_dir(cfg['checkpoint_dir'], run_name)
    print(f"\n{'='*60}")
    print("  STACK CNN — Precipitation Prediction")
    print(f"{'='*60}")
    print(f"  Device:  {device}")
    print(f"  Dataset: {cfg['pickle_path']}")
    print(f"  Epochs:  {cfg['max_epochs']}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # Data
    train_ds = RadarGaugeDataset(cfg['pickle_path'], dem_path=cfg['dem_path'], split='train', augment=True, aug_prob=0.5)
    val_ds = RadarGaugeDataset(cfg['pickle_path'], dem_path=cfg['dem_path'], split='val', augment=False)

    # Apply filters
    exclude = cfg.get('exclude_stations', [])
    train_ds.samples = filter_stations(train_ds.samples, exclude)
    train_ds.samples = filter_nan_radar(train_ds.samples)
    train_ds.samples = filter_biased_extremes(train_ds.samples)
    train_ds.samples = filter_bad_samples(train_ds.samples)
    train_ds.samples = filter_suspect_station_days(train_ds.samples)
    train_ds.samples = filter_gauge_dumps(train_ds.samples)
    val_ds.samples = filter_stations(val_ds.samples, exclude)
    val_ds.samples = filter_nan_radar(val_ds.samples)
    val_ds.samples = filter_biased_extremes(val_ds.samples)
    val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_suspect_station_days(val_ds.samples)
    val_ds.samples = filter_gauge_dumps(val_ds.samples)

    print(f"\nAfter filtering — Train: {len(train_ds.samples)}, Val: {len(val_ds.samples)}")

    if cfg.get('no_sampler', False):
        print("  Using uniform sampling (no weighted sampler)")
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    else:
        sampler = create_sampler(train_ds.samples, sampler_type=cfg.get('sampler_type', 'moderate'))
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Model
    patch_pixels = train_ds.samples[0]['radar_patch'].shape[-1]
    cfg['output_size'] = patch_pixels
    model = PrecipitationStackModel(
        latent_dim=cfg['latent_dim'],
        add_bias=cfg['add_bias'],
        output_size=patch_pixels,
        scalar_output=cfg.get('scalar_output', False),
    ).to(device)
    model.apply(init_weights)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    save_config(run_dir, cfg, n_params, len(train_ds.samples), len(val_ds.samples), model)

    criterion = GaugePixelLoss(max_precip=cfg['max_precip'], loss_type=cfg['loss_type'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt_path = None

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

    # Save last
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'run_dir': str(run_dir),
    }, str(run_dir / 'last.pt'))

    # Update config.json with final training info
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
    parser = argparse.ArgumentParser(description="Train Stack CNN precipitation model")
    parser.add_argument('--pickle', default=DEFAULT_CONFIG['pickle_path'])
    parser.add_argument('--dem', default=DEFAULT_CONFIG['dem_path'])
    parser.add_argument('--ckpt-dir', default=DEFAULT_CONFIG['checkpoint_dir'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['max_epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'])
    parser.add_argument('--loss', choices=['mae', 'mse', 'weighted_mae', 'weighted_mae_sq'], default=DEFAULT_CONFIG['loss_type'])
    parser.add_argument('--add-bias', action='store_true')
    parser.add_argument('--no-sampler', action='store_true', help='Disable weighted sampler (use uniform sampling)')
    parser.add_argument('--sampler-type', choices=['light', 'moderate', 'heavy'], default='moderate',
                        help='Sampler intensity preset (default: moderate)')
    parser.add_argument('--scalar-output', action='store_true', help='Predict single scalar instead of spatial map')
    parser.add_argument('--exclude-stations', nargs='+', default=[], help='Station names to exclude (e.g. Dangermond_Bunker_Hill)')
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
    cfg['loss_type'] = args.loss
    cfg['add_bias'] = args.add_bias
    cfg['no_sampler'] = args.no_sampler
    cfg['sampler_type'] = args.sampler_type
    cfg['scalar_output'] = args.scalar_output
    cfg['exclude_stations'] = args.exclude_stations

    train(cfg, run_name=args.run_name)
