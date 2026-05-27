"""
train.py — Train the 10-minute single-scan precipitation model.

Run from the project root:
    python -m models.stack_10min.train
    python -m models.stack_10min.train --pickle dataset/outputs/10min/radar_gauge_10min.pkl --epochs 80
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

from models.stack_10min.model import PrecipModel10min, init_weights
from models.stack_10min.dataset import RadarGaugeDataset10min

# ── DEFAULT CONFIG ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'pickle_path':    'dataset/outputs/10min/radar_gauge_10min.pkl',
    'dem_path':       'dem/preserve_dem_10m_utm.tif',
    'checkpoint_dir': 'models/checkpoints/stack_10min',
    'lr':             1e-4,
    'weight_decay':   1e-4,
    'max_epochs':     100,
    'batch_size':     64,
    'patience':       15,
    'latent_dim':     256,
    'loss_type':      'huber',
    'max_precip':     50.0,
}


# ── LOSS ──────────────────────────────────────────────────────────────────────

def compute_loss(pred, target, loss_type='huber', max_precip=50.0):
    """Compute loss on raw mm values."""
    valid = (target >= 0) & (target <= max_precip)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_v = pred[valid]
    tgt_v = target[valid]

    if loss_type == 'huber':
        return nn.functional.huber_loss(pred_v, tgt_v, delta=1.0)
    elif loss_type == 'mae':
        return torch.abs(pred_v - tgt_v).mean()
    elif loss_type == 'mse':
        return ((pred_v - tgt_v) ** 2).mean()
    elif loss_type == 'weighted_mae':
        weights = 1.0 + tgt_v
        return (weights * torch.abs(pred_v - tgt_v)).mean()
    return nn.functional.huber_loss(pred_v, tgt_v, delta=1.0)


# ── SAMPLER ───────────────────────────────────────────────────────────────────

SAMPLER_PRESETS = {
    'light': {
        'dry': 0.9, 'trace': 1.0, 'light': 1.3, 'moderate': 1.8, 'heavy': 2.5,
    },
    'moderate': {
        'dry': 0.8, 'trace': 1.0, 'light': 1.5, 'moderate': 2.5, 'heavy': 4.0,
    },
    'heavy': {
        'dry': 0.5, 'trace': 0.8, 'light': 2.0, 'moderate': 4.0, 'heavy': 6.0,
    },
}


def create_sampler(samples, sampler_type='moderate'):
    """10-min thresholds: dry<0.1, trace<0.5, light<1.5, moderate<3, heavy>=3"""
    preset = SAMPLER_PRESETS.get(sampler_type, SAMPLER_PRESETS['moderate'])
    targets = np.array([s['precip_mm'] for s in samples])
    weights = np.ones(len(targets))
    weights[targets < 0.1] = preset['dry']
    weights[(targets >= 0.1) & (targets < 0.5)] = preset['trace']
    weights[(targets >= 0.5) & (targets < 1.5)] = preset['light']
    weights[(targets >= 1.5) & (targets < 3.0)] = preset['moderate']
    weights[targets >= 3.0] = preset['heavy']
    weights = weights / weights.sum() * len(weights)
    print(f"  Using '{sampler_type}' sampler: {preset}")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── FILTERS ───────────────────────────────────────────────────────────────────

def filter_stations(samples, exclude_stations):
    if not exclude_stations:
        return samples
    exclude_set = set(exclude_stations)
    filtered = [s for s in samples if s.get('station_name', '') not in exclude_set]
    removed = len(samples) - len(filtered)
    print(f"✓ filter_stations: removed {removed} (excluded: {', '.join(exclude_stations)})")
    return filtered


def filter_nan_radar(samples):
    """Remove samples where reflectivity is entirely NaN."""
    filtered = []
    for s in samples:
        ref_data = s['radar_patch'][0, :, :]  # (H, W) — first field is reflectivity
        max_dbz = np.nanmax(ref_data)
        if np.isnan(max_dbz):
            continue
        filtered.append(s)
    removed = len(samples) - len(filtered)
    print(f"✓ filter_nan_radar: removed {removed} (all-NaN reflectivity)")
    return filtered


def filter_bad_samples(samples, max_precip=50.0):
    """Remove physically implausible samples."""
    filtered = []
    for s in samples:
        target = s['precip_mm']
        max_dbz = np.nanmax(s['radar_patch'][0, :, :])
        if np.isnan(max_dbz):
            continue
        if target > max_precip:
            continue
        # High rain but very low reflectivity
        if target > 2.0 and max_dbz < 15.0:
            continue
        # Very high reflectivity but no rain
        if max_dbz > 55.0 and target < 0.1:
            continue
        filtered.append(s)
    removed = len(samples) - len(filtered)
    print(f"✓ filter_bad_samples: removed {removed}")
    return filtered


def filter_suspect_gauges(samples, precip_threshold=5.0, dbz_threshold=50.0):
    """
    Remove samples where high precip is not supported by radar — likely gauge artifacts.
    
    A sample is SUSPECT if:
    - precip_mm >= precip_threshold AND
    - max radar reflectivity < dbz_threshold (radar doesn't support extreme rain)
    
    Keeps "likely real" (radar confirms) and "inconclusive" samples.
    """
    from collections import defaultdict

    # Build cross-station lookup for co-occurrence check
    by_timestamp = defaultdict(list)
    for s in samples:
        by_timestamp[s['bin_start']].append(s)

    filtered = []
    removed_count = 0
    for s in samples:
        target = s['precip_mm']
        if target < precip_threshold:
            filtered.append(s)
            continue

        ref_data = s['radar_patch'][0, :, :]
        ref_valid = ref_data[~np.isnan(ref_data) & (ref_data != -9999.0)]
        max_dbz = np.nanmax(ref_valid) if len(ref_valid) > 0 else 0.0

        # Radar supports heavy rain — keep it
        if max_dbz >= dbz_threshold:
            filtered.append(s)
            continue

        # Radar doesn't support it — check if other stations are also elevated
        same_time = by_timestamp[s['bin_start']]
        other_precip = [x['precip_mm'] for x in same_time if x['station_id'] != s['station_id']]
        others_elevated = len(other_precip) > 0 and np.max(other_precip) > precip_threshold * 0.3

        if others_elevated:
            # Inconclusive — other stations partially confirm, keep it
            filtered.append(s)
        else:
            # SUSPECT: high precip, low radar, no corroboration
            removed_count += 1

    print(f"✓ filter_suspect_gauges: removed {removed_count} "
          f"(precip>={precip_threshold}mm but max_dBZ<{dbz_threshold} and isolated)")
    return filtered


def filter_biased_extremes(samples):
    OVER = {'Dangermond_Bunker Hill', 'Dangermond_Cistern', 'Dangermond_Cojo HQ',
            'Dangermond_Jalachichi', 'Dangermond_Repeator'}
    UNDER = {'Dangermond_Cojo Gate', 'Dangermond_Sutter'}
    filtered = []
    for s in samples:
        name = s.get('station_name', '')
        target = s['precip_mm']
        max_dbz = np.nanmax(s['radar_patch'][0, :, :])
        if np.isnan(max_dbz):
            filtered.append(s)
            continue
        if name in OVER:
            if target > 8.0:
                continue
            if max_dbz > 30.0 and target < 0.1:
                continue
        if name in UNDER:
            if max_dbz > 30.0 and target < 0.1:
                continue
        filtered.append(s)
    removed = len(samples) - len(filtered)
    print(f"✓ filter_biased_extremes: removed {removed}")
    return filtered


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device, cfg):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        radar = batch['radar'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        pred = model(radar)

        loss = compute_loss(pred, target, loss_type=cfg['loss_type'], max_precip=cfg['max_precip'])
        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, dataloader, device, cfg):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            radar = batch['radar'].to(device)
            target = batch['target'].to(device)

            pred = model(radar)
            loss = compute_loss(pred, target, loss_type=cfg['loss_type'], max_precip=cfg['max_precip'])
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # R² in mm-space
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return total_loss / max(n_batches, 1), r2


# ── MAIN ──────────────────────────────────────────────────────────────────────

def create_run_dir(base_dir, run_name=None):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    folder_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = Path(base_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir, cfg, n_params, train_samples, val_samples, model):
    config_data = {
        'timestamp': datetime.now().isoformat(),
        'model_type': '10min_single_scan_cnn',
        'loss_type': cfg.get('loss_type', 'huber'),
        'lr': cfg.get('lr'),
        'weight_decay': cfg.get('weight_decay'),
        'batch_size': cfg.get('batch_size'),
        'patience': cfg.get('patience'),
        'max_epochs': cfg.get('max_epochs'),
        'latent_dim': cfg.get('latent_dim'),
        'pickle_path': cfg.get('pickle_path'),
        'dem_path': cfg.get('dem_path'),
        'n_parameters': n_params,
        'n_input_channels': RadarGaugeDataset10min.n_input_channels(),
        'fields_used': RadarGaugeDataset10min.FIELDS,
        'temporal_resolution': '10min',
        'scans_per_sample': 1,
        'dropout_rate': cfg.get('dropout_rate', 0.2),
        'train_samples_after_filter': train_samples,
        'val_samples_after_filter': val_samples,
        'sampler_type': cfg.get('sampler_type', 'none'),
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
    print("  10-MIN SINGLE-SCAN CNN — Precipitation Prediction")
    print(f"{'='*60}")
    print(f"  Device:  {device}")
    print(f"  Dataset: {cfg['pickle_path']}")
    print(f"  Epochs:  {cfg['max_epochs']}")
    print(f"  Loss:    {cfg['loss_type']}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # Data
    train_ds = RadarGaugeDataset10min(cfg['pickle_path'], dem_path=cfg['dem_path'], split='train', augment=True)
    val_ds = RadarGaugeDataset10min(cfg['pickle_path'], dem_path=cfg['dem_path'], split='val', augment=False)

    # Apply filters
    exclude = cfg.get('exclude_stations', [])
    train_ds.samples = filter_stations(train_ds.samples, exclude)
    train_ds.samples = filter_nan_radar(train_ds.samples)
    train_ds.samples = filter_biased_extremes(train_ds.samples)
    train_ds.samples = filter_bad_samples(train_ds.samples, max_precip=cfg['max_precip'])
    train_ds.samples = filter_suspect_gauges(train_ds.samples)
    val_ds.samples = filter_stations(val_ds.samples, exclude)
    val_ds.samples = filter_nan_radar(val_ds.samples)
    val_ds.samples = filter_biased_extremes(val_ds.samples)
    val_ds.samples = filter_bad_samples(val_ds.samples, max_precip=cfg['max_precip'])
    val_ds.samples = filter_suspect_gauges(val_ds.samples)

    print(f"\nAfter filtering — Train: {len(train_ds.samples)}, Val: {len(val_ds.samples)}")

    if cfg.get('no_sampler', False):
        print("  Using uniform sampling (no weighted sampler)")
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    else:
        sampler = create_sampler(train_ds.samples, sampler_type=cfg.get('sampler_type', 'moderate'))
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = PrecipModel10min(
        latent_dim=cfg['latent_dim'],
        dropout_rate=cfg.get('dropout_rate', 0.2),
    ).to(device)
    model.apply(init_weights)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    save_config(run_dir, cfg, n_params, len(train_ds.samples), len(val_ds.samples), model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt_path = None

    for epoch in range(1, cfg['max_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, cfg)
        val_loss, val_r2 = validate(model, val_loader, device, cfg)
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

    # Update config.json with final info
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
    parser = argparse.ArgumentParser(description="Train 10-min single-scan precipitation model")
    parser.add_argument('--pickle', default=DEFAULT_CONFIG['pickle_path'])
    parser.add_argument('--dem', default=DEFAULT_CONFIG['dem_path'])
    parser.add_argument('--ckpt-dir', default=DEFAULT_CONFIG['checkpoint_dir'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['max_epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'])
    parser.add_argument('--loss', choices=['huber', 'mae', 'mse', 'weighted_mae'], default=DEFAULT_CONFIG['loss_type'])
    parser.add_argument('--no-sampler', action='store_true', help='Disable weighted sampler')
    parser.add_argument('--sampler-type', choices=['light', 'moderate', 'heavy'], default='moderate')
    parser.add_argument('--exclude-stations', nargs='+', default=[])
    parser.add_argument('--run-name', default=None)
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
    cfg['no_sampler'] = args.no_sampler
    cfg['sampler_type'] = args.sampler_type
    cfg['exclude_stations'] = args.exclude_stations

    train(cfg, run_name=args.run_name)
