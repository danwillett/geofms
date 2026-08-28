"""
ablation.py — Feature ablation study for the U-Net precipitation model.

Systematically zeros out feature groups to measure their contribution to
prediction performance.

Run from the project root:
    python -m models.unet.ablation
    python -m models.unet.ablation --run-dir models/checkpoints/unet_dualpol/2026-05-28_...
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from models.unet.model import PrecipUNet
from models.unet.dataset import (
    RadarGaugeDataset, resolve_fields, compute_n_input_channels, N_SCANS,
)
from models.unet.evaluate import find_checkpoint, load_model
from models.unet.train import (
    filter_nan_radar, filter_biased_extremes,
    filter_bad_samples, filter_suspect_station_days,
    filter_radar_unsupported, filter_gauge_dumps, filter_stations,
)

DEFAULT_PICKLE = 'dataset/outputs/radar_gauge_dataset_tr22_24_26_vl_23_25.pkl'
DEFAULT_DEM = 'dem/preserve_dem_10m_utm.tif'
DEFAULT_CKPT_DIR = 'models/checkpoints/stack_dualpol'


def _ensure_utf8_stdio():
    """Avoid UnicodeEncodeError on Windows cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


def _write_ablation_results(results_path, lines):
    """Write ablation section, replacing any previous ablation block."""
    marker = '  ABLATION RESULTS'
    if results_path.exists():
        content = results_path.read_text(encoding='utf-8')
        idx = content.find(marker)
        if idx != -1:
            block_start = content.rfind('=' * 60, 0, idx)
            content = content[:block_start].rstrip() + '\n' if block_start != -1 else content[:idx].rstrip() + '\n'
            results_path.write_text(content, encoding='utf-8')
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def build_feature_groups(fields, use_mask=True, use_temporal_pos=True, use_dem=True,
                         use_feature_masks=False):
    """Dynamically build feature group channel indices from a field list.

    Channel layout must match RadarGaugeDataset.__getitem__:
    fields → mask → feature_masks → temporal_pos → dem.
    """
    groups = {}
    n_fields = len(fields)

    # Individual field groups
    for i, field in enumerate(fields):
        groups[field] = list(range(i * N_SCANS, (i + 1) * N_SCANS))

    # Composite groups
    dualpol_fields = ['differential_reflectivity', 'cross_correlation_ratio',
                      'specific_differential_phase']
    vertical_fields = ['echo_top_height', 'max_z_height', 'vil',
                       'low_level_ref', 'column_depth_fraction']

    dualpol_channels = []
    for f in dualpol_fields:
        if f in fields:
            idx = fields.index(f)
            dualpol_channels.extend(range(idx * N_SCANS, (idx + 1) * N_SCANS))
    if dualpol_channels:
        groups['all_dualpol'] = dualpol_channels

    vert_channels = []
    for f in vertical_fields:
        if f in fields:
            idx = fields.index(f)
            vert_channels.extend(range(idx * N_SCANS, (idx + 1) * N_SCANS))
    if vert_channels:
        groups['all_vertical'] = vert_channels

    # Auxiliary channels (after field channels) — order matches the dataset.
    offset = n_fields * N_SCANS
    if use_mask:
        groups['mask'] = list(range(offset, offset + N_SCANS))
        offset += N_SCANS
    if use_feature_masks:
        from models.unet.dataset import FEATURE_MASK_SOURCES
        feat_mask_channels = list(range(offset, offset + len(FEATURE_MASK_SOURCES) * N_SCANS))
        groups['feature_masks'] = feat_mask_channels
        offset += len(FEATURE_MASK_SOURCES) * N_SCANS
    if use_temporal_pos:
        groups['temporal_pos'] = list(range(offset, offset + N_SCANS))
        offset += N_SCANS
    if use_dem:
        groups['dem'] = [offset]

    return groups


def run_ablation_inference(model, dataloader, device, zero_channels=None, log_target=True):
    """Run inference with optional channel zeroing."""
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            radar = batch['radar'].to(device)
            target = batch['target']
            gauge_pixel = batch['gauge_pixel']

            if zero_channels is not None:
                radar[:, zero_channels, :, :] = 0.0

            pred_map = model(radar)
            pred_map = pred_map.cpu()

            if pred_map.dim() == 1:
                pred_at_gauge = pred_map
            else:
                batch_idx = torch.arange(pred_map.shape[0])
                if isinstance(gauge_pixel, (tuple, list)):
                    y, x = gauge_pixel
                    if isinstance(y, torch.Tensor):
                        pred_at_gauge = pred_map[batch_idx, y, x]
                    else:
                        pred_at_gauge = pred_map[:, y, x]
                elif isinstance(gauge_pixel, torch.Tensor) and gauge_pixel.dim() == 2:
                    y = gauge_pixel[:, 0].long()
                    x = gauge_pixel[:, 1].long()
                    pred_at_gauge = pred_map[batch_idx, y, x]
                else:
                    pred_at_gauge = pred_map[:, 4, 4]

            if log_target:
                pred_at_gauge = torch.expm1(pred_at_gauge)
                target = torch.expm1(target)

            all_preds.extend(pred_at_gauge.numpy().tolist())
            all_targets.extend(target.numpy().tolist())

    preds_mm = np.array(all_preds)
    targets_mm = np.array(all_targets)

    valid = targets_mm >= 0
    preds_mm = preds_mm[valid]
    targets_mm = targets_mm[valid]

    return preds_mm, targets_mm


def compute_metrics(preds_mm, targets_mm):
    """Compute R², MAE, RMSE, and heavy rain metrics."""
    mae = np.mean(np.abs(targets_mm - preds_mm))
    rmse = np.sqrt(np.mean((targets_mm - preds_mm) ** 2))
    ss_res = np.sum((targets_mm - preds_mm) ** 2)
    ss_tot = np.sum((targets_mm - targets_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    heavy = targets_mm > 5
    heavy_mae = np.mean(np.abs(targets_mm[heavy] - preds_mm[heavy])) if heavy.sum() > 0 else float('nan')
    heavy_bias = np.mean(targets_mm[heavy] - preds_mm[heavy]) if heavy.sum() > 0 else float('nan')
    heavy_n = int(heavy.sum())

    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'heavy_mae': heavy_mae,
        'heavy_bias': heavy_bias,
        'heavy_n': heavy_n,
    }


def run_ablation(checkpoint_path=None, checkpoint_dir=None, pickle_path=None, dem_path=None, run_dir=None):
    _ensure_utf8_stdio()
    pickle_path = pickle_path or DEFAULT_PICKLE
    dem_path = dem_path or DEFAULT_DEM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = find_checkpoint(checkpoint_path, checkpoint_dir or DEFAULT_CKPT_DIR, run_dir=run_dir)

    # Try to get run_dir from checkpoint metadata if not provided
    if not run_dir:
        ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
        run_dir = ckpt_data.get('run_dir')

    model, cfg = load_model(ckpt, device)

    # Prefer paths from checkpoint config for reproducibility
    pickle_path = cfg.get('pickle_path') or pickle_path
    dem_path = cfg.get('dem_path') or dem_path
    exclude = cfg.get('exclude_stations', [])

    # Reconstruct dataset with same feature config as training
    fields = resolve_fields(cfg.get('fields'))
    use_dem = cfg.get('use_dem', True)
    use_mask = cfg.get('use_mask', True)
    use_temporal_pos = cfg.get('use_temporal_pos', True)
    use_feature_masks = cfg.get('use_feature_masks', False)

    ds_kwargs = dict(
        dem_path=dem_path,
        fields=fields,
        use_dem=use_dem,
        use_mask=use_mask,
        use_temporal_pos=use_temporal_pos,
        log_target=cfg.get('log_target', True),
        use_feature_masks=use_feature_masks,
    )
    val_ds = RadarGaugeDataset(pickle_path, split='val', augment=False, **ds_kwargs)
    val_ds.samples = filter_stations(val_ds.samples, exclude)
    val_ds.samples = filter_nan_radar(val_ds.samples)
    filter_mode = cfg.get('filter_mode', 'blunt')
    if filter_mode == 'radar':
        val_ds.samples = filter_radar_unsupported(val_ds.samples)
    else:
        val_ds.samples = filter_biased_extremes(val_ds.samples)
        val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_suspect_station_days(val_ds.samples)
    val_ds.samples = filter_gauge_dumps(val_ds.samples)

    # Build dynamic feature groups for this model's configuration
    feature_groups = build_feature_groups(fields, use_mask, use_temporal_pos, use_dem,
                                           use_feature_masks)
    n_channels = compute_n_input_channels(fields, use_mask, use_temporal_pos, use_dem,
                                          use_feature_masks)

    print(f"\nAblation dataset: {len(val_ds.samples)} validation samples")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # Baseline (no ablation)
    print(f"\n{'='*70}")
    print("  FEATURE ABLATION STUDY")
    print(f"{'='*70}")
    print(f"\n  Checkpoint: {ckpt}")
    print(f"  Samples:    {len(val_ds.samples)}")
    print(f"  Channels:   {n_channels}")
    print(f"  Fields:     {fields}")
    print(f"{'='*70}")

    log_target = cfg.get('log_target', True)

    print("\n  Running baseline (all features)...")
    preds_mm, targets_mm = run_ablation_inference(model, val_loader, device, zero_channels=None, log_target=log_target)
    baseline = compute_metrics(preds_mm, targets_mm)

    print(f"  Baseline: R²={baseline['r2']:.4f}  MAE={baseline['mae']:.3f}  "
          f"RMSE={baseline['rmse']:.3f}  Heavy MAE={baseline['heavy_mae']:.3f}")

    # Run each ablation
    results = {'baseline': baseline}

    for group_name, channels in feature_groups.items():
        print(f"\n  Ablating: {group_name} (channels {channels[0]}-{channels[-1]}, n={len(channels)})...")
        preds_mm, targets_mm = run_ablation_inference(model, val_loader, device, zero_channels=channels, log_target=log_target)
        metrics = compute_metrics(preds_mm, targets_mm)
        results[group_name] = metrics

        delta_r2 = metrics['r2'] - baseline['r2']
        delta_mae = metrics['mae'] - baseline['mae']
        sign_r2 = '+' if delta_r2 >= 0 else ''
        sign_mae = '+' if delta_mae >= 0 else ''
        print(f"    R²={metrics['r2']:.4f} ({sign_r2}{delta_r2:.4f})  "
              f"MAE={metrics['mae']:.3f} ({sign_mae}{delta_mae:.3f})  "
              f"RMSE={metrics['rmse']:.3f}  Heavy MAE={metrics['heavy_mae']:.3f}")

    # Summary table
    print(f"\n\n{'='*70}")
    print("  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Feature':<16} {'R²':>8} {'ΔR²':>8} {'MAE':>8} {'ΔMAE':>8} {'RMSE':>8} {'H.MAE':>8} {'Impact':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    print(f"  {'BASELINE':<16} {baseline['r2']:>8.4f} {'---':>8} {baseline['mae']:>8.3f} {'---':>8} "
          f"{baseline['rmse']:>8.3f} {baseline['heavy_mae']:>8.3f} {'---':>10}")

    ranked = []
    for group_name in feature_groups:
        m = results[group_name]
        delta_r2 = m['r2'] - baseline['r2']
        delta_mae = m['mae'] - baseline['mae']

        if delta_r2 < -0.02:
            impact = 'HIGH'
        elif delta_r2 < -0.005:
            impact = 'MEDIUM'
        elif delta_r2 < 0.005:
            impact = 'LOW'
        else:
            impact = 'NEGATIVE'

        ranked.append((group_name, delta_r2, delta_mae, m, impact))

    ranked.sort(key=lambda x: x[1])

    for group_name, delta_r2, delta_mae, m, impact in ranked:
        print(f"  {group_name:<16} {m['r2']:>8.4f} {delta_r2:>+8.4f} {m['mae']:>8.3f} {delta_mae:>+8.3f} "
              f"{m['rmse']:>8.3f} {m['heavy_mae']:>8.3f} {impact:>10}")

    print(f"\n  Interpretation:")
    print(f"  - Negative ΔR² = feature HELPS (removing it hurts performance)")
    print(f"  - Positive ΔR² = feature HURTS (removing it improves performance)")
    print(f"  - Impact: HIGH (ΔR² < -0.02), MEDIUM (-0.02 to -0.005), LOW (±0.005), NEGATIVE (> +0.005)")
    print(f"\n{'='*70}\n")

    # Write ablation results to results.txt (replace prior ablation block)
    if run_dir:
        results_path = Path(run_dir) / 'results.txt'
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  ABLATION RESULTS")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Checkpoint: {ckpt}")
        lines.append(f"  Samples:    {len(val_ds.samples)}")
        lines.append("")
        lines.append(f"  {'Feature':<16} {'R2':>8} {'dR2':>8} {'MAE':>8} {'dMAE':>8} {'RMSE':>8} {'H.MAE':>8} {'Impact':>10}")
        lines.append(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        lines.append(f"  {'BASELINE':<16} {baseline['r2']:>8.4f} {'---':>8} {baseline['mae']:>8.3f} {'---':>8} "
                     f"{baseline['rmse']:>8.3f} {baseline['heavy_mae']:>8.3f} {'---':>10}")

        for group_name, delta_r2, delta_mae, m, impact in ranked:
            lines.append(f"  {group_name:<16} {m['r2']:>8.4f} {delta_r2:>+8.4f} {m['mae']:>8.3f} {delta_mae:>+8.3f} "
                         f"{m['rmse']:>8.3f} {m['heavy_mae']:>8.3f} {impact:>10}")

        lines.append("")
        lines.append("  Interpretation:")
        lines.append("  - Negative dR2 = feature HELPS (removing it hurts performance)")
        lines.append("  - Positive dR2 = feature HURTS (removing it improves performance)")
        lines.append("")

        _write_ablation_results(results_path, lines)
        print(f"  OK: Wrote ablation results to: {results_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature ablation study for Stack CNN")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--run-dir', default=None, help='Run directory (auto-finds checkpoint and saves results there)')
    parser.add_argument('--pickle', default=DEFAULT_PICKLE)
    parser.add_argument('--dem', default=DEFAULT_DEM)
    args = parser.parse_args()

    run_ablation(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        pickle_path=args.pickle,
        dem_path=args.dem,
        run_dir=args.run_dir,
    )
