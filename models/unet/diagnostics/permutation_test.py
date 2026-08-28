"""
permutation_test.py — Batch-wise permutation importance for U-Net QPE models.

For each feature group, shuffles that group's channels across the batch dimension
(real values from other samples, decorrelated from the target). Positive delta
loss / negative delta R2 means the model was using that feature productively.

Unlike zero-ablation, permutation preserves each channel's marginal distribution
while breaking the joint relationship with co-located fields and rain rate.

Run from the project root:
    python -m models.unet.diagnostics.permutation_test --run-dir models/checkpoints/unet_dualpol/<run_name>
    python -m models.unet.diagnostics.permutation_test --run-dir ... --n-repeats 5 --fields-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.unet.ablation import build_feature_groups
from models.unet.dataset import RadarGaugeDataset, resolve_fields
from models.unet.evaluate import find_checkpoint, load_model
from models.unet.train import (
    GaugePixelLoss,
    filter_bad_samples,
    filter_biased_extremes,
    filter_gauge_dumps,
    filter_nan_radar,
    filter_radar_unsupported,
    filter_stations,
    filter_suspect_station_days,
)

DEFAULT_PICKLE = 'dataset/outputs/3d/radar_gauge_dataset_vertlowmeltbb_9500.pkl'
DEFAULT_DEM = 'dem/preserve_dem_10m_utm.tif'
DEFAULT_CKPT_DIR = 'models/checkpoints/unet_dualpol'

AUX_GROUPS = {'mask', 'feature_masks', 'temporal_pos', 'dem'}
COMPOSITE_GROUPS = {'all_dualpol', 'all_vertical'}


def _ensure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


def permute_channels(radar: torch.Tensor, channel_indices: list[int], generator: torch.Generator) -> torch.Tensor:
    """Shuffle a channel group across the batch, keeping all scans from one donor sample."""
    if not channel_indices:
        return radar
    out = radar.clone()
    batch_size = radar.shape[0]
    idx = torch.randperm(batch_size, generator=generator).to(radar.device)
    donors = radar[idx]
    out[:, channel_indices, :, :] = donors[:, channel_indices, :, :]
    return out


def _pred_at_gauge(pred_map: torch.Tensor, gauge_pixel) -> torch.Tensor:
    if pred_map.dim() == 1:
        return pred_map
    batch_idx = torch.arange(pred_map.shape[0], device=pred_map.device)
    if isinstance(gauge_pixel, (tuple, list)):
        y, x = gauge_pixel
        if isinstance(y, torch.Tensor):
            return pred_map[batch_idx, y.to(pred_map.device), x.to(pred_map.device)]
        return pred_map[:, y, x]
    if isinstance(gauge_pixel, torch.Tensor) and gauge_pixel.dim() == 2:
        y = gauge_pixel[:, 0].long().to(pred_map.device)
        x = gauge_pixel[:, 1].long().to(pred_map.device)
        return pred_map[batch_idx, y, x]
    return pred_map[:, 4, 4]


def build_val_loader(cfg, pickle_path, dem_path, batch_size):
    ds_kwargs = dict(
        dem_path=dem_path,
        fields=resolve_fields(cfg.get('fields')),
        use_dem=cfg.get('use_dem', True),
        use_mask=cfg.get('use_mask', True),
        use_temporal_pos=cfg.get('use_temporal_pos', True),
        log_target=cfg.get('log_target', True),
        use_feature_masks=cfg.get('use_feature_masks', False),
    )
    val_ds = RadarGaugeDataset(pickle_path, split='val', augment=False, **ds_kwargs)
    exclude = cfg.get('exclude_stations', [])
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
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader, len(val_ds.samples)


def build_loss_fn(cfg):
    return GaugePixelLoss(
        loss_type=cfg.get('loss_type', 'mae'),
        log_target=cfg.get('log_target', True),
        under_weight=cfg.get('under_weight', 2.0),
        huber_delta=cfg.get('huber_delta', 2.0),
    )


def eval_pass(
    model,
    loader,
    loss_fn,
    device,
    log_target,
    permute_channels_idx: list[int] | None = None,
    generator: torch.Generator | None = None,
):
    """One validation pass; returns mean training loss and mm-space metrics."""
    total_loss = 0.0
    n_loss = 0
    all_preds_mm = []
    all_targets_mm = []

    with torch.no_grad():
        for batch in loader:
            radar = batch['radar'].to(device)
            target = batch['target'].to(device)
            gauge_pixel = batch['gauge_pixel']

            if permute_channels_idx is not None:
                radar = permute_channels(radar, permute_channels_idx, generator)

            pred_map = model(radar)
            batch_loss = loss_fn(pred_map, target, gauge_pixel)
            total_loss += batch_loss.item() * target.shape[0]
            n_loss += target.shape[0]

            pred_at_gauge = _pred_at_gauge(pred_map, gauge_pixel).cpu()
            target_cpu = target.cpu()
            if log_target:
                pred_at_gauge = torch.expm1(pred_at_gauge)
                target_cpu = torch.expm1(target_cpu)

            all_preds_mm.extend(pred_at_gauge.numpy().tolist())
            all_targets_mm.extend(target_cpu.numpy().tolist())

    preds_mm = np.asarray(all_preds_mm)
    targets_mm = np.asarray(all_targets_mm)
    valid = targets_mm >= 0
    preds_mm = preds_mm[valid]
    targets_mm = targets_mm[valid]

    mae = float(np.mean(np.abs(targets_mm - preds_mm)))
    rmse = float(np.sqrt(np.mean((targets_mm - preds_mm) ** 2)))
    ss_res = float(np.sum((targets_mm - preds_mm) ** 2))
    ss_tot = float(np.sum((targets_mm - targets_mm.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    heavy = targets_mm > 5.0
    heavy_mae = float(np.mean(np.abs(targets_mm[heavy] - preds_mm[heavy]))) if heavy.any() else float('nan')

    mean_loss = total_loss / max(n_loss, 1)
    return {
        'loss': mean_loss,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'heavy_mae': heavy_mae,
        'n': int(valid.sum()),
    }


def select_groups(feature_groups, fields_only=False, skip_composite=False, only_groups=None):
    names = []
    for name in feature_groups:
        if fields_only and name in AUX_GROUPS:
            continue
        if skip_composite and name in COMPOSITE_GROUPS:
            continue
        if only_groups and name not in only_groups:
            continue
        names.append(name)
    return names


def permutation_importance(
    model,
    loader,
    loss_fn,
    feature_groups,
    device,
    log_target,
    n_repeats=3,
    fields_only=False,
    skip_composite=True,
    only_groups=None,
    seed=0,
):
    """Permute each feature group across the batch and measure metric deltas."""
    group_names = select_groups(feature_groups, fields_only, skip_composite, only_groups)

    print("\n  Computing baseline...")
    baseline = eval_pass(model, loader, loss_fn, device, log_target)
    print(
        f"  Baseline: loss={baseline['loss']:.4f}  R2={baseline['r2']:.4f}  "
        f"MAE={baseline['mae']:.3f}  Heavy MAE={baseline['heavy_mae']:.3f}"
    )

    results = {'baseline': baseline, 'groups': {}}

    for group_name in group_names:
        channels = feature_groups[group_name]
        deltas = []
        print(f"\n  Permuting: {group_name} ({len(channels)} channels)...")

        for repeat in range(n_repeats):
            gen = torch.Generator()
            gen.manual_seed(seed + repeat * 1000 + hash(group_name) % 1000)
            metrics = eval_pass(
                model, loader, loss_fn, device, log_target,
                permute_channels_idx=channels,
                generator=gen,
            )
            deltas.append({
                'loss': metrics['loss'] - baseline['loss'],
                'r2': metrics['r2'] - baseline['r2'],
                'mae': metrics['mae'] - baseline['mae'],
                'rmse': metrics['rmse'] - baseline['rmse'],
                'heavy_mae': metrics['heavy_mae'] - baseline['heavy_mae'],
            })

        arr = {k: np.array([d[k] for d in deltas], dtype=np.float64) for k in deltas[0]}
        entry = {
            'n_channels': len(channels),
            'loss_mean': float(arr['loss'].mean()),
            'loss_std': float(arr['loss'].std(ddof=0)),
            'r2_mean': float(arr['r2'].mean()),
            'r2_std': float(arr['r2'].std(ddof=0)),
            'mae_mean': float(arr['mae'].mean()),
            'mae_std': float(arr['mae'].std(ddof=0)),
            'heavy_mae_mean': float(arr['heavy_mae'].mean()),
            'heavy_mae_std': float(arr['heavy_mae'].std(ddof=0)),
        }
        results['groups'][group_name] = entry
        print(
            f"    dLoss={entry['loss_mean']:+.4f} +/- {entry['loss_std']:.4f}  "
            f"dR2={entry['r2_mean']:+.4f}  dMAE={entry['mae_mean']:+.3f}"
        )

    return results


def classify_impact(delta_loss_mean, delta_r2_mean):
    if delta_loss_mean > 0.02 or delta_r2_mean < -0.02:
        return 'HIGH'
    if delta_loss_mean > 0.005 or delta_r2_mean < -0.005:
        return 'MEDIUM'
    if delta_loss_mean < -0.005 or delta_r2_mean > 0.005:
        return 'NEGATIVE'
    return 'LOW'


def print_summary(results, n_repeats):
    baseline = results['baseline']
    ranked = sorted(
        results['groups'].items(),
        key=lambda kv: kv[1]['loss_mean'],
        reverse=True,
    )

    print(f"\n{'=' * 78}")
    print("  PERMUTATION IMPORTANCE SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Repeats per feature: {n_repeats}")
    print(
        f"  Baseline: loss={baseline['loss']:.4f}  R2={baseline['r2']:.4f}  "
        f"MAE={baseline['mae']:.3f}  Heavy MAE={baseline['heavy_mae']:.3f}"
    )
    print()
    print(
        f"  {'Feature':<28} {'dLoss':>10} {'dR2':>10} {'dMAE':>10} {'dH.MAE':>10} {'Impact':>10}"
    )
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'BASELINE':<28} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {'---':>10}")

    for name, m in ranked:
        impact = classify_impact(m['loss_mean'], m['r2_mean'])
        print(
            f"  {name:<28} {m['loss_mean']:>+10.4f} {m['r2_mean']:>+10.4f} "
            f"{m['mae_mean']:>+10.3f} {m['heavy_mae_mean']:>+10.3f} {impact:>10}"
        )

    print()
    print("  Interpretation:")
    print("  - Positive dLoss / negative dR2 = feature helps (permute hurt performance)")
    print("  - Near-zero = model not using it much on validation")
    print("  - Negative dLoss / positive dR2 = feature may add noise or confusion")
    print("  - Correlated features (e.g. Z_H + Z_DR) can share importance; both may look weaker")
    print(f"{'=' * 78}\n")


def write_results(run_dir, ckpt, n_samples, n_repeats, results):
    if not run_dir:
        return
    path = Path(run_dir) / 'permutation_importance.txt'
    baseline = results['baseline']
    ranked = sorted(results['groups'].items(), key=lambda kv: kv[1]['loss_mean'], reverse=True)

    lines = [
        "=" * 60,
        "  PERMUTATION IMPORTANCE",
        "=" * 60,
        "",
        f"  Checkpoint: {ckpt}",
        f"  Samples:    {n_samples}",
        f"  Repeats:    {n_repeats}",
        "",
        f"  Baseline loss: {baseline['loss']:.4f}  R2: {baseline['r2']:.4f}  "
        f"MAE: {baseline['mae']:.3f} mm/hr",
        "",
        f"  {'Feature':<28} {'dLoss':>10} {'dR2':>10} {'dMAE':>10} {'dH.MAE':>10} {'Impact':>10}",
        f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}",
        f"  {'BASELINE':<28} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {'---':>10}",
    ]
    for name, m in ranked:
        impact = classify_impact(m['loss_mean'], m['r2_mean'])
        lines.append(
            f"  {name:<28} {m['loss_mean']:>+10.4f} {m['r2_mean']:>+10.4f} "
            f"{m['mae_mean']:>+10.3f} {m['heavy_mae_mean']:>+10.3f} {impact:>10}"
        )
    lines.extend([
        "",
        "  Interpretation:",
        "  - Positive dLoss = permuting this feature increased loss (feature helps)",
        "  - Negative dLoss = permuting decreased loss (feature may hurt or add noise)",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"  OK: Wrote results to: {path}")


def run_permutation_test(
    checkpoint_path=None,
    checkpoint_dir=None,
    pickle_path=None,
    dem_path=None,
    run_dir=None,
    n_repeats=3,
    batch_size=64,
    fields_only=False,
    include_composite=False,
    only_groups=None,
    seed=0,
):
    _ensure_utf8_stdio()
    pickle_path = pickle_path or DEFAULT_PICKLE
    dem_path = dem_path or DEFAULT_DEM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = find_checkpoint(checkpoint_path, checkpoint_dir or DEFAULT_CKPT_DIR, run_dir=run_dir)

    if not run_dir:
        ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
        run_dir = ckpt_data.get('run_dir')

    model, cfg = load_model(ckpt, device)
    pickle_path = cfg.get('pickle_path') or pickle_path
    dem_path = cfg.get('dem_path') or dem_path

    fields = resolve_fields(cfg.get('fields'))
    use_dem = cfg.get('use_dem', True)
    use_mask = cfg.get('use_mask', True)
    use_temporal_pos = cfg.get('use_temporal_pos', True)
    use_feature_masks = cfg.get('use_feature_masks', False)
    log_target = cfg.get('log_target', True)

    feature_groups = build_feature_groups(
        fields, use_mask, use_temporal_pos, use_dem, use_feature_masks,
    )
    loader, n_samples = build_val_loader(cfg, pickle_path, dem_path, batch_size)
    loss_fn = build_loss_fn(cfg).to(device)

    print(f"\n{'=' * 78}")
    print("  PERMUTATION IMPORTANCE")
    print(f"{'=' * 78}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Samples:    {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Loss:       {cfg.get('loss_type', 'mae')} (log_target={log_target})")
    print(f"  Fields:     {len(fields)} radar fields + aux channels")
    print(f"{'=' * 78}")

    results = permutation_importance(
        model, loader, loss_fn, feature_groups, device, log_target,
        n_repeats=n_repeats,
        fields_only=fields_only,
        skip_composite=not include_composite,
        only_groups=only_groups,
        seed=seed,
    )

    print_summary(results, n_repeats)
    write_results(run_dir, ckpt, n_samples, n_repeats, results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch-wise permutation importance for U-Net precipitation models",
    )
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--run-dir', default=None, help='Run directory (auto-finds best checkpoint)')
    parser.add_argument('--pickle', default=DEFAULT_PICKLE)
    parser.add_argument('--dem', default=DEFAULT_DEM)
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Permutation repeats per feature (default: 3)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--fields-only', action='store_true',
                        help='Skip aux groups (mask, feature_masks, temporal_pos, dem)')
    parser.add_argument('--include-composite', action='store_true',
                        help='Also permute all_dualpol / all_vertical composite groups')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Permute only these group names (e.g. reflectivity dem)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_permutation_test(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        pickle_path=args.pickle,
        dem_path=args.dem,
        run_dir=args.run_dir,
        n_repeats=args.n_repeats,
        batch_size=args.batch_size,
        fields_only=args.fields_only,
        include_composite=args.include_composite,
        only_groups=args.only,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
