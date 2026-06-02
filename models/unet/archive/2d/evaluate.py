"""
evaluate.py — Evaluate a trained Stack CNN precipitation model.

Run from the project root:
    python -m models.stack.evaluate --checkpoint checkpoints/stack_dualpol/best-epoch=XX-val_loss=X.XXXX.pt
    python -m models.stack.evaluate --run-dir models/checkpoints/stack_dualpol/2026-05-21_12-04_mae_3block
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from models.unet.model import PrecipUNet, init_weights
from models.stack.dataset import RadarGaugeDataset
from models.unet.train import filter_bad_samples, filter_biased_extremes, filter_nan_radar, filter_suspect_station_days, filter_stations, filter_radar_unsupported

DEFAULT_PICKLE   = 'dataset/outputs/radar_gauge_dataset_with_offsets_9500.pkl'
DEFAULT_DEM      = 'dem/preserve_dem_10m_utm.tif'
DEFAULT_CKPT_DIR = 'models/checkpoints/unet_dualpol'
DEFAULT_OUTPUT   = 'evaluation_figures/unet_dualpol'


def find_checkpoint(checkpoint_path=None, checkpoint_dir=None, run_dir=None):
    if checkpoint_path and Path(checkpoint_path).exists():
        return checkpoint_path
    if run_dir:
        rd = Path(run_dir)
        best_files = sorted(rd.glob('best-*.pt'), key=os.path.getmtime, reverse=True)
        if best_files:
            return str(best_files[0])
        last = rd / 'last.pt'
        if last.exists():
            return str(last)
    ckpt_dir = Path(checkpoint_dir or DEFAULT_CKPT_DIR)
    best_files = sorted(ckpt_dir.glob('best-*.pt'), key=os.path.getmtime, reverse=True)
    if best_files:
        return str(best_files[0])
    last = ckpt_dir / 'last.pt'
    if last.exists():
        return str(last)
    raise FileNotFoundError(f"No checkpoints found in {run_dir or ckpt_dir}")


def load_model(checkpoint_path, device):
    print(f"Loading model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})

    model = PrecipUNet(
        base_filters=cfg.get('base_filters', 64),
        dropout_rate=cfg.get('dropout_rate', 0.15),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    return model, cfg


def run_inference(model, val_loader, device):
    all_preds = []
    all_targets = []
    all_station_names = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            radar = batch['radar'].to(device)
            target = torch.expm1(batch['target'])
            gauge_pixel = batch['gauge_pixel']
            bias_flag = batch.get('bias_flag')
            station_names = batch.get('station_name', [])

            if model.add_bias and bias_flag is not None:
                pred_map = model(radar, bias_flag.to(device))
            else:
                pred_map = model(radar)

            pred_map = pred_map.cpu()

            # Scalar output: pred_map is already (B,)
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
                    pred_at_gauge = pred_map[:, 2, 2]

            all_preds.extend(pred_at_gauge.numpy().tolist())
            all_targets.extend(target.numpy().tolist())
            all_station_names.extend(station_names)

    preds_mm = np.array(all_preds)
    targets_mm = np.array(all_targets)

    valid = targets_mm >= 0
    preds_mm = preds_mm[valid]
    targets_mm = targets_mm[valid]
    station_names = np.array(all_station_names)[valid]

    print(f"✓ Collected {valid.sum()} valid samples")
    return preds_mm, targets_mm, station_names


def compute_metrics(preds_mm, targets_mm):
    mae = np.mean(np.abs(targets_mm - preds_mm))
    rmse = np.sqrt(np.mean((targets_mm - preds_mm) ** 2))
    ss_res = np.sum((targets_mm - preds_mm) ** 2)
    ss_tot = np.sum((targets_mm - targets_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return dict(r2=r2, mae=mae, rmse=rmse)


def print_report(preds_mm, targets_mm, metrics):
    print(f"\n{'='*60}")
    print("  STACK CNN MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"\n  R²:         {metrics['r2']:.3f}")
    print(f"  MAE:        {metrics['mae']:.3f} mm/hr")
    print(f"  RMSE:       {metrics['rmse']:.3f} mm/hr")
    print(f"  Pred max:   {preds_mm.max():.2f} mm/hr")
    print(f"  # Pred >5mm: {int(np.sum(preds_mm > 5))}")

    heavy = targets_mm > 5
    if heavy.sum():
        h_mae = np.mean(np.abs(targets_mm[heavy] - preds_mm[heavy]))
        h_bias = np.mean(targets_mm[heavy] - preds_mm[heavy])
        direction = "underpredicting" if h_bias > 0 else "overpredicting"
        print(f"\n  Heavy Rain (actual >5mm):")
        print(f"    Samples:   {heavy.sum()}")
        print(f"    MAE:       {h_mae:.3f} mm/hr")
        print(f"    Bias:      {h_bias:.3f} mm/hr ({direction})")
        print(f"    Pred >5mm: {int(np.sum(preds_mm[heavy] > 5))} / {heavy.sum()}")


def write_eval_results(run_dir, preds_mm, targets_mm, metrics):
    """Write evaluation results to results.txt in the run directory."""
    if not run_dir:
        return

    results_path = Path(run_dir) / 'results.txt'
    heavy = targets_mm > 5

    lines = []
    lines.append("=" * 60)
    lines.append("  EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  R²:            {metrics['r2']:.4f}")
    lines.append(f"  MAE:           {metrics['mae']:.3f} mm/hr")
    lines.append(f"  RMSE:          {metrics['rmse']:.3f} mm/hr")
    lines.append(f"  Pred max:      {preds_mm.max():.2f} mm/hr")
    lines.append(f"  Pred min:      {preds_mm.min():.2f} mm/hr")
    lines.append(f"  # Pred >5mm:   {int(np.sum(preds_mm > 5))}")
    lines.append(f"  # Samples:     {len(preds_mm)}")
    lines.append("")

    if heavy.sum():
        h_mae = np.mean(np.abs(targets_mm[heavy] - preds_mm[heavy]))
        h_bias = np.mean(targets_mm[heavy] - preds_mm[heavy])
        direction = "underpredicting" if h_bias > 0 else "overpredicting"
        lines.append("  Heavy Rain (actual >5mm):")
        lines.append(f"    Samples:   {int(heavy.sum())}")
        lines.append(f"    MAE:       {h_mae:.3f} mm/hr")
        lines.append(f"    Bias:      {h_bias:.3f} mm/hr ({direction})")
        lines.append(f"    Pred >5mm: {int(np.sum(preds_mm[heavy] > 5))} / {int(heavy.sum())}")
        lines.append("")

    with open(results_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  ✓ Saved evaluation results to: {results_path}")


def plot_evaluation(preds_mm, targets_mm, metrics, output_dir, run_dir=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Scatter
    ax = axes[0, 0]
    ax.scatter(targets_mm, preds_mm, alpha=0.3, s=20, c='steelblue')
    max_val = max(targets_mm.max(), preds_mm.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual (mm/hr)')
    ax.set_ylabel('Predicted (mm/hr)')
    ax.set_title(f"Predicted vs Actual\nR²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f} mm/hr")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE by category
    ax = axes[0, 1]
    categories = ['Dry\n(0-0.1)', 'Light\n(0.1-1)', 'Moderate\n(1-5)', 'Heavy\n(5-10)', 'Extreme\n(>10)']
    bounds = [0, 0.1, 1, 5, 10, 1e6]
    maes, biases, counts = [], [], []
    for lo, hi in zip(bounds, bounds[1:]):
        m = (targets_mm >= lo) & (targets_mm < hi)
        if m.sum():
            err = targets_mm[m] - preds_mm[m]
            maes.append(np.mean(np.abs(err)))
            biases.append(np.mean(err))
            counts.append(m.sum())
        else:
            maes.append(0); biases.append(0); counts.append(0)
    colors = ['lightblue', 'skyblue', 'steelblue', 'royalblue', 'darkblue']
    bars = ax.bar(range(len(categories)), maes, color=colors)
    for bar, n, b in zip(bars, counts, biases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'n={n}\nbias={b:.2f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel('MAE (mm/hr)')
    ax.set_title('Error by Precipitation Category')
    ax.grid(True, alpha=0.3, axis='y')

    # Residuals
    ax = axes[1, 0]
    residuals = targets_mm - preds_mm
    ax.scatter(preds_mm, residuals, alpha=0.3, s=20, c='coral')
    ax.axhline(0, color='r', lw=2, linestyle='--')
    ax.set_xlabel('Predicted (mm/hr)')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title('Residual Plot\n(above zero = underpredicting)')
    ax.grid(True, alpha=0.3)

    # Distribution
    ax = axes[1, 1]
    ax.hist(targets_mm, bins=50, alpha=0.5, label='Actual', color='green', density=True)
    ax.hist(preds_mm, bins=50, alpha=0.5, label='Predicted', color='blue', density=True)
    ax.set_xlabel('Precipitation (mm/hr)')
    ax.set_ylabel('Density')
    ax.set_title('Precipitation Distribution')
    ax.legend()
    ax.set_xlim(0, max(targets_mm.max(), preds_mm.max()))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = out / 'stack_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved evaluation plot to: {save_path}")

    if run_dir:
        run_save_path = Path(run_dir) / 'evaluation.png'
        plt.savefig(run_save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved evaluation plot to run dir: {run_save_path}")

    plt.close()


def plot_station_bias(preds_mm, targets_mm, station_names, output_dir, run_dir=None):
    """Plot per-station bias broken down by dry/light rain vs heavy rain."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    unique_stations = sorted(set(station_names))
    if not unique_stations:
        return

    station_short = [s.replace('Dangermond_', '') for s in unique_stations]

    # Compute per-station bias for two regimes
    dry_bias = []     # actual < 2mm
    heavy_bias = []   # actual >= 2mm
    dry_counts = []
    heavy_counts = []

    for station in unique_stations:
        mask = station_names == station
        s_preds = preds_mm[mask]
        s_targets = targets_mm[mask]

        dry_mask = s_targets < 2.0
        heavy_mask = s_targets >= 2.0

        if dry_mask.sum() > 0:
            dry_bias.append(np.mean(s_preds[dry_mask] - s_targets[dry_mask]))
            dry_counts.append(int(dry_mask.sum()))
        else:
            dry_bias.append(0.0)
            dry_counts.append(0)

        if heavy_mask.sum() > 0:
            heavy_bias.append(np.mean(s_preds[heavy_mask] - s_targets[heavy_mask]))
            heavy_counts.append(int(heavy_mask.sum()))
        else:
            heavy_bias.append(0.0)
            heavy_counts.append(0)

    dry_bias = np.array(dry_bias)
    heavy_bias = np.array(heavy_bias)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(len(unique_stations))
    width = 0.35

    # Top chart: bias by station, grouped by regime
    ax = axes[0]
    bars1 = ax.bar(x - width/2, dry_bias, width, label='Dry/Light (<2mm)', color='skyblue', edgecolor='steelblue')
    bars2 = ax.bar(x + width/2, heavy_bias, width, label='Moderate/Heavy (≥2mm)', color='salmon', edgecolor='darkred')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(station_short, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Bias (pred - actual, mm/hr)')
    ax.set_title('Per-Station Prediction Bias\n(positive = overpredicting, negative = underpredicting)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, n in zip(bars1, dry_counts):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{n}', ha='center', va='bottom', fontsize=7, color='steelblue')
    for bar, n in zip(bars2, heavy_counts):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{n}', ha='center', va='bottom', fontsize=7, color='darkred')

    # Bottom chart: MAE by station, grouped by regime
    ax = axes[1]
    dry_mae = []
    heavy_mae = []
    for station in unique_stations:
        mask = station_names == station
        s_preds = preds_mm[mask]
        s_targets = targets_mm[mask]

        dry_mask = s_targets < 2.0
        heavy_mask = s_targets >= 2.0

        dry_mae.append(np.mean(np.abs(s_preds[dry_mask] - s_targets[dry_mask])) if dry_mask.sum() > 0 else 0.0)
        heavy_mae.append(np.mean(np.abs(s_preds[heavy_mask] - s_targets[heavy_mask])) if heavy_mask.sum() > 0 else 0.0)

    ax.bar(x - width/2, dry_mae, width, label='Dry/Light (<2mm)', color='skyblue', edgecolor='steelblue')
    ax.bar(x + width/2, heavy_mae, width, label='Moderate/Heavy (≥2mm)', color='salmon', edgecolor='darkred')
    ax.set_xticks(x)
    ax.set_xticklabels(station_short, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('MAE (mm/hr)')
    ax.set_title('Per-Station MAE by Precipitation Regime')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = out / 'station_bias.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved station bias plot to: {save_path}")

    if run_dir:
        run_save_path = Path(run_dir) / 'station_bias.png'
        plt.savefig(run_save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved station bias plot to run dir: {run_save_path}")

    plt.close()


def evaluate(checkpoint_path=None, checkpoint_dir=None, pickle_path=None, dem_path=None, output_dir=None, run_dir=None, exclude_stations=None):
    pickle_path = pickle_path or DEFAULT_PICKLE
    dem_path = dem_path or DEFAULT_DEM
    output_dir = output_dir or DEFAULT_OUTPUT

    # If run_dir provided, derive checkpoint from it
    if run_dir and not checkpoint_path:
        checkpoint_dir = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = find_checkpoint(checkpoint_path, checkpoint_dir, run_dir=run_dir)

    # Try to get run_dir from checkpoint metadata if not provided
    if not run_dir:
        ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
        run_dir = ckpt_data.get('run_dir')

    model, cfg = load_model(ckpt, device)

    # Use exclude_stations from config if not explicitly provided
    exclude = exclude_stations or cfg.get('exclude_stations', [])

    val_ds = RadarGaugeDataset(pickle_path, dem_path=dem_path, split='val', augment=False)
    val_ds.samples = filter_stations(val_ds.samples, exclude)
    val_ds.samples = filter_nan_radar(val_ds.samples)
    filter_mode = cfg.get('filter_mode', 'blunt')
    if filter_mode == 'radar':
        val_ds.samples = filter_radar_unsupported(val_ds.samples)
    else:
        val_ds.samples = filter_biased_extremes(val_ds.samples)
        val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_suspect_station_days(val_ds.samples)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    preds_mm, targets_mm, station_names = run_inference(model, val_loader, device)
    metrics = compute_metrics(preds_mm, targets_mm)
    print_report(preds_mm, targets_mm, metrics)
    plot_evaluation(preds_mm, targets_mm, metrics, output_dir, run_dir=run_dir)
    plot_station_bias(preds_mm, targets_mm, station_names, output_dir, run_dir=run_dir)
    write_eval_results(run_dir, preds_mm, targets_mm, metrics)

    # Test evaluation (daily gauges)
    evaluate_test(model, cfg, pickle_path, dem_path, device, output_dir, run_dir)

    return metrics, run_dir


def evaluate_test(model, cfg, pickle_path, dem_path, device, output_dir, run_dir=None):
    """Evaluate model on daily cumulative gauge test set."""
    import pickle as pkl

    with open(pickle_path, 'rb') as f:
        dataset = pkl.load(f)

    test_samples = dataset.get('test', [])
    if not test_samples:
        print("\n  No test samples in pickle — skipping daily gauge evaluation.")
        return

    print(f"\n{'='*60}")
    print("  TEST EVALUATION (daily cumulative gauges)")
    print(f"{'='*60}")
    print(f"  Hourly test samples: {len(test_samples)}")

    test_ds = RadarGaugeDataset(pickle_path, dem_path=dem_path, split='test', augment=False)
    test_ds.samples = filter_nan_radar(test_ds.samples)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Run hourly inference
    hourly_preds = []
    hourly_meta = []
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            radar = batch['radar'].to(device)
            gauge_pixel = batch['gauge_pixel']

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
                    pred_at_gauge = pred_map[:, 2, 2]

            for i in range(pred_at_gauge.shape[0]):
                hourly_preds.append(max(0.0, pred_at_gauge[i].item()))
                sample = test_ds.samples[sample_idx]
                hourly_meta.append({
                    'date': sample['date'],
                    'station_id': sample['station_id'],
                    'station_name': sample.get('station_name', ''),
                    'daily_precip_mm': sample['daily_precip_mm'],
                })
                sample_idx += 1

    # Aggregate hourly predictions to daily totals
    from collections import defaultdict
    daily_groups = defaultdict(lambda: {'pred_sum': 0.0, 'count': 0, 'actual': 0.0, 'station_name': ''})

    for pred, meta in zip(hourly_preds, hourly_meta):
        key = (meta['date'], meta['station_id'])
        daily_groups[key]['pred_sum'] += pred
        daily_groups[key]['count'] += 1
        daily_groups[key]['actual'] = meta['daily_precip_mm']
        daily_groups[key]['station_name'] = meta['station_name']

    pred_daily = np.array([v['pred_sum'] for v in daily_groups.values()])
    actual_daily = np.array([v['actual'] for v in daily_groups.values()])
    hours_per_day = np.array([v['count'] for v in daily_groups.values()])
    station_names_daily = [v['station_name'] for v in daily_groups.values()]

    # Only keep days with sufficient coverage (>=18 hours)
    valid = hours_per_day >= 18
    pred_daily = pred_daily[valid]
    actual_daily = actual_daily[valid]
    station_names_daily = np.array(station_names_daily)[valid]

    if len(pred_daily) == 0:
        print("  No valid day-station groups with >=18 hours. Skipping.")
        return

    # Metrics
    test_metrics = compute_metrics(pred_daily, actual_daily)

    print(f"\n  Day-station groups: {len(pred_daily)} (≥18 hrs coverage)")
    print(f"  Avg hours/day:     {hours_per_day[valid].mean():.1f}")
    print(f"\n  R²:         {test_metrics['r2']:.3f}")
    print(f"  MAE:        {test_metrics['mae']:.3f} mm/day")
    print(f"  RMSE:       {test_metrics['rmse']:.3f} mm/day")
    print(f"  Pred range: {pred_daily.min():.2f} – {pred_daily.max():.2f} mm/day")
    print(f"  Actual range: {actual_daily.min():.2f} – {actual_daily.max():.2f} mm/day")

    # Save test results
    if run_dir:
        results_path = Path(run_dir) / 'test_results.txt'
        lines = [
            "=" * 60,
            "  TEST EVALUATION (daily cumulative gauges)",
            "=" * 60,
            "",
            f"  Day-station groups: {len(pred_daily)}",
            f"  R²:          {test_metrics['r2']:.4f}",
            f"  MAE:         {test_metrics['mae']:.3f} mm/day",
            f"  RMSE:        {test_metrics['rmse']:.3f} mm/day",
            f"  Pred range:  {pred_daily.min():.2f} – {pred_daily.max():.2f} mm/day",
            f"  Actual range: {actual_daily.min():.2f} – {actual_daily.max():.2f} mm/day",
            "",
        ]
        with open(results_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"  ✓ Saved test results to: {results_path}")

    # Plot
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(actual_daily, pred_daily, alpha=0.5, s=40, c='teal')
    max_val = max(actual_daily.max(), pred_daily.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Daily Rainfall (mm/day)')
    ax.set_ylabel('Predicted Daily Rainfall (mm/day)')
    ax.set_title(f"Daily Test — Predicted vs Actual\nR²={test_metrics['r2']:.3f}, MAE={test_metrics['mae']:.3f} mm/day")
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    unique_stations = sorted(set(station_names_daily))
    station_short = [s.replace('Dangermond_', '') for s in unique_stations]
    station_mae = []
    station_bias = []
    for station in unique_stations:
        mask = station_names_daily == station
        s_preds = pred_daily[mask]
        s_actual = actual_daily[mask]
        station_mae.append(np.mean(np.abs(s_preds - s_actual)))
        station_bias.append(np.mean(s_preds - s_actual))

    x_pos = np.arange(len(unique_stations))
    colors = ['salmon' if b > 0 else 'skyblue' for b in station_bias]
    ax.bar(x_pos, station_bias, color=colors, edgecolor='gray')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(station_short, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Bias (pred - actual, mm/day)')
    ax.set_title('Daily Test — Per-Station Bias')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = out / 'test_daily_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if run_dir:
        plt.savefig(Path(run_dir) / 'test_daily_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved test evaluation plot to: {save_path}")

    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Stack CNN precipitation model")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--run-dir', default=None, help='Run directory (auto-finds checkpoint and saves results there)')
    parser.add_argument('--pickle', default=DEFAULT_PICKLE)
    parser.add_argument('--dem', default=DEFAULT_DEM)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        pickle_path=args.pickle,
        dem_path=args.dem,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
    )
