"""
evaluate.py — Evaluate a trained 10-minute single-scan precipitation model.

Run from the project root:
    python -m models.stack_10min.evaluate --run-dir models/checkpoints/stack_10min/2026-05-26_...
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from models.stack_10min.model import PrecipModel10min
from models.stack_10min.dataset import RadarGaugeDataset10min
from models.stack_10min.train import filter_nan_radar, filter_bad_samples, filter_biased_extremes, filter_stations, filter_suspect_gauges

DEFAULT_PICKLE = 'dataset/outputs/10min/radar_gauge_10min.pkl'
DEFAULT_DEM = 'dem/preserve_dem_10m_utm.tif'
DEFAULT_CKPT_DIR = 'models/checkpoints/stack_10min'
DEFAULT_OUTPUT = 'evaluation_figures/stack_10min'


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

    model = PrecipModel10min(
        latent_dim=cfg.get('latent_dim', 256),
        dropout_rate=cfg.get('dropout_rate', 0.2),
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
            target = batch['target']
            station_names = batch.get('station_name', [])

            pred = model(radar).cpu()

            all_preds.extend(pred.numpy().tolist())
            all_targets.extend(target.numpy().tolist())
            all_station_names.extend(station_names)

    preds_mm = np.array(all_preds)
    targets_mm = np.array(all_targets)
    station_names = np.array(all_station_names)

    # Clip predictions to non-negative
    preds_mm = np.maximum(preds_mm, 0.0)

    print(f"✓ Collected {len(preds_mm)} samples")
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
    print("  10-MIN MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"\n  R²:         {metrics['r2']:.3f}")
    print(f"  MAE:        {metrics['mae']:.3f} mm/10min")
    print(f"  RMSE:       {metrics['rmse']:.3f} mm/10min")
    print(f"  Pred max:   {preds_mm.max():.2f} mm/10min")
    print(f"  Pred range: [{preds_mm.min():.3f}, {preds_mm.max():.3f}]")

    # Heavy rain for 10-min: >1.5 mm/10min ≈ 9 mm/hr
    heavy = targets_mm > 1.5
    if heavy.sum():
        h_mae = np.mean(np.abs(targets_mm[heavy] - preds_mm[heavy]))
        h_bias = np.mean(targets_mm[heavy] - preds_mm[heavy])
        direction = "underpredicting" if h_bias > 0 else "overpredicting"
        print(f"\n  Heavy Rain (actual >1.5 mm/10min):")
        print(f"    Samples:   {heavy.sum()}")
        print(f"    MAE:       {h_mae:.3f} mm/10min")
        print(f"    Bias:      {h_bias:.3f} mm/10min ({direction})")


def plot_scatter(preds_mm, targets_mm, output_dir, metrics):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(targets_mm, preds_mm, alpha=0.2, s=8, c='steelblue')

    max_val = max(targets_mm.max(), preds_mm.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='1:1')
    ax.set_xlabel('Actual (mm/10min)', fontsize=12)
    ax.set_ylabel('Predicted (mm/10min)', fontsize=12)
    ax.set_title(f'10-min Precipitation — R²={metrics["r2"]:.3f}, MAE={metrics["mae"]:.3f}', fontsize=13)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scatter_path = Path(output_dir) / 'scatter_10min.png'
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"  Scatter plot → {scatter_path}")


def plot_station_bias(preds_mm, targets_mm, station_names, output_dir):
    unique_stations = sorted(set(station_names))
    station_biases = []
    station_counts = []

    for station in unique_stations:
        mask = station_names == station
        if mask.sum() < 5:
            continue
        bias = np.mean(preds_mm[mask] - targets_mm[mask])
        station_biases.append(bias)
        station_counts.append(mask.sum())

    if not station_biases:
        return

    # Sort by bias
    sort_idx = np.argsort(station_biases)
    sorted_stations = [unique_stations[i] for i in sort_idx if station_counts[sort_idx.tolist().index(i)] >= 5]

    biases_sorted = np.array(station_biases)[sort_idx]
    short_names = [s.replace('Dangermond_', '') for s in sorted_stations]

    fig, ax = plt.subplots(figsize=(10, max(6, len(short_names) * 0.35)))
    colors = ['steelblue' if b < 0 else 'coral' for b in biases_sorted]
    ax.barh(range(len(short_names)), biases_sorted, color=colors, alpha=0.8)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Bias (pred - actual) mm/10min', fontsize=11)
    ax.set_title('Station Prediction Bias (10-min)', fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bias_path = Path(output_dir) / 'station_bias_10min.png'
    fig.tight_layout()
    fig.savefig(bias_path, dpi=150)
    plt.close(fig)
    print(f"  Station bias → {bias_path}")


def write_eval_results(run_dir, preds_mm, targets_mm, metrics):
    if not run_dir:
        return
    results_path = Path(run_dir) / 'results.txt'
    lines = [
        "=" * 60,
        "  10-MIN MODEL EVALUATION RESULTS",
        "=" * 60,
        "",
        f"  R²:            {metrics['r2']:.4f}",
        f"  MAE:           {metrics['mae']:.3f} mm/10min",
        f"  RMSE:          {metrics['rmse']:.3f} mm/10min",
        f"  Pred max:      {preds_mm.max():.2f} mm/10min",
        f"  Pred min:      {preds_mm.min():.3f} mm/10min",
        f"  # Samples:     {len(preds_mm)}",
        "",
    ]

    heavy = targets_mm > 1.5
    if heavy.sum():
        h_mae = np.mean(np.abs(targets_mm[heavy] - preds_mm[heavy]))
        h_bias = np.mean(targets_mm[heavy] - preds_mm[heavy])
        direction = "underpredicting" if h_bias > 0 else "overpredicting"
        lines.append("  Heavy Rain (actual >1.5 mm/10min):")
        lines.append(f"    Samples:   {int(heavy.sum())}")
        lines.append(f"    MAE:       {h_mae:.3f} mm/10min")
        lines.append(f"    Bias:      {h_bias:.3f} mm/10min ({direction})")

    with open(results_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ Results written to: {results_path}")


def evaluate(checkpoint_path=None, checkpoint_dir=None, pickle_path=None,
             dem_path=None, output_dir=None, run_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pickle_path = pickle_path or DEFAULT_PICKLE
    dem_path = dem_path or DEFAULT_DEM
    output_dir = output_dir or DEFAULT_OUTPUT

    ckpt_path = find_checkpoint(checkpoint_path, checkpoint_dir, run_dir)
    model, cfg = load_model(ckpt_path, device)

    # Determine output directory
    if run_dir:
        output_dir = str(Path(run_dir))
    elif not output_dir:
        output_dir = str(Path(ckpt_path).parent)

    val_ds = RadarGaugeDataset10min(pickle_path, dem_path=dem_path, split='val')
    val_ds.samples = filter_nan_radar(val_ds.samples)
    val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_biased_extremes(val_ds.samples)
    val_ds.samples = filter_suspect_gauges(val_ds.samples)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    preds_mm, targets_mm, station_names = run_inference(model, val_loader, device)
    metrics = compute_metrics(preds_mm, targets_mm)

    print_report(preds_mm, targets_mm, metrics)
    plot_scatter(preds_mm, targets_mm, output_dir, metrics)
    plot_station_bias(preds_mm, targets_mm, station_names, output_dir)
    write_eval_results(run_dir, preds_mm, targets_mm, metrics)

    return metrics, run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate 10-min single-scan model")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--run-dir', default=None)
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
