"""
evaluate.py — Evaluate a trained TerraMind precipitation model.

Run from the project root:
    python -m models.gfm.evaluate --checkpoint checkpoints/terramind_dualpol/best-epoch=XX-val_loss=0.0000.ckpt

Or point at a checkpoint directory and it will find the best one automatically:
    python -m models.gfm.evaluate --checkpoint-dir checkpoints/terramind_dualpol
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from terratorch.tasks import PixelwiseRegressionTask

from models.gfm.dataset import RadarDEMDataModule, create_heavy_rain_sampler


# ── DEFAULT CONFIG ──────────────────────────────────────────────────────────────
DEFAULT_PICKLE   = "deep_learning/radar_gauge_dataset_9x9.pkl"
DEFAULT_CKPT_DIR = "checkpoints/terramind_dualpol"
DEFAULT_OUTPUT   = "evaluation_figures"
# ───────────────────────────────────────────────────────────────────────────────


def find_checkpoint(checkpoint_path: str = None, checkpoint_dir: str = None) -> str:
    """Resolve the best available checkpoint path."""
    if checkpoint_path and Path(checkpoint_path).exists():
        return checkpoint_path

    ckpt_dir = Path(checkpoint_dir or DEFAULT_CKPT_DIR)
    best_files = sorted(ckpt_dir.glob("best-*.ckpt"), key=os.path.getmtime, reverse=True)
    if best_files:
        return str(best_files[0])

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def load_model(checkpoint_path: str) -> tuple:
    """Load model from checkpoint and return (model, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {checkpoint_path}")
    model = PixelwiseRegressionTask.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    print("✓ Model loaded successfully!")
    return model, device


def run_inference(model, datamodule, device) -> tuple:
    """
    Run inference over the validation set.

    Returns:
        preds_mm, targets_mm : np.ndarray
            Predictions and targets converted from log space to real mm/hr.
    """
    all_preds_log   = []
    all_targets_log = []

    datamodule.setup()
    print("Running inference on validation set...")

    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            image_gpu  = {k: v.to(device) for k, v in batch["image"].items()}
            model_out  = model.model(image_gpu)
            pred       = model_out.output                        # (B, 1, H, W)
            pred_center   = pred[:, 2, 2].cpu().numpy()      # centre pixel
            target_center = batch["mask"][:, 2, 2].numpy()

            all_preds_log.extend(pred_center.tolist())
            all_targets_log.extend(target_center.tolist())

    preds_log   = np.array(all_preds_log)
    targets_log = np.array(all_targets_log)

    # Remove ignore-index values
    valid = targets_log > -9000
    preds_mm   = np.expm1(preds_log[valid])
    targets_mm = np.expm1(targets_log[valid])

    print(f"✓ Collected {valid.sum()} valid samples")
    return preds_mm, targets_mm


def compute_metrics(preds_mm: np.ndarray, targets_mm: np.ndarray) -> dict:
    """Compute R², MAE, RMSE in real mm/hr space."""
    mae  = np.mean(np.abs(targets_mm - preds_mm))
    rmse = np.sqrt(np.mean((targets_mm - preds_mm) ** 2))
    ss_res = np.sum((targets_mm - preds_mm) ** 2)
    ss_tot = np.sum((targets_mm - targets_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(r2=r2, mae=mae, rmse=rmse)


def print_report(preds_mm, targets_mm, metrics):
    """Print detailed evaluation report to stdout."""
    print(f"\n{'='*60}")
    print(f"📊 TERRAMIND MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"\n  R²:         {metrics['r2']:.3f}")
    print(f"  MAE:        {metrics['mae']:.3f} mm/hr")
    print(f"  RMSE:       {metrics['rmse']:.3f} mm/hr")
    print(f"  Pred max:   {preds_mm.max():.2f} mm/hr")
    print(f"  # Pred >5mm: {int(np.sum(preds_mm > 5))}")

    print(f"\n🎯 Heavy Rain Performance (actual >5mm):")
    heavy = targets_mm > 5
    if heavy.sum():
        h_mae  = np.mean(np.abs(targets_mm[heavy] - preds_mm[heavy]))
        h_bias = np.mean(targets_mm[heavy] - preds_mm[heavy])
        direction = "underpredicting" if h_bias > 0 else "overpredicting"
        print(f"  Samples:       {heavy.sum()}")
        print(f"  MAE:           {h_mae:.3f} mm/hr")
        print(f"  Bias:          {h_bias:.3f} mm/hr ({direction})")
        print(f"  Pred >5mm:     {int(np.sum(preds_mm[heavy] > 5))} / {heavy.sum()}")
    else:
        print("  No heavy-rain samples in validation set.")


def plot_evaluation(preds_mm, targets_mm, metrics, output_dir: str):
    """Save 4-panel evaluation figure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── 1: Scatter ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(targets_mm, preds_mm, alpha=0.3, s=20, c="steelblue")
    max_val = max(targets_mm.max(), preds_mm.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual (mm/hr)")
    ax.set_ylabel("Predicted (mm/hr)")
    ax.set_title(
        f"Predicted vs Actual\nR²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f} mm/hr"
    )
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 2: MAE by category ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    categories = ["Dry\n(0-0.1)", "Light\n(0.1-1)", "Moderate\n(1-5)", "Heavy\n(5-10)", "Extreme\n(>10)"]
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

    colors = ["lightblue", "skyblue", "steelblue", "royalblue", "darkblue"]
    bars = ax.bar(range(len(categories)), maes, color=colors)
    for bar, n, b in zip(bars, counts, biases):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"n={n}\nbias={b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("MAE (mm/hr)")
    ax.set_title("Error by Precipitation Category")
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3: Residuals ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    residuals = targets_mm - preds_mm
    ax.scatter(preds_mm, residuals, alpha=0.3, s=20, c="coral")
    ax.axhline(0, color="r", lw=2, linestyle="--")
    ax.set_xlabel("Predicted (mm/hr)")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot\n(above zero = underpredicting)")
    ax.grid(True, alpha=0.3)

    # ── 4: Distribution ────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(targets_mm, bins=50, alpha=0.5, label="Actual",    color="green", density=True)
    ax.hist(preds_mm,   bins=50, alpha=0.5, label="Predicted", color="blue",  density=True)
    ax.set_xlabel("Precipitation (mm/hr)")
    ax.set_ylabel("Density")
    ax.set_title("Precipitation Distribution")
    ax.legend()
    ax.set_xlim(0, max(targets_mm.max(), preds_mm.max()))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = out / "terramind_evaluation.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved evaluation plot to: {save_path}")
    plt.show()


def evaluate(
    checkpoint_path: str = None,
    checkpoint_dir:  str = DEFAULT_CKPT_DIR,
    pickle_path:     str = DEFAULT_PICKLE,
    output_dir:      str = DEFAULT_OUTPUT,
):
    ckpt  = find_checkpoint(checkpoint_path, checkpoint_dir)
    model, device = load_model(ckpt)

    datamodule = RadarDEMDataModule(
        pickle_path=pickle_path,
        weight_sampler=create_heavy_rain_sampler,
        batch_size=32,
    )

    preds_mm, targets_mm = run_inference(model, datamodule, device)
    metrics = compute_metrics(preds_mm, targets_mm)
    print_report(preds_mm, targets_mm, metrics)
    plot_evaluation(preds_mm, targets_mm, metrics, output_dir)

    # Test evaluation (daily gauges)
    evaluate_test(model, device, pickle_path, output_dir)

    return metrics


def evaluate_test(model, device, pickle_path, output_dir):
    """Evaluate GFM model on daily cumulative gauge test set."""
    import pickle as pkl
    from torch.utils.data import DataLoader

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

    from models.gfm.dataset import RadarDEMDataset, filter_bad_samples
    test_samples = filter_bad_samples(test_samples)

    test_ds = RadarDEMDataset(test_samples, dem_path='dem/preserve_dem_10m_utm.tif', patch_size_m=4620, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    hourly_preds = []
    hourly_meta = []
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            image_gpu = {k: v.to(device) for k, v in batch["image"].items()}
            model_out = model.model(image_gpu)
            pred = model_out.output
            pred_center = pred[:, 2, 2].cpu().numpy()

            # Convert from log-space to mm
            preds_mm_batch = np.expm1(pred_center)

            for i in range(preds_mm_batch.shape[0]):
                hourly_preds.append(max(0.0, float(preds_mm_batch[i])))
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

    valid = hours_per_day >= 18
    pred_daily = pred_daily[valid]
    actual_daily = actual_daily[valid]
    station_names_daily = np.array(station_names_daily)[valid]

    if len(pred_daily) == 0:
        print("  No valid day-station groups with >=18 hours. Skipping.")
        return

    test_metrics = compute_metrics(pred_daily, actual_daily)

    print(f"\n  Day-station groups: {len(pred_daily)} (≥18 hrs coverage)")
    print(f"  Avg hours/day:     {hours_per_day[valid].mean():.1f}")
    print(f"\n  R²:         {test_metrics['r2']:.3f}")
    print(f"  MAE:        {test_metrics['mae']:.3f} mm/day")
    print(f"  RMSE:       {test_metrics['rmse']:.3f} mm/day")
    print(f"  Pred range: {pred_daily.min():.2f} – {pred_daily.max():.2f} mm/day")
    print(f"  Actual range: {actual_daily.min():.2f} – {actual_daily.max():.2f} mm/day")

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
    station_bias = []
    for station in unique_stations:
        mask = station_names_daily == station
        s_preds = pred_daily[mask]
        s_actual = actual_daily[mask]
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
    plt.close()
    print(f"  ✓ Saved test evaluation plot to: {save_path}")

    return test_metrics


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TerraMind precipitation model")
    parser.add_argument("--checkpoint",     default=None,            help="Path to a specific .ckpt file")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CKPT_DIR, help="Directory to search for best checkpoint")
    parser.add_argument("--pickle",         default=DEFAULT_PICKLE,   help="Path to dataset pickle")
    parser.add_argument("--output-dir",     default=DEFAULT_OUTPUT,   help="Directory for saved figures")
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        checkpoint_dir  = args.checkpoint_dir,
        pickle_path     = args.pickle,
        output_dir      = args.output_dir,
    )