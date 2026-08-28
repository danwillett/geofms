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

from models.gfm.model import UpscalingPrecipTask
from models.gfm.dataset import RadarDEMDataModule, create_heavy_rain_sampler


# ── DEFAULT CONFIG ──────────────────────────────────────────────────────────────
DEFAULT_PICKLE   = "dataset/outputs/3d/radar_gauge_dataset_subml_daygroup_offsets_9500.pkl"
DEFAULT_CKPT_DIR = "models/checkpoints/terramind_dualpol"
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
    model = UpscalingPrecipTask.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    print("✓ Model loaded successfully!")
    return model, device


def _center_predict(model, batch, device):
    """Upscale a batch (GPU) and return (pred_center, target_center) raw arrays.

    Mirrors UpscalingPrecipTask's step-time upscaling since we call model.model()
    directly (bypassing the LightningModule step hooks).
    """
    batch_up  = UpscalingPrecipTask._upscale_batch(batch)
    image_gpu = {k: v.to(device) for k, v in batch_up["image"].items()}
    pred      = model.model(image_gpu).output
    if pred.dim() == 4:           # (B, 1, H, W) -> (B, H, W)
        pred = pred[:, 0]
    out    = batch["mask"].shape[-1]
    center = out // 2
    pred_center   = pred[:, center, center].cpu().numpy()
    target_center = batch["mask"][:, center, center].numpy()
    return pred_center, target_center


def estimate_jensen_correction(pred_log, target_log):
    """Jensen bias term for log1p targets: c = Var(y_log - ŷ_log) / 2.

    Applied at inference as expm1(ŷ_log + c) to approximate E[mm | X] when the
    model predicts a point estimate in log1p space. Estimated once on validation.
    """
    pred_log = np.asarray(pred_log, dtype=np.float64)
    target_log = np.asarray(target_log, dtype=np.float64)
    residuals = target_log - pred_log
    return 0.5 * float(np.var(residuals))


def log1p_to_mm(pred_log, correction=0.0):
    """Map log1p-space predictions to mm/hr with optional Jensen correction."""
    return np.maximum(np.expm1(np.asarray(pred_log, dtype=np.float64) + correction), 0.0)


def run_inference(model, datamodule, device) -> tuple:
    """
    Run inference over the validation set.

    Returns the RAW model-space predictions/targets (log1p space when the model
    was trained with log_target). The caller handles the mm conversion and the
    Jensen correction so the correction can be estimated once and reused for the
    daily-test aggregation.
    """
    all_preds, all_targets = [], []

    datamodule.setup()
    print("Running inference on validation set...")

    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            pred_center, target_center = _center_predict(model, batch, device)
            all_preds.extend(pred_center.tolist())
            all_targets.extend(target_center.tolist())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)

    valid = targets > -9000
    print(f"✓ Collected {valid.sum()} valid samples")
    return preds[valid], targets[valid]


def compute_metrics(preds_mm: np.ndarray, targets_mm: np.ndarray) -> dict:
    """Compute R², MAE, RMSE in real mm/hr space."""
    mae  = np.mean(np.abs(targets_mm - preds_mm))
    rmse = np.sqrt(np.mean((targets_mm - preds_mm) ** 2))
    ss_res = np.sum((targets_mm - preds_mm) ** 2)
    ss_tot = np.sum((targets_mm - targets_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(r2=r2, mae=mae, rmse=rmse)


def print_report(preds_mm, targets_mm, metrics, metrics_jensen=None, jensen_c=None):
    """Print detailed evaluation report to stdout."""
    print(f"\n{'='*60}")
    print(f"📊 TERRAMIND MODEL EVALUATION")
    print(f"{'='*60}")
    if metrics_jensen is not None:
        print(f"\n  R² (mm, naive expm1):      {metrics['r2']:.3f}")
        print(f"  R² (mm, Jensen-corrected): {metrics_jensen['r2']:.3f}")
        print(f"  Jensen c (= σ²_ε/2):       {jensen_c:.6f}")
        print(f"  MAE (naive):    {metrics['mae']:.3f} mm/hr")
        print(f"  MAE (Jensen):   {metrics_jensen['mae']:.3f} mm/hr")
        print(f"  RMSE (naive):   {metrics['rmse']:.3f} mm/hr")
        print(f"  RMSE (Jensen):  {metrics_jensen['rmse']:.3f} mm/hr")
    else:
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
    plt.close()


def plot_jensen_comparison(targets, preds_naive, preds_jensen,
                           metrics_naive, metrics_jensen, jensen_c,
                           output_dir, filename, unit="mm/hr"):
    """Side-by-side naive-expm1 vs Jensen-corrected scatter (log-space runs only).

    The main evaluation figures always chart the naive predictions so that
    log- and raw-space runs are directly comparable; this extra figure shows
    what the Jensen correction does for a given log-space run.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_val = max(float(targets.max()), float(preds_naive.max()), float(preds_jensen.max()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    panels = [
        (axes[0], preds_naive,  metrics_naive,  "Naive expm1"),
        (axes[1], preds_jensen, metrics_jensen, f"Jensen-corrected (c={jensen_c:.4f})"),
    ]
    for ax, preds, m, label in panels:
        ax.scatter(targets, preds, alpha=0.3, s=20, c="steelblue")
        ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect")
        ax.set_xlabel(f"Actual ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(f"{label}\nR²={m['r2']:.3f}, MAE={m['mae']:.3f} {unit}")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Jensen comparison plot to: {save_path}")


def evaluate(
    checkpoint_path: str = None,
    checkpoint_dir:  str = DEFAULT_CKPT_DIR,
    pickle_path:     str = DEFAULT_PICKLE,
    output_dir:      str = DEFAULT_OUTPUT,
    output_size:     int = 9,
    log_target:      bool = True,
    fields=None,
    use_feature_masks: bool = True,
    filter_mode:     str = 'blunt',
    dem_norm:        str = 'minmax',
    run_dir:         str = None,
):
    ckpt  = find_checkpoint(checkpoint_path, checkpoint_dir)
    model, device = load_model(ckpt)

    # Prefer saving figures into the run directory (mirrors stack/unet).
    fig_dir = run_dir if run_dir else output_dir

    datamodule = RadarDEMDataModule(
        pickle_path=pickle_path,
        output_size=output_size,
        fields=fields,
        use_feature_masks=use_feature_masks,
        log_target=log_target,
        dem_norm=dem_norm,
        weight_sampler=create_heavy_rain_sampler,
        batch_size=32,
        filter_mode=filter_mode,
    )

    preds_raw, targets_raw = run_inference(model, datamodule, device)

    # Convert to mm, estimating the Jensen retransformation correction on the
    # validation residuals (log space) so it can be reused for the daily test.
    if log_target:
        jensen_c        = estimate_jensen_correction(preds_raw, targets_raw)
        preds_mm        = log1p_to_mm(preds_raw, 0.0)
        targets_mm      = np.expm1(targets_raw)
        preds_mm_jensen = log1p_to_mm(preds_raw, jensen_c)
        metrics         = compute_metrics(preds_mm, targets_mm)
        metrics_jensen  = compute_metrics(preds_mm_jensen, targets_mm)
    else:
        jensen_c        = 0.0
        preds_mm        = np.maximum(preds_raw, 0.0)
        targets_mm      = targets_raw
        metrics         = compute_metrics(preds_mm, targets_mm)
        metrics_jensen  = None

    print_report(preds_mm, targets_mm, metrics, metrics_jensen=metrics_jensen,
                 jensen_c=jensen_c if log_target else None)
    # Main chart always uses NAIVE (native) predictions so log- and raw-space
    # runs are comparable across experiments.
    plot_evaluation(preds_mm, targets_mm, metrics, fig_dir)
    # Extra side-by-side naive vs Jensen chart for log-space runs only.
    if log_target and metrics_jensen is not None:
        plot_jensen_comparison(targets_mm, preds_mm, preds_mm_jensen,
                               metrics, metrics_jensen, jensen_c,
                               fig_dir, "terramind_jensen_comparison.png", unit="mm/hr")

    # Test evaluation (daily gauges)
    test_summary = evaluate_test(
        model, device, pickle_path, fig_dir,
        output_size=output_size, log_target=log_target,
        fields=fields, use_feature_masks=use_feature_masks,
        dem_norm=dem_norm, jensen_c=jensen_c)

    # ── Persist val + test metrics into the run's config.json so each run keeps
    #    its own numbers (the sweep summary otherwise gets overwritten). ──
    val_summary = {
        'r2_naive':   float(metrics['r2']),
        'mae_naive':  float(metrics['mae']),
        'rmse_naive': float(metrics['rmse']),
        'r2_jensen':  (float(metrics_jensen['r2']) if metrics_jensen else None),
        'mae_jensen': (float(metrics_jensen['mae']) if metrics_jensen else None),
        'jensen_c':   (float(jensen_c) if log_target else None),
    }
    final_dir = run_dir if run_dir else str(Path(ckpt).parent)
    _persist_metrics(final_dir, val_summary, test_summary)

    # Headline = naive metrics, so all runs (incl. raw-space) share one basis.
    # Both naive and Jensen are still printed and charted above.
    return metrics, test_summary, final_dir


def _persist_metrics(run_dir, val_summary, test_summary):
    """Append val/test metric summaries to the run's config.json if present."""
    import json
    cfg_path = Path(run_dir) / 'config.json'
    if not cfg_path.exists():
        return
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        cfg['val_metrics']  = val_summary
        cfg['test_metrics'] = test_summary
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"  ✓ Wrote val/test metrics into {cfg_path}")
    except Exception as e:
        print(f"  ⚠ Could not persist metrics to {cfg_path}: {e}")


def evaluate_test(model, device, pickle_path, output_dir,
                  output_size=9, log_target=True, fields=None, use_feature_masks=True,
                  dem_norm='minmax', jensen_c=0.0):
    """Evaluate GFM model on daily cumulative gauge test set.

    Returns a summary dict of daily test metrics (naive + Jensen) or None.
    """
    import pickle as pkl
    from torch.utils.data import DataLoader

    with open(pickle_path, 'rb') as f:
        dataset = pkl.load(f)

    test_samples = dataset.get('test', [])
    if not test_samples:
        print("\n  No test samples in pickle — skipping daily gauge evaluation.")
        return None

    print(f"\n{'='*60}")
    print("  TEST EVALUATION (daily cumulative gauges)")
    print(f"{'='*60}")
    print(f"  Hourly test samples: {len(test_samples)}")

    from models.gfm.dataset import RadarDEMDataset, PICKLE_FIELD_ORDER
    from models.unet.train import filter_bad_samples
    meta         = dataset.get('metadata', {})
    field_order  = list(meta.get('fields', PICKLE_FIELD_ORDER))
    patch_size_m = meta.get('patch_size_m', 9500)
    test_samples = filter_bad_samples(test_samples)

    test_ds = RadarDEMDataset(
        test_samples,
        field_order=field_order,
        dem_path='dem/preserve_dem_10m_utm.tif',
        patch_size_m=patch_size_m,
        output_size=output_size,
        augment=False,
        use_feature_masks=use_feature_masks,
        fields=fields,
        log_target=log_target,
        dem_norm=dem_norm,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    hourly_preds = []
    hourly_preds_jensen = []
    hourly_meta = []
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            pred_center, _ = _center_predict(model, batch, device)

            # Convert from log-space to mm if trained in log space
            if log_target:
                preds_mm_batch = log1p_to_mm(pred_center, 0.0)
                preds_mm_jensen_batch = log1p_to_mm(pred_center, jensen_c)
            else:
                preds_mm_batch = np.maximum(pred_center, 0.0)
                preds_mm_jensen_batch = preds_mm_batch

            for i in range(preds_mm_batch.shape[0]):
                hourly_preds.append(float(preds_mm_batch[i]))
                hourly_preds_jensen.append(float(preds_mm_jensen_batch[i]))
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
    daily_groups = defaultdict(lambda: {'pred_sum': 0.0, 'pred_jensen_sum': 0.0,
                                         'count': 0, 'actual': 0.0, 'station_name': ''})

    for pred, pred_j, meta in zip(hourly_preds, hourly_preds_jensen, hourly_meta):
        key = (meta['date'], meta['station_id'])
        daily_groups[key]['pred_sum'] += pred
        daily_groups[key]['pred_jensen_sum'] += pred_j
        daily_groups[key]['count'] += 1
        daily_groups[key]['actual'] = meta['daily_precip_mm']
        daily_groups[key]['station_name'] = meta['station_name']

    pred_daily = np.array([v['pred_sum'] for v in daily_groups.values()])
    pred_daily_jensen = np.array([v['pred_jensen_sum'] for v in daily_groups.values()])
    actual_daily = np.array([v['actual'] for v in daily_groups.values()])
    hours_per_day = np.array([v['count'] for v in daily_groups.values()])
    station_names_daily = [v['station_name'] for v in daily_groups.values()]

    valid = hours_per_day >= 18
    pred_daily = pred_daily[valid]
    pred_daily_jensen = pred_daily_jensen[valid]
    actual_daily = actual_daily[valid]
    station_names_daily = np.array(station_names_daily)[valid]

    if len(pred_daily) == 0:
        print("  No valid day-station groups with >=18 hours. Skipping.")
        return None

    test_metrics = compute_metrics(pred_daily, actual_daily)
    test_metrics_jensen = compute_metrics(pred_daily_jensen, actual_daily) if log_target else None

    print(f"\n  Day-station groups: {len(pred_daily)} (≥18 hrs coverage)")
    print(f"  Avg hours/day:     {hours_per_day[valid].mean():.1f}  (max {hours_per_day[valid].max()})")
    if test_metrics_jensen is not None:
        print(f"\n  R² (mm/day, naive expm1):      {test_metrics['r2']:.3f}")
        print(f"  R² (mm/day, Jensen-corrected): {test_metrics_jensen['r2']:.3f}")
        print(f"  MAE (naive):   {test_metrics['mae']:.3f} mm/day")
        print(f"  MAE (Jensen):  {test_metrics_jensen['mae']:.3f} mm/day")
    else:
        print(f"\n  R²:         {test_metrics['r2']:.3f}")
        print(f"  MAE:        {test_metrics['mae']:.3f} mm/day")
        print(f"  RMSE:       {test_metrics['rmse']:.3f} mm/day")
    print(f"  Pred range: {pred_daily.min():.2f} – {pred_daily.max():.2f} mm/day")
    print(f"  Actual range: {actual_daily.min():.2f} – {actual_daily.max():.2f} mm/day")

    # Extra side-by-side naive vs Jensen chart for log-space runs only.
    if test_metrics_jensen is not None:
        plot_jensen_comparison(actual_daily, pred_daily, pred_daily_jensen,
                               test_metrics, test_metrics_jensen, jensen_c,
                               output_dir, "test_daily_jensen_comparison.png",
                               unit="mm/day")

    # Main daily plot uses NAIVE predictions (consistent with validation chart).
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

    return {
        'r2_naive':   float(test_metrics['r2']),
        'mae_naive':  float(test_metrics['mae']),
        'rmse_naive': float(test_metrics['rmse']),
        'r2_jensen':  (float(test_metrics_jensen['r2']) if test_metrics_jensen else None),
        'mae_jensen': (float(test_metrics_jensen['mae']) if test_metrics_jensen else None),
        'n_groups':   int(len(pred_daily)),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TerraMind precipitation model")
    parser.add_argument("--checkpoint",     default=None,            help="Path to a specific .ckpt file")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CKPT_DIR, help="Directory to search for best checkpoint")
    parser.add_argument("--pickle",         default=DEFAULT_PICKLE,   help="Path to dataset pickle")
    parser.add_argument("--output-dir",     default=DEFAULT_OUTPUT,   help="Directory for saved figures")
    parser.add_argument("--output-size",    type=int, default=9,      help="Radar crop size (must match training run)")
    parser.add_argument("--no-log",         action="store_true",      help="Model was trained in raw mm-space")
    parser.add_argument("--dem-norm",       default='minmax', choices=['minmax', 'terramind'],
                        help="DEM normalization (must match the training run)")
    parser.add_argument("--run-dir",        default=None,             help="Run directory to save figures into")
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        checkpoint_dir  = args.checkpoint_dir,
        pickle_path     = args.pickle,
        output_dir      = args.output_dir,
        output_size     = args.output_size,
        log_target      = not args.no_log,
        dem_norm        = args.dem_norm,
        run_dir         = args.run_dir,
    )