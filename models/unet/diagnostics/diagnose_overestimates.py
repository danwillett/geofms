"""
diagnose_overestimates.py — Analyze samples where the U-Net predicts >8mm
but actual precipitation is <5mm.

Investigates whether these are model errors, gauge malfunctions, or
legitimate radar signals that didn't reach the gauge.

Run from project root:
    python -m models.unet.diagnostics.diagnose_overestimates --run-dir models/checkpoints/unet_dualpol/<run_name>
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from torch.utils.data import DataLoader

from models.unet.evaluate import load_model, find_checkpoint
from models.unet.dataset import (
    RadarGaugeDataset, resolve_fields, compute_n_input_channels,
    PICKLE_FIELD_ORDER, FIELD_NORMS
)
from models.unet.train import (
    filter_nan_radar, filter_biased_extremes, filter_bad_samples,
    filter_suspect_station_days, filter_radar_unsupported
)


def extract_radar_features(sample, fields):
    """Extract raw radar features at the center pixel for a sample."""
    radar_patch = sample['radar_patch']  # (12, N_fields, H, W)
    center_y = radar_patch.shape[2] // 2
    center_x = radar_patch.shape[3] // 2

    features = {}

    # Legacy fields + new bright-band discriminators (rhohv_min, melting layer,
    # low-level RhoHV/ZDR) that the model now receives as inputs.
    for field_name in ['reflectivity', 'echo_top_height', 'max_z_height',
                       'vil', 'low_level_ref', 'column_depth_fraction',
                       'rhohv_min', 'melting_layer_height',
                       'low_level_rhohv', 'low_level_zdr',
                       'vertical_reflectivity_gradient']:
        if field_name in PICKLE_FIELD_ORDER:
            idx = PICKLE_FIELD_ORDER.index(field_name)
            if idx < radar_patch.shape[1]:
                center_vals = radar_patch[:, idx, center_y, center_x]
                valid = center_vals[center_vals != -9999.0]
                valid = valid[~np.isnan(valid)]
                features[f'{field_name}_center_max'] = np.nanmax(valid) if len(valid) > 0 else np.nan
                features[f'{field_name}_center_mean'] = np.nanmean(valid) if len(valid) > 0 else np.nan

                patch_vals = radar_patch[:, idx, :, :]
                valid_patch = patch_vals[patch_vals != -9999.0]
                valid_patch = valid_patch[~np.isnan(valid_patch)]
                features[f'{field_name}_patch_max'] = np.nanmax(valid_patch) if len(valid_patch) > 0 else np.nan

    return features


def run_diagnostic(run_dir, pred_threshold=8.0, actual_threshold=5.0):
    """Main diagnostic: find and analyze severe overpredictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = find_checkpoint(run_dir=run_dir)
    model, cfg = load_model(checkpoint_path, device)

    pickle_path = cfg.get('pickle_path', 'dataset/outputs/3d/radar_gauge_dataset_vertical_9500.pkl')
    dem_path = cfg.get('dem_path', 'dem/preserve_dem_10m_utm.tif')
    fields = resolve_fields(cfg.get('fields'))
    log_target = cfg.get('log_target', True)

    ds_kwargs = dict(
        dem_path=dem_path,
        fields=fields,
        use_dem=cfg.get('use_dem', True),
        use_mask=cfg.get('use_mask', True),
        use_temporal_pos=cfg.get('use_temporal_pos', True),
        use_feature_masks=cfg.get('use_feature_masks', False),
        log_target=log_target,
    )

    val_ds = RadarGaugeDataset(pickle_path, split='val', augment=False, **ds_kwargs)
    val_ds.samples = filter_nan_radar(val_ds.samples)

    filter_mode = cfg.get('filter_mode', 'blunt')
    if filter_mode == 'radar':
        val_ds.samples = filter_radar_unsupported(val_ds.samples)
    else:
        val_ds.samples = filter_biased_extremes(val_ds.samples)
        val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_suspect_station_days(val_ds.samples)

    print(f"\n  Validation samples: {len(val_ds.samples)}")
    print(f"  Overprediction criteria: pred >= {pred_threshold}mm AND actual < {actual_threshold}mm")

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Run inference and collect per-sample data
    all_preds = []
    all_actuals = []
    all_indices = []

    with torch.no_grad():
        idx_offset = 0
        for batch in val_loader:
            radar = batch['radar'].to(device)
            gauge_pixel = batch['gauge_pixel']

            pred_map = model(radar).cpu()

            if pred_map.dim() == 1:
                pred_at_gauge = pred_map
            else:
                batch_idx = torch.arange(pred_map.shape[0])
                if isinstance(gauge_pixel, torch.Tensor) and gauge_pixel.dim() == 2:
                    y = gauge_pixel[:, 0].long()
                    x = gauge_pixel[:, 1].long()
                    pred_at_gauge = pred_map[batch_idx, y, x]
                elif isinstance(gauge_pixel, (tuple, list)):
                    y, x = gauge_pixel
                    if isinstance(y, torch.Tensor):
                        pred_at_gauge = pred_map[batch_idx, y, x]
                    else:
                        pred_at_gauge = pred_map[:, y, x]
                else:
                    center = pred_map.shape[-1] // 2
                    pred_at_gauge = pred_map[:, center, center]

            preds = pred_at_gauge.numpy()
            targets = batch['target'].numpy()

            if log_target:
                preds = np.expm1(preds)
                targets = np.expm1(targets)

            for i in range(len(preds)):
                all_preds.append(preds[i])
                all_actuals.append(targets[i])
                all_indices.append(idx_offset + i)
            idx_offset += len(preds)

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Identify overprediction samples
    overpredict_mask = (all_preds >= pred_threshold) & (all_actuals < actual_threshold)
    correct_low_mask = (all_preds < 2.0) & (all_actuals < 2.0)

    n_over = overpredict_mask.sum()
    n_correct_low = correct_low_mask.sum()
    print(f"\n  Overprediction samples (pred>={pred_threshold}, actual<{actual_threshold}): {n_over}")
    print(f"  Correct low samples (pred<2, actual<2): {n_correct_low}")

    if n_over == 0:
        print("  No overprediction samples found. Try lowering pred_threshold.")
        return

    # Extract radar features for overprediction samples
    over_indices = np.where(overpredict_mask)[0]
    correct_indices = np.where(correct_low_mask)[0]
    if len(correct_indices) > 500:
        correct_indices = np.random.choice(correct_indices, 500, replace=False)

    print(f"\n{'='*70}")
    print("  OVERPREDICTION ANALYSIS")
    print(f"{'='*70}")

    # Station breakdown
    over_stations = [val_ds.samples[i].get('station_name', 'unknown') for i in over_indices]
    station_counts = Counter(over_stations)
    total_per_station = Counter(s.get('station_name', 'unknown') for s in val_ds.samples)

    print(f"\n  Station breakdown ({n_over} overpredictions):")
    print(f"  {'Station':<30} {'Count':>6} {'% of station':>12} {'% of overpr':>12}")
    print(f"  {'-'*30} {'-'*6} {'-'*12} {'-'*12}")
    for station, count in station_counts.most_common(15):
        station_short = station.replace('Dangermond_', '')
        pct_station = 100 * count / total_per_station[station] if total_per_station[station] > 0 else 0
        pct_over = 100 * count / n_over
        print(f"  {station_short:<30} {count:>6} {pct_station:>10.1f}% {pct_over:>10.1f}%")

    # Temporal clustering
    over_hours = defaultdict(list)
    for i in over_indices:
        s = val_ds.samples[i]
        key = (s.get('station_name', ''), s.get('date', ''))
        over_hours[key].append(s.get('hour_start', ''))

    multi_hour_events = {k: v for k, v in over_hours.items() if len(v) >= 2}
    print(f"\n  Temporal clustering:")
    print(f"    Station-days with multiple overprediction hours: {len(multi_hour_events)}")
    print(f"    Single-hour overpredictions: {len(over_hours) - len(multi_hour_events)}")
    if multi_hour_events:
        print(f"    Top clustered events:")
        for (station, date), hours in sorted(multi_hour_events.items(), key=lambda x: -len(x[1]))[:5]:
            station_short = station.replace('Dangermond_', '')
            print(f"      {station_short} on {date}: {len(hours)} consecutive hours")

    # Radar feature comparison
    print(f"\n  Radar features at gauge pixel (center):")
    print(f"  {'Feature':<30} {'Overpredict':>15} {'Correct Low':>15} {'Difference':>12}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*12}")

    over_features = [extract_radar_features(val_ds.samples[i], fields) for i in over_indices]
    correct_features = [extract_radar_features(val_ds.samples[i], fields) for i in correct_indices]

    feature_keys = ['reflectivity_center_max', 'reflectivity_patch_max',
                    'echo_top_height_center_max', 'vil_center_max',
                    'max_z_height_center_max', 'low_level_ref_center_max',
                    'column_depth_fraction_center_mean',
                    # new — direct bright-band discriminators (model inputs)
                    'rhohv_min_center_mean', 'melting_layer_height_center_mean',
                    'low_level_rhohv_center_mean', 'low_level_zdr_center_mean',
                    'vertical_reflectivity_gradient_center_mean']

    over_medians = {}
    correct_medians = {}
    for key in feature_keys:
        over_vals = [f[key] for f in over_features if not np.isnan(f.get(key, np.nan))]
        correct_vals = [f[key] for f in correct_features if not np.isnan(f.get(key, np.nan))]
        over_med = np.median(over_vals) if over_vals else np.nan
        correct_med = np.median(correct_vals) if correct_vals else np.nan
        over_medians[key] = over_med
        correct_medians[key] = correct_med
        diff = over_med - correct_med if not (np.isnan(over_med) or np.isnan(correct_med)) else np.nan
        print(f"  {key:<30} {over_med:>15.2f} {correct_med:>15.2f} {diff:>+12.2f}")

    # Prediction and actual distributions
    print(f"\n  Prediction distribution for overprediction samples:")
    print(f"    Pred range:   {all_preds[overpredict_mask].min():.1f} – {all_preds[overpredict_mask].max():.1f} mm/hr")
    print(f"    Pred mean:    {all_preds[overpredict_mask].mean():.2f} mm/hr")
    print(f"    Actual range: {all_actuals[overpredict_mask].min():.1f} – {all_actuals[overpredict_mask].max():.1f} mm/hr")
    print(f"    Actual mean:  {all_actuals[overpredict_mask].mean():.2f} mm/hr")

    # Radar signal assessment
    over_dbz = [f.get('reflectivity_center_max', np.nan) for f in over_features]
    over_dbz_valid = [v for v in over_dbz if not np.isnan(v)]
    if over_dbz_valid:
        high_radar = sum(1 for v in over_dbz_valid if v > 30)
        med_radar = sum(1 for v in over_dbz_valid if 15 <= v <= 30)
        low_radar = sum(1 for v in over_dbz_valid if v < 15)
        print(f"\n  Radar signal strength for overpredictions:")
        print(f"    High (>30 dBZ): {high_radar} ({100*high_radar/len(over_dbz_valid):.1f}%) — likely real rain, gauge missed")
        print(f"    Medium (15-30):  {med_radar} ({100*med_radar/len(over_dbz_valid):.1f}%) — borderline")
        print(f"    Low (<15 dBZ):  {low_radar} ({100*low_radar/len(over_dbz_valid):.1f}%) — model confused by non-precip echo")

    # Generate diagnostic plots
    output_dir = Path(run_dir) if run_dir else Path('evaluation_figures/unet_dualpol')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Overprediction Diagnostics (pred≥{pred_threshold}mm, actual<{actual_threshold}mm)\n'
                 f'n={n_over} samples', fontsize=14, fontweight='bold')

    # 1. Station breakdown bar chart
    ax = axes[0, 0]
    top_stations = station_counts.most_common(10)
    names = [s.replace('Dangermond_', '')[:15] for s, _ in top_stations]
    counts = [c for _, c in top_stations]
    ax.barh(range(len(names)), counts, color='#e74c3c', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Count')
    ax.set_title('Overpredictions by Station')
    ax.invert_yaxis()

    # 2. Reflectivity distribution comparison
    ax = axes[0, 1]
    over_ref = [f.get('reflectivity_center_max', np.nan) for f in over_features]
    correct_ref = [f.get('reflectivity_center_max', np.nan) for f in correct_features]
    over_ref = [v for v in over_ref if not np.isnan(v)]
    correct_ref = [v for v in correct_ref if not np.isnan(v)]
    bins = np.linspace(-10, 60, 30)
    ax.hist(correct_ref, bins=bins, alpha=0.5, density=True, label='Correct low', color='#2ecc71')
    ax.hist(over_ref, bins=bins, alpha=0.7, density=True, label='Overpredictions', color='#e74c3c')
    ax.axvline(30, color='black', linestyle='--', alpha=0.5, label='30 dBZ')
    ax.set_xlabel('Max Reflectivity at Gauge (dBZ)')
    ax.set_ylabel('Density')
    ax.set_title('Reflectivity: Overpredict vs Correct Low')
    ax.legend(fontsize=8)

    # 3. Echo top height comparison
    ax = axes[0, 2]
    over_eth = [f.get('echo_top_height_center_max', np.nan) for f in over_features]
    correct_eth = [f.get('echo_top_height_center_max', np.nan) for f in correct_features]
    over_eth = [v for v in over_eth if not np.isnan(v)]
    correct_eth = [v for v in correct_eth if not np.isnan(v)]
    if over_eth and correct_eth:
        bins_eth = np.linspace(0, max(max(over_eth), max(correct_eth)) * 1.1, 25)
        ax.hist(correct_eth, bins=bins_eth, alpha=0.5, density=True, label='Correct low', color='#2ecc71')
        ax.hist(over_eth, bins=bins_eth, alpha=0.7, density=True, label='Overpredictions', color='#e74c3c')
        ax.set_xlabel('Echo Top Height (m)')
        ax.set_ylabel('Density')
        ax.set_title('Echo Top Height: Overpredict vs Correct Low')
        ax.legend(fontsize=8)

    # 4. Scatter: prediction vs max reflectivity
    ax = axes[1, 0]
    all_ref_center = []
    for i in range(len(val_ds.samples)):
        if i < len(all_preds):
            s = val_ds.samples[i]
            rp = s['radar_patch']
            if 'reflectivity' in PICKLE_FIELD_ORDER:
                ridx = PICKLE_FIELD_ORDER.index('reflectivity')
                if ridx < rp.shape[1]:
                    cy, cx = rp.shape[2]//2, rp.shape[3]//2
                    cvals = rp[:, ridx, cy, cx]
                    valid = cvals[(cvals != -9999.0) & ~np.isnan(cvals)]
                    all_ref_center.append(np.max(valid) if len(valid) > 0 else np.nan)
                else:
                    all_ref_center.append(np.nan)
            else:
                all_ref_center.append(np.nan)
    all_ref_center = np.array(all_ref_center[:len(all_preds)])
    valid_mask = ~np.isnan(all_ref_center)
    sc = ax.scatter(all_ref_center[valid_mask], all_preds[valid_mask],
                    c=all_actuals[valid_mask], cmap='YlOrRd', alpha=0.3, s=10,
                    vmin=0, vmax=20)
    ax.axhline(pred_threshold, color='red', linestyle='--', alpha=0.7, label=f'Pred={pred_threshold}mm')
    ax.set_xlabel('Max Reflectivity at Gauge (dBZ)')
    ax.set_ylabel('Prediction (mm/hr)')
    ax.set_title('Prediction vs Reflectivity\n(colored by actual precip)')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Actual (mm/hr)')

    # 5. VIL comparison
    ax = axes[1, 1]
    over_vil = [f.get('vil_center_max', np.nan) for f in over_features]
    correct_vil = [f.get('vil_center_max', np.nan) for f in correct_features]
    over_vil = [v for v in over_vil if not np.isnan(v)]
    correct_vil = [v for v in correct_vil if not np.isnan(v)]
    if over_vil and correct_vil:
        bins_vil = np.linspace(0, max(max(over_vil), max(correct_vil)) * 1.1, 25)
        ax.hist(correct_vil, bins=bins_vil, alpha=0.5, density=True, label='Correct low', color='#2ecc71')
        ax.hist(over_vil, bins=bins_vil, alpha=0.7, density=True, label='Overpredictions', color='#e74c3c')
        ax.set_xlabel('VIL at Gauge')
        ax.set_ylabel('Density')
        ax.set_title('VIL: Overpredict vs Correct Low')
        ax.legend(fontsize=8)

    # 6. Prediction vs Actual for overprediction zone
    ax = axes[1, 2]
    zone_mask = (all_preds > 3) & (all_actuals < 10)
    ax.scatter(all_actuals[zone_mask], all_preds[zone_mask], alpha=0.3, s=15, c='#3498db')
    ax.scatter(all_actuals[overpredict_mask], all_preds[overpredict_mask],
               alpha=0.7, s=25, c='#e74c3c', label='Severe overpredict')
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5)
    ax.axhline(pred_threshold, color='red', linestyle=':', alpha=0.5)
    ax.axvline(actual_threshold, color='blue', linestyle=':', alpha=0.5)
    ax.set_xlabel('Actual (mm/hr)')
    ax.set_ylabel('Prediction (mm/hr)')
    ax.set_title('Overprediction Zone Detail')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, max(20, all_preds[overpredict_mask].max() * 1.1))

    plt.tight_layout()
    out_path = output_dir / 'overestimate_diagnostics.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved diagnostic plot to: {out_path}")
    plt.close()

    # Save CSV of overprediction samples for manual inspection
    csv_path = output_dir / 'overestimate_samples.csv'
    with open(csv_path, 'w') as f:
        f.write('station,date,hour,actual_mm,predicted_mm,max_dbz_center,max_dbz_patch,echo_top,vil\n')
        for i, feat in zip(over_indices, over_features):
            s = val_ds.samples[i]
            f.write(f"{s.get('station_name','')},{s.get('date','')},{s.get('hour_start','')},")
            f.write(f"{all_actuals[i]:.2f},{all_preds[i]:.2f},")
            f.write(f"{feat.get('reflectivity_center_max', ''):.1f},")
            f.write(f"{feat.get('reflectivity_patch_max', ''):.1f},")
            f.write(f"{feat.get('echo_top_height_center_max', ''):.1f},")
            f.write(f"{feat.get('vil_center_max', ''):.1f}\n")
    print(f"  ✓ Saved sample details to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose U-Net overpredictions')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to the model run directory')
    parser.add_argument('--pred-threshold', type=float, default=8.0,
                        help='Minimum prediction to flag as overprediction (mm/hr)')
    parser.add_argument('--actual-threshold', type=float, default=5.0,
                        help='Maximum actual precip for flagging (mm/hr)')
    args = parser.parse_args()

    run_diagnostic(args.run_dir, args.pred_threshold, args.actual_threshold)


if __name__ == '__main__':
    main()
