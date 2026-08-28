"""
diagnose_underestimates.py — Analyze samples where actual rain is heavy
(> 5mm) but the U-Net predicts less than a fraction (default 3/4) of it.

These are cases where the radar/model UNDER-reads relative to the gauge.
Likely mechanisms, and the radar signatures that distinguish them:
  - Warm-rain / collision-coalescence: low echo top, low max-Z height,
    modest reflectivity but high rain rate, low ZDR (small drops),
    high KDP relative to Z (lots of liquid).
  - Beam overshoot: radar samples too high, misses low-level growth.
  - Orographic / seeder-feeder: low-level reflectivity exceeds aloft.

The goal is to decide which NEW radar-structure metrics are worth
deriving from the 3D volume before regenerating the zarr.

Run from project root:
    python -m models.unet.diagnose_underestimates --run-dir models/checkpoints/unet_dualpol/<run_name>
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
from models.unet.diagnose_overpredict_weather import query_hourly_avg


# Fields we extract at the gauge pixel. Dual-pol fields are included because
# they are the key discriminators for the warm-rain underestimation mechanism.
SCALAR_MAX_FIELDS = [
    'reflectivity', 'echo_top_height', 'max_z_height', 'vil',
    'low_level_ref', 'column_depth_fraction',
    # new — low-level / warm-rain structure
    'lowest_gate_reflectivity', 'beam_height',
    'vertical_reflectivity_gradient', 'melting_layer_height',
]
MEAN_FIELDS = [
    'differential_reflectivity', 'cross_correlation_ratio',
    'specific_differential_phase',
    # new — low-level dual-pol (sampled near the surface, not at max-Z) and
    # the column-min RhoHV. These are the direct warm-rain discriminators.
    'low_level_zdr', 'low_level_rhohv', 'low_level_kdp', 'rhohv_min',
]


def _valid_center(radar_patch, field_name):
    """Return the valid (non-sentinel, non-NaN) center-pixel time series."""
    if field_name not in PICKLE_FIELD_ORDER:
        return np.array([])
    idx = PICKLE_FIELD_ORDER.index(field_name)
    if idx >= radar_patch.shape[1]:
        return np.array([])
    cy, cx = radar_patch.shape[2] // 2, radar_patch.shape[3] // 2
    vals = radar_patch[:, idx, cy, cx]
    vals = vals[(vals != -9999.0) & ~np.isnan(vals)]
    return vals


def extract_radar_features(sample, fields):
    """Extract raw radar features at the center pixel for a sample."""
    radar_patch = sample['radar_patch']  # (12, N_fields, H, W)
    features = {}

    for field_name in SCALAR_MAX_FIELDS:
        valid = _valid_center(radar_patch, field_name)
        features[f'{field_name}_center_max'] = np.nanmax(valid) if len(valid) else np.nan
        features[f'{field_name}_center_mean'] = np.nanmean(valid) if len(valid) else np.nan
        # patch max (over all pixels and scans)
        if field_name in PICKLE_FIELD_ORDER:
            idx = PICKLE_FIELD_ORDER.index(field_name)
            if idx < radar_patch.shape[1]:
                patch_vals = radar_patch[:, idx, :, :]
                patch_vals = patch_vals[(patch_vals != -9999.0) & ~np.isnan(patch_vals)]
                features[f'{field_name}_patch_max'] = (
                    np.nanmax(patch_vals) if len(patch_vals) else np.nan)

    for field_name in MEAN_FIELDS:
        valid = _valid_center(radar_patch, field_name)
        features[f'{field_name}_center_mean'] = np.nanmean(valid) if len(valid) else np.nan

    # Derived: low-level enhancement (orographic / seeder-feeder proxy).
    # low_level_ref (0-2km mean) minus the column-max reflectivity. A value
    # near 0 (or positive) means the wettest layer is near the surface.
    ll = features.get('low_level_ref_center_max', np.nan)
    refl = features.get('reflectivity_center_max', np.nan)
    features['lowlevel_minus_colmax'] = (ll - refl) if not (np.isnan(ll) or np.isnan(refl)) else np.nan

    return features


def run_diagnostic(run_dir, actual_threshold=5.0, ratio=0.75):
    """Main diagnostic: find and analyze severe underpredictions."""
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
    print(f"  Underprediction criteria: actual > {actual_threshold}mm AND pred < {ratio:.2f} × actual")

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_actuals = [], []
    with torch.no_grad():
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
            all_preds.extend(preds.tolist())
            all_actuals.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Underprediction: heavy actual rain, but prediction falls short.
    underpredict_mask = (all_actuals > actual_threshold) & (all_preds < ratio * all_actuals)
    # Comparison: heavy rain the model got roughly right (within +/-25%).
    correct_heavy_mask = (
        (all_actuals > actual_threshold)
        & (all_preds >= ratio * all_actuals)
        & (all_preds <= (2.0 - ratio) * all_actuals)
    )

    n_under = int(underpredict_mask.sum())
    n_correct = int(correct_heavy_mask.sum())
    n_heavy = int((all_actuals > actual_threshold).sum())
    print(f"\n  Heavy-rain samples (actual > {actual_threshold}mm): {n_heavy}")
    print(f"  Underprediction samples: {n_under} ({100*n_under/max(n_heavy,1):.1f}% of heavy rain)")
    print(f"  Correct heavy samples (within ±{100*(1-ratio):.0f}%): {n_correct}")

    if n_under == 0:
        print("  No underprediction samples found. Try lowering actual_threshold or raising ratio.")
        return

    under_indices = np.where(underpredict_mask)[0]
    correct_indices = np.where(correct_heavy_mask)[0]
    if len(correct_indices) > 500:
        correct_indices = np.random.choice(correct_indices, 500, replace=False)

    print(f"\n{'='*70}")
    print("  UNDERPREDICTION ANALYSIS")
    print(f"{'='*70}")

    # Station breakdown
    under_stations = [val_ds.samples[i].get('station_name', 'unknown') for i in under_indices]
    station_counts = Counter(under_stations)
    # Heavy-rain count per station (denominator)
    heavy_per_station = Counter(
        val_ds.samples[i].get('station_name', 'unknown')
        for i in np.where(all_actuals > actual_threshold)[0]
    )
    print(f"\n  Station breakdown ({n_under} underpredictions):")
    print(f"  {'Station':<30} {'Count':>6} {'% of heavy':>12} {'% of under':>12}")
    print(f"  {'-'*30} {'-'*6} {'-'*12} {'-'*12}")
    for station, count in station_counts.most_common(15):
        station_short = station.replace('Dangermond_', '')
        pct_heavy = 100 * count / heavy_per_station[station] if heavy_per_station[station] > 0 else 0
        pct_under = 100 * count / n_under
        print(f"  {station_short:<30} {count:>6} {pct_heavy:>10.1f}% {pct_under:>10.1f}%")

    # Temporal clustering. hour_start is a datetime, so derive the calendar
    # day from it (the samples carry no separate 'date' key).
    def _sample_date(s):
        h = s.get('hour_start', None)
        if h is None:
            return ''
        if hasattr(h, 'date'):
            return h.date()
        return str(h).split(' ')[0]

    under_hours = defaultdict(list)
    for i in under_indices:
        s = val_ds.samples[i]
        key = (s.get('station_name', ''), _sample_date(s))
        under_hours[key].append(s.get('hour_start', ''))
    multi_hour_events = {k: v for k, v in under_hours.items() if len(v) >= 2}
    n_distinct_days = len(set(d for (_, d) in under_hours.keys()))
    print(f"\n  Temporal clustering:")
    print(f"    Distinct storm days: {n_distinct_days}")
    print(f"    Station-days with multiple underprediction hours: {len(multi_hour_events)}")
    print(f"    Single-hour underpredictions: {len(under_hours) - len(multi_hour_events)}")
    if multi_hour_events:
        print(f"    Top clustered station-days:")
        for (station, date), hours in sorted(multi_hour_events.items(), key=lambda x: -len(x[1]))[:8]:
            station_short = station.replace('Dangermond_', '')
            print(f"      {station_short} on {date}: {len(hours)} hours")

    # Radar feature comparison (under-read vs correctly-predicted heavy rain)
    print(f"\n  Radar features at gauge pixel (underpredict vs correct heavy):")
    print(f"  {'Feature':<32} {'Underpredict':>14} {'Correct Heavy':>14} {'Difference':>12}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*12}")

    under_features = [extract_radar_features(val_ds.samples[i], fields) for i in under_indices]
    correct_features = [extract_radar_features(val_ds.samples[i], fields) for i in correct_indices]

    feature_keys = [
        'reflectivity_center_max', 'reflectivity_patch_max',
        'echo_top_height_center_max', 'max_z_height_center_max',
        'vil_center_max', 'low_level_ref_center_max',
        'lowlevel_minus_colmax', 'column_depth_fraction_center_mean',
        'differential_reflectivity_center_mean',
        'cross_correlation_ratio_center_mean',
        'specific_differential_phase_center_mean',
        # new — low-level / warm-rain discriminators (now model inputs)
        'lowest_gate_reflectivity_center_max',
        'beam_height_center_max',
        'vertical_reflectivity_gradient_center_max',
        'melting_layer_height_center_max',
        'low_level_zdr_center_mean',
        'low_level_rhohv_center_mean',
        'low_level_kdp_center_mean',
        'rhohv_min_center_mean',
    ]
    for key in feature_keys:
        u_vals = [f[key] for f in under_features if not np.isnan(f.get(key, np.nan))]
        c_vals = [f[key] for f in correct_features if not np.isnan(f.get(key, np.nan))]
        u_med = np.median(u_vals) if u_vals else np.nan
        c_med = np.median(c_vals) if c_vals else np.nan
        diff = u_med - c_med if not (np.isnan(u_med) or np.isnan(c_med)) else np.nan
        print(f"  {key:<32} {u_med:>14.3f} {c_med:>14.3f} {diff:>+12.3f}")

    # Reflectivity efficiency: how much reflectivity per mm of actual rain.
    # Warm rain produces LOW reflectivity for its rain rate, so this ratio
    # should be smaller for underpredictions than for correct heavy rain.
    print(f"\n  Reflectivity-per-rainrate (median dBZ / actual mm/hr):")
    u_eff = [under_features[k]['reflectivity_center_max'] / all_actuals[i]
             for k, i in enumerate(under_indices)
             if not np.isnan(under_features[k].get('reflectivity_center_max', np.nan)) and all_actuals[i] > 0]
    c_eff = [correct_features[k]['reflectivity_center_max'] / all_actuals[i]
             for k, i in enumerate(correct_indices)
             if not np.isnan(correct_features[k].get('reflectivity_center_max', np.nan)) and all_actuals[i] > 0]
    if u_eff and c_eff:
        print(f"    Underpredict:  {np.median(u_eff):.2f} dBZ per mm/hr")
        print(f"    Correct heavy: {np.median(c_eff):.2f} dBZ per mm/hr")
        if np.median(u_eff) < np.median(c_eff):
            print(f"    → Lower reflectivity-per-rainrate supports the warm-rain hypothesis.")

    # Prediction / actual distribution
    print(f"\n  Prediction shortfall for underprediction samples:")
    print(f"    Actual range: {all_actuals[underpredict_mask].min():.1f} – {all_actuals[underpredict_mask].max():.1f} mm/hr")
    print(f"    Actual mean:  {all_actuals[underpredict_mask].mean():.2f} mm/hr")
    print(f"    Pred mean:    {all_preds[underpredict_mask].mean():.2f} mm/hr")
    ratios = all_preds[underpredict_mask] / np.maximum(all_actuals[underpredict_mask], 1e-6)
    print(f"    Pred/actual ratio (median): {np.median(ratios):.2f}")

    # Echo-top regime: warm rain = shallow echoes.
    under_eth = [f.get('echo_top_height_center_max', np.nan) for f in under_features]
    under_eth = [v for v in under_eth if not np.isnan(v)]
    if under_eth:
        shallow = sum(1 for v in under_eth if v < 3000)
        mid = sum(1 for v in under_eth if 3000 <= v < 6000)
        deep = sum(1 for v in under_eth if v >= 6000)
        n = len(under_eth)
        print(f"\n  Echo-top regime for underpredictions:")
        print(f"    Shallow (<3km): {shallow} ({100*shallow/n:.1f}%) — warm-rain candidate")
        print(f"    Mid (3-6km):    {mid} ({100*mid/n:.1f}%)")
        print(f"    Deep (>6km):    {deep} ({100*deep/n:.1f}%)")

    # ── Air temperature (warm-rain test) ──
    # Warm rain should occur at WARMER temps — the opposite of the cold
    # bright-band overpredictions. Query the gauge station's air temperature.
    under_temp, correct_temp = {}, {}
    try:
        from database.config import connect, create_session
        engine = connect()
        session = create_session(engine)

        def _collect_temps(indices, store):
            by_station = defaultdict(list)
            for i in indices:
                s = val_ds.samples[i]
                by_station[s.get('station_name', '')].append((i, s.get('hour_start', '')))
            for station, items in by_station.items():
                hours = [str(h) for _, h in items]
                temps = query_hourly_avg(session, station, '%Air Temp Avg%', hours)
                for (i, h) in items:
                    v = temps.get(str(h), np.nan)
                    if not np.isnan(v):
                        store[i] = v

        _collect_temps(under_indices, under_temp)
        _collect_temps(correct_indices, correct_temp)
        session.close()
    except Exception as e:
        print(f"\n  ⚠ Temperature query skipped ({e})")

    u_temps = np.array(list(under_temp.values()))
    c_temps = np.array(list(correct_temp.values()))
    if len(u_temps) and len(c_temps):
        print(f"\n  Air Temperature (°C) — warm-rain test:")
        print(f"  {'Metric':<14} {'Underpredict':>14} {'Correct Heavy':>14} {'Difference':>12}")
        print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*12}")
        print(f"  {'Mean':<14} {u_temps.mean():>13.1f}° {c_temps.mean():>13.1f}° {u_temps.mean()-c_temps.mean():>+11.1f}°")
        print(f"  {'Median':<14} {np.median(u_temps):>13.1f}° {np.median(c_temps):>13.1f}° {np.median(u_temps)-np.median(c_temps):>+11.1f}°")
        print(f"  {'N samples':<14} {len(u_temps):>14} {len(c_temps):>14}")
        warm = (u_temps >= 12).mean() * 100
        print(f"  {'% ≥ 12°C':<14} {warm:>13.1f}% {(c_temps>=12).mean()*100:>13.1f}%")

        # Correlation: warmer temp → larger shortfall (lower pred/actual)?
        ratios_t = [all_preds[i] / max(all_actuals[i], 1e-6) for i in under_temp.keys()]
        if len(ratios_t) > 10:
            r = np.corrcoef(u_temps, ratios_t)[0, 1]
            print(f"\n  Correlation (temp vs. pred/actual ratio among underpredictions): r = {r:.3f}")
            if r < -0.15:
                print(f"    → Warmer = larger shortfall (supports warm-rain underestimation)")
            elif r > 0.15:
                print(f"    → Cooler = larger shortfall (unexpected for warm rain)")
            else:
                print(f"    → Weak correlation")
        if np.median(u_temps) > np.median(c_temps):
            print(f"    → Underpredictions skew WARMER than correct heavy rain (consistent with warm rain).")

    # ── Plots ──
    output_dir = Path(run_dir) if run_dir else Path('evaluation_figures/unet_dualpol')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f'Underprediction Diagnostics (actual>{actual_threshold}mm, pred<{ratio:.2f}×actual)\n'
                 f'n={n_under} samples', fontsize=14, fontweight='bold')

    def _hist(ax, ukey, title, xlabel, bins=None, clip=None):
        u = [f.get(ukey, np.nan) for f in under_features]
        c = [f.get(ukey, np.nan) for f in correct_features]
        u = [v for v in u if not np.isnan(v)]
        c = [v for v in c if not np.isnan(v)]
        if not u or not c:
            return
        if bins is None:
            lo = min(min(u), min(c))
            hi = max(max(u), max(c))
            if clip:
                lo, hi = clip
            bins = np.linspace(lo, hi, 25)
        ax.hist(c, bins=bins, alpha=0.5, density=True, label=f'Correct heavy (n={len(c)})', color='#2ecc71')
        ax.hist(u, bins=bins, alpha=0.7, density=True, label=f'Underpredict (n={len(u)})', color='#9b59b6')
        ax.axvline(np.median(u), color='purple', linestyle='--', alpha=0.7)
        ax.axvline(np.median(c), color='green', linestyle='--', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(fontsize=8)

    # 1. Station breakdown
    ax = axes[0, 0]
    top_stations = station_counts.most_common(10)
    names = [s.replace('Dangermond_', '')[:15] for s, _ in top_stations]
    counts = [c for _, c in top_stations]
    ax.barh(range(len(names)), counts, color='#9b59b6', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Count')
    ax.set_title('Underpredictions by Station')
    ax.invert_yaxis()

    # 2. Reflectivity
    _hist(axes[0, 1], 'reflectivity_center_max',
          'Reflectivity: Under vs Correct Heavy', 'Max Reflectivity at Gauge (dBZ)',
          clip=(-10, 60))

    # 3. Echo top height (warm-rain discriminator)
    _hist(axes[0, 2], 'echo_top_height_center_max',
          'Echo Top Height: Under vs Correct Heavy', 'Echo Top Height (m)')

    # 4. Scatter: prediction vs actual for heavy rain (colored by echo top)
    ax = axes[1, 0]
    heavy_mask = all_actuals > actual_threshold
    eth_all = np.full(len(all_preds), np.nan)
    # compute echo top for heavy samples only (cheaper)
    for i in np.where(heavy_mask)[0]:
        valid = _valid_center(val_ds.samples[i]['radar_patch'], 'echo_top_height')
        eth_all[i] = np.nanmax(valid) if len(valid) else np.nan
    hm = heavy_mask & ~np.isnan(eth_all)
    sc = ax.scatter(all_actuals[hm], all_preds[hm], c=eth_all[hm], cmap='viridis',
                    alpha=0.5, s=20, vmin=0, vmax=8000)
    lim = max(all_actuals[heavy_mask].max(), all_preds[heavy_mask].max())
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.plot([0, lim], [0, ratio * lim], 'r:', alpha=0.7, label=f'pred={ratio:.2f}×actual')
    ax.set_xlabel('Actual (mm/hr)')
    ax.set_ylabel('Prediction (mm/hr)')
    ax.set_title('Heavy Rain: Pred vs Actual\n(colored by echo top)')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Echo top (m)')

    # 5. KDP (warm-rain heavy liquid discriminator)
    _hist(axes[1, 1], 'specific_differential_phase_center_mean',
          'KDP: Under vs Correct Heavy', 'KDP at Gauge (deg/km)')

    # 6. ZDR (small drops = low ZDR for warm rain)
    _hist(axes[1, 2], 'differential_reflectivity_center_mean',
          'ZDR: Under vs Correct Heavy', 'ZDR at Gauge (dB)')

    # 7. Temperature distribution (warm-rain test)
    ax = axes[2, 0]
    if len(u_temps) and len(c_temps):
        t_lo = min(u_temps.min(), c_temps.min())
        t_hi = max(u_temps.max(), c_temps.max())
        bins_t = np.linspace(t_lo, t_hi, 25)
        ax.hist(c_temps, bins=bins_t, alpha=0.5, density=True,
                label=f'Correct heavy (n={len(c_temps)})', color='#2ecc71')
        ax.hist(u_temps, bins=bins_t, alpha=0.7, density=True,
                label=f'Underpredict (n={len(u_temps)})', color='#9b59b6')
        ax.axvline(np.median(u_temps), color='purple', linestyle='--', alpha=0.7)
        ax.axvline(np.median(c_temps), color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Air Temperature (°C)')
    ax.set_ylabel('Density')
    ax.set_title('Temperature: Under vs Correct Heavy')
    ax.legend(fontsize=8)

    # 8. Temperature vs pred/actual ratio (warmer = bigger shortfall?)
    ax = axes[2, 1]
    if len(under_temp) > 0:
        t_arr = np.array([under_temp[i] for i in under_temp.keys()])
        r_arr = np.array([all_preds[i] / max(all_actuals[i], 1e-6) for i in under_temp.keys()])
        ax.scatter(t_arr, r_arr, alpha=0.6, s=25, c='#9b59b6')
        ax.axhline(ratio, color='red', linestyle=':', alpha=0.7, label=f'ratio={ratio:.2f}')
    ax.set_xlabel('Air Temperature (°C)')
    ax.set_ylabel('Pred / Actual ratio')
    ax.set_title('Temperature vs. Prediction Shortfall')
    ax.legend(fontsize=8)

    # 9. Temperature vs echo top, colored by pred/actual ratio
    ax = axes[2, 2]
    if len(under_temp) > 0:
        t_arr, eth_arr, ratio_arr = [], [], []
        for i in under_temp.keys():
            valid = _valid_center(val_ds.samples[i]['radar_patch'], 'echo_top_height')
            if len(valid):
                t_arr.append(under_temp[i])
                eth_arr.append(np.nanmax(valid))
                ratio_arr.append(all_preds[i] / max(all_actuals[i], 1e-6))
        if t_arr:
            sc = ax.scatter(t_arr, eth_arr, c=ratio_arr, cmap='plasma',
                            alpha=0.7, s=30, vmin=0, vmax=ratio)
            ax.set_xlabel('Air Temperature (°C)')
            ax.set_ylabel('Echo top height (m)')
            ax.set_title('Warm-rain space: Temp vs Echo Top\n(colored by pred/actual)')
            plt.colorbar(sc, ax=ax, label='Pred/Actual')

    plt.tight_layout()
    out_path = output_dir / 'underestimate_diagnostics.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved diagnostic plot to: {out_path}")
    plt.close()

    # CSV
    csv_path = output_dir / 'underestimate_samples.csv'
    with open(csv_path, 'w') as f:
        f.write('station,date,hour,actual_mm,predicted_mm,ratio,temp_c,max_dbz_center,'
                'echo_top,max_z_height,low_level_ref,zdr,rhohv,kdp\n')
        for k, i in enumerate(under_indices):
            s = val_ds.samples[i]
            feat = under_features[k]
            def g(key):
                v = feat.get(key, np.nan)
                return f"{v:.2f}" if not np.isnan(v) else ''
            temp_v = under_temp.get(i, np.nan)
            temp_s = f"{temp_v:.1f}" if not np.isnan(temp_v) else ''
            f.write(f"{s.get('station_name','')},{_sample_date(s)},{s.get('hour_start','')},")
            f.write(f"{all_actuals[i]:.2f},{all_preds[i]:.2f},{all_preds[i]/max(all_actuals[i],1e-6):.2f},{temp_s},")
            f.write(f"{g('reflectivity_center_max')},{g('echo_top_height_center_max')},")
            f.write(f"{g('max_z_height_center_max')},{g('low_level_ref_center_max')},")
            f.write(f"{g('differential_reflectivity_center_mean')},")
            f.write(f"{g('cross_correlation_ratio_center_mean')},")
            f.write(f"{g('specific_differential_phase_center_mean')}\n")
    print(f"  ✓ Saved sample details to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose U-Net underpredictions')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to the model run directory')
    parser.add_argument('--actual-threshold', type=float, default=5.0,
                        help='Minimum actual precip to flag as heavy rain (mm/hr)')
    parser.add_argument('--ratio', type=float, default=0.75,
                        help='Flag underprediction when pred < ratio × actual')
    args = parser.parse_args()
    run_diagnostic(args.run_dir, args.actual_threshold, args.ratio)


if __name__ == '__main__':
    main()
