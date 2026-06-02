"""
diagnose_outliers.py — Investigate extreme hourly precipitation samples in the U-Net pickle.

Examines high-rainfall samples to understand:
1. Radar reflectivity and dual-pol features across the 12 scans
2. Temporal consistency (how many scans show significant reflectivity?)
3. Station identity and cross-station comparison
4. Whether the dump_ratio flag suggests a gauge artifact

Run from project root:
    python -m models.unet.diagnose_outliers --pickle dataset/outputs/radar_gauge_dataset_with_offsets_9500.pkl --threshold 20
"""

import argparse
import pickle
import numpy as np
from collections import defaultdict


FIELD_NAMES = [
    'reflectivity',
    'differential_reflectivity (ZDR)',
    'cross_correlation_ratio (RhoHV)',
    'differential_phase (PhiDP)',
    'specific_differential_phase (KDP)',
]


def analyze_radar_patch(radar_patch, radar_indices):
    """
    Analyze a (12, N_fields, H, W) radar patch.
    Returns dict of statistics across time and fields.
    """
    n_scans, n_fields, H, W = radar_patch.shape
    center_y, center_x = H // 2, W // 2

    stats = {}

    # Valid scans (not None in radar_indices)
    valid_scans = [i for i, idx in enumerate(radar_indices) if idx is not None]
    stats['n_valid_scans'] = len(valid_scans)
    stats['n_total_scans'] = n_scans

    for f_idx, field_name in enumerate(FIELD_NAMES):
        field_data = radar_patch[:, f_idx, :, :]  # (12, H, W)

        # Replace sentinel values
        field_clean = field_data.copy()
        field_clean[field_clean == -9999.0] = np.nan

        # Stats across all valid scans
        valid_data = field_clean[valid_scans] if valid_scans else field_clean
        valid_flat = valid_data[~np.isnan(valid_data)]

        if len(valid_flat) == 0:
            stats[field_name] = {
                'max': np.nan, 'mean': np.nan, 'center_mean': np.nan,
                'scans_above_30': 0, 'temporal_std': np.nan,
            }
            continue

        # Center pixel across time
        center_vals = field_clean[valid_scans, center_y, center_x] if valid_scans else []
        center_valid = center_vals[~np.isnan(center_vals)] if len(center_vals) > 0 else []

        # For reflectivity: count scans with significant values
        scans_above_30 = 0
        if f_idx == 0:  # reflectivity
            for scan_idx in valid_scans:
                scan_max = np.nanmax(field_clean[scan_idx])
                if not np.isnan(scan_max) and scan_max > 30:
                    scans_above_30 += 1

        # Temporal variability at center pixel
        temporal_std = np.nanstd(center_valid) if len(center_valid) > 1 else 0.0

        stats[field_name] = {
            'max': float(np.nanmax(valid_flat)),
            'mean': float(np.nanmean(valid_flat)),
            'center_mean': float(np.nanmean(center_valid)) if len(center_valid) > 0 else np.nan,
            'scans_above_30': scans_above_30,
            'temporal_std': float(temporal_std),
        }

    return stats


def diagnose_outliers(pickle_path, threshold_mm=20.0, split='both'):
    print(f"\nLoading pickle: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)

    if split == 'both':
        all_samples = dataset['train'] + dataset['val']
        print(f"Total samples: {len(all_samples)} (train={len(dataset['train'])}, val={len(dataset['val'])})")
    else:
        all_samples = dataset[split]
        print(f"Total {split} samples: {len(all_samples)}")

    # Find outliers
    outliers = [s for s in all_samples if s['hourly_precip_mm'] >= threshold_mm]
    print(f"\nSamples with precip >= {threshold_mm} mm/hr: {len(outliers)}")

    if not outliers:
        print("No outliers found.")
        return

    # Group by hour for cross-station comparison
    by_hour = defaultdict(list)
    for s in all_samples:
        by_hour[s['hour_start']].append(s)

    # Analyze each outlier
    print(f"\n{'='*90}")
    print(f"  OUTLIER ANALYSIS (threshold = {threshold_mm} mm/hr)")
    print(f"{'='*90}")

    for i, s in enumerate(sorted(outliers, key=lambda x: -x['hourly_precip_mm'])):
        precip = s['hourly_precip_mm']
        station = s.get('station_name', 'Unknown')
        hour_start = s['hour_start']
        radar_indices = s.get('radar_indices', [])
        dump_ratio = s.get('dump_ratio', None)
        max_bin = s.get('max_bin_mm', None)
        n_active = s.get('n_active_bins', None)

        # Radar analysis
        radar_stats = analyze_radar_patch(s['radar_patch'], radar_indices)

        print(f"\n{'─'*90}")
        print(f"  Outlier #{i+1}: {precip:.2f} mm/hr")
        print(f"{'─'*90}")
        print(f"  Station:       {station}")
        print(f"  Hour:          {hour_start}")
        print(f"  Valid scans:   {radar_stats['n_valid_scans']}/{radar_stats['n_total_scans']}")

        if dump_ratio is not None:
            print(f"\n  Gauge diagnostics:")
            print(f"    Dump ratio:    {dump_ratio:.3f} (1.0 = all rain in 1 bin)")
            print(f"    Max 10-min:    {max_bin:.2f} mm")
            print(f"    Active bins:   {n_active}/6")

        print(f"\n  Radar features (across {radar_stats['n_valid_scans']} valid scans):")
        for field_name in FIELD_NAMES:
            fs = radar_stats.get(field_name, {})
            if not fs or np.isnan(fs.get('max', np.nan)):
                print(f"    {field_name:<40s} NO DATA")
                continue

            line = f"    {field_name:<40s}"
            line += f" max={fs['max']:7.1f}"
            line += f"  mean={fs['mean']:7.1f}"
            line += f"  center={fs['center_mean']:7.1f}"

            if field_name == FIELD_NAMES[0]:  # reflectivity
                line += f"  scans>30dBZ={fs['scans_above_30']}/{radar_stats['n_valid_scans']}"

            line += f"  t_std={fs['temporal_std']:.2f}"
            print(line)

        # Reflectivity temporal profile
        ref_data = s['radar_patch'][:, 0, :, :].copy()
        ref_data[ref_data == -9999.0] = np.nan
        center_y, center_x = ref_data.shape[1] // 2, ref_data.shape[2] // 2
        print(f"\n  Reflectivity timeline (patch max per scan):")
        timeline_str = "    "
        for scan_i in range(12):
            if radar_indices[scan_i] is None:
                timeline_str += "  --- "
            else:
                scan_max = np.nanmax(ref_data[scan_i])
                if np.isnan(scan_max):
                    timeline_str += "  NaN "
                else:
                    timeline_str += f"{scan_max:5.1f} "
        print(timeline_str)
        print(f"    {'t=0':>5s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} "
              f"{'':>5s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} {'t=55m':>5s}")

        # Cross-station comparison
        same_hour = by_hour[hour_start]
        other_readings = [(x['station_name'], x['hourly_precip_mm']) for x in same_hour
                          if x['station_id'] != s['station_id']]

        print(f"\n  Other stations at same hour ({len(other_readings)} stations):")
        if other_readings:
            other_sorted = sorted(other_readings, key=lambda x: -x[1])[:8]
            for name, val in other_sorted:
                short_name = name.replace('Dangermond_', '')
                print(f"    {short_name:<25s} {val:.2f} mm/hr")
            all_other = [r[1] for r in other_readings]
            print(f"    {'───'}")
            print(f"    Mean: {np.mean(all_other):.2f} mm/hr  |  "
                  f"Max: {np.max(all_other):.2f} mm/hr  |  "
                  f"Zeros: {sum(1 for p in all_other if p < 0.1)}/{len(all_other)}")

        # Verdict
        ref_stats = radar_stats.get(FIELD_NAMES[0], {})
        max_dbz = ref_stats.get('max', 0)
        scans_strong = ref_stats.get('scans_above_30', 0)
        is_dump = dump_ratio is not None and dump_ratio >= 0.9

        print(f"\n  Quick assessment:")
        if max_dbz < 25:
            print(f"    ⚠ Very low reflectivity ({max_dbz:.0f} dBZ) — LIKELY ARTIFACT")
        elif max_dbz < 40 and scans_strong < 3:
            print(f"    ⚠ Moderate reflectivity but few strong scans — SUSPECT")
        elif max_dbz >= 45 and scans_strong >= 6:
            print(f"    ✓ Strong persistent reflectivity — LIKELY REAL")
        else:
            print(f"    ? Mixed signal (max={max_dbz:.0f} dBZ, {scans_strong} strong scans)")

        if is_dump:
            print(f"    ⚠ High dump ratio ({dump_ratio:.2f}) — rain concentrated in 1 bin")

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"\n  {'Station':<22s} {'Precip':>7s} {'MaxZ':>6s} {'MeanZ':>6s} "
          f"{'Scans>30':>8s} {'DumpR':>6s} {'ActiveBins':>10s} {'Assessment'}")
    print(f"  {'─'*22} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*6} {'─'*10} {'─'*15}")

    for s in sorted(outliers, key=lambda x: -x['hourly_precip_mm']):
        station = s.get('station_name', 'Unknown').replace('Dangermond_', '')
        precip = s['hourly_precip_mm']
        radar_stats = analyze_radar_patch(s['radar_patch'], s.get('radar_indices', []))
        ref_stats = radar_stats.get(FIELD_NAMES[0], {})

        max_dbz = ref_stats.get('max', 0)
        mean_dbz = ref_stats.get('mean', 0)
        scans_strong = ref_stats.get('scans_above_30', 0)
        n_valid = radar_stats['n_valid_scans']
        dump_r = s.get('dump_ratio', None)
        n_active = s.get('n_active_bins', None)

        dump_str = f"{dump_r:.2f}" if dump_r is not None else "N/A"
        active_str = f"{n_active}/6" if n_active is not None else "N/A"

        if max_dbz < 25:
            verdict = "ARTIFACT"
        elif max_dbz < 40 and scans_strong < 3:
            verdict = "SUSPECT"
        elif max_dbz >= 45 and scans_strong >= 6:
            verdict = "REAL"
        else:
            verdict = "UNCLEAR"

        print(f"  {station:<22s} {precip:>7.1f} {max_dbz:>6.1f} {mean_dbz:>6.1f} "
              f"{scans_strong:>3d}/{n_valid:<4d} {dump_str:>6s} {active_str:>10s} {verdict}")

    # Distribution summary
    print(f"\n  Total outliers: {len(outliers)}")
    verdicts = defaultdict(int)
    for s in outliers:
        rs = analyze_radar_patch(s['radar_patch'], s.get('radar_indices', []))
        ref = rs.get(FIELD_NAMES[0], {})
        mx = ref.get('max', 0)
        sc = ref.get('scans_above_30', 0)
        if mx < 25:
            verdicts['ARTIFACT'] += 1
        elif mx < 40 and sc < 3:
            verdicts['SUSPECT'] += 1
        elif mx >= 45 and sc >= 6:
            verdicts['REAL'] += 1
        else:
            verdicts['UNCLEAR'] += 1

    for v in ['REAL', 'UNCLEAR', 'SUSPECT', 'ARTIFACT']:
        if verdicts[v]:
            print(f"    {v}: {verdicts[v]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose extreme hourly precipitation outliers")
    parser.add_argument('--pickle', default='dataset/outputs/radar_gauge_dataset_with_offsets_9500.pkl')
    parser.add_argument('--threshold', type=float, default=20.0,
                        help='Precipitation threshold in mm/hr (default: 20.0)')
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both')
    args = parser.parse_args()

    diagnose_outliers(args.pickle, threshold_mm=args.threshold, split=args.split)
