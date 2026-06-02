"""
tests.py — Diagnostic analysis of the radar-gauge pickle before training.

Run from the project root:
    python -m models.stack.tests
"""

import pickle
import numpy as np
from pathlib import Path

PICKLE_PATH = 'dataset/outputs/radar_gauge_dataset.pkl'

PICKLE_FIELD_ORDER = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
]


def load_pickle(path=PICKLE_PATH):
    print(f"Loading pickle: {path}")
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Val samples:   {len(dataset['val'])}")
    print(f"  Metadata:      {list(dataset['metadata'].keys())}")
    return dataset


def analyze_field_distributions(samples, split_name='train'):
    """Compute per-field value ranges and distribution stats (ignoring NaN and -9999)."""
    print(f"\n{'='*70}")
    print(f"  FIELD DISTRIBUTIONS — {split_name.upper()} ({len(samples)} samples)")
    print(f"{'='*70}")

    n_fields = len(PICKLE_FIELD_ORDER)
    n_scans = samples[0]['radar_patch'].shape[0]

    for f_idx, field_name in enumerate(PICKLE_FIELD_ORDER):
        all_vals = []
        nan_count = 0
        sentinel_count = 0
        total_pixels = 0

        for s in samples:
            arr = s['radar_patch'][:, f_idx, :, :]  # (12, H, W)
            total_pixels += arr.size

            sentinel_mask = (arr == -9999.0)
            sentinel_count += sentinel_mask.sum()

            nan_mask = np.isnan(arr)
            nan_count += nan_mask.sum()

            valid = arr[~nan_mask & ~sentinel_mask]
            if len(valid) > 0:
                all_vals.append(valid)

        if all_vals:
            combined = np.concatenate(all_vals)
            pct = np.percentile(combined, [1, 5, 25, 50, 75, 95, 99])
            print(f"\n  {field_name}:")
            print(f"    Range:       [{combined.min():.2f}, {combined.max():.2f}]")
            print(f"    Mean:        {combined.mean():.3f}")
            print(f"    Std:         {combined.std():.3f}")
            print(f"    Percentiles: p1={pct[0]:.2f}  p5={pct[1]:.2f}  p25={pct[2]:.2f}  "
                  f"p50={pct[3]:.2f}  p75={pct[4]:.2f}  p95={pct[5]:.2f}  p99={pct[6]:.2f}")
            print(f"    NaN pixels:      {nan_count:,} / {total_pixels:,} ({100*nan_count/total_pixels:.1f}%)")
            print(f"    Sentinel(-9999): {sentinel_count:,} / {total_pixels:,} ({100*sentinel_count/total_pixels:.1f}%)")
            print(f"    Valid pixels:    {len(combined):,} / {total_pixels:,} ({100*len(combined)/total_pixels:.1f}%)")
        else:
            print(f"\n  {field_name}: NO VALID DATA")
            print(f"    NaN pixels:      {nan_count:,} / {total_pixels:,}")
            print(f"    Sentinel(-9999): {sentinel_count:,} / {total_pixels:,}")


def analyze_nan_scans(samples, split_name='train'):
    """Analyze how many individual scans and full samples have all-NaN reflectivity."""
    print(f"\n{'='*70}")
    print(f"  NaN SCAN ANALYSIS — {split_name.upper()} ({len(samples)} samples)")
    print(f"{'='*70}")

    n_scans = samples[0]['radar_patch'].shape[0]

    total_scans = 0
    all_nan_scans = 0
    samples_all_nan = 0
    samples_some_nan = 0
    nan_scan_counts = []  # per-sample count of all-NaN scans

    for s in samples:
        refl = s['radar_patch'][:, 0, :, :]  # (12, H, W) reflectivity only
        sample_nan_scans = 0

        for t in range(n_scans):
            total_scans += 1
            scan_data = refl[t]
            if np.all(np.isnan(scan_data)):
                all_nan_scans += 1
                sample_nan_scans += 1

        nan_scan_counts.append(sample_nan_scans)

        if sample_nan_scans == n_scans:
            samples_all_nan += 1
        elif sample_nan_scans > 0:
            samples_some_nan += 1

    nan_scan_counts = np.array(nan_scan_counts)

    print(f"\n  Scan-level:")
    print(f"    Total scans:       {total_scans:,}")
    print(f"    All-NaN scans:     {all_nan_scans:,} ({100*all_nan_scans/total_scans:.1f}%)")
    print(f"    Valid scans:       {total_scans - all_nan_scans:,}")

    print(f"\n  Sample-level ({n_scans} scans per sample):")
    print(f"    Fully NaN (0/{n_scans} valid):  {samples_all_nan:,} ({100*samples_all_nan/len(samples):.2f}%)")
    print(f"    Partially NaN:                  {samples_some_nan:,} ({100*samples_some_nan/len(samples):.1f}%)")
    print(f"    Fully valid ({n_scans}/{n_scans}):       {len(samples) - samples_all_nan - samples_some_nan:,}")

    print(f"\n  Distribution of NaN scans per sample:")
    for k in range(n_scans + 1):
        count = (nan_scan_counts == k).sum()
        if count > 0:
            print(f"    {k:2d} NaN scans: {count:,} samples ({100*count/len(samples):.1f}%)")


def analyze_radar_indices(samples, split_name='train'):
    """Check radar_indices for None slots (scans the binning couldn't fill)."""
    print(f"\n{'='*70}")
    print(f"  RADAR INDICES (binned slots) — {split_name.upper()}")
    print(f"{'='*70}")

    none_counts = []
    for s in samples:
        indices = s['radar_indices']
        n_none = sum(1 for idx in indices if idx is None)
        none_counts.append(n_none)

    none_counts = np.array(none_counts)
    n_scans = len(samples[0]['radar_indices'])

    print(f"\n  Samples with all slots filled:   {(none_counts == 0).sum():,} ({100*(none_counts == 0).mean():.1f}%)")
    print(f"  Samples with some None slots:    {(none_counts > 0).sum():,} ({100*(none_counts > 0).mean():.1f}%)")
    print(f"  Avg None slots per sample:       {none_counts.mean():.1f}")

    print(f"\n  Distribution of None slots per sample:")
    for k in range(n_scans + 1):
        count = (none_counts == k).sum()
        if count > 0:
            print(f"    {k:2d} None slots: {count:,} samples ({100*count/len(samples):.1f}%)")


def analyze_targets(samples, split_name='train'):
    """Target precipitation distribution."""
    print(f"\n{'='*70}")
    print(f"  TARGET DISTRIBUTION — {split_name.upper()}")
    print(f"{'='*70}")

    targets = np.array([s['hourly_precip_mm'] for s in samples])

    print(f"\n  Range:   [{targets.min():.3f}, {targets.max():.3f}] mm/hr")
    print(f"  Mean:    {targets.mean():.3f} mm/hr")
    print(f"  Median:  {np.median(targets):.3f} mm/hr")
    print(f"  Std:     {targets.std():.3f} mm/hr")

    bins = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 40, 100]
    print(f"\n  Category breakdown:")
    for lo, hi in zip(bins, bins[1:]):
        count = ((targets >= lo) & (targets < hi)).sum()
        print(f"    [{lo:5.1f}, {hi:5.1f}) mm/hr: {count:,} ({100*count/len(targets):.1f}%)")
    count = (targets >= bins[-1]).sum()
    print(f"    [{bins[-1]:5.1f}, inf ) mm/hr: {count:,} ({100*count/len(targets):.1f}%)")


def plot_reflectivity_vs_precip(samples, split_name='train', output_dir='evaluation_figures/stack_dualpol'):
    """Scatter plot of max reflectivity vs measured precipitation."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_dbz_list = []
    precip_list = []

    for s in samples:
        refl = s['radar_patch'][:, 0, :, :]  # (12, H, W)
        max_dbz = np.nanmax(refl)
        if np.isnan(max_dbz):
            continue
        max_dbz_list.append(max_dbz)
        precip_list.append(s['hourly_precip_mm'])

    max_dbz_arr = np.array(max_dbz_list)
    precip_arr = np.array(precip_list)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: full scatter
    ax = axes[0]
    sc = ax.scatter(max_dbz_arr, precip_arr, alpha=0.15, s=10, c='steelblue')
    ax.set_xlabel('Max Reflectivity (dBZ)')
    ax.set_ylabel('Hourly Precipitation (mm/hr)')
    ax.set_title(f'Max Reflectivity vs Precipitation — {split_name}\n(n={len(max_dbz_arr):,})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 70)
    ax.set_ylim(0, min(precip_arr.max() * 1.05, 50))

    # Marshall-Palmer reference: Z = 200 * R^1.6 → R = (Z/200)^(1/1.6)
    dbz_range = np.linspace(0, 60, 100)
    z_linear = 10 ** (dbz_range / 10.0)
    r_mp = (z_linear / 200.0) ** (1.0 / 1.6)
    ax.plot(dbz_range, r_mp, 'r--', lw=2, label='Marshall-Palmer Z-R')
    ax.legend()

    # Right: binned mean/median
    ax = axes[1]
    bins = np.arange(-20, 65, 5)
    bin_centers = []
    bin_means = []
    bin_medians = []
    bin_p75 = []
    bin_counts = []

    for lo, hi in zip(bins, bins[1:]):
        mask = (max_dbz_arr >= lo) & (max_dbz_arr < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_means.append(precip_arr[mask].mean())
            bin_medians.append(np.median(precip_arr[mask]))
            bin_p75.append(np.percentile(precip_arr[mask], 75))
            bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_medians = np.array(bin_medians)
    bin_p75 = np.array(bin_p75)

    ax.plot(bin_centers, bin_means, 'b-o', lw=2, markersize=5, label='Mean precip')
    ax.plot(bin_centers, bin_medians, 'g-s', lw=2, markersize=5, label='Median precip')
    ax.fill_between(bin_centers, bin_medians, bin_p75, alpha=0.2, color='blue', label='IQR (p50-p75)')
    ax.plot(dbz_range, r_mp, 'r--', lw=2, label='Marshall-Palmer')
    ax.set_xlabel('Max Reflectivity (dBZ)')
    ax.set_ylabel('Precipitation (mm/hr)')
    ax.set_title(f'Binned Reflectivity vs Precipitation — {split_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 65)
    ax.set_ylim(0, min(max(bin_p75.max(), bin_means.max()) * 1.2, 30))

    # Annotate bin counts
    for x, n in zip(bin_centers, bin_counts):
        ax.text(x, -0.8, f'{n}', ha='center', fontsize=7, color='gray')

    plt.tight_layout()
    save_path = out / f'reflectivity_vs_precip_{split_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    from models.stack.train import (
        filter_nan_radar, filter_biased_extremes,
        filter_bad_samples, filter_suspect_station_days,
    )

    dataset = load_pickle()

    for split_name in ['train', 'val']:
        samples = dataset[split_name]
        analyze_field_distributions(samples, split_name)
        analyze_nan_scans(samples, split_name)
        analyze_radar_indices(samples, split_name)
        analyze_targets(samples, split_name)
        plot_reflectivity_vs_precip(samples, split_name)

        # Post-filter scatter
        filtered = filter_nan_radar(samples)
        filtered = filter_biased_extremes(filtered)
        filtered = filter_bad_samples(filtered)
        filtered = filter_suspect_station_days(filtered)
        plot_reflectivity_vs_precip(filtered, f'{split_name}_filtered')

    print(f"\n{'='*70}")
    print("  TESTS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
