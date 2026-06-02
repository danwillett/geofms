"""
Inspect vertical structure features in a radar-gauge pickle.

Produces per-field statistics and histograms to sanity-check the new
derived features (echo_top_height, max_z_height, vil, low_level_ref,
column_depth_fraction) before training.

Usage:
    python -m dataset.eval_pickle dataset/outputs/3d/radar_gauge_dataset_vertical_9500.pkl
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PICKLE_FIELD_ORDER = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
    'echo_top_height',
    'max_z_height',
    'vil',
    'low_level_ref',
    'column_depth_fraction',
]

VERTICAL_FIELDS = [
    'echo_top_height',
    'max_z_height',
    'vil',
    'low_level_ref',
    'column_depth_fraction',
]

FIELD_UNITS = {
    'echo_top_height': 'm',
    'max_z_height': 'm',
    'vil': 'kg/m²',
    'low_level_ref': 'dBZ',
    'column_depth_fraction': '(0-1)',
}


def extract_field_values(samples, field_name, max_samples=5000):
    """Extract all pixel values for a given field across samples (center scan)."""
    field_idx = PICKLE_FIELD_ORDER.index(field_name)
    rng = np.random.default_rng(42)

    if len(samples) > max_samples:
        indices = rng.choice(len(samples), max_samples, replace=False)
    else:
        indices = range(len(samples))

    values = []
    for i in indices:
        patch = samples[i]['radar_patch']  # (12, n_fields, H, W)
        field_data = patch[:, field_idx, :, :]  # (12, H, W)
        valid_mask = ~np.isnan(field_data) & (field_data != 0)
        values.append(field_data[valid_mask].flatten())

    return np.concatenate(values) if values else np.array([])


def print_stats(name, vals):
    """Print summary statistics for a field."""
    print(f"\n{'─'*60}")
    print(f"  {name} ({FIELD_UNITS[name]})")
    print(f"{'─'*60}")
    print(f"  Count (non-zero): {len(vals):,}")
    if len(vals) == 0:
        print("  ⚠ No non-zero values found!")
        return
    print(f"  Mean:    {np.mean(vals):.4f}")
    print(f"  Median:  {np.median(vals):.4f}")
    print(f"  Std:     {np.std(vals):.4f}")
    print(f"  Min:     {np.min(vals):.4f}")
    print(f"  P5:      {np.percentile(vals, 5):.4f}")
    print(f"  P25:     {np.percentile(vals, 25):.4f}")
    print(f"  P75:     {np.percentile(vals, 75):.4f}")
    print(f"  P95:     {np.percentile(vals, 95):.4f}")
    print(f"  P99:     {np.percentile(vals, 99):.4f}")
    print(f"  Max:     {np.max(vals):.4f}")


def correlate_with_precip(samples, field_name, max_samples=5000):
    """Get center-pixel field mean vs precipitation for scatter context."""
    field_idx = PICKLE_FIELD_ORDER.index(field_name)
    rng = np.random.default_rng(42)

    if len(samples) > max_samples:
        indices = rng.choice(len(samples), max_samples, replace=False)
    else:
        indices = range(len(samples))

    field_means = []
    precip_vals = []
    for i in indices:
        s = samples[i]
        patch = s['radar_patch']  # (12, n_fields, H, W)
        cy, cx = patch.shape[2] // 2, patch.shape[3] // 2
        center_vals = patch[:, field_idx, cy, cx]
        valid = center_vals[~np.isnan(center_vals)]
        if len(valid) > 0:
            field_means.append(np.mean(valid))
            precip_vals.append(s['hourly_precip_mm'])

    return np.array(field_means), np.array(precip_vals)


def plot_histograms(field_values_dict, output_path):
    """Create a figure with histograms for each vertical field."""
    n_fields = len(VERTICAL_FIELDS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, field in enumerate(VERTICAL_FIELDS):
        ax = axes[i]
        vals = field_values_dict[field]
        if len(vals) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(field)
            continue

        p1, p99 = np.percentile(vals, [1, 99])
        clipped = vals[(vals >= p1) & (vals <= p99)]
        ax.hist(clipped, bins=80, color='steelblue', alpha=0.8, edgecolor='none')
        ax.axvline(np.median(vals), color='red', ls='--', lw=1.5, label=f'median={np.median(vals):.1f}')
        ax.set_title(f'{field} ({FIELD_UNITS[field]})', fontsize=11, fontweight='bold')
        ax.set_xlabel(FIELD_UNITS[field])
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Use last subplot for a summary text
    ax = axes[-1]
    ax.axis('off')
    summary_lines = []
    for field in VERTICAL_FIELDS:
        vals = field_values_dict[field]
        if len(vals) > 0:
            summary_lines.append(f"{field}: μ={np.mean(vals):.2f}, σ={np.std(vals):.2f}")
    ax.text(0.1, 0.9, "Summary (non-zero pixels)\n" + "\n".join(summary_lines),
            transform=ax.transAxes, va='top', fontsize=10, family='monospace')

    plt.suptitle('Vertical Structure Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Histograms saved → {output_path}")
    plt.close()


def plot_precip_correlation(samples, output_path):
    """Scatter plots of field center-pixel mean vs hourly precip."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, field in enumerate(VERTICAL_FIELDS):
        ax = axes[i]
        fmeans, precip = correlate_with_precip(samples, field)
        if len(fmeans) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(field)
            continue

        ax.scatter(fmeans, precip, alpha=0.15, s=8, color='steelblue')
        ax.set_xlabel(f'{field} ({FIELD_UNITS[field]})')
        ax.set_ylabel('Hourly Precip (mm)')
        ax.set_title(field, fontsize=11, fontweight='bold')

        valid = ~np.isnan(fmeans) & ~np.isnan(precip)
        if valid.sum() > 10:
            r = np.corrcoef(fmeans[valid], precip[valid])[0, 1]
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                    va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].axis('off')
    plt.suptitle('Vertical Features vs Hourly Precipitation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Correlations saved → {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate vertical structure features in pickle')
    parser.add_argument('pickle_path', type=str, help='Path to the pickle file')
    parser.add_argument('--max-samples', type=int, default=5000, help='Max samples to analyze')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Which split to analyze')
    args = parser.parse_args()

    pkl_path = Path(args.pickle_path)
    output_dir = pkl_path.parent

    print(f"Loading pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    samples = dataset[args.split]
    print(f"Split: {args.split} ({len(samples):,} samples)")

    patch = samples[0]['radar_patch']
    print(f"Patch shape: {patch.shape}  (scans, fields, H, W)")
    print(f"Fields in pickle: {patch.shape[1]}")

    if patch.shape[1] < 10:
        print("\n⚠ Pickle only has {patch.shape[1]} fields — vertical features not present!")
        print("  Regenerate with the updated create_pickle.py that includes vertical fields.")
        return

    field_values = {}
    for field in VERTICAL_FIELDS:
        print(f"\n  Extracting {field}...", end='', flush=True)
        vals = extract_field_values(samples, field, max_samples=args.max_samples)
        field_values[field] = vals
        print(f" {len(vals):,} values")

    print("\n" + "=" * 60)
    print("  VERTICAL STRUCTURE FEATURE STATISTICS")
    print("=" * 60)
    for field in VERTICAL_FIELDS:
        print_stats(field, field_values[field])

    hist_path = output_dir / f'vertical_features_hist_{args.split}.png'
    plot_histograms(field_values, hist_path)

    corr_path = output_dir / f'vertical_features_vs_precip_{args.split}.png'
    plot_precip_correlation(samples, corr_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
