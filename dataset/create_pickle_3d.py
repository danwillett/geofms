"""
create_pickle_3d.py — Build a lazy-index radar-gauge pickle for 3D volume training.

Stores scan indices and spatial windows only; volumes are read from zarr at train time.

Run from the project root:
    python -m dataset.create_pickle_3d \\
        --radar radar/outputs/dualpol_3d_500m_2022-01-01_2026-04-04.zarr \\
        --days weather/days/top_100_days_2022-01-01_2026-04-04.txt \\
        --patch-size 4500 \\
        --z-max 4000 \\
        --output dataset/outputs/3d/radar_gauge_dataset_3d_4500.pkl
"""

import argparse
import gc
import pickle
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm

from dataset.create_pickle import get_station_bias
from dataset.volume_io import (
    RAW_FIELDS_3D,
    bin_scans_to_fixed_slots,
    compute_spatial_window,
    max_reflectivity_over_scans,
    open_3d_zarr_store,
    resolve_z_indices,
    sample_radar_scans_for_hour,
)
from weather.pull_weather import get_hourly_precipitation_by_station, get_offset_hourly_precipitation_by_station


def build_index_sample(
    precip,
    store,
    time_vals,
    x_vals,
    y_vals,
    z_indices,
    patch_size_m,
    resolution_m,
    crs,
):
    """Create one lazy-load index sample (no volume arrays)."""
    radar_times, radar_indices = sample_radar_scans_for_hour(
        time_vals, precip['hour_start'], n_scans=12
    )
    if len(radar_indices) < 6:
        return None

    radar_times, radar_indices = bin_scans_to_fixed_slots(
        radar_times, radar_indices, precip['hour_start'], n_bins=12
    )

    window = compute_spatial_window(
        precip['lat'], precip['lon'],
        x_vals, y_vals, patch_size_m, resolution_m, crs=crs,
    )
    max_dbz = max_reflectivity_over_scans(store, radar_indices, z_indices, window)
    if np.isnan(max_dbz):
        return None

    return {
        'hour_start': precip['hour_start'],
        'station_id': precip['station_id'],
        'station_name': precip['station_name'],
        'station_lat': precip['lat'],
        'station_lon': precip['lon'],
        'bias_flag': get_station_bias(precip['station_name']),
        'hourly_precip_mm': precip['hourly_precip_mm'],
        'dump_ratio': precip.get('dump_ratio'),
        'max_bin_mm': precip.get('max_bin_mm'),
        'n_active_bins': precip.get('n_active_bins'),
        'radar_times': radar_times,
        'radar_indices': radar_indices,
        'n_valid_radar': sum(1 for idx in radar_indices if idx is not None),
        'max_reflectivity_dbz': max_dbz,
        **window,
    }


def create_training_samples(
    radar_zarr_path,
    output_path,
    dem_path=None,
    train_years=None,
    val_years=None,
    start_date=None,
    end_date=None,
    day_filter_file=None,
    min_rainfall_mm=0.0,
    max_valid_rainfall=100.0,
    patch_size_m=4500,
    z_max_m=4000.0,
    z_stride=1,
    half_hour_offsets=False,
    include_test=False,
):
    print("=" * 60)
    print("3D RADAR-GAUGE LAZY INDEX DATASET")
    print("=" * 60)

    if day_filter_file:
        print(f"\nLoading dates from: {day_filter_file}")
        with open(day_filter_file) as f:
            dates = [
                datetime.strptime(line.strip(), '%Y-%m-%d').date()
                for line in f if line.strip() and not line.startswith('#')
            ]
        if not dates:
            raise ValueError("No dates found in filter file!")
        print(f"  → {len(dates)} days")
        start_date = min(dates)
        end_date = max(dates)
    else:
        print(f"\nDate range: {start_date} → {end_date}")
        dates = None

    print(f"\n1. Opening 3D radar zarr (metadata only): {radar_zarr_path}")
    store, present, min_time, time_vals, x_vals, y_vals, z_vals, resolution_m, crs = (
        open_3d_zarr_store(radar_zarr_path)
    )
    z_indices = resolve_z_indices(z_vals, z_max_m, z_stride)
    print(f"  Fields present : {present}")
    print(f"  Time steps     : {min_time}")
    print(f"  Z levels used  : {len(z_indices)} / {len(z_vals)} (≤ {z_max_m} m, stride={z_stride})")
    print(f"  Resolution     : {resolution_m} m")

    if dem_path:
        print(f"\n  DEM will be extracted on-the-fly from: {dem_path}")

    print(f"\n2. Loading hourly precipitation (min={min_rainfall_mm} mm)…")
    hourly_precip = get_hourly_precipitation_by_station(
        start_date, end_date, min_rainfall_mm=min_rainfall_mm
    )
    if dates:
        date_set = set(dates)
        hourly_precip = [h for h in hourly_precip if h['hour_start'].date() in date_set]
    print(f"  → {len(hourly_precip)} station-hours")

    if not hourly_precip:
        raise ValueError("No hourly precipitation found — check date range / DB connection.")

    print(f"\n3. Building lazy index samples (patch_size_m={patch_size_m})…")
    samples = []
    skipped_outliers = 0

    for i, precip in enumerate(tqdm(hourly_precip, desc="Hours")):
        if precip['hourly_precip_mm'] > max_valid_rainfall:
            skipped_outliers += 1
            continue

        sample = build_index_sample(
            precip, store, time_vals, x_vals, y_vals, z_indices,
            patch_size_m, resolution_m, crs,
        )
        if sample is not None:
            samples.append(sample)

        if i % 500 == 0:
            gc.collect()

    print(f"\n4. Created {len(samples)} samples")
    if skipped_outliers:
        print(f"   ⚠ Skipped {skipped_outliers} outlier readings (>{max_valid_rainfall} mm/hr)")

    print("\n5. Train/val split…")
    if train_years and val_years:
        print(f"   Temporal split — train: {train_years}  val: {val_years}")
        train_samples, val_samples = [], []
        for s in samples:
            yr = s['hour_start'].year
            if yr in train_years:
                train_samples.append(s)
            elif yr in val_years:
                val_samples.append(s)
        if not train_samples:
            raise ValueError(f"No training samples for years {train_years}!")
        if not val_samples:
            raise ValueError(f"No validation samples for years {val_years}!")
    else:
        print("   Random 80/20 split")
        np.random.seed(42)
        idx = np.random.permutation(len(samples))
        split = int(0.8 * len(samples))
        train_samples = [samples[i] for i in idx[:split]]
        val_samples = [samples[i] for i in idx[split:]]

    print(f"   Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    if half_hour_offsets:
        print("\n5b. Generating 30-min offset samples (training only)…")
        offset_precip = get_offset_hourly_precipitation_by_station(
            start_date, end_date, min_rainfall_mm=min_rainfall_mm, offset_minutes=30
        )
        if dates:
            offset_precip = [h for h in offset_precip if h['hour_start'].date() in set(dates)]
        if train_years:
            offset_precip = [h for h in offset_precip if h['hour_start'].year in train_years]

        offset_samples = []
        for i, precip in enumerate(tqdm(offset_precip, desc="Offset hours")):
            if precip['hourly_precip_mm'] > max_valid_rainfall:
                continue
            sample = build_index_sample(
                precip, store, time_vals, x_vals, y_vals, z_indices,
                patch_size_m, resolution_m, crs,
            )
            if sample is not None:
                sample['is_offset'] = True
                offset_samples.append(sample)
            if i % 500 == 0:
                gc.collect()

        print(f"   Created {len(offset_samples)} offset samples for training")
        train_samples.extend(offset_samples)
        print(f"   Train total (with offsets): {len(train_samples)}  |  Val: {len(val_samples)}")

    test_samples = []
    if include_test:
        print("\n  ⚠ --include-test not yet implemented for 3D lazy pickles; skipping test set.")

    patch_pixels = int(patch_size_m / resolution_m)
    dataset = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
        'metadata': {
            'radar_zarr_path': radar_zarr_path,
            'lazy_load': True,
            'dem_path': dem_path,
            'fields': present,
            'n_fields': len(present),
            'patch_size_m': patch_size_m,
            'patch_pixels': patch_pixels,
            'z_max_m': z_max_m,
            'z_stride': z_stride,
            'z_indices': z_indices.tolist(),
            'n_z_levels': len(z_indices),
            'z_values_m': z_vals[z_indices].tolist(),
            'resolution_m': resolution_m,
            'crs': crs,
            'volume_shape': f"(12, {len(present)}, {len(z_indices)}, {patch_pixels}, {patch_pixels})",
            'start_date': str(start_date),
            'end_date': str(end_date),
            'day_filter_file': day_filter_file,
            'specific_days': [str(d) for d in dates] if dates else None,
            'split_type': 'temporal' if train_years else 'random',
            'train_years': train_years or 'N/A',
            'val_years': val_years or 'N/A',
            'created': datetime.now().isoformat(),
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'n_test': len(test_samples),
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n6. Saving to {output_path}…")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    all_s = train_samples + val_samples
    print(f"\n✅ Dataset saved!")
    print("=" * 60)
    print(f"  Total samples   : {len(all_s)}")
    print(f"  Stations        : {len(set(s['station_id'] for s in all_s))}")
    print(f"  Avg valid scans : {np.mean([s['n_valid_radar'] for s in all_s]):.1f} / 12")
    print(f"  Volume shape    : {dataset['metadata']['volume_shape']}")
    return str(output_path)


def inspect_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print("=" * 60)
    print("3D LAZY DATASET INSPECTION")
    print("=" * 60)
    print("\nMetadata:")
    for k, v in dataset['metadata'].items():
        print(f"  {k}: {v}")

    print(f"\nTrain: {len(dataset['train'])}  |  Val: {len(dataset['val'])}")
    if dataset['train']:
        s = dataset['train'][0]
        print("\nFirst training sample:")
        print(f"  Hour         : {s['hour_start']}")
        print(f"  Station      : {s['station_name']}")
        print(f"  Window       : y=[{s['y_start']}:{s['y_end']}] x=[{s['x_start']}:{s['x_end']}]")
        print(f"  Max reflectivity: {s.get('max_reflectivity_dbz', 'N/A')} dBZ")
        print(f"  Rainfall     : {s['hourly_precip_mm']:.2f} mm/hr")
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create lazy-index 3D radar-gauge pickle")
    parser.add_argument('--radar', default=None, help='Path to dualpol_3d zarr (required unless --inspect)')
    parser.add_argument('--output', default='dataset/outputs/3d/radar_gauge_dataset_3d.pkl')
    parser.add_argument('--days', default=None, help='Path to date-filter file')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--dem', default=None, help='DEM GeoTIFF path')
    parser.add_argument('--patch-size', type=int, default=4500,
                        help='Patch size in metres (default: 4500 → 9×9 @ 500 m/px)')
    parser.add_argument('--z-max', type=float, default=4000.0,
                        help='Max height in metres for Z gate (default: 4000)')
    parser.add_argument('--z-stride', type=int, default=1,
                        help='Subsample Z levels (default: 1 = all gates ≤ z-max)')
    parser.add_argument('--min-rainfall', type=float, default=None)
    parser.add_argument('--max-rainfall', type=float, default=100.0)
    parser.add_argument('--train-years', type=int, nargs='+', default=None)
    parser.add_argument('--val-years', type=int, nargs='+', default=None)
    parser.add_argument('--half-hour-offsets', action='store_true')
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--inspect', action='store_true')

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.output)
    else:
        if not args.radar:
            parser.error("--radar is required when building the pickle (omit it only with --inspect)")
        if not args.days and not (args.start and args.end):
            parser.error("Provide --days OR both --start and --end")
        if bool(args.train_years) != bool(args.val_years):
            parser.error("Provide both --train-years and --val-years, or neither")

        min_rain = args.min_rainfall
        if min_rain is None:
            min_rain = 0.0 if args.days else 0.5

        create_training_samples(
            radar_zarr_path=args.radar,
            output_path=args.output,
            dem_path=args.dem,
            train_years=args.train_years,
            val_years=args.val_years,
            start_date=args.start,
            end_date=args.end,
            day_filter_file=args.days,
            min_rainfall_mm=min_rain,
            max_valid_rainfall=args.max_rainfall,
            patch_size_m=args.patch_size,
            z_max_m=args.z_max,
            z_stride=args.z_stride,
            half_hour_offsets=args.half_hour_offsets,
            include_test=args.include_test,
        )
