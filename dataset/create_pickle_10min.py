"""
create_pickle_10min.py — Build a training/validation pickle with 10-minute resolution.

Each sample pairs a SINGLE radar scan with the 10-minute precipitation accumulation
at the nearest gauge. This produces ~6x more samples than the hourly approach and
simplifies the model task (no temporal integration required).

Patch shape stored: (n_fields, patch_pixels, patch_pixels)
  n_fields = 5 (reflectivity, ZDR, RhoHV, PhiDP, KDP)

Usage:
    python -m dataset.create_pickle_10min \
        --radar radar/outputs/dualpol_500m_2022-01-01_2026-04-04.zarr \
        --days weather/days/top_100_days_2022-01-01_2026-04-04.txt \
        --patch-size 4500 \
        --train-years 2022 2024 2026 --val-years 2023 2025 \
        --output dataset/outputs/10min/radar_gauge_10min.pkl
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gc
import pickle
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer, CRS
from tqdm import tqdm

from weather.pull_weather import get_hourly_precipitation_by_station

# ── Dual-pol fields ───────────────────────────────────────────────────────────
FIELDS = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
]


# ── Radar patch extraction (single scan) ──────────────────────────────────────

def extract_single_scan_patch(radar_ds, scan_idx, station_lat, station_lon, patch_size_m=4500):
    """
    Extract a multi-field radar patch for a single scan centred on a gauge station.

    Returns
    -------
    patch : ndarray, shape (n_fields, patch_pixels, patch_pixels)
        -9999.0 sentinel preserved for missing field data.
    """
    n_fields = len(FIELDS)

    try:
        radar_crs_str = radar_ds[FIELDS[0]].attrs.get('crs', 'EPSG:32610')
        radar_crs = CRS.from_string(radar_crs_str)
        wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(wgs84, radar_crs, always_xy=True)
        station_x, station_y = transformer.transform(station_lon, station_lat)

        resolution_m = radar_ds.attrs.get('resolution_m', 500)
        patch_pixels = int(patch_size_m / resolution_m)
        half_pixels = patch_pixels // 2

        x_idx = int(np.abs(radar_ds.x.values - station_x).argmin())
        y_idx = int(np.abs(radar_ds.y.values - station_y).argmin())

        x_start = max(0, x_idx - half_pixels)
        x_end = x_start + patch_pixels
        y_start = max(0, y_idx - half_pixels)
        y_end = y_start + patch_pixels

        if x_end > len(radar_ds.x):
            x_end = len(radar_ds.x)
            x_start = max(0, x_end - patch_pixels)
        if y_end > len(radar_ds.y):
            y_end = len(radar_ds.y)
            y_start = max(0, y_end - patch_pixels)

        out = np.full((n_fields, patch_pixels, patch_pixels), np.nan, dtype=np.float32)

        for f_idx, field in enumerate(FIELDS):
            if field not in radar_ds:
                continue
            data = radar_ds[field].isel(time=scan_idx, y=slice(y_start, y_end), x=slice(x_start, x_end)).values
            h, w = data.shape
            out[f_idx, :h, :w] = data

        return out

    except Exception as e:
        resolution_m = radar_ds.attrs.get('resolution_m', 500)
        patch_pixels = int(patch_size_m / resolution_m)
        return np.full((n_fields, patch_pixels, patch_pixels), np.nan, dtype=np.float32)


# ── 10-min precipitation from database ────────────────────────────────────────

def get_10min_precipitation_by_station(start_date=None, end_date=None, min_rainfall_mm=0.0):
    """
    Get raw 10-minute precipitation measurements for each station.
    
    Each datapoint in the database is already a 10-min accumulation.
    We floor timestamps to 10-min bins and sum (in case of multiple reports per bin).
    """
    from database.config import connect, create_session
    from database.models import DendraStation, DendraDatastream, DendraDatapoint

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    engine = connect()
    session = create_session(engine)

    rain_ds = session.query(DendraDatastream).filter(
        DendraDatastream.name.in_(["Rainfall", "Rainfall Sum"])
    ).all()
    rain_ds_ids = [ds.id for ds in rain_ds]
    print(f"Found {len(rain_ds_ids)} rainfall datastream(s)")

    station_info = {}
    for ds in rain_ds:
        station = session.query(DendraStation).filter(
            DendraStation.id == ds.station_id
        ).first()
        if station:
            station_info[ds.id] = {
                'station_id': station.id,
                'name': station.name,
                'lat': station.latitude,
                'lon': station.longitude
            }

    query = session.query(
        DendraDatapoint.timestamp_utc,
        DendraDatapoint.datastream_id,
        DendraDatapoint.value
    ).filter(DendraDatapoint.datastream_id.in_(rain_ds_ids))

    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(DendraDatapoint.timestamp_utc >= start_datetime)
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        query = query.filter(DendraDatapoint.timestamp_utc < end_datetime)

    query = query.order_by(DendraDatapoint.timestamp_utc)

    print(f"Querying 10-min precipitation data from {start_date} to {end_date}...")
    results = query.all()
    print(f"Found {len(results)} raw measurements")

    if len(results) == 0:
        return []

    df = pd.DataFrame(results, columns=['timestamp_utc', 'datastream_id', 'rainfall_mm'])

    df['station_id'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('station_id'))
    df['station_name'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('name'))
    df['lat'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lat'))
    df['lon'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lon'))

    # Floor to 10-min bins and sum within each bin
    df['bin_10min'] = df['timestamp_utc'].dt.floor('10min')

    binned = df.groupby(['bin_10min', 'station_id']).agg({
        'rainfall_mm': 'sum',
        'station_name': 'first',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()

    # Filter by minimum rainfall
    binned = binned[binned['rainfall_mm'] >= min_rainfall_mm]

    samples = []
    for _, row in binned.iterrows():
        samples.append({
            'bin_start': row['bin_10min'],
            'station_id': row['station_id'],
            'station_name': row['station_name'],
            'lat': row['lat'],
            'lon': row['lon'],
            'precip_mm': row['rainfall_mm']
        })

    print(f"\nFound {len(samples)} 10-min samples with rainfall >= {min_rainfall_mm}mm")
    print(f"  Covering {len(set(s['station_id'] for s in samples))} stations")
    if samples:
        print(f"  Time range: {min(s['bin_start'] for s in samples)} to {max(s['bin_start'] for s in samples)}")
        print(f"  Precip range: {min(s['precip_mm'] for s in samples):.3f} - {max(s['precip_mm'] for s in samples):.2f} mm")

    return samples


# ── Find closest radar scan to a given time ───────────────────────────────────

def find_closest_scan(radar_times, target_time, max_offset_minutes=5):
    """
    Find the radar scan index closest to target_time.
    Returns None if no scan is within max_offset_minutes.
    """
    target_np = np.datetime64(target_time, 'ns')
    diffs = np.abs(radar_times - target_np)
    min_idx = int(np.argmin(diffs))
    min_diff_minutes = diffs[min_idx] / np.timedelta64(1, 'm')

    if min_diff_minutes <= max_offset_minutes:
        return min_idx
    return None


# ── Main sample creation ──────────────────────────────────────────────────────

def create_10min_samples(
    radar_zarr_path,
    output_path,
    train_years=None,
    val_years=None,
    start_date=None,
    end_date=None,
    day_filter_file=None,
    min_rainfall_mm=0.0,
    max_valid_rainfall=50.0,
    patch_size_m=4500,
):
    """
    Create 10-minute resolution radar-gauge samples.
    Each sample: single radar scan + 10-min precipitation accumulation.
    """
    print("\n" + "=" * 60)
    print("  10-MINUTE RADAR-GAUGE DATA ALIGNMENT")
    print("=" * 60)

    # --- dates ---------------------------------------------------------------
    if day_filter_file:
        print(f"\nLoading dates from: {day_filter_file}")
        with open(day_filter_file) as f:
            dates = [
                datetime.strptime(l.strip(), '%Y-%m-%d').date()
                for l in f if l.strip() and not l.startswith('#')
            ]
        if not dates:
            raise ValueError("No dates found in filter file!")
        print(f"  → {len(dates)} days")
        start_date = min(dates)
        end_date = max(dates)
    else:
        print(f"\nDate range: {start_date} → {end_date}")
        dates = None

    # --- load zarr -----------------------------------------------------------
    print(f"\n1. Loading radar zarr: {radar_zarr_path}")
    import zarr as _zarr

    _store = _zarr.open(radar_zarr_path, mode='r')

    min_time_len = None
    present = []
    for field in FIELDS:
        if field in _store:
            present.append(field)
            t_len = _store[field].shape[0]
            if min_time_len is None or t_len < min_time_len:
                min_time_len = t_len

    if not present:
        raise ValueError(f"No recognised fields in zarr! Available: {list(_store.keys())}")

    print(f"  Fields present: {present}")
    print(f"  Truncating all fields to {min_time_len} time steps")

    data_vars = {}
    for field in present:
        arr = _store[field][:min_time_len]
        data_vars[field] = xr.DataArray(
            arr,
            dims=('time', 'y', 'x'),
            attrs=dict(_store[field].attrs),
        )

    time_raw = _store['time'][:min_time_len]
    if hasattr(time_raw, 'filled'):
        time_raw = time_raw.filled(fill_value=np.datetime64('NaT'))
    time_vals = pd.to_datetime(time_raw)

    coords = {
        'time': time_vals,
        'y': _store['y'][:],
        'x': _store['x'][:],
    }
    top_attrs = dict(_store.attrs)
    radar_ds = xr.Dataset(data_vars, coords=coords, attrs=top_attrs)

    print("  Loading into memory…")
    radar_ds = radar_ds.load()
    print("  ✓ Loaded")

    radar_times = radar_ds.time.values

    if pd.isna(radar_times).all():
        print("  ⚠ All time values are NaT!")
    else:
        print(f"  Time range: {radar_ds.time.min().values} → {radar_ds.time.max().values}")
        print(f"  Total scans: {len(radar_times)}")

    # --- 10-min precipitation ------------------------------------------------
    print(f"\n2. Loading 10-min precipitation (min={min_rainfall_mm} mm)…")
    precip_samples = get_10min_precipitation_by_station(
        start_date, end_date, min_rainfall_mm=min_rainfall_mm
    )
    if dates:
        precip_samples = [s for s in precip_samples if s['bin_start'].date() in set(dates)]
    print(f"  → {len(precip_samples)} 10-min station measurements after date filter")

    if not precip_samples:
        raise ValueError("No 10-min precipitation found!")

    # --- build samples -------------------------------------------------------
    print(f"\n3. Extracting single-scan radar patches (patch_size_m={patch_size_m})…")
    print(f"   Skipping readings > {max_valid_rainfall} mm/10min")

    samples = []
    skipped_outliers = 0
    skipped_no_scan = 0

    for precip in tqdm(precip_samples, desc="Samples"):
        if precip['precip_mm'] > max_valid_rainfall:
            skipped_outliers += 1
            continue

        # Find the scan closest to the midpoint of the 10-min window
        bin_midpoint = precip['bin_start'] + timedelta(minutes=5)
        scan_idx = find_closest_scan(radar_times, bin_midpoint, max_offset_minutes=5)

        if scan_idx is None:
            skipped_no_scan += 1
            continue

        radar_patch = extract_single_scan_patch(
            radar_ds, scan_idx,
            precip['lat'], precip['lon'],
            patch_size_m=patch_size_m,
        )

        scan_time = pd.Timestamp(radar_times[scan_idx]).to_pydatetime()

        samples.append({
            'bin_start': precip['bin_start'],
            'scan_time': scan_time,
            'scan_idx': scan_idx,
            'station_id': precip['station_id'],
            'station_name': precip['station_name'],
            'station_lat': precip['lat'],
            'station_lon': precip['lon'],
            'precip_mm': precip['precip_mm'],
            'radar_patch': radar_patch,
        })

        if len(samples) % 5000 == 0:
            gc.collect()

    print(f"\n  Created {len(samples)} samples")
    print(f"  Skipped: {skipped_outliers} outliers, {skipped_no_scan} no-scan-available")

    # --- train/val split -----------------------------------------------------
    if train_years and val_years:
        print(f"\n4. Temporal split: train={train_years}, val={val_years}")
        train_samples = [s for s in samples if s['bin_start'].year in train_years]
        val_samples = [s for s in samples if s['bin_start'].year in val_years]
    else:
        print("\n4. Random 80/20 split")
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        split_idx = int(0.8 * len(samples))
        train_samples = [samples[i] for i in indices[:split_idx]]
        val_samples = [samples[i] for i in indices[split_idx:]]

    print(f"   Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    # --- save ----------------------------------------------------------------
    patch_pixels = int(patch_size_m / radar_ds.attrs.get('resolution_m', 500))
    dataset = {
        'train': train_samples,
        'val': val_samples,
        'metadata': {
            'radar_zarr': radar_zarr_path,
            'fields': FIELDS,
            'n_fields': len(present),
            'patch_size_m': patch_size_m,
            'patch_pixels': patch_pixels,
            'radar_patch_shape': f"({len(present)}, {patch_pixels}, {patch_pixels})",
            'temporal_resolution': '10min',
            'scans_per_sample': 1,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'day_filter_file': day_filter_file,
            'split_type': 'temporal' if train_years else 'random',
            'train_years': train_years or 'N/A',
            'val_years': val_years or 'N/A',
            'created': datetime.now().isoformat(),
            'n_train': len(train_samples),
            'n_val': len(val_samples),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\n5. Saving to {output_path}…")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    # --- summary -------------------------------------------------------------
    print(f"\n✅ Dataset saved!")
    print("=" * 60)
    all_s = train_samples + val_samples
    rainy = [s for s in all_s if s['precip_mm'] >= 0.1]
    print(f"  Total samples   : {len(all_s)}")
    print(f"  Stations        : {len(set(s['station_id'] for s in all_s))}")
    print(f"  Rainy (≥0.1 mm) : {len(rainy)} ({100*len(rainy)/max(len(all_s),1):.1f}%)")
    print(f"  Precip range    : {np.min([s['precip_mm'] for s in all_s]):.3f}"
          f" – {np.max([s['precip_mm'] for s in all_s]):.2f} mm/10min")
    print(f"  Patch shape     : {dataset['metadata']['radar_patch_shape']}")

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create 10-min radar-gauge pickle (single scan)")

    parser.add_argument('--radar', required=True,
                        help='Path to dual-pol radar zarr')
    parser.add_argument('--output', default='dataset/outputs/10min/radar_gauge_10min.pkl',
                        help='Output pickle path')
    parser.add_argument('--days', default=None,
                        help='Path to date-filter file')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--patch-size', type=int, default=4500,
                        help='Patch size in metres (default: 4500 → 9×9 @ 500 m/px)')
    parser.add_argument('--min-rainfall', type=float, default=0.0,
                        help='Min 10-min rainfall mm (default: 0.0)')
    parser.add_argument('--max-rainfall', type=float, default=50.0,
                        help='Max valid rainfall mm/10min (default: 50.0)')
    parser.add_argument('--train-years', type=int, nargs='+', default=None)
    parser.add_argument('--val-years', type=int, nargs='+', default=None)

    args = parser.parse_args()

    if not args.days and not (args.start and args.end):
        parser.error("Provide --days OR both --start and --end")
    if bool(args.train_years) != bool(args.val_years):
        parser.error("Provide both --train-years and --val-years, or neither")

    create_10min_samples(
        radar_zarr_path=args.radar,
        output_path=args.output,
        train_years=args.train_years,
        val_years=args.val_years,
        start_date=args.start,
        end_date=args.end,
        day_filter_file=args.days,
        min_rainfall_mm=args.min_rainfall,
        max_valid_rainfall=args.max_rainfall,
        patch_size_m=args.patch_size,
    )
