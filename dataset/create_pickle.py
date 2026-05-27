"""
create_pickle.py — Build a training/validation pickle from the dual-pol zarr.

Patch shape stored: (12, n_fields, patch_pixels, patch_pixels)
  n_fields = 5 (reflectivity, ZDR, RhoHV, PhiDP, KDP)
  patch_pixels = patch_size_m / 500  (e.g. 9 for 4500m, 5 for 2500m)

Run from the project root:
    python -m pickle.create_pickle --radar KVBX_preserve_500m.zarr \
        --days my_rainy_days_150.txt \
        --output radar_gauge_dataset.pkl \
        --train-years 2020 2021 2022 2023 \
        --val-years 2024 2025 \
        --dem dem/preserve_dem_10m_utm.tif \
        --patch-size 4500
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

from weather.pull_weather import get_hourly_precipitation_by_station, get_offset_hourly_precipitation_by_station

# ── Dual-pol fields — must match RadarDEMDataset.FIELDS exactly ──────────────
FIELDS = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
]

# Station bias flags  (1 = over-estimator, -1 = under-estimator, 0 = unknown)
STATION_BIAS = {
    'Dangermond_Bunker Hill': 1,
    'Dangermond_Cistern':     1,
    'Dangermond_Cojo HQ':     1,
    'Dangermond_Jalachichi':  1,
    'Dangermond_Repeater':    1,
    'Dangermond_Cojo Gate':  -1,
    'Dangermond_Sutter':     -1,
}


def get_station_bias(station_name):
    return STATION_BIAS.get(station_name, 0)


# ── Scan sampling ─────────────────────────────────────────────────────────────

def sample_radar_scans_for_hour(radar_ds, hour_start, n_scans=12):
    """
    Return up to n_scans evenly-spaced radar time indices falling in [hour_start, hour_start+1h).
    Returns (list[datetime], list[int]).
    """
    hour_end = hour_start + timedelta(hours=1)
    mask = (
        (radar_ds.time.values >= np.datetime64(hour_start)) &
        (radar_ds.time.values <  np.datetime64(hour_end))
    )
    hour_indices = np.where(mask)[0]

    if len(hour_indices) == 0:
        return [], []

    if len(hour_indices) >= n_scans:
        sample_pos = np.linspace(0, len(hour_indices) - 1, n_scans, dtype=int)
        selected   = hour_indices[sample_pos]
    else:
        selected = hour_indices

    times   = [pd.Timestamp(radar_ds.time.values[i]).to_pydatetime() for i in selected]
    indices = selected.tolist()
    return times, indices


def bin_scans_to_fixed_slots(radar_times, radar_indices, hour_start, n_bins=12):
    """
    Map variable-count scans into n_bins fixed 5-minute slots.
    Slots with no scan are None.  Returns (binned_times, binned_indices).
    """
    bin_minutes = 60 / n_bins
    binned_times   = [None] * n_bins
    binned_indices = [None] * n_bins

    for scan_time, scan_idx in zip(radar_times, radar_indices):
        minutes_in = (scan_time - hour_start).total_seconds() / 60
        slot = int(minutes_in / bin_minutes)
        slot = max(0, min(n_bins - 1, slot))
        if binned_indices[slot] is None:   # keep first if collision
            binned_indices[slot] = scan_idx
            binned_times[slot]   = scan_time

    return binned_times, binned_indices


# ── Dual-pol patch extraction ─────────────────────────────────────────────────

def extract_radar_patch_at_station(radar_ds, time_indices, station_lat, station_lon,
                                   patch_size_m=4500):
    """
    Extract a multi-field radar patch centred on a gauge station.

    Returns
    -------
    patch : ndarray, shape (n_times, n_fields, patch_pixels, patch_pixels)
        NaN for missing scans; -9999.0 sentinel preserved for missing field data.
    """
    n_times  = len(time_indices)
    n_fields = len(FIELDS)

    try:
        # --- coordinate transform ------------------------------------------
        radar_crs_str = radar_ds[FIELDS[0]].attrs.get('crs', 'EPSG:32610')
        radar_crs = CRS.from_string(radar_crs_str)
        wgs84     = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(wgs84, radar_crs, always_xy=True)
        station_x, station_y = transformer.transform(station_lon, station_lat)

        # --- pixel window --------------------------------------------------
        resolution_m = radar_ds.attrs.get('resolution_m', 500)
        patch_pixels = int(patch_size_m / resolution_m)
        half_pixels  = patch_pixels // 2

        x_idx = int(np.abs(radar_ds.x.values - station_x).argmin())
        y_idx = int(np.abs(radar_ds.y.values - station_y).argmin())

        x_start = max(0, x_idx - half_pixels)
        x_end   = x_start + patch_pixels
        y_start = max(0, y_idx - half_pixels)
        y_end   = y_start + patch_pixels

        # Guard against boundary overshoot
        if x_end > len(radar_ds.x):
            x_end   = len(radar_ds.x)
            x_start = max(0, x_end - patch_pixels)
        if y_end > len(radar_ds.y):
            y_end   = len(radar_ds.y)
            y_start = max(0, y_end - patch_pixels)

        # --- build patch array --------------------------------------------
        out = np.full((n_times, n_fields, patch_pixels, patch_pixels), np.nan, dtype=np.float32)

        for t_idx, scan_idx in enumerate(time_indices):
            if scan_idx is None:
                continue  # leave as NaN

            for f_idx, field in enumerate(FIELDS):
                if field not in radar_ds:
                    continue
                # Shape stored in zarr: (time, y, x) — no Z dimension
                arr = radar_ds[field].isel(
                    time=scan_idx,
                    x=slice(x_start, x_end),
                    y=slice(y_start, y_end),
                ).values.astype(np.float32)

                # Pad if we landed on an edge
                h, w = arr.shape
                if (h, w) != (patch_pixels, patch_pixels):
                    padded = np.full((patch_pixels, patch_pixels), np.nan, dtype=np.float32)
                    padded[:h, :w] = arr
                    arr = padded

                out[t_idx, f_idx] = arr

        return out   # (n_times, n_fields, patch_pixels, patch_pixels)

    except Exception as e:
        print(f"  ⚠ Error extracting patch at ({station_lat:.4f}, {station_lon:.4f}): {e}")
        return np.full((n_times, n_fields, patch_pixels, patch_pixels), np.nan, dtype=np.float32)


# ── Main pipeline ─────────────────────────────────────────────────────────────

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
    half_hour_offsets=False,
    include_test=False,
):
    """
    Build aligned radar-gauge samples and save to a pickle file.

    Radar patch shape: (12, n_fields, patch_pixels, patch_pixels)
      n_fields     = len(FIELDS) = 5
      patch_pixels = patch_size_m / 500
    """
    print("=" * 60)
    print("DUAL-POL RADAR-GAUGE DATA ALIGNMENT")
    print("=" * 60)

    # --- resolve date range ------------------------------------------------
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
        end_date   = max(dates)
    else:
        print(f"\nDate range: {start_date} → {end_date}")
        dates = None

    # --- load zarr ---------------------------------------------------------
    # Open each field independently via zarr to handle partially-written stores.
    # When the writer is mid-scan, different fields may have n vs n+1 time steps.
    # We read every field separately, then truncate all to the minimum common
    # time length before building an xarray Dataset — no conflicts possible.
    print(f"\n1. Loading radar zarr: {radar_zarr_path}")
    import zarr as _zarr

    _store = _zarr.open(radar_zarr_path, mode='r')

    # Find minimum time length across all present fields
    field_lengths = {f: _store[f].shape[0] for f in FIELDS if f in _store}
    if not field_lengths:
        raise ValueError(f"No expected fields found in {radar_zarr_path}. Keys: {list(_store.keys())}")

    min_time = min(field_lengths.values())
    print(f"  Field time lengths : {field_lengths}")
    print(f"  → Truncating all fields to min time: {min_time}")

    time_vals    = _store['time'][:min_time]
    # Zarr stores time as raw int64 nanoseconds since epoch; decode to datetime64[ns]
    time_decoded = time_vals.astype('datetime64[ns]')
    x_vals       = _store['x'][:]
    y_vals       = _store['y'][:]

    data_vars = {
        f: xr.Variable(['time', 'y', 'x'], _store[f][:min_time])
        for f in FIELDS if f in _store
    }
    coords = {
        'time': xr.Variable('time', time_decoded),
        'x':    xr.Variable('x',    x_vals),
        'y':    xr.Variable('y',    y_vals),
    }
    radar_ds = xr.Dataset(data_vars, coords=coords, attrs=dict(_store.attrs))

    # Copy per-field attributes (crs, units, etc.)
    for f in FIELDS:
        if f in _store and f in radar_ds:
            radar_ds[f].attrs = dict(_store[f].attrs)

    # Validate expected fields
    present = [f for f in FIELDS if f in radar_ds]
    missing = [f for f in FIELDS if f not in radar_ds]
    print(f"  Fields present : {present}")
    if missing:
        print(f"  ⚠ Fields MISSING: {missing}  (these will be NaN in patches)")

    print(f"  Dimensions    : {dict(radar_ds.dims)}")
    print("  Loading to RAM…")
    radar_ds = radar_ds.load()
    print("  ✓ Loaded")

    if pd.isna(radar_ds.time.values).all():
        print("  ⚠ All time values are NaT — check zarr encoding!")
    else:
        print(f"  Time range: {radar_ds.time.min().values} → {radar_ds.time.max().values}")

    if dem_path:
        print(f"\n  DEM will be extracted on-the-fly from: {dem_path}")

    # --- hourly precipitation ----------------------------------------------
    print(f"\n2. Loading hourly precipitation (min={min_rainfall_mm} mm)…")
    hourly_precip = get_hourly_precipitation_by_station(
        start_date, end_date, min_rainfall_mm=min_rainfall_mm
    )
    if dates:
        hourly_precip = [h for h in hourly_precip if h['hour_start'].date() in set(dates)]
    print(f"  → {len(hourly_precip)} station-hours")

    if not hourly_precip:
        raise ValueError("No hourly precipitation found — check date range / DB connection.")

    # --- build samples -----------------------------------------------------
    print(f"\n3. Extracting radar patches (patch_size_m={patch_size_m})…")
    print(f"   Skipping readings > {max_valid_rainfall} mm/hr")
    samples        = []
    skipped_outliers = 0

    for i, precip in enumerate(tqdm(hourly_precip, desc="Hours")):
        if precip['hourly_precip_mm'] > max_valid_rainfall:
            skipped_outliers += 1
            continue

        radar_times, radar_indices = sample_radar_scans_for_hour(
            radar_ds, precip['hour_start'], n_scans=12
        )

        if len(radar_indices) < 6:
            continue  # not enough temporal context

        # Bin into 12 fixed 5-min slots
        radar_times, radar_indices = bin_scans_to_fixed_slots(
            radar_times, radar_indices, precip['hour_start'], n_bins=12
        )

        # Extract dual-pol patch  → (12, n_fields, y, x)
        radar_patch = extract_radar_patch_at_station(
            radar_ds,
            radar_indices,
            precip['lat'],
            precip['lon'],
            patch_size_m=patch_size_m,
        )

        samples.append({
            'hour_start':       precip['hour_start'],
            'station_id':       precip['station_id'],
            'station_name':     precip['station_name'],
            'station_lat':      precip['lat'],
            'station_lon':      precip['lon'],
            'bias_flag':        get_station_bias(precip['station_name']),
            'hourly_precip_mm': precip['hourly_precip_mm'],
            'dump_ratio':       precip.get('dump_ratio'),
            'max_bin_mm':       precip.get('max_bin_mm'),
            'n_active_bins':    precip.get('n_active_bins'),
            'radar_times':      radar_times,
            'radar_indices':    radar_indices,
            # Shape: (12, n_fields=5, patch_pixels, patch_pixels)
            'radar_patch':      radar_patch,
            'n_valid_radar':    sum(1 for idx in radar_indices if idx is not None),
        })

        if i % 500 == 0:
            gc.collect()

    print(f"\n4. Created {len(samples)} samples")
    if skipped_outliers:
        print(f"   ⚠ Skipped {skipped_outliers} outlier readings (>{max_valid_rainfall} mm/hr)")

    # --- train/val split ---------------------------------------------------
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
            else:
                print(f"   ⚠ Skipping year {yr} (not in train/val sets)")
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
        val_samples   = [samples[i] for i in idx[split:]]

    print(f"   Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    # --- half-hour offset samples (training only) ---------------------------
    if half_hour_offsets:
        print("\n5b. Generating 30-min offset samples (training only)…")
        offset_precip = get_offset_hourly_precipitation_by_station(
            start_date, end_date, min_rainfall_mm=min_rainfall_mm, offset_minutes=30
        )
        if dates:
            offset_precip = [h for h in offset_precip if h['hour_start'].date() in set(dates)]

        # Only keep offset samples that fall within training years
        if train_years:
            offset_precip = [h for h in offset_precip if h['hour_start'].year in train_years]

        print(f"   → {len(offset_precip)} offset station-hours (training years only)")

        offset_samples = []
        offset_skipped = 0
        for i, precip in enumerate(tqdm(offset_precip, desc="Offset hours")):
            if precip['hourly_precip_mm'] > max_valid_rainfall:
                offset_skipped += 1
                continue

            radar_times, radar_indices = sample_radar_scans_for_hour(
                radar_ds, precip['hour_start'], n_scans=12
            )

            if len(radar_indices) < 6:
                continue

            radar_times, radar_indices = bin_scans_to_fixed_slots(
                radar_times, radar_indices, precip['hour_start'], n_bins=12
            )

            radar_patch = extract_radar_patch_at_station(
                radar_ds,
                radar_indices,
                precip['lat'],
                precip['lon'],
                patch_size_m=patch_size_m,
            )

            offset_samples.append({
                'hour_start':       precip['hour_start'],
                'station_id':       precip['station_id'],
                'station_name':     precip['station_name'],
                'station_lat':      precip['lat'],
                'station_lon':      precip['lon'],
                'bias_flag':        get_station_bias(precip['station_name']),
                'hourly_precip_mm': precip['hourly_precip_mm'],
                'dump_ratio':       precip.get('dump_ratio'),
                'max_bin_mm':       precip.get('max_bin_mm'),
                'n_active_bins':    precip.get('n_active_bins'),
                'radar_times':      radar_times,
                'radar_indices':    radar_indices,
                'radar_patch':      radar_patch,
                'n_valid_radar':    sum(1 for idx in radar_indices if idx is not None),
                'is_offset':        True,
            })

            if i % 500 == 0:
                gc.collect()

        print(f"   Created {len(offset_samples)} offset samples for training")
        if offset_skipped:
            print(f"   ⚠ Skipped {offset_skipped} offset outliers")

        train_samples.extend(offset_samples)
        print(f"   Train total (with offsets): {len(train_samples)}  |  Val: {len(val_samples)}")

    # --- test samples (daily gauges) -----------------------------------------
    test_samples = []
    if include_test:
        # Use explicit dates if available, else generate from start/end range
        test_dates = dates
        if test_dates is None:
            from datetime import date as date_cls
            sd = datetime.strptime(start_date, '%Y-%m-%d').date() if isinstance(start_date, str) else start_date
            ed = datetime.strptime(end_date, '%Y-%m-%d').date() if isinstance(end_date, str) else end_date
            test_dates = [sd + timedelta(days=i) for i in range((ed - sd).days + 1)]

        test_samples = create_test_samples(
            radar_ds, test_dates, patch_size_m,
            min_rainfall_mm=1.0, max_valid_rainfall=200.0,
        )

    # --- save --------------------------------------------------------------
    patch_pixels = int(patch_size_m / radar_ds.attrs.get('resolution_m', 500))
    dataset = {
        'train': train_samples,
        'val':   val_samples,
        'test':  test_samples,
        'metadata': {
            'radar_zarr':      radar_zarr_path,
            'dem_path':        dem_path,
            'fields':          FIELDS,
            'n_fields':        len(present),
            'patch_size_m':    patch_size_m,
            'patch_pixels':    patch_pixels,
            'radar_patch_shape': f"(12, {len(present)}, {patch_pixels}, {patch_pixels})",
            'start_date':      str(start_date),
            'end_date':        str(end_date),
            'day_filter_file': day_filter_file,
            'specific_days':   [str(d) for d in dates] if dates else None,
            'split_type':      'temporal' if train_years else 'random',
            'train_years':     train_years or 'N/A',
            'val_years':       val_years   or 'N/A',
            'created':         datetime.now().isoformat(),
            'n_train':         len(train_samples),
            'n_val':           len(val_samples),
            'n_test':          len(test_samples),
            'test_gauge_type': 'daily_cumulative' if include_test else None,
        },
    }

    print(f"\n6. Saving to {output_path}…")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    # --- summary -----------------------------------------------------------
    print(f"\n✅ Dataset saved!")
    print("=" * 60)
    all_s = train_samples + val_samples
    rainy = [s for s in all_s if s['hourly_precip_mm'] >= 0.5]
    print(f"  Total samples   : {len(all_s)}")
    print(f"  Stations        : {len(set(s['station_id'] for s in all_s))}")
    print(f"  Rainy (≥0.5 mm) : {len(rainy)} ({100*len(rainy)/len(all_s):.1f}%)")
    print(f"  Avg valid scans : {np.mean([s['n_valid_radar'] for s in all_s]):.1f} / 12")
    print(f"  Rainfall range  : {np.min([s['hourly_precip_mm'] for s in all_s]):.2f}"
          f" – {np.max([s['hourly_precip_mm'] for s in all_s]):.2f} mm/hr")
    print(f"  Patch shape     : {dataset['metadata']['radar_patch_shape']}")

    if test_samples:
        print(f"\n  Test set (daily gauges):")
        print(f"    Samples       : {len(test_samples)}")
        print(f"    Stations      : {len(set(s['station_id'] for s in test_samples))}")
        unique_days = len(set((s['date'], s['station_id']) for s in test_samples))
        print(f"    Day-station   : {unique_days}")

    return output_path


def inspect_dataset(dataset_path):
    """Print a summary of an existing pickle."""
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print("\nMetadata:")
    for k, v in dataset['metadata'].items():
        print(f"  {k}: {v}")

    print(f"\nTrain: {len(dataset['train'])}  |  Val: {len(dataset['val'])}")

    if dataset['train']:
        s = dataset['train'][0]
        print("\nFirst training sample:")
        print(f"  Hour         : {s['hour_start']}")
        print(f"  Station      : {s['station_name']} (ID: {s['station_id']})")
        print(f"  Location     : ({s['station_lat']:.4f}, {s['station_lon']:.4f})")
        print(f"  Radar patch  : {s['radar_patch'].shape}")
        print(f"  Rainfall     : {s['hourly_precip_mm']:.2f} mm/hr")
        print(f"  Valid scans  : {s['n_valid_radar']} / 12")

    return dataset


# ── TEST DATASET (daily gauges) ───────────────────────────────────────────────

def get_daily_gauge_stations():
    """Get stations with daily cumulative gauges, excluding hourly training stations."""
    from database.config import connect, create_session
    from database.models import DendraStation, DendraDatastream

    engine = connect()
    session = create_session(engine)

    daily_ds = session.query(DendraDatastream).filter(
        DendraDatastream.name.in_(["Rainfall Cumulative", "Ranchbot Cumulative Daily Rainfall"])
    ).all()

    hourly_ds = session.query(DendraDatastream).filter(
        DendraDatastream.name.in_(["Rainfall", "Rainfall Sum"])
    ).all()
    hourly_station_ids = set(ds.station_id for ds in hourly_ds)

    daily_only = [ds for ds in daily_ds if ds.station_id not in hourly_station_ids]

    seen_station_ids = set()
    daily_stations = []
    daily_only_sorted = sorted(daily_only, key=lambda ds: ds.name != "Ranchbot Cumulative Daily Rainfall")

    for ds in daily_only_sorted:
        if ds.station_id in seen_station_ids:
            continue
        station = session.query(DendraStation).filter(DendraStation.id == ds.station_id).first()
        if station and station.latitude and station.longitude:
            daily_stations.append({
                'station_id': station.id,
                'station_name': station.name,
                'lat': station.latitude,
                'lon': station.longitude,
                'datastream_id': ds.id,
            })
            seen_station_ids.add(ds.station_id)

    return daily_stations


def get_daily_rainfall(datastream_ids, dates):
    """Get daily rainfall totals for given datastreams and dates."""
    from database.config import connect, create_session
    from database.models import DendraDatapoint

    engine = connect()
    session = create_session(engine)

    all_data = []
    for date in tqdm(dates, desc="Loading daily rainfall"):
        start_dt = datetime.combine(date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)

        results = session.query(
            DendraDatapoint.datastream_id,
            DendraDatapoint.value
        ).filter(
            DendraDatapoint.datastream_id.in_(datastream_ids),
            DendraDatapoint.timestamp_utc >= start_dt,
            DendraDatapoint.timestamp_utc < end_dt
        ).all()

        if results:
            df = pd.DataFrame(results, columns=['datastream_id', 'value'])
            daily = df.groupby('datastream_id')['value'].sum().reset_index()
            daily['date'] = date
            daily.columns = ['datastream_id', 'rainfall_mm', 'date']
            all_data.append(daily)

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def create_test_samples(radar_ds, dates, patch_size_m, min_rainfall_mm=1.0, max_valid_rainfall=200.0):
    """
    Create test samples from daily cumulative gauges.
    Uses the same radar extraction pipeline as training data.
    """
    print("\n" + "=" * 60)
    print("  CREATING TEST SAMPLES (daily gauges)")
    print("=" * 60)

    daily_stations = get_daily_gauge_stations()
    if not daily_stations:
        print("  ⚠ No daily-only gauge stations found. Skipping test set.")
        return []

    print(f"  Found {len(daily_stations)} daily-only stations:")
    for s in daily_stations:
        print(f"    - {s['station_name']}")

    datastream_ids = [s['datastream_id'] for s in daily_stations]
    daily_rain = get_daily_rainfall(datastream_ids, dates)

    if daily_rain.empty:
        print("  ⚠ No daily rainfall data found.")
        return []

    daily_rain = daily_rain[
        (daily_rain['rainfall_mm'] >= min_rainfall_mm) &
        (daily_rain['rainfall_mm'] <= max_valid_rainfall)
    ]
    print(f"  {len(daily_rain)} day-station measurements after filtering")

    station_lookup = {s['datastream_id']: s for s in daily_stations}
    test_samples = []
    skipped = 0

    for _, row in tqdm(daily_rain.iterrows(), total=len(daily_rain), desc="Test samples"):
        station = station_lookup[row['datastream_id']]
        date = row['date']
        daily_rainfall = row['rainfall_mm']

        for hour in range(24):
            hour_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)

            radar_times, radar_indices = sample_radar_scans_for_hour(radar_ds, hour_start, n_scans=12)
            if len(radar_indices) < 6:
                skipped += 1
                continue

            radar_times, radar_indices = bin_scans_to_fixed_slots(
                radar_times, radar_indices, hour_start, n_bins=12
            )

            radar_patch = extract_radar_patch_at_station(
                radar_ds, radar_indices, station['lat'], station['lon'],
                patch_size_m=patch_size_m,
            )

            test_samples.append({
                'hour_start':       hour_start,
                'date':             date,
                'station_id':       station['station_id'],
                'station_name':     station['station_name'],
                'station_lat':      station['lat'],
                'station_lon':      station['lon'],
                'bias_flag':        0,
                'daily_precip_mm':  daily_rainfall,
                'hourly_precip_mm': daily_rainfall / 24.0,  # approximate for compatibility
                'radar_times':      radar_times,
                'radar_indices':    radar_indices,
                'radar_patch':      radar_patch,
                'n_valid_radar':    sum(1 for idx in radar_indices if idx is not None),
                'is_test':          True,
            })

        if len(test_samples) % 500 == 0:
            gc.collect()

    print(f"  Created {len(test_samples)} test samples ({skipped} hours skipped)")
    unique_days = len(set((s['date'], s['station_id']) for s in test_samples))
    print(f"  Unique day-station combinations: {unique_days}")

    return test_samples


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create dual-pol radar-gauge pickle")

    parser.add_argument('--radar',      required=True,
                        help='Path to dual-pol radar zarr (e.g. KVBX_preserve_500m_dualpol.zarr)')
    parser.add_argument('--output',     default='radar_gauge_dataset.pkl',
                        help='Output pickle path')
    parser.add_argument('--days',       default=None,
                        help='Path to date-filter file (e.g. my_rainy_days_150.txt)')
    parser.add_argument('--start',      default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end',        default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--dem',        default=None,
                        help='DEM GeoTIFF path (e.g. dem/preserve_dem_10m_utm.tif)')
    parser.add_argument('--patch-size', type=int, default=4500,
                        help='Patch size in metres (default: 4500 → 9×9 @ 500 m/px)')
    parser.add_argument('--min-rainfall', type=float, default=None,
                        help='Min hourly rainfall mm (default: 0.0 with --days, 0.5 without)')
    parser.add_argument('--max-rainfall', type=float, default=100.0,
                        help='Max valid rainfall mm/hr (default: 100.0)')
    parser.add_argument('--train-years', type=int, nargs='+', default=None)
    parser.add_argument('--val-years',   type=int, nargs='+', default=None)
    parser.add_argument('--half-hour-offsets', action='store_true',
                        help='Add 30-min offset samples to training set (nearly doubles training data)')
    parser.add_argument('--include-test', action='store_true',
                        help='Also generate test samples from daily cumulative gauges')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect an existing pickle instead of building one')

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.output)
    else:
        if not args.days and not (args.start and args.end):
            parser.error("Provide --days OR both --start and --end")
        if bool(args.train_years) != bool(args.val_years):
            parser.error("Provide both --train-years and --val-years, or neither")

        min_rain = args.min_rainfall
        if min_rain is None:
            min_rain = 0.0 if args.days else 0.5

        create_training_samples(
            radar_zarr_path  = args.radar,
            output_path      = args.output,
            dem_path         = args.dem,
            train_years      = args.train_years,
            val_years        = args.val_years,
            start_date       = args.start,
            end_date         = args.end,
            day_filter_file  = args.days,
            min_rainfall_mm  = min_rain,
            max_valid_rainfall = args.max_rainfall,
            patch_size_m     = args.patch_size,
            half_hour_offsets = args.half_hour_offsets,
            include_test     = args.include_test,
        )