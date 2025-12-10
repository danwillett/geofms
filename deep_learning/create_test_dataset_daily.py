"""
Create test dataset using DAILY rainfall gauges

This script creates a test dataset using more stable daily rainfall gauges,
providing an independent evaluation of model performance.

Daily gauges use:
- "Rainfall Cumulative" datastream
- "Ranchbot Cumulative Daily Rainfall" datastream

These are separate from the hourly "Rainfall" / "Rainfall Sum" gauges
used for training/validation.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm

from database.config import connect, create_session
from database.models import DendraStation, DendraDatastream, DendraDatapoint
from radar.constants import RADAR_MISSING_VALUE


def get_daily_gauge_stations():
    """
    Get stations with daily rainfall gauges, excluding hourly stations
    
    Returns:
    --------
    daily_stations : list of dict
        Each dict contains: station_id, station_name, lat, lon, datastream_id
    """
    engine = connect()
    session = create_session(engine)
    
    # Find daily rainfall datastreams
    print("Finding daily rainfall datastreams...")
    daily_ds = session.query(DendraDatastream).filter(
        DendraDatastream.name.in_(["Rainfall Cumulative", "Ranchbot Cumulative Daily Rainfall"])
    ).all()
    
    print(f"  Found {len(daily_ds)} daily rainfall datastreams")
    
    # Find hourly rainfall datastreams (to exclude these stations)
    hourly_ds = session.query(DendraDatastream).filter(
        DendraDatastream.name.in_(["Rainfall", "Rainfall Sum"])
    ).all()
    
    hourly_station_ids = set(ds.station_id for ds in hourly_ds)
    print(f"  Found {len(hourly_station_ids)} stations with hourly gauges (will exclude)")
    
    # Filter to daily-only stations
    daily_only = [ds for ds in daily_ds if ds.station_id not in hourly_station_ids]
    print(f"  ✓ Kept {len(daily_only)} daily-only datastreams")
    
    # === DEDUPLICATE BY STATION_ID ===
    # If a station has multiple datastreams, keep only one (prefer Ranchbot)
    seen_station_ids = set()
    daily_stations = []
    
    # Sort to prefer "Ranchbot Cumulative Daily Rainfall" over "Rainfall Cumulative"
    daily_only_sorted = sorted(daily_only, key=lambda ds: ds.name != "Ranchbot Cumulative Daily Rainfall")
    
    for ds in daily_only_sorted:
        if ds.station_id in seen_station_ids:
            print(f"  ⚠️  Skipping duplicate datastream for station {ds.station_id}")
            continue
        
        station = session.query(DendraStation).filter(
            DendraStation.id == ds.station_id
        ).first()
        

        if station and station.latitude and station.longitude:
            daily_stations.append({
                'station_id': station.id,
                'station_name': station.name,
                'lat': station.latitude,
                'lon': station.longitude,
                'datastream_id': ds.id,
                'datastream_name': ds.name
            })
            seen_station_ids.add(ds.station_id)
    
    print(f"\n✓ Found {len(daily_stations)} unique daily-only stations with coordinates")
    for s in daily_stations:
        print(f"  - {s['station_name']} ({s['datastream_name']})")
    
    return daily_stations


def get_daily_rainfall_on_days(datastream_ids, dates):
    """
    Get daily rainfall for specific dates
    
    Parameters:
    -----------
    datastream_ids : list of int
        Daily rainfall datastream IDs
    dates : list of datetime.date
        Dates to query
    
    Returns:
    --------
    daily_rain : pandas.DataFrame
        Columns: date, datastream_id, rainfall_mm
    """
    engine = connect()
    session = create_session(engine)
    
    print(f"\nQuerying daily rainfall for {len(dates)} days...")
    
    all_data = []
    
    for date in tqdm(dates, desc="Loading daily rainfall"):
        # Query range: full day
        start_dt = datetime.combine(date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        
        # Get all datapoints for this day
        query = session.query(
            DendraDatapoint.datastream_id,
            DendraDatapoint.timestamp_utc,
            DendraDatapoint.value
        ).filter(
            DendraDatapoint.datastream_id.in_(datastream_ids),
            DendraDatapoint.timestamp_utc >= start_dt,
            DendraDatapoint.timestamp_utc < end_dt
        )
        
        results = query.all()
        
        # Group by datastream and take the sum value (cumulative gauges reset daily)
        if len(results) > 0:
            df = pd.DataFrame(results, columns=['datastream_id', 'timestamp_utc', 'value'])
            
            # For cumulative gauges, the sum value is the daily total
            daily = df.groupby('datastream_id')['value'].sum().reset_index()
            daily['date'] = date
            daily.columns = ['datastream_id', 'rainfall_mm', 'date']
            
            all_data.append(daily)
    
    if len(all_data) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    print(f"  Found {len(result)} daily measurements")
    
    return result


def extract_radar_patch_at_station(radar_ds, time_indices, station_lat, station_lon, 
                                    patch_size_m=2640):
    """
    Extract radar patch around a rain gauge station
    
    (Same as in prepare_radar_gauge_data.py)
    """
    try:
        from pyproj import Transformer, CRS
        
        # Get radar CRS from metadata
        radar_crs_str = radar_ds.reflectivity.attrs.get('crs', 'EPSG:32610')
        radar_crs = CRS.from_string(radar_crs_str)
        wgs84 = CRS.from_epsg(4326)
        
        transformer = Transformer.from_crs(wgs84, radar_crs, always_xy=True)
        station_x, station_y = transformer.transform(station_lon, station_lat)
        
        # Calculate patch size in pixels
        resolution_m = radar_ds.attrs.get('resolution_m', 500)
        patch_pixels = int(patch_size_m / resolution_m)
        half_pixels = patch_pixels // 2
        
        # Find nearest pixel indices
        x_idx = np.abs(radar_ds.x.values - station_x).argmin()
        y_idx = np.abs(radar_ds.y.values - station_y).argmin()
        
        # Define pixel window
        x_start = max(0, x_idx - half_pixels)
        x_end = x_start + patch_pixels
        y_start = max(0, y_idx - half_pixels)
        y_end = y_start + patch_pixels
        
        # Extract patches for each time
        patches = []
        for idx in time_indices:
            if idx is None:
                shape = (len(radar_ds.z), patch_pixels, patch_pixels)
                patches.append(np.full(shape, np.nan))
            else:
                patch = radar_ds.reflectivity.isel(
                    time=idx,
                    x=slice(x_start, x_end),
                    y=slice(y_start, y_end)
                ).values
                
                if patch.shape[-2:] != (patch_pixels, patch_pixels):
                    print(f"Warning: Patch size mismatch, padding...")
                    padded = np.full((len(radar_ds.z), patch_pixels, patch_pixels), np.nan)
                    h, w = patch.shape[-2:]
                    padded[:, :h, :w] = patch
                    patches.append(padded)
                else:
                    patches.append(patch)
        
        result = np.stack(patches, axis=0)
        return result
        
    except Exception as e:
        print(f"Error extracting patch: {e}")
        shape = (len(time_indices), len(radar_ds.z), 10, 10)
        return np.full(shape, np.nan)


def extract_dem_patch(dem_path, station_lon, station_lat, patch_size_m=2640):
    """
    Extract DEM patch (same as prepare_radar_gauge_data.py)
    """
    import rioxarray as rxr
    from pyproj import Transformer
    from scipy.ndimage import zoom
    import os
    
    if not os.path.exists(dem_path):
        return None
    
    # Load DEM
    dem = rxr.open_rasterio(dem_path)
    
    # Convert station coords to UTM
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
    station_x, station_y = transformer.transform(station_lon, station_lat)
    
    # Calculate patch bounds
    half_size = patch_size_m / 2
    minx = station_x - half_size
    maxx = station_x + half_size
    miny = station_y - half_size
    maxy = station_y + half_size
    
    # Clip to patch
    patch = dem.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    
    # Force exact size: 264×264 @ 10m
    expected_pixels = int(patch_size_m / 10)
    current_shape = patch.shape
    
    if current_shape[-2:] != (expected_pixels, expected_pixels):
        patch_data = patch.values[0]
        zoom_y = expected_pixels / current_shape[1]
        zoom_x = expected_pixels / current_shape[2]
        resampled = zoom(patch_data, (zoom_y, zoom_x), order=1)
        
        if resampled.shape != (expected_pixels, expected_pixels):
            final = np.zeros((expected_pixels, expected_pixels), dtype=resampled.dtype)
            h, w = min(resampled.shape[0], expected_pixels), min(resampled.shape[1], expected_pixels)
            final[:h, :w] = resampled[:h, :w]
            resampled = final
        
        patch_data = resampled[np.newaxis, ...]
    else:
        patch_data = patch.values
    
    return patch_data


def sample_radar_scans_for_hour(radar_ds, hour_start, n_scans=12):
    """
    Sample radar scans from a given hour
    
    Same function as in prepare_radar_gauge_data.py - ensures consistency
    with training data format.
    """
    from datetime import timedelta
    
    hour_end = hour_start + timedelta(hours=1)
    
    # Get all radar scans within the hour
    mask = (radar_ds.time.values >= np.datetime64(hour_start)) & \
           (radar_ds.time.values < np.datetime64(hour_end))
    
    hour_indices = np.where(mask)[0]
    
    if len(hour_indices) == 0:
        # No scans in this hour
        return [], []
    
    # Create fixed time bins (12 bins of 5 minutes each)
    n_bins = 12
    bin_duration_minutes = 60 / n_bins  # 5 minutes per bin
    
    # Initialize bins (all None/empty)
    binned_indices = [None] * n_bins
    binned_times = [None] * n_bins
    
    if len(hour_indices) >= n_scans:
        # Take evenly spaced scans
        sample_indices = np.linspace(0, len(hour_indices)-1, n_scans, dtype=int)
        selected_indices = hour_indices[sample_indices]
        radar_times = [pd.Timestamp(radar_ds.time.values[idx]).to_pydatetime() 
                       for idx in selected_indices]
        
        # Assign each scan to its time bin
        for scan_time, scan_idx in zip(radar_times, selected_indices):
            # Calculate which bin this scan belongs to
            minutes_into_hour = (scan_time - hour_start).total_seconds() / 60
            bin_idx = int(minutes_into_hour / bin_duration_minutes)
            
            # Clamp to valid range [0, 11]
            bin_idx = max(0, min(n_bins - 1, bin_idx))
            
            # Assign to bin (if multiple scans in same bin, keep first)
            if binned_indices[bin_idx] is None:
                binned_indices[bin_idx] = scan_idx
                binned_times[bin_idx] = scan_time
    else:
        # Use all available scans, assign to bins
        radar_times = [pd.Timestamp(radar_ds.time.values[idx]).to_pydatetime() 
                       for idx in hour_indices]
        for scan_time, scan_idx in zip(radar_times, hour_indices):
            minutes_into_hour = (scan_time - hour_start).total_seconds() / 60
            bin_idx = int(minutes_into_hour / bin_duration_minutes)
            bin_idx = max(0, min(n_bins - 1, bin_idx))
            
            if binned_indices[bin_idx] is None:
                binned_indices[bin_idx] = scan_idx
                binned_times[bin_idx] = scan_time
    
    return binned_times, binned_indices


def create_test_dataset_daily(
    radar_zarr_path,
    output_path,
    day_filter_file,
    dem_path='dem/preserve_dem_10m_utm.tif',
    min_rainfall_mm=1.0,
    max_valid_rainfall=200.0,
    patch_size_m=2640
):
    """
    Create test dataset from daily rainfall gauges
    
    Parameters:
    -----------
    radar_zarr_path : str
        Path to radar zarr file
    output_path : str
        Where to save test dataset (pickle)
    day_filter_file : str
        File with dates to use (same as training)
    dem_path : str
        Path to DEM file
    min_rainfall_mm : float
        Minimum daily rainfall to include (default: 1.0mm)
    max_valid_rainfall : float
        Maximum valid daily rainfall (default: 200mm)
    """
    
    print("="*60)
    print("CREATING TEST DATASET FROM DAILY GAUGES")
    print("="*60)
    
    # Step 1: Load dates
    print(f"\n1. Loading dates from {day_filter_file}...")
    with open(day_filter_file, 'r') as f:
        dates = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                dates.append(datetime.strptime(line, '%Y-%m-%d').date())
    
    print(f"   Found {len(dates)} days")
    
    # Step 2: Get daily gauge stations
    print("\n2. Finding daily gauge stations...")
    daily_stations = get_daily_gauge_stations()
    
    if len(daily_stations) == 0:
        print("❌ No daily-only stations found!")
        return None
    
    # Step 3: Load radar data INTO MEMORY (faster than dask lazy loading)
    print("\n3. Loading radar data into memory...")
    radar_ds = xr.open_zarr(radar_zarr_path, consolidated=False)
    print(f"   Radar shape: {radar_ds.reflectivity.shape}")
    print("   Loading full dataset to RAM (this may take a minute)...")
    radar_ds = radar_ds.load()  # Force load everything - no more dask overhead!
    print("   ✓ Radar data loaded to memory")
    
    # Note: DEM is extracted on-the-fly in the Dataset class, not here
    if dem_path:
        print(f"\n   ℹ️  DEM path stored in metadata: {dem_path}")
        print(f"   (DEM will be extracted on-the-fly during evaluation)")
    
    # Step 4: Get daily rainfall
    datastream_ids = [s['datastream_id'] for s in daily_stations]
    daily_rain = get_daily_rainfall_on_days(datastream_ids, dates)
    
    if len(daily_rain) == 0:
        print("❌ No daily rainfall found!")
        return None
    
    # Filter by rainfall threshold
    daily_rain = daily_rain[
        (daily_rain['rainfall_mm'] >= min_rainfall_mm) & 
        (daily_rain['rainfall_mm'] <= max_valid_rainfall)
    ]
    
    print(f"\n   ✓ {len(daily_rain)} daily measurements after filtering")
    print(f"   Rainfall range: {daily_rain['rainfall_mm'].min():.1f} - {daily_rain['rainfall_mm'].max():.1f} mm/day")
    
    # Step 5: Create test samples (24 hourly samples per day)
    print("\n4. Creating hourly test samples...")
    print("   Strategy: 24 hourly samples per day → sum predictions at eval time")
    
    # Create station lookup
    station_lookup = {s['datastream_id']: s for s in daily_stations}
    
    test_samples = []
    skipped_hours = 0
    
    for _, row in tqdm(daily_rain.iterrows(), total=len(daily_rain), desc="Processing days"):
        station = station_lookup[row['datastream_id']]
        date = row['date']
        daily_rainfall = row['rainfall_mm']
        
        # DEM is extracted on-the-fly in the Dataset, not stored in pickle
        
        # Create 24 hourly samples for this day
        hourly_samples_for_day = []
        
        for hour in range(24):
            hour_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            
            # Sample radar scans for this hour (same as training data)
            radar_times, radar_indices = sample_radar_scans_for_hour(
                radar_ds,
                hour_start,
                n_scans=12
            )
            
            # Need at least 6 valid scans
            n_valid = len([idx for idx in radar_indices if idx is not None])
            if n_valid < 6:
                skipped_hours += 1
                continue
            
            # Extract radar patch
            radar_patch = extract_radar_patch_at_station(
                radar_ds,
                radar_indices,
                station['lat'],
                station['lon'],
                patch_size_m=patch_size_m
            )
            
            # Create hourly sample
            sample = {
                'date': date,
                'hour_start': hour_start,
                'station_id': station['station_id'],
                'station_name': station['station_name'],
                'station_lat': station['lat'],
                'station_lon': station['lon'],
                'datastream_name': station['datastream_name'],
                'daily_precip_mm': daily_rainfall,  # Store daily total (same for all hours)
                'radar_times': radar_times,
                'radar_indices': radar_indices,
                'radar_patch': radar_patch,
                # DEM extracted on-the-fly using station_lat/lon
                'n_valid_radar': n_valid
            }
            
            hourly_samples_for_day.append(sample)
        
        # Add all hourly samples from this day
        test_samples.extend(hourly_samples_for_day)
    
    if skipped_hours > 0:
        print(f"\n   ⚠️  Skipped {skipped_hours} hours with <6 valid radar scans")
    
    print(f"\n5. Created {len(test_samples)} test samples")
    
    # Step 6: Save dataset
    print(f"\n6. Saving to {output_path}...")
    
    dataset = {
        'test': test_samples,
        'metadata': {
            'radar_zarr': radar_zarr_path,
            'dem_path': dem_path,  # DEM extracted on-the-fly in Dataset
            'day_filter_file': day_filter_file,
            'dates': [str(d) for d in dates],
            'created': datetime.now().isoformat(),
            'n_test': len(test_samples),
            'gauge_type': 'daily_cumulative',
            'datastreams': ['Rainfall Cumulative', 'Ranchbot Cumulative Daily Rainfall'],
            'stations': daily_stations,
            'min_rainfall_mm': min_rainfall_mm,
            'max_valid_rainfall': max_valid_rainfall,
            'radar_resolution_m': radar_ds.attrs.get('resolution_m', 500),
            'patch_size_m': patch_size_m
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n✅ Test dataset saved!")
    print("="*60)
    
    # Print summary
    print("\nTest Dataset Summary:")
    print(f"  Total hourly samples: {len(test_samples)}")
    
    # Count unique days
    unique_days = set((s['date'], s['station_id']) for s in test_samples)
    print(f"  Unique day-station combinations: {len(unique_days)}")
    
    # Samples per day
    from collections import Counter
    days_counter = Counter(s['date'] for s in test_samples)
    avg_hours_per_day = np.mean(list(days_counter.values()))
    print(f"  Avg hourly samples per day: {avg_hours_per_day:.1f} / 24")
    
    # Station breakdown
    print(f"\n  Stations: {len(set(s['station_id'] for s in test_samples))}")
    for station in daily_stations:
        n_samples = len([s for s in test_samples if s['station_id'] == station['station_id']])
        n_days = len(set(s['date'] for s in test_samples if s['station_id'] == station['station_id']))
        print(f"    - {station['station_name']}: {n_samples} hourly samples ({n_days} days)")
    
    print(f"\n  Date range: {min(s['date'] for s in test_samples)} to {max(s['date'] for s in test_samples)}")
    print(f"  Avg valid radar scans per hour: {np.mean([s['n_valid_radar'] for s in test_samples]):.1f} / 12")
    
    # Daily rainfall stats (unique per day)
    daily_values = list(set(s['daily_precip_mm'] for s in test_samples))
    print(f"\n  Daily rainfall statistics:")
    print(f"    Mean: {np.mean(daily_values):.2f} mm/day")
    print(f"    Range: {np.min(daily_values):.2f} - {np.max(daily_values):.2f} mm/day")
    
    print(f"\n  💡 Evaluation: Run model on all hourly samples, group by (date, station_id), sum predictions → compare to daily_precip_mm")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create test dataset from daily gauges")
    parser.add_argument('--radar', type=str, default='KVBX_preserve_500m.zarr',
                        help='Path to radar zarr file')
    parser.add_argument('--days', type=str, required=True,
                        help='Path to file with dates (e.g., my_rainy_days_76.txt)')
    parser.add_argument('--output', type=str, default='radar_gauge_test_daily_76.pkl',
                        help='Output file path')
    parser.add_argument('--dem', type=str, default='dem/preserve_dem_10m_utm.tif',
                        help='Path to DEM file')
    parser.add_argument('--min-rainfall', type=float, default=1.0,
                        help='Minimum daily rainfall in mm (default: 1.0)')
    parser.add_argument('--max-rainfall', type=float, default=200.0,
                        help='Maximum valid daily rainfall in mm (default: 200.0)')
    parser.add_argument('--patch-size', type=int, default=2640,
                        help='Patch size in meters (default: 2640 for 5x5 @ 500m). Use 4620 for 9x9.')
    
    args = parser.parse_args()
    
    dataset = create_test_dataset_daily(
        radar_zarr_path=args.radar,
        output_path=args.output,
        day_filter_file=args.days,
        dem_path=args.dem,
        min_rainfall_mm=args.min_rainfall,
        max_valid_rainfall=args.max_rainfall,
        patch_size_m=args.patch_size
    )

