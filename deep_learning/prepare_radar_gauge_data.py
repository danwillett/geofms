"""
Data preparation script: Align NEXRAD radar with rain gauge measurements

This script creates training/validation datasets by pairing:
- NEXRAD radar scans (temporal sequences)
- Rain gauge measurements (10-minute intervals)
- Context data (DEM, LULC - future)

Output: Aligned dataset ready for ML training
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

from weather.pull_weather import get_hourly_precipitation_by_station
from radar.constants import RADAR_MISSING_VALUE

import gc

# station bias... add a flag for over/under estimation
            
STATION_BIAS = {
    'Dangermond_Bunker Hill': 1, 
    'Dangermond_Cistern': 1,
    'Dangermond_Cojo HQ': 1,
    'Dangermond_Jalachichi': 1,
    'Dangermond_Repeater': 1,
    'Dangermond_Cojo Gate': -1,
    'Dangermond_Sutter': -1
}

def get_station_bias(station_name):
    """Get bias flag for a station"""
    return STATION_BIAS.get(station_name, 0)


def sample_radar_scans_for_hour(radar_ds, hour_start, n_scans=12):  # Changed from 6
    """
    Sample radar scans from a given hour
    
    Parameters:
    -----------
    n_scans : int or None
        If None, uses all available scans (up to max_scans)
        If int, samples that many evenly-spaced scans
    max_scans : int
        Maximum scans to return (default: 12)
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
    
    if n_scans is None:
        # Use all available scans (cap at 12)
        selected_indices = hour_indices[:12]
    elif len(hour_indices) >= n_scans:
        # Take evenly spaced scans
        sample_indices = np.linspace(0, len(hour_indices)-1, n_scans, dtype=int)
        selected_indices = hour_indices[sample_indices]
    else:
        # Use all available
        selected_indices = hour_indices
    
    # Convert to times and indices
    radar_times = [pd.Timestamp(radar_ds.time.values[idx]).to_pydatetime() 
                   for idx in selected_indices]
    indices = selected_indices.tolist()
    
    return radar_times, indices


def extract_radar_patch_at_station(radar_ds, time_indices, station_lat, station_lon, 
                                    patch_size_m=2640):
    """
    Extract radar patch around a rain gauge station
    
    Parameters:
    -----------
    radar_ds : xarray.Dataset
        Radar data
    time_indices : list of int
        Time indices to extract (length = 6 for hourly sequence)
    station_lat, station_lon : float
        Station location
    patch_size_m : int
        Patch size in meters (default: 2640m for TerraMesh)
    
    Returns:
    --------
    patch : numpy array
        Shape: (n_times, z, patch_pixels, patch_pixels)
        Or all NaN if extraction fails
    """
    try:
        # Convert station lat/lon to radar grid coordinates
        # (Assumes radar grid is in UTM - need to transform)
        from pyproj import Transformer, CRS
        
        # Get radar CRS from metadata
        radar_crs_str = radar_ds.reflectivity.attrs.get('crs', 'EPSG:32610')  # Default to UTM 10N
        radar_crs = CRS.from_string(radar_crs_str)
        wgs84 = CRS.from_epsg(4326)
        
        transformer = Transformer.from_crs(wgs84, radar_crs, always_xy=True)
        station_x, station_y = transformer.transform(station_lon, station_lat)
        
        # Calculate patch size in pixels
        resolution_m = radar_ds.attrs.get('resolution_m', 500)
        patch_pixels = int(patch_size_m / resolution_m)  # Should be 5 for 2640m / 500m
        half_pixels = patch_pixels // 2  # 2 pixels on each side
        
        # Find nearest pixel indices to station location
        x_idx = np.abs(radar_ds.x.values - station_x).argmin()
        y_idx = np.abs(radar_ds.y.values - station_y).argmin()
        
        # Define pixel window (fixed size, centered on station)
        x_start = max(0, x_idx - half_pixels)
        x_end = x_start + patch_pixels
        y_start = max(0, y_idx - half_pixels)
        y_end = y_start + patch_pixels
        
        # Extract patches for each time
        patches = []
        for idx in time_indices:
            if idx is None:
                # Missing radar scan - fill with NaN
                shape = (len(radar_ds.z), patch_pixels, patch_pixels)
                patches.append(np.full(shape, np.nan))
            else:
                # Extract patch using pixel indices (ensures fixed size)
                patch = radar_ds.reflectivity.isel(
                    time=idx,
                    x=slice(x_start, x_end),
                    y=slice(y_start, y_end)
                ).load().values
                
                # Verify patch size (should always be patch_pixels × patch_pixels)
                if patch.shape[-2:] != (patch_pixels, patch_pixels):
                    # Edge case: station near boundary, pad if needed
                    print(f"Warning: Patch size mismatch at station ({station_lat:.4f}, {station_lon:.4f})")
                    print(f"  Got {patch.shape[-2:]} but expected ({patch_pixels}, {patch_pixels})")
                    print(f"  Padding to correct size...")
                    padded = np.full((len(radar_ds.z), patch_pixels, patch_pixels), np.nan)
                    h, w = patch.shape[-2:]
                    padded[:, :h, :w] = patch
                    patches.append(padded)
                else:
                    patches.append(patch)
        
        # Stack along time dimension
        result = np.stack(patches, axis=0)  # (n_times, z, y, x)
        return result
        
    except Exception as e:
        print(f"Error extracting patch: {e}")
        # Return NaN patch
        shape = (len(time_indices), len(radar_ds.z), 10, 10)  # Fallback shape
        return np.full(shape, np.nan)

def create_training_samples(radar_zarr_path, output_path, dem_path='dem/preserve_dem_10m_utm.tif', train_years=None, val_years=None, 
                           start_date=None, end_date=None, 
                           day_filter_file=None, min_rainfall_mm=0.0, max_valid_rainfall=100.0, patch_size_m=2640):
    """
    Main function: Create aligned radar-gauge training samples
    
    Works at the DAY level: For each day in day_filter_file (or date range),
    extracts ALL hours (rainy and non-rainy) to create balanced training data.
    
    Parameters:
    -----------
    radar_zarr_path : str
        Path to radar zarr file (e.g., "KVBX_preserve_500m.zarr")
    output_path : str
        Where to save the prepared dataset (pickle file)
    train_years : list of int, optional
        List of years to include in training set (e.g., [2023, 2024])
        If provided, uses temporal split by year. If None, uses random split.
    val_years : list of int, optional
        List of years to include in validation set (e.g., [2024])
        Required if train_years is provided.
    start_date, end_date : str or datetime.date, optional
        Date range for data extraction (if day_filter_file not provided)
    day_filter_file : str, optional
        Path to file with specific dates (like my_rainy_days.txt)
        If provided, only these days will be processed.
        ALL hours from these days will be included (rainy and non-rainy).
    min_rainfall_mm : float, optional
        Minimum hourly rainfall to include (default: 0.0 = include all hours)
        Set to 0.0 when using day_filter_file to get both rainy and non-rainy hours
        Set higher (e.g., 0.5) to focus only on rainy hours
    max_valid_rainfall : float, optional
        Maximum physically plausible rainfall rate in mm/hr (default: 100.0)
        Samples with rainfall above this threshold are skipped (likely sensor errors).
        World record is ~305 mm/hr, but 100+ mm/hr is extremely rare.
    dem_path : str, optional
        Path to DEM GeoTIFF file (e.g., 'dem/preserve_dem_10m_utm.tif')
        If provided, DEM patches will be extracted for each sample.
        If None, only radar data will be included.
    patch_size: int, required
        Size of patches produced (eg 2640)
    Returns:
    --------
    dataset : dict
        Dictionary with 'train' and 'val' keys containing lists of samples
    """
    
    print("="*60)
    print("RADAR-GAUGE DATA ALIGNMENT")
    print("="*60)
    
    # Determine date range
    if day_filter_file:
        print(f"\nUsing specific dates from: {day_filter_file}")
        # Load dates from file
        with open(day_filter_file, 'r') as f:
            dates = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    dates.append(datetime.strptime(line, '%Y-%m-%d').date())
        
        if len(dates) == 0:
            print("No dates found in filter file!")
            return None
        
        print(f"   Found {len(dates)} days: {dates}")
        # Use min/max as date range
        start_date = min(dates)
        end_date = max(dates)
    else:
        print(f"\nUsing date range: {start_date} to {end_date}")
        dates = None  # Will use all days in range
    
    # Step 1: Load radar data INTO MEMORY (faster than dask lazy loading)
    print("\n1. Loading radar data into memory...")
    radar_ds = xr.open_zarr(radar_zarr_path, consolidated=False)
    print(f"   Radar shape: {radar_ds.reflectivity.shape}")
    print("   Loading full dataset to RAM (this may take a minute)...")
    radar_ds = radar_ds.load()  # Force load everything - no more dask overhead!
    print("   ✓ Radar data loaded to memory")
    
    # Note: DEM is extracted on-the-fly in the Dataset class, not here
    if dem_path:
        print(f"\n   ℹ️  DEM path stored in metadata: {dem_path}")
        print(f"   (DEM will be extracted on-the-fly during training)")
    
    # Check if times are valid
    if pd.isna(radar_ds.time.values).all():
        print("   ⚠️  WARNING: All time values are NaT!")
        print("   Attempting to read raw time values...")
        radar_ds_raw = xr.open_zarr(radar_zarr_path, decode_times=False, consolidated=False)
        print(f"   Raw time values (first 5): {radar_ds_raw.time.values[:5]}")
        print(f"   Time encoding: {radar_ds_raw.time.attrs}")
    else:
        print(f"   Time range: {radar_ds.time.min().values} to {radar_ds.time.max().values}")
    
    # Step 2: Get hourly precipitation (ALL hours from specified days)
    print("\n2. Loading hourly precipitation data...")
    print(f"   Min rainfall threshold: {min_rainfall_mm}mm")
    hourly_precip = get_hourly_precipitation_by_station(start_date, end_date, min_rainfall_mm=min_rainfall_mm)
    
    if min_rainfall_mm == 0.0:
        print(f"   Found {len(hourly_precip)} hourly samples (including non-rainy hours)")
    else:
        print(f"   Found {len(hourly_precip)} hourly samples with rainfall >= {min_rainfall_mm}mm")
    
    # Filter to specific dates if provided
    if dates:
        hourly_precip = [h for h in hourly_precip if h['hour_start'].date() in dates]
        print(f"   Filtered to {len(hourly_precip)} samples on specified days")
    
    if len(hourly_precip) == 0:
        print("No hourly precipitation found! Check your date range.")
        return None
    
    # Step 3: Sample radar scans for each hour
    print("\n3. Sampling radar scans for each hour...")
    print(f"   Filtering out readings > {max_valid_rainfall} mm/hr (sensor errors)")
    samples = []
    skipped_outliers = 0
    
    for i, precip in enumerate(tqdm(hourly_precip, desc="Processing hours")):
        # Skip samples with extreme rainfall (likely sensor errors)
        if precip['hourly_precip_mm'] > max_valid_rainfall:
            skipped_outliers += 1
            continue
        
        # Sample ~12 radar scans from this hour
        radar_times, radar_indices = sample_radar_scans_for_hour(
            radar_ds,
            precip['hour_start'],
            n_scans=12
        )
        
        # Need at least 6 radar scans
        if len(radar_indices) >= 6:
            # Create fixed time bins (12 bins of 5 minutes each)
            n_bins = 12
            bin_duration_minutes = 60 / n_bins  # 5 minutes per bin
            
            # Initialize bins (all None/empty)
            binned_indices = [None] * n_bins
            binned_times = [None] * n_bins
            
            # Assign each scan to its time bin
            for scan_time, scan_idx in zip(radar_times, radar_indices):
                # Calculate which bin this scan belongs to
                minutes_into_hour = (scan_time - precip['hour_start']).total_seconds() / 60
                bin_idx = int(minutes_into_hour / bin_duration_minutes)
                
                # Clamp to valid range [0, 11]
                bin_idx = max(0, min(n_bins - 1, bin_idx))
                
                # Assign to bin (if multiple scans in same bin, keep first)
                if binned_indices[bin_idx] is None:
                    binned_indices[bin_idx] = scan_idx
                    binned_times[bin_idx] = scan_time
            
            # Use binned data (preserves temporal structure)
            radar_indices = binned_indices
            radar_times = binned_times
            
            # Extract radar patch around station
            radar_patch = extract_radar_patch_at_station(
                radar_ds,
                radar_indices,
                precip['lat'],
                precip['lon'],
                patch_size_m=patch_size_m
            )
            

            # Create sample
            sample = {
                'hour_start': precip['hour_start'],
                'station_id': precip['station_id'],
                'station_name': precip['station_name'],
                'station_lat': precip['lat'],
                'station_lon': precip['lon'],
                'bias_flag': get_station_bias(precip['station_name']),
                'hourly_precip_mm': precip['hourly_precip_mm'],  # ← Single target value!
                'radar_times': radar_times,
                'radar_indices': radar_indices,
                'radar_patch': radar_patch,  # (12, z, y, x) - was (6, z, y, x)     # (1, 264, 264) or None
                'n_valid_radar': len([idx for idx in radar_indices if idx is not None])
            }
            
            # DEM will be extracted on-the-fly in Dataset using station_lat/lon
            # (already stored above - no extra storage needed!)
            if i % 500 == 0:
                gc.collect()
            samples.append(sample)
    
    print(f"\n4. Created {len(samples)} aligned samples")
    if skipped_outliers > 0:
        print(f"   ⚠️  Skipped {skipped_outliers} samples with rainfall > {max_valid_rainfall} mm/hr (sensor errors)")
        print(f"   ✓  Retained {len(samples)} valid samples")
    
    # Step 4: Train/val split (by year or random)
    print("\n5. Splitting train/validation...")
    
    if train_years is not None and val_years is not None:
        # Temporal split by year
        print(f"   Using temporal split:")
        print(f"     Train years: {train_years}")
        print(f"     Val years: {val_years}")
        
        train_samples = []
        val_samples = []
        
        for sample in samples:
            sample_year = sample['hour_start'].year
            
            if sample_year in train_years:
                train_samples.append(sample)
            elif sample_year in val_years:
                val_samples.append(sample)
            else:
                # Sample from year not in train or val - skip it
                print(f"   ⚠️  Skipping sample from year {sample_year} (not in train_years or val_years)")
        
        print(f"   Train: {len(train_samples)} samples from years {train_years}")
        print(f"   Val: {len(val_samples)} samples from years {val_years}")
        
        if len(train_samples) == 0:
            raise ValueError(f"No training samples found for years {train_years}!")
        if len(val_samples) == 0:
            raise ValueError(f"No validation samples found for years {val_years}!")
            
    else:
        # Random split (80/20)
        print(f"   Using random split (80/20)...")
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        split_idx = int(0.8 * len(samples))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        
        print(f"   Train: {len(train_samples)} samples")
        print(f"   Val: {len(val_samples)} samples")
    
    # Step 5: Save dataset
    print(f"\n6. Saving to {output_path}...")
    dataset = {
        'train': train_samples,
        'val': val_samples,
        'metadata': {
            'radar_zarr': radar_zarr_path,
            'dem_path': dem_path,  # DEM extracted on-the-fly in Dataset
            'start_date': str(start_date),
            'end_date': str(end_date),
            'day_filter_file': day_filter_file,
            'specific_days': [str(d) for d in dates] if dates else None,
            'split_type': 'temporal' if train_years is not None else 'random',
            'train_years': train_years if train_years is not None else 'N/A',
            'val_years': val_years if val_years is not None else 'N/A',
            'created': datetime.now().isoformat(),
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'radar_resolution_m': radar_ds.attrs.get('resolution_m', 500),
            'patch_size_m': patch_size_m 
        }
    }




    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n✅ Dataset saved!")
    print("="*60)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Stations: {len(set(s['station_id'] for s in samples))}")
    print(f"  Date range: {min(s['hour_start'] for s in samples)} to {max(s['hour_start'] for s in samples)}")
    
    # Rainy vs non-rainy breakdown
    rainy_samples = [s for s in samples if s['hourly_precip_mm'] >= 0.5]
    non_rainy_samples = [s for s in samples if s['hourly_precip_mm'] < 0.5]
    print(f"\n  Rainy hours (≥0.5mm): {len(rainy_samples)} ({100*len(rainy_samples)/len(samples):.1f}%)")
    print(f"  Non-rainy hours (<0.5mm): {len(non_rainy_samples)} ({100*len(non_rainy_samples)/len(samples):.1f}%)")
    
    print(f"\n  Avg valid radar scans per sample: {np.mean([s['n_valid_radar'] for s in samples]):.1f} / 12")
    print(f"  Avg hourly rainfall: {np.mean([s['hourly_precip_mm'] for s in samples]):.2f} mm/hr")
    print(f"  Rainfall range: {np.min([s['hourly_precip_mm'] for s in samples]):.2f} - {np.max([s['hourly_precip_mm'] for s in samples]):.2f} mm/hr")
    
    return dataset


def inspect_dataset(dataset_path):
    """
    Load and inspect a prepared dataset
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print("="*60)
    print("DATASET INSPECTION")
    print("="*60)
    
    print("\nMetadata:")
    for key, value in dataset['metadata'].items():
        print(f"  {key}: {value}")
    
    print(f"\nTrain samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['val'])}")
    
    # Inspect first sample
    if len(dataset['train']) > 0:
        sample = dataset['train'][0]
        print("\nFirst training sample:")
        print(f"  Hour: {sample['hour_start']}")
        print(f"  Station: {sample['station_name']} (ID: {sample['station_id']})")
        print(f"  Location: ({sample['station_lat']:.4f}, {sample['station_lon']:.4f})")
        print(f"  Radar patch shape: {sample['radar_patch'].shape}")
        print(f"  Hourly precipitation: {sample['hourly_precip_mm']:.2f} mm")
        print(f"  Valid radar scans: {sample['n_valid_radar']} / 12")
        print(f"  Radar scan times: {[str(t)[:19] if t else 'None' for t in sample['radar_times']]}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare aligned radar-gauge dataset")
    parser.add_argument('--radar', type=str, default='KVBX_preserve_500m.zarr',
                        help='Path to radar zarr file')
    parser.add_argument('--start', type=str,
                        help='Start date (YYYY-MM-DD) - required if not using --days')
    parser.add_argument('--end', type=str,
                        help='End date (YYYY-MM-DD) - required if not using --days')
    parser.add_argument('--days', type=str,
                        help='Path to file with specific dates (e.g., my_rainy_days.txt)')
    parser.add_argument('--output', type=str, default='radar_gauge_dataset.pkl',
                        help='Output file path')
    parser.add_argument('--min-rainfall', type=float, default=None,
                        help='Minimum hourly rainfall in mm (default: 0.0 with --days, 0.5 without)')
    parser.add_argument('--max-rainfall', type=float, default=100.0,
                        help='Maximum valid rainfall in mm/hr (default: 100.0). Samples above this are skipped as sensor errors.')
    parser.add_argument('--train-years', type=int, nargs='+', default=None,
                        help='Years to include in training set (e.g., --train-years 2023). Use temporal split if provided.')
    parser.add_argument('--val-years', type=int, nargs='+', default=None,
                        help='Years to include in validation set (e.g., --val-years 2024). Required if --train-years provided.')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect existing dataset instead of creating new one')
    parser.add_argument('--dem', type=str,
                        help='Path to dem file with (e.g., ./dem/preserver_dem_10m_utim.tif)')
    
    parser.add_argument('--patch-size', type=int, default=2640,
                    help='Patch size in meters (default: 2640 for 5x5 @ 500m). Use 4620 for 9x9.')
    
    args = parser.parse_args()
    
    if args.inspect:
        dataset = inspect_dataset(args.output)
    else:
        # Validate arguments
        if not args.days and (not args.start or not args.end):
            parser.error("Must provide either --days or both --start and --end")
        
        # Validate year split arguments
        if (args.train_years is not None) != (args.val_years is not None):
            parser.error("Must provide both --train-years and --val-years together, or neither")
        
        # Set default min_rainfall based on usage
        if args.min_rainfall is not None:
            min_rainfall = args.min_rainfall
        elif args.days:
            # When using day filter, include ALL hours (rainy and non-rainy)
            min_rainfall = 0.0
            print(f"Using --days: Including all hours (min_rainfall=0.0) for balanced dataset")
        else:
            # When using date range, default to filtering light rain
            min_rainfall = 0.5
            print(f"Using date range: Filtering to rainy hours only (min_rainfall=0.5)")
        
        dataset = create_training_samples(
            radar_zarr_path=args.radar,
            output_path=args.output,
            train_years=args.train_years,
            val_years=args.val_years,
            start_date=args.start,
            end_date=args.end,
            day_filter_file=args.days,
            min_rainfall_mm=min_rainfall,
            max_valid_rainfall=args.max_rainfall,
            dem_path=args.dem,
            patch_size_m=args.patch_size
        )

