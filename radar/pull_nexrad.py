# Requirements:
# pip install s3fs pyart xarray numpy tqdm pyproj shapely

import numpy as np
import datetime as dt
import s3fs
import pyart
import xarray as xr
from pyproj import Transformer, Geod
from tqdm import tqdm
import io

import rasterio
from shapely.geometry import box
import cftime
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import time

def pull_nexrad(day_filter=None, day_filter_file=None, apply_qc=True):
    """
    Pull NEXRAD radar data and grid to Cartesian coordinates
    
    Parameters:
    -----------
    day_filter : list of datetime.date, optional
        List of specific days to process (overrides START_DATE/END_DATE if provided)
    day_filter_file : str, optional
        Path to file containing list of days to process (one per line, YYYY-MM-DD format)
        Overrides START_DATE/END_DATE if provided
    apply_qc : bool, optional
        Apply dual-polarization quality control to remove clutter (default: True)
        Uses correlation coefficient (RhoHV) to filter non-meteorological echoes
    
    If both day_filter and day_filter_file are None, processes all days between
    START_DATE and END_DATE.
    """
    
    # ---------------- USER PARAMETERS ----------------
    STATION = "KVBX"
    RADAR_LAT, RADAR_LON = 34.83855, -120.397917
    RESOLUTION = 500.0  # radar horizontal spacing in meters (native resolution)
    Z_MIN, Z_MAX, Z_RES = 0.0, 15000.0, 375.0
    PRESERVE_BBOX = (-120.5130681, 34.5775648, -120.3456914, 34.4344052)  # study AOI (west, north, east, south)
    PATCH_SIZE_M = 2640  # TerraMesh patch size in meters
    UTM_CRS = "EPSG:32610"  # UTM Zone 10N for California coast
    BUFFER_M = 5000  # Add 5km buffer around preserve

    START_DATE = dt.date(2022, 10, 1)
    END_DATE   = dt.date(2025, 5, 12)

    # Threading settings
    MAX_WORKERS = 30  # Number of parallel downloads/processing threads
    FIELD = 'reflectivity'
    
    # Quality control settings
    APPLY_QC = apply_qc  # Apply dual-pol QC to remove clutter
    QC_FIELDS = ['reflectivity']  # Fields to apply QC mask to
    # -------------------------------------------------
    
    print(f"\n📐 Resolution: {RESOLUTION}m (native radar resolution)")
    print(f"   Patch size: {PATCH_SIZE_M}m → {int(PATCH_SIZE_M/RESOLUTION)}×{int(PATCH_SIZE_M/RESOLUTION)} pixels per patch")
    
    if APPLY_QC:
        print("\n✅ Quality control ENABLED - will filter ground clutter & noise using RhoHV")
    else:
        print("\n⚠️  Quality control DISABLED - raw reflectivity will be used")


    # Convert AOI from lat/lon to UTM coordinates
    from pyproj import CRS
    wgs84 = CRS.from_epsg(4326)
    utm_crs = CRS.from_string(UTM_CRS)
    transformer_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    
    # PRESERVE_BBOX is (west, north, east, south) in lat/lon
    west_utm, north_utm = transformer_to_utm.transform(PRESERVE_BBOX[0], PRESERVE_BBOX[1])
    east_utm, south_utm = transformer_to_utm.transform(PRESERVE_BBOX[2], PRESERVE_BBOX[3])
    
    # Add buffer and align to PATCH_SIZE_M boundaries for clean patch extraction
    xmin = np.floor((west_utm - BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M
    xmax = np.ceil((east_utm + BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M
    ymin = np.floor((south_utm - BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M
    ymax = np.ceil((north_utm + BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M
    
    # Calculate grid dimensions
    num_patches_x = int((xmax - xmin) / PATCH_SIZE_M)
    num_patches_y = int((ymax - ymin) / PATCH_SIZE_M)
    grid_width_m = xmax - xmin
    grid_height_m = ymax - ymin
    
    print(f"\n{'='*60}")
    print(f"Radar Grid Configuration:")
    print(f"  Study area (lat/lon): {PRESERVE_BBOX}")
    print(f"  Buffer: {BUFFER_M}m")
    print(f"  Patch alignment: {PATCH_SIZE_M}m boundaries")
    print(f"  ")
    print(f"  UTM Bounding Box ({UTM_CRS}):")
    print(f"    X: [{xmin:.0f}, {xmax:.0f}] meters ({num_patches_x} patches)")
    print(f"    Y: [{ymin:.0f}, {ymax:.0f}] meters ({num_patches_y} patches)")
    print(f"    Dimensions: {grid_width_m:.0f}m × {grid_height_m:.0f}m")
    print(f"  ")
    print(f"  Resolution: {RESOLUTION}m")
    print(f"  Pixels per patch: {int(PATCH_SIZE_M/RESOLUTION)}×{int(PATCH_SIZE_M/RESOLUTION)}")
    print(f"{'='*60}\n")

    # Convert UTM bounds to radar-local azimuthal equidistant coordinates for PyART gridding
    transformer = Transformer.from_crs(
        utm_crs, 
        f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84", 
        always_xy=True
    )
    xmin_m, ymin_m = transformer.transform(xmin, ymin)
    xmax_m, ymax_m = transformer.transform(xmax, ymax)

    # Compute radar grid
    nx = int(np.ceil((xmax_m - xmin_m) / RESOLUTION))
    ny = int(np.ceil((ymax_m - ymin_m) / RESOLUTION))
    nz = int(np.ceil((Z_MAX - Z_MIN) / Z_RES))

    grid_limits = ((Z_MIN, Z_MAX), (ymin_m, ymax_m), (xmin_m, xmax_m))
    grid_shape = (nz, ny, nx)

    print(f"PyART Gridding Parameters:")
    print(f"  Grid shape (z,y,x): {grid_shape}")
    print(f"  Resolution: {RESOLUTION}m horizontal, {Z_RES}m vertical")
    print(f"  Radar-local coordinates (for PyART gridding):")
    print(f"    X: [{xmin_m:.0f}, {xmax_m:.0f}] meters from radar")
    print(f"    Y: [{ymin_m:.0f}, {ymax_m:.0f}] meters from radar")
    print(f"    Z: [{Z_MIN:.0f}, {Z_MAX:.0f}] meters altitude")
    print(f"  Total grid size: {nx*ny*nz:,} cells")
    print(f"  Radar location: ({RADAR_LAT:.4f}, {RADAR_LON:.4f})")
    print(f"\n  ℹ️  PyART grids in radar-local coords, output stored in UTM for alignment with DEM/LULC\n")

    # S3 setup
    bucket = "unidata-nexrad-level2"
    fs = s3fs.S3FileSystem(anon=True)

    def list_daily_files(station, date, max_retries=3):
        prefix = f"{date.year}/{date:%m}/{date:%d}/{station}/"
        
        for attempt in range(max_retries):
            try:
                keys = fs.ls(f"s3://{bucket}/{prefix}")
                
                # Filter out MDM files (metadata) - keep only actual radar data files
                keys = [k for k in keys if not k.endswith("_MDM")]
                
                # keys are like "unidata-nexrad-level2/2024/01/01/KVTX/KVTX20240101_120000_V06"
                # convert to s3:// path for open
                files = [f"{bucket}/{k.split('/',1)[1]}" if k.startswith(bucket + "/") else k for k in keys]
                return [f"s3://{p}" if not p.startswith("s3://") else p for p in files]
                
            except FileNotFoundError:
                # This date doesn't exist in the bucket - that's OK, just skip it
                return []
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a 404/NoSuchKey error (date doesn't exist)
                if "404" in error_msg or "NoSuchKey" in error_msg or "does not exist" in error_msg.lower():
                    return []
                
                # For other errors, retry with backoff
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                    print(f"S3 list error for {date} (attempt {attempt+1}/{max_retries}): {error_msg}")
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"S3 list failed for {date} after {max_retries} attempts: {error_msg}")
                    return []
        
        return []

    def apply_dualpol_qc_inline(radar, fields_to_qc=['reflectivity']):
        """
        Apply dual-polarization quality control inline
        Uses correlation coefficient (RhoHV) to mask non-meteorological echoes
        
        Returns:
        --------
        radar : pyart.Radar
            Radar with QC applied
        qc_applied : bool
            True if QC was applied, False if RhoHV not available
        """
        # Check if dual-pol field is available
        if 'cross_correlation_ratio' not in radar.fields:
            # Try alternate names
            if 'RHOHV' in radar.fields:
                rhohv_name = 'RHOHV'
            elif 'correlation_coefficient' in radar.fields:
                rhohv_name = 'correlation_coefficient'
            else:
                return radar, False  # No RhoHV available, skip QC
        else:
            rhohv_name = 'cross_correlation_ratio'
        
        # Get correlation coefficient
        rhohv = radar.fields[rhohv_name]['data']
        
        # Create QC mask: RhoHV < 0.9 indicates non-meteorological echoes
        qc_mask = rhohv < 0.9
        
        # Apply mask to requested fields
        for field_name in fields_to_qc:
            if field_name in radar.fields:
                data = radar.fields[field_name]['data']
                # Mask out bad data
                data = np.ma.masked_where(qc_mask, data)
                radar.fields[field_name]['data'] = data
        
        return radar, True
    
    def grid_radar_file_from_s3(s3_path, max_retries=3):
        """Download and grid a single radar file with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # open remote file and read with pyart
                with fs.open(s3_path.replace("s3://", ""), 'rb') as f:
                    radar = pyart.io.read_nexrad_archive(f)
                
                # Apply quality control before gridding
                qc_applied = False
                if APPLY_QC:
                    radar, qc_applied = apply_dualpol_qc_inline(radar, fields_to_qc=QC_FIELDS)
                    # Track QC statistics
                    with lock:
                        if qc_applied:
                            qc_stats['files_with_qc'] += 1
                        else:
                            qc_stats['files_without_qc'] += 1
                
                # grid only requested limits/shape, and request the field
                grid = pyart.map.grid_from_radars(
                    [radar],
                    grid_shape=grid_shape,
                    grid_limits=grid_limits,
                    grid_origin=(RADAR_LAT, RADAR_LON),
                    fields=[FIELD],
                    form='linear'  # or 'barnes' depending on your preference
                )
                
                # extract field array (z,y,x)
                arr = grid.fields[FIELD]['data'].filled(np.nan) if hasattr(grid.fields[FIELD]['data'], 'filled') else grid.fields[FIELD]['data']
                
                # get timestamp; pyart grid.time is typically of shape (1,)
                try:
                    time_val = float(grid.time['data'][0])
                    units = grid.time['units']  # e.g. "seconds since 2024-01-01T00:00:00Z"

                    # Extract base time from units
                    base_time_str = re.search(r"since\s+([0-9T:\-\.Z]+)", units).group(1)
                    base_time = datetime.fromisoformat(base_time_str.replace("Z", "+00:00"))

                    scan_time = base_time + timedelta(seconds=time_val)
                    
                    # Validate scan_time
                    if scan_time is None:
                        raise ValueError(f"Invalid scan_time computed: {scan_time}")
                    
                except Exception as time_error:
                    # Fallback: try to get time from radar object directly
                    print(f"Warning: Time extraction failed ({time_error}), using radar.time")
                    try:
                        # PyART radar objects have a time attribute
                        radar_time = radar.time
                        scan_time = datetime.utcfromtimestamp(radar_time['data'][0])
                    except:
                        # Last resort: parse from filename
                        # KVBX20240401_120000_V06 -> 2024-04-01 12:00:00
                        filename = s3_path.split('/')[-1]
                        time_match = re.search(r'(\d{8})_(\d{6})', filename)
                        if time_match:
                            date_str = time_match.group(1)  # YYYYMMDD
                            time_str = time_match.group(2)  # HHMMSS
                            scan_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                        else:
                            raise ValueError(f"Could not extract time from {s3_path}")
                
                return arr, scan_time
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    
        # If we get here, all retries failed
        raise last_error

    # Setup for resumable processing
    zarr_path = f"{STATION}_preserve_{int(RESOLUTION)}m.zarr"
    processed_log = "processed_files.txt"
    lock = threading.Lock()
    
    # QC statistics tracking (thread-safe)
    qc_stats = {'files_with_qc': 0, 'files_without_qc': 0}
    
    # Load processed files if exists
    if os.path.exists(processed_log):
        with open(processed_log) as f:
            processed_files = set(line.strip() for line in f)
        print(f"Found {len(processed_files)} already processed files")
        if os.path.exists(zarr_path):
            print(f"Resuming - will append to existing zarr file")
    else:
        processed_files = set()

    def process_and_save_file(s3_path):
        """Process a single radar file and immediately append to zarr"""
        # Check if already processed (thread-safe check)
        with lock:
            if s3_path in processed_files:
                return None
            # Mark as being processed to prevent duplicate work
            processed_files.add(s3_path)
        
        try:
            # Process the file (this happens outside the lock for parallelism)
            arr, scan_time = grid_radar_file_from_s3(s3_path)
            
            # Write to zarr (must be thread-safe)
            with lock:
                first_write = not os.path.exists(zarr_path)
                
                # Debug: Print scan time being written
                print(f"Writing scan at: {scan_time} (type: {type(scan_time)})")
                
                if first_write:
                    # First write - create the zarr store with full coordinates
                    # Coordinates are in UTM (matching Sentinel-2/TerraMesh)
                    # Note: radar is gridded in radar-local coords, but we approximate with UTM grid
                    # This is valid for small areas (~20km) where projection distortion is minimal
                    
                    # Convert to numpy datetime64 explicitly
                    time_np = np.datetime64(scan_time, 'ns')
                    print(f"  As numpy datetime64: {time_np}")
                    
                    new_da = xr.DataArray(
                        arr[None, ...],  # Add time dimension
                        dims=('time', 'z', 'y', 'x'),
                        coords={
                            'time': [time_np],
                            'z': np.linspace(Z_MIN + Z_RES/2, Z_MAX - Z_RES/2, nz),
                            'y': np.linspace(ymin + RESOLUTION/2, ymax - RESOLUTION/2, ny),
                            'x': np.linspace(xmin + RESOLUTION/2, xmax - RESOLUTION/2, nx)
                        },
                        name='reflectivity',
                        attrs={
                            'crs': UTM_CRS,
                            'crs_wkt': utm_crs.to_wkt(),
                            'radar_lat': RADAR_LAT,
                            'radar_lon': RADAR_LON,
                            'radar_station': STATION,
                            'utm_bounds': f'({xmin}, {ymin}, {xmax}, {ymax})',
                            'patch_size_m': PATCH_SIZE_M,
                            'resolution_m': RESOLUTION,
                            'description': 'NEXRAD reflectivity at 500m resolution, aligned to 2640m patch boundaries in UTM'
                        }
                    )
                    # Use explicit encoding to avoid time decoding issues
                    encoding = {
                        'time': {'units': 'nanoseconds since 1970-01-01', 'calendar': 'proleptic_gregorian'}
                    }
                    new_da.to_zarr(zarr_path, mode='w', encoding=encoding)
                else:
                    # For appending, only include time coordinate (spatial coords already exist)
                    # Don't provide encoding - the time variable already exists with its encoding
                    
                    # Convert to numpy datetime64 explicitly
                    time_np = np.datetime64(scan_time, 'ns')
                    
                    new_da = xr.DataArray(
                        arr[None, ...],  # Add time dimension
                        dims=('time', 'z', 'y', 'x'),
                        coords={
                            'time': [time_np]
                        },
                        name='reflectivity'
                    )
                    
                    # Retry zarr write if Windows file locking causes issues
                    max_write_retries = 3
                    for write_attempt in range(max_write_retries):
                        try:
                            new_da.to_zarr(zarr_path, append_dim='time')
                            break  # Success!
                        except PermissionError as pe:
                            if write_attempt < max_write_retries - 1:
                                time.sleep(0.2 * (write_attempt + 1))  # 0.2s, 0.4s, 0.6s
                            else:
                                raise  # Give up after 3 attempts
                
                # Log successful processing after successful write
                with open(processed_log, 'a') as f:
                    f.write(s3_path + "\n")
            
            return s3_path
            
        except Exception as e:
            # More detailed error reporting
            error_type = type(e).__name__
            error_msg = str(e)
            filename = s3_path.split('/')[-1]
            print(f"Failed {filename}: {error_type} - {error_msg[:100]}")
            
            # Remove from processed set since it failed
            with lock:
                processed_files.discard(s3_path)
            return None

    # Handle day filtering
    if day_filter_file is not None:
        # Load days from file
        print(f"Loading day filter from: {day_filter_file}")
        with open(day_filter_file, 'r') as f:
            dates = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    dates.append(datetime.strptime(line, '%Y-%m-%d').date())
        print(f"Loaded {len(dates)} days from filter file")
        
    elif day_filter is not None:
        # Use provided list of days
        dates = day_filter
        print(f"Using day filter with {len(dates)} days")
        
    else:
        # Use date range
        dates = [START_DATE + dt.timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
        print(f"Processing date range: {START_DATE} to {END_DATE}")
    
    # Main loop: collect all files first, then process with threading
    all_files = []
    missing_dates = []
    
    print(f"\nCollecting file list...")
    print(f"Total days to check: {len(dates)}")
    
    for d in tqdm(dates, desc="Scanning dates"):
        files = list_daily_files(STATION, d)
        if files:
            all_files.extend(files)
        else:
            missing_dates.append(d)
    
    if missing_dates:
        print(f"\nNote: {len(missing_dates)} dates had no data or were inaccessible")
        if len(missing_dates) <= 10:
            print(f"  Missing dates: {', '.join(str(d) for d in missing_dates)}")
        else:
            print(f"  First missing: {missing_dates[0]}, Last missing: {missing_dates[-1]}")
    
    print(f"\nTotal files to process: {len(all_files)}")
    files_to_process = [f for f in all_files if f not in processed_files]
    print(f"  New files: {len(files_to_process)}")
    print(f"  Already processed: {len(all_files) - len(files_to_process)}")
    
    if len(files_to_process) == 0:
        print("All files already processed!")
        print(f"Zarr file available at: {zarr_path}")
        return
    
    # Process files with threading - each file is saved immediately
    successful = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files
        future_to_path = {executor.submit(process_and_save_file, s3path): s3path for s3path in all_files}
        
        # Track results as they complete
        for future in tqdm(as_completed(future_to_path), total=len(all_files), desc="Processing files"):
            result = future.result()
            if result is not None:
                successful += 1
    
    skipped = len(all_files) - len(files_to_process)
    failed = len(files_to_process) - successful
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already processed): {skipped}")
    print(f"  Total files: {len(all_files)}")
    
    # Print QC statistics
    if APPLY_QC:
        total_qc_files = qc_stats['files_with_qc'] + qc_stats['files_without_qc']
        if total_qc_files > 0:
            print(f"\nQuality Control Statistics:")
            print(f"  Files with QC applied: {qc_stats['files_with_qc']} ({100*qc_stats['files_with_qc']/total_qc_files:.1f}%)")
            print(f"  Files without RhoHV: {qc_stats['files_without_qc']} ({100*qc_stats['files_without_qc']/total_qc_files:.1f}%)")
    
    print(f"\nZarr file saved to: {zarr_path}")
    print(f"\nNote: Time dimension may be unsorted (files appended as they completed).")
    print(f"To sort by time, use: xr.open_zarr('{zarr_path}').sortby('time')")
    print(f"{'='*60}")

def load_radar_for_terramesh(zarr_path):
    """
    Load radar data aligned to 2640m patch boundaries in UTM coordinates.
    
    The zarr file contains radar data at 500m resolution, stored in UTM coordinates
    for easy alignment with DEM and LULC data.
    
    Parameters:
    -----------
    zarr_path : str
        Path to the radar zarr file
    
    Returns:
    --------
    ds : xarray.Dataset
        Dataset with radar data in UTM coordinates
    """
    import xarray as xr
    
    # Load the radar data
    ds = xr.open_zarr(zarr_path)
    
    # Display metadata
    attrs = ds.reflectivity.attrs
    print(f"Radar data loaded successfully!")
    print(f"  Coordinate system: {attrs['crs']}")
    print(f"  Radar station: {attrs['radar_station']} at ({attrs['radar_lat']}, {attrs['radar_lon']})")
    print(f"  UTM bounds: {attrs['utm_bounds']}")
    print(f"  Resolution: {attrs['resolution_m']}m")
    print(f"  Patch size: {attrs['patch_size_m']}m")
    print(f"  Pixels per patch: {int(attrs['patch_size_m'] / attrs['resolution_m'])}×{int(attrs['patch_size_m'] / attrs['resolution_m'])}")
    print(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"  Number of time steps: {len(ds.time)}")
    print(f"\n✅ Data ready for multi-modal fusion with DEM/LULC!")
    
    return ds


if __name__ == "__main__":
    pull_nexrad()


# ==============================================================================
# USAGE NOTES FOR MULTI-MODAL DEEP LEARNING
# ==============================================================================
#
# This script creates NEXRAD radar data at 500m resolution, aligned to 2640m patch
# boundaries in UTM coordinates. This makes it ready to fuse with DEM and LULC data
# for multi-modal precipitation prediction using TerraMind.
#
# KEY POINTS:
# -----------
# 1. Resolution Strategy:
#    - Radar: 500m (native resolution, ~250m beam width)
#    - DEM/LULC: 10m (TerraMind native resolution)
#    - Output: 5×5 grid at 500m (precipitation predictions)
#    
#    Each modality is processed at its optimal resolution, then fused in the model.
#
# 2. Coordinate System:
#    - All data in UTM Zone 10N (EPSG:32610)
#    - PyART grids radar in radar-local azimuthal equidistant coordinates
#    - Output is stored in UTM for alignment with DEM/LULC
#    - For areas <50km from radar, projection distortion is <0.5%
#
# 3. Patch Alignment:
#    - Bounding box aligned to 2640m boundaries (TerraMind patch size)
#    - Radar: 5×5 pixels per 2640m patch (at 500m resolution)
#    - DEM/LULC: 264×264 pixels per 2640m patch (at 10m resolution)
#    - Same geographic area, different pixel densities
#
# 4. Usage Example:
#    a) Load radar data:
#       from radar.pull_nexrad import load_radar_for_terramesh
#       radar_ds = load_radar_for_terramesh("KVBX_preserve_500m.zarr")
#
#    b) Extract patch around rain gauge:
#       gauge_x, gauge_y = 233456, 3823456  # UTM coordinates
#       patch_size = 2640  # meters
#       
#       # Extract 5×5 radar patch at 500m (covers 2500m × 2500m)
#       radar_patch = radar_ds.sel(
#           time=target_time, method='nearest',
#           x=slice(gauge_x - patch_size/2, gauge_x + patch_size/2),
#           y=slice(gauge_y - patch_size/2, gauge_y + patch_size/2)
#       ).reflectivity.values  # Shape: (z, 5, 5)
#
#       # Extract 264×264 DEM patch at 10m (covers 2640m × 2640m)
#       dem_patch = dem_ds.sel(
#           x=slice(gauge_x - patch_size/2, gauge_x + patch_size/2),
#           y=slice(gauge_y - patch_size/2, gauge_y + patch_size/2)
#       ).values  # Shape: (264, 264)
#
#    c) Feed to model:
#       # Radar encoder processes 500m data
#       radar_features = radar_encoder(radar_patch)  # CNN on (6, Z, 5, 5)
#       
#       # TerraMind processes 10m data
#       terrain_features = terramind(dem_patch, lulc_patch)  # (264, 264)
#       
#       # Fusion
#       fused = fusion_layer([radar_features, terrain_features])
#       precip_map = decoder(fused)  # Output: (5, 5) at 500m resolution
#
# 5. Quality Control:
#    - Set apply_qc=True (default) to filter ground clutter
#    - Uses dual-pol correlation coefficient (RhoHV < 0.9 = non-meteorological)
#    - Essential for precipitation estimation
#
# 6. Data Acquisition:
#    - DEM: Download Copernicus 30m DEM, resample to 10m
#    - LULC: Download ESRI 10m Land Cover
#    - Both should use same UTM bounds (xmin, ymin, xmax, ymax) as radar
#
# ==============================================================================