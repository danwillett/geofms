# Requirements:
# pip install s3fs pyart xarray numpy tqdm pyproj

import numpy as np
import datetime as dt
import s3fs
import pyart
import xarray as xr
from pyproj import Transformer, Geod
from tqdm import tqdm
import io

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
    STATION = "KVBX"   # Vandenberg AFB
    # Radar lat/lon (fill in from station metadata or lookup)
    RADAR_LAT = 34.83855 
    RADAR_LON = -120.397917

    # Preserve bounding box (lon_min, lat_min, lon_max, lat_max) - replace with your values
    PRESERVE_BBOX = (-120.5130681, 34.5775648, -120.3456914, 34.4344052)

    START_DATE = dt.date(2022, 10, 1)
    END_DATE   = dt.date(2025, 5, 12)

    # desired horizontal resolution (meters)
    RESOLUTION = 500.0

    # vertical grid settings (z in meters)
    Z_MIN = 0.0
    Z_MAX = 15000.0   # 15 km tops
    Z_RES = 375.0     # vertical spacing (example)
    
    # Threading settings
    MAX_WORKERS = 30  # Number of parallel downloads/processing threads
    FIELD = 'reflectivity'
    
    # Quality control settings
    APPLY_QC = apply_qc  # Apply dual-pol QC to remove clutter
    QC_FIELDS = ['reflectivity']  # Fields to apply QC mask to
    # -------------------------------------------------
    
    if APPLY_QC:
        print("\n✅ Quality control ENABLED - will filter ground clutter & noise using RhoHV")
    else:
        print("\n⚠️  Quality control DISABLED - raw reflectivity will be used")

    # Derived vertical shape
    nz = int(np.ceil((Z_MAX - Z_MIN) / Z_RES))

    # pyproj transforms: use Azimuthal Equidistant centered on the radar
    aeqd = Transformer.from_crs(
        f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84",
        "epsg:3857", # intermediate, but we only use forward transform below via inverse usage; simpler to use transform directly
        always_xy=True
    )
    # Actually we want direct lon/lat -> local meters. Use Transformer with proj string forward:
    transformer = Transformer.from_crs("EPSG:4326",
                                    f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84",
                                    always_xy=True)

    def bbox_to_local_meters(bbox):
        lon_min, lat_min, lon_max, lat_max = bbox
        # get four corners (lon, lat) -> (x, y) in meters relative to radar center
        x1, y1 = transformer.transform(lon_min, lat_min)
        x2, y2 = transformer.transform(lon_max, lat_max)
        # compute axis-aligned min/max
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return xmin, xmax, ymin, ymax

    # check distance from radar center to preserve centroid to ensure coverage
    geod = Geod(ellps="WGS84")
    lon_c = 0.5 * (PRESERVE_BBOX[0] + PRESERVE_BBOX[2])
    lat_c = 0.5 * (PRESERVE_BBOX[1] + PRESERVE_BBOX[3])
    az12, az21, dist_m = geod.inv(RADAR_LON, RADAR_LAT, lon_c, lat_c)  # meters

    print(f"Distance radar -> preserve centroid: {dist_m/1000:.1f} km")

    # reasonable NEXRAD range threshold (meters)
    RANGE_THRESHOLD = 250_000.0
    if dist_m > RANGE_THRESHOLD:
        print("Warning: preserve centroid is >250 km from radar - limited or no coverage likely.")

    # compute local bbox in meters and build grid limits
    xmin, xmax, ymin, ymax = bbox_to_local_meters(PRESERVE_BBOX)
    # expand a little buffer (optional)
    buffer_m = 1000.0
    xmin -= buffer_m; xmax += buffer_m; ymin -= buffer_m; ymax += buffer_m

    # compute grid nx, ny for chosen resolution
    nx = int(np.ceil((xmax - xmin) / RESOLUTION))
    ny = int(np.ceil((ymax - ymin) / RESOLUTION))
    nz = int(np.ceil((Z_MAX - Z_MIN) / Z_RES))

    print(f"Grid dims (z,y,x): {nz}, {ny}, {nx}  (resolution {RESOLUTION} m)")

    # pyart grid_limits expects ((zmin,zmax),(ymin,ymax),(xmin,xmax))
    grid_limits = ((Z_MIN, Z_MAX), (ymin, ymax), (xmin, xmax))
    grid_shape = (nz, ny, nx)

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
                time_val = float(grid.time['data'][0])
                units = grid.time['units']  # e.g. "seconds since 2024-01-01T00:00:00Z"

                # Extract base time from units
                base_time_str = re.search(r"since\s+([0-9T:\-\.Z]+)", units).group(1)
                base_time = datetime.fromisoformat(base_time_str.replace("Z", "+00:00"))

                scan_time = base_time + timedelta(seconds=time_val)
                return arr, scan_time
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    
        # If we get here, all retries failed
        raise last_error

    # Setup for resumable processing
    zarr_path = f"{STATION}_preserve_500m.zarr"
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
                
                if first_write:
                    # First write - create the zarr store with full coordinates
                    new_da = xr.DataArray(
                        arr[None, ...],  # Add time dimension
                        dims=('time', 'z', 'y', 'x'),
                        coords={
                            'time': [np.datetime64(scan_time)],
                            'z': np.linspace(Z_MIN + Z_RES/2, Z_MAX - Z_RES/2, nz),
                            'y': np.linspace(ymin + RESOLUTION/2, ymax - RESOLUTION/2, ny),
                            'x': np.linspace(xmin + RESOLUTION/2, xmax - RESOLUTION/2, nx)
                        },
                        name='reflectivity'
                    )
                    # Use explicit encoding to avoid time decoding issues
                    encoding = {
                        'time': {'units': 'nanoseconds since 1970-01-01', 'calendar': 'proleptic_gregorian'}
                    }
                    new_da.to_zarr(zarr_path, mode='w', encoding=encoding)
                else:
                    # For appending, only include time coordinate (spatial coords already exist)
                    # Don't provide encoding - the time variable already exists with its encoding
                    new_da = xr.DataArray(
                        arr[None, ...],  # Add time dimension
                        dims=('time', 'z', 'y', 'x'),
                        coords={
                            'time': [np.datetime64(scan_time)]
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

if __name__ == "__main__":
    pull_nexrad()