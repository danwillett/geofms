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
    Z_MIN, Z_MAX, Z_RES = 0.0, 8000.0, 375.0  # 22 levels; coastal CA echo tops rarely exceed 7-8km (Ralph 2004, Neiman 2002)
    PRESERVE_BBOX = (-120.5130681, 34.5775648, -120.3456914, 34.4344052)  # study AOI (west, north, east, south)
    PATCH_SIZE_M = 2640  # TerraMesh patch size in meters
    UTM_CRS = "EPSG:32610"  # UTM Zone 10N for California coast
    BUFFER_M = 5000  # Add 5km buffer around preserve

    START_DATE = dt.date(2022, 10, 1)
    END_DATE   = dt.date(2026, 3, 27)

    # Threading settings
    MAX_WORKERS = 10  # Number of parallel downloads/processing threads
    FIELDS = [
        'reflectivity',
        'differential_reflectivity',   # ZDR
        'cross_correlation_ratio',      # RhoHV
        'differential_phase',           # PhiDP
        'specific_differential_phase',  # KDP (computed from PhiDP)
    ]
    
    # Quality control settings
    APPLY_QC = apply_qc
    QC_FIELDS = ['reflectivity', 'differential_reflectivity']  # Apply QC mask to Z and ZDR
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

                # Compute KDP from PhiDP — gracefully skip if PhiDP unavailable or computation fails
                if 'differential_phase' in radar.fields:
                    try:
                        kdp, phidp_cor, *_ = pyart.retrieve.kdp_maesaka(radar)  # * handles 2- or 3-value return across PyART versions
                        radar.add_field('specific_differential_phase', kdp, replace_existing=True)
                    except Exception as kdp_err:
                        print(f"  KDP computation failed, skipping: {kdp_err}")

                # Only grid fields that are actually present in this radar volume
                fields_to_grid = [f for f in FIELDS if f in radar.fields]

                grid = pyart.map.grid_from_radars(
                    [radar],
                    grid_shape=grid_shape,
                    grid_limits=grid_limits,
                    grid_origin=(RADAR_LAT, RADAR_LON),
                    fields=fields_to_grid,
                    form='linear'
                )

                # Extract each gridded field into a dict of (z, y, x) float arrays
                field_arrays_3d = {}
                ref_data = None
                for field_name in fields_to_grid:
                    raw = grid.fields[field_name]['data']
                    arr = raw.filled(np.nan) if hasattr(raw, 'filled') else np.array(raw, dtype=float)

                    # KDP noise mask: unreliable in light rain / below-threshold regions
                    if field_name == 'specific_differential_phase' and ref_data is not None:
                        arr = np.where(ref_data >= 20.0, arr, np.nan)

                    field_arrays_3d[field_name] = arr
                    if field_name == 'reflectivity':
                        ref_data = arr  # save for KDP mask above

                # --- Collapse Z dimension → 2D (y, x) ---
                # Reflectivity : column maximum (CMAX) — standard approach
                # All dual-pol : sampled at the HEIGHT of max reflectivity (collocated)
                #   → ZDR/KDP/RhoHV at the dominant echo level, not independent maxima
                # This is ~40× smaller than storing all 40 Z levels and matches what
                # the model does anyway (torch.max over Z at training time).
                field_arrays = {}
                if 'reflectivity' in field_arrays_3d:
                    ref_3d = field_arrays_3d['reflectivity']               # (z, y, x)
                    field_arrays['reflectivity'] = np.nanmax(ref_3d, axis=0)  # (y, x)

                    # Z-index of max reflectivity at every (y, x) pixel.
                    # Substitute -inf for NaN so argmax handles all-NaN (clear-air) columns
                    # without raising ValueError. Those pixels return z=0 and will sample NaN
                    # from the dual-pol arrays anyway → correctly NaN in the output.
                    ref_safe = np.where(np.isfinite(ref_3d), ref_3d, -np.inf)
                    z_idx = np.argmax(ref_safe, axis=0)                    # (y, x)
                    ny_g, nx_g = ref_3d.shape[1], ref_3d.shape[2]
                    yy, xx = np.meshgrid(np.arange(ny_g), np.arange(nx_g), indexing='ij')

                    # Sample every dual-pol field at the level of max-Z
                    for field_name, arr_3d in field_arrays_3d.items():
                        if field_name == 'reflectivity':
                            continue
                        field_arrays[field_name] = arr_3d[z_idx, yy, xx]  # (y, x)
                else:
                    # Fallback: nanmax across Z for all fields if reflectivity missing
                    for field_name, arr_3d in field_arrays_3d.items():
                        field_arrays[field_name] = np.nanmax(arr_3d, axis=0)

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
                
                return field_arrays, scan_time
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    
        # If we get here, all retries failed
        raise last_error

    # Setup for resumable processing
    zarr_path = f"{STATION}_preserve_{int(RESOLUTION)}m_{START_DATE}_{END_DATE}.zarr"
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
            field_arrays, scan_time = grid_radar_file_from_s3(s3_path)

            # Write to zarr (must be thread-safe)
            with lock:
                first_write = not os.path.exists(zarr_path)

                # Convert to numpy datetime64 explicitly
                time_np = np.datetime64(scan_time, 'ns')
                print(f"Writing scan at: {scan_time}")

                # Metadata attrs stored on each variable
                shared_attrs = {
                    'crs': UTM_CRS,
                    'crs_wkt': utm_crs.to_wkt(),
                    'radar_lat': RADAR_LAT,
                    'radar_lon': RADAR_LON,
                    'radar_station': STATION,
                    'utm_bounds': f'({xmin}, {ymin}, {xmax}, {ymax})',
                    'patch_size_m': PATCH_SIZE_M,
                    'resolution_m': RESOLUTION,
                    'description': (
                        'NEXRAD dual-pol radar at 500m resolution, Z-collapsed to 2D (y, x). '
                        'reflectivity = column maximum (CMAX) over 0–8000m (22 levels at 375m spacing). '
                        '8000m ceiling justified for coastal CA: echo tops rarely exceed 7–8km '
                        '(Ralph et al. 2004; Neiman et al. 2002). '
                        'ZDR, RhoHV, PhiDP, KDP = sampled at height of max reflectivity (collocated). '
                        'KDP masked where Z < 20 dBZ. '
                        'Aligned to 2640m patch boundaries in UTM.'
                    ),
                }

                def make_dataset(field_arrays, time_np, include_spatial=True):
                    """Build an xr.Dataset from a dict of 2D (y, x) arrays for one timestep."""
                    coords = {'time': [time_np]}
                    if include_spatial:
                        coords.update({
                            'y': np.linspace(ymin + RESOLUTION/2, ymax - RESOLUTION/2, ny),
                            'x': np.linspace(xmin + RESOLUTION/2, xmax - RESOLUTION/2, nx),
                        })
                    data_vars = {
                        name: xr.DataArray(
                            arr[None, ...],  # add time dim → (1, y, x)
                            dims=('time', 'y', 'x'),
                            attrs=shared_attrs,
                        )
                        for name, arr in field_arrays.items()
                    }
                    return xr.Dataset(data_vars, coords=coords)

                if first_write:
                    # First write — create zarr store with full spatial coordinates
                    # (valid approximation for small AOI <50 km from radar)
                    print(f"  As numpy datetime64: {time_np}")
                    ds = make_dataset(field_arrays, time_np, include_spatial=True)
                    encoding = {
                        'time': {'units': 'nanoseconds since 1970-01-01', 'calendar': 'proleptic_gregorian'}
                    }
                    ds.to_zarr(zarr_path, mode='w', encoding=encoding)
                else:
                    # Append — spatial coords already in store, only supply time
                    ds = make_dataset(field_arrays, time_np, include_spatial=False)

                    # Retry zarr write if Windows file locking causes issues
                    max_write_retries = 3
                    for write_attempt in range(max_write_retries):
                        try:
                            ds.to_zarr(zarr_path, append_dim='time')
                            break  # success
                        except PermissionError:
                            if write_attempt < max_write_retries - 1:
                                time.sleep(0.2 * (write_attempt + 1))  # 0.2s, 0.4s, 0.6s
                            else:
                                raise  # give up after 3 attempts

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
    Load dual-pol radar data aligned to 2640m patch boundaries in UTM coordinates.

    The zarr file contains a 2D (time, y, x) multi-variable Dataset at 500m
    resolution — the Z dimension has been pre-collapsed for storage efficiency:

    Variables
    ---------
    reflectivity                : Z  (dBZ)  — column maximum reflectivity (CMAX)
    differential_reflectivity   : ZDR (dB)  — drop oblateness at level of max-Z
    cross_correlation_ratio     : RhoHV     — echo quality at level of max-Z
    differential_phase          : PhiDP (°) — cumulative phase at level of max-Z
    specific_differential_phase : KDP (°/km)— rain rate proxy at level of max-Z
                                  (NaN where Z < 20 dBZ)

    All dual-pol fields are sampled at the height of maximum reflectivity
    (collocated), which is physically correct and ~40× smaller than storing
    all 40 Z levels.

    Parameters
    ----------
    zarr_path : str
        Path to the radar zarr file.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with shape (time, y, x) for all dual-pol variables.
    """
    import xarray as xr

    # Load the radar dataset
    ds = xr.open_zarr(zarr_path)

    # Read shared metadata from reflectivity (always present)
    attrs = ds['reflectivity'].attrs
    print(f"Radar data loaded successfully!")
    print(f"  Coordinate system : {attrs.get('crs', 'N/A')}")
    print(f"  Radar station     : {attrs.get('radar_station')} "
          f"at ({attrs.get('radar_lat')}, {attrs.get('radar_lon')})")
    print(f"  UTM bounds        : {attrs.get('utm_bounds')}")
    print(f"  Resolution        : {attrs.get('resolution_m')}m")
    print(f"  Patch size        : {attrs.get('patch_size_m')}m  "
          f"→ {int(attrs.get('patch_size_m', 0) / attrs.get('resolution_m', 1))}×"
          f"{int(attrs.get('patch_size_m', 0) / attrs.get('resolution_m', 1))} pixels/patch")
    print(f"  Variables         : {list(ds.data_vars)}")
    print(f"  Time range        : {ds.time.min().values} to {ds.time.max().values}")
    print(f"  Number of scans   : {len(ds.time)}")
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