# Requirements:
# pip install s3fs pyart xarray numpy tqdm pyproj shapely

import numpy as np
import datetime as dt
import s3fs
import pyart
import xarray as xr
from pyproj import Transformer, CRS
from tqdm import tqdm
import io
import re
import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# MODULE-LEVEL WORKER FUNCTION
# Must be at module level (not nested) so Python's multiprocessing can pickle it.
# Each worker process runs this in its own memory space → no GIL contention.
# ==============================================================================

def _process_one_file(args):
    """
    Download, grid, and Z-collapse a single NEXRAD file.
    Runs in a worker process — completely independent of all other workers.

    Returns (field_arrays, scan_time) on success, or None on failure.
    """
    (s3_path, grid_shape, grid_limits, radar_lat, radar_lon,
     fields, apply_qc, qc_fields) = args

    # Each worker creates its own S3 filesystem — not shared across processes
    import s3fs as _s3fs
    import pyart as _pyart
    import numpy as _np
    import re as _re
    from datetime import datetime as _datetime, timedelta as _timedelta

    fs = _s3fs.S3FileSystem(anon=True)

    def _apply_qc(radar, fields_to_qc):
        for name in ['cross_correlation_ratio', 'RHOHV', 'correlation_coefficient']:
            if name in radar.fields:
                rhohv = radar.fields[name]['data']
                qc_mask = rhohv < 0.9
                for f in fields_to_qc:
                    if f in radar.fields:
                        radar.fields[f]['data'] = _np.ma.masked_where(
                            qc_mask, radar.fields[f]['data'])
                return radar, True
        return radar, False

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with fs.open(s3_path.replace("s3://", ""), 'rb') as f:
                radar = _pyart.io.read_nexrad_archive(f)

            if apply_qc:
                radar, _ = _apply_qc(radar, qc_fields)
            
            MAX_RANGE_M = 60_000  # 60 km covers preserve + buffer
            gate_spacing = radar.range['meters_between_gates']
            gates_needed = int(MAX_RANGE_M / gate_spacing)

            if radar.ngates > gates_needed:
                for field_name in list(radar.fields.keys()):
                    radar.fields[field_name]['data'] = \
                        radar.fields[field_name]['data'][:, :gates_needed]
                radar.ngates = gates_needed
                radar.range['data'] = radar.range['data'][:gates_needed]

            # Compute KDP from PhiDP
            if 'differential_phase' in radar.fields:
                try:
                    kdp, phidp_cor, *_ = _pyart.retrieve.kdp_maesaka(radar)
                    radar.add_field('specific_differential_phase', kdp, replace_existing=True)
                except Exception:
                    # Fill with NaN so all fields stay in lockstep in zarr
                    nan_arr = np.full_like(
                        radar.fields['differential_phase']['data'], np.nan, dtype=float
                    )
                    radar.add_field(
                        'specific_differential_phase',
                        {'data': nan_arr, 'units': 'deg/km', 'long_name': 'specific_differential_phase'},
                        replace_existing=True,
                    )

            fields_to_grid = [f for f in fields if f in radar.fields]

            grid = _pyart.map.grid_from_radars(
                [radar],
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                grid_origin=(radar_lat, radar_lon),
                fields=fields_to_grid,
                form='linear'
            )

            # Extract 3D arrays
            field_arrays_3d = {}
            ref_data = None
            for field_name in fields_to_grid:
                raw = grid.fields[field_name]['data']
                arr = raw.filled(_np.nan) if hasattr(raw, 'filled') else _np.array(raw, dtype=float)
                if field_name == 'specific_differential_phase' and ref_data is not None:
                    arr = _np.where(ref_data >= 20.0, arr, _np.nan)
                field_arrays_3d[field_name] = arr
                if field_name == 'reflectivity':
                    ref_data = arr

            # Collapse Z → 2D + derive vertical structure features
            field_arrays = {}
            if 'reflectivity' in field_arrays_3d:
                ref_3d = field_arrays_3d['reflectivity']  # (nz, ny, nx)
                nz_grid = ref_3d.shape[0]

                # Vertical level heights (metres above radar)
                z_min, z_max = grid_limits[0]
                z_res = (z_max - z_min) / nz_grid
                z_heights = _np.linspace(z_min + z_res / 2, z_max - z_res / 2, nz_grid)

                # --- Standard 2D composites (existing behaviour) ---
                field_arrays['reflectivity'] = _np.nanmax(ref_3d, axis=0)
                ref_safe = _np.where(_np.isfinite(ref_3d), ref_3d, -_np.inf)
                z_idx = _np.argmax(ref_safe, axis=0)

                # --- Derived vertical features ---

                # Echo top height: highest level with Z >= 18 dBZ (metres)
                echo_mask = ref_3d >= 18.0
                has_echo = _np.any(echo_mask, axis=0)
                echo_top_idx = nz_grid - 1 - _np.argmax(echo_mask[::-1, :, :], axis=0)
                echo_top = z_heights[echo_top_idx]
                field_arrays['echo_top_height'] = _np.where(has_echo, echo_top, 0.0)

                # Max reflectivity height (metres)
                max_z_height = z_heights[z_idx]
                max_z_height = _np.where(
                    _np.isfinite(field_arrays['reflectivity']), max_z_height, 0.0
                )
                field_arrays['max_z_height'] = max_z_height

                # VIL — Vertically Integrated Liquid (kg/m²)
                ref_linear = 10.0 ** (ref_3d / 10.0)
                ref_linear = _np.where(_np.isfinite(ref_linear), ref_linear, 0.0)
                field_arrays['vil'] = 3.44e-6 * _np.nansum(
                    ref_linear ** (4.0 / 7.0) * z_res, axis=0
                )

                # Low-level mean reflectivity (0–2 km)
                low_mask = z_heights <= 2000.0
                if low_mask.any():
                    field_arrays['low_level_ref'] = _np.nanmean(ref_3d[low_mask, :, :], axis=0)
                else:
                    field_arrays['low_level_ref'] = field_arrays['reflectivity'].copy()

                # Column depth fraction: fraction of levels with Z > 10 dBZ
                precip_levels = _np.sum(ref_3d > 10.0, axis=0).astype(_np.float32)
                field_arrays['column_depth_fraction'] = precip_levels / nz_grid

                # --- Collapse dual-pol fields at height of max Z ---
                ny_g, nx_g = ref_3d.shape[1], ref_3d.shape[2]
                yy, xx = _np.meshgrid(_np.arange(ny_g), _np.arange(nx_g), indexing='ij')
                for field_name, arr_3d in field_arrays_3d.items():
                    if field_name != 'reflectivity':
                        field_arrays[field_name] = arr_3d[z_idx, yy, xx]
            else:
                for field_name, arr_3d in field_arrays_3d.items():
                    field_arrays[field_name] = _np.nanmax(arr_3d, axis=0)

            # Extract timestamp
            try:
                time_val = float(grid.time['data'][0])
                units = grid.time['units']
                base_time_str = _re.search(r"since\s+([0-9T:\-\.Z]+)", units).group(1)
                base_time = _datetime.fromisoformat(base_time_str.replace("Z", "+00:00"))
                scan_time = base_time + _timedelta(seconds=time_val)
            except Exception:
                try:
                    scan_time = _datetime.utcfromtimestamp(radar.time['data'][0])
                except Exception:
                    filename = s3_path.split('/')[-1]
                    m = _re.search(r'(\d{8})_(\d{6})', filename)
                    if m:
                        scan_time = _datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
                    else:
                        raise ValueError(f"Cannot extract time from {s3_path}")

            return field_arrays, scan_time

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                filename = s3_path.split('/')[-1]
                print(f"  ✗ Failed {filename}: {type(e).__name__} - {str(e)[:120]}")
                return None


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def pull_nexrad_multi(day_filter=None, day_filter_file=None, apply_qc=True, start_date=None, end_date=None):
    """
    Pull NEXRAD radar data using multiprocessing for true CPU parallelism.

    Key difference from pull_nexrad.py (ThreadPoolExecutor):
    - Workers run in SEPARATE PROCESSES → no GIL → true parallel gridding/KDP
    - Zarr writes happen in the MAIN PROCESS after results arrive → no locking needed
    - Each worker creates its own s3fs / pyart resources → no sharing required

    Parameters mirror pull_nexrad() exactly so main.py can swap them.
    """

    # ── USER PARAMETERS ────────────────────────────────────────────────────────
    STATION          = "KVBX"
    RADAR_LAT        = 34.83855
    RADAR_LON        = -120.397917
    RESOLUTION       = 500.0
    Z_MIN, Z_MAX, Z_RES = 0.0, 8000.0, 375.0
    PRESERVE_BBOX    = (-120.5130681, 34.5775648, -120.3456914, 34.4344052)
    PATCH_SIZE_M     = 2640
    UTM_CRS          = "EPSG:32610"
    BUFFER_M         = 5000

    START_DATE       = start_date
    END_DATE         = end_date

    # With multiprocessing, each worker uses a full CPU core.
    # Set this to your physical core count (NOT logical/hyperthreaded).
    # Check Task Manager → Performance → CPU → Cores.
    MAX_WORKERS      = 6

    FIELDS = [
        'reflectivity',
        'differential_reflectivity',
        'cross_correlation_ratio',
        'differential_phase',
        'specific_differential_phase',
    ]
    APPLY_QC   = apply_qc
    QC_FIELDS  = ['reflectivity', 'differential_reflectivity']
    # ───────────────────────────────────────────────────────────────────────────

    # ── COORDINATE SETUP ───────────────────────────────────────────────────────
    wgs84   = CRS.from_epsg(4326)
    utm_crs = CRS.from_string(UTM_CRS)
    to_utm  = Transformer.from_crs(wgs84, utm_crs, always_xy=True)

    west_utm,  north_utm = to_utm.transform(PRESERVE_BBOX[0], PRESERVE_BBOX[1])
    east_utm,  south_utm = to_utm.transform(PRESERVE_BBOX[2], PRESERVE_BBOX[3])

    xmin = np.floor((west_utm  - BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M
    xmax = np.ceil( (east_utm  + BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M
    ymin = np.floor((south_utm - BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M
    ymax = np.ceil( (north_utm + BUFFER_M) / PATCH_SIZE_M) * PATCH_SIZE_M

    transformer = Transformer.from_crs(
        utm_crs,
        f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84",
        always_xy=True,
    )
    xmin_m, ymin_m = transformer.transform(xmin, ymin)
    xmax_m, ymax_m = transformer.transform(xmax, ymax)

    nx = int(np.ceil((xmax_m - xmin_m) / RESOLUTION))
    ny = int(np.ceil((ymax_m - ymin_m) / RESOLUTION))
    nz = int(np.ceil((Z_MAX   - Z_MIN)  / Z_RES))

    grid_limits = ((Z_MIN, Z_MAX), (ymin_m, ymax_m), (xmin_m, xmax_m))
    grid_shape  = (nz, ny, nx)

    print(f"Grid shape (z, y, x): {grid_shape}")
    print(f"MAX_WORKERS (processes): {MAX_WORKERS}")

    # ── S3 FILE LISTING (done in main process — lightweight I/O) ──────────────
    bucket = "unidata-nexrad-level2"
    fs     = s3fs.S3FileSystem(anon=True)

    def list_daily_files(station, date, max_retries=3):
        prefix = f"{date.year}/{date:%m}/{date:%d}/{station}/"
        for attempt in range(max_retries):
            try:
                keys  = fs.ls(f"s3://{bucket}/{prefix}")
                keys  = [k for k in keys if not k.endswith("_MDM")]
                files = [f"{bucket}/{k.split('/',1)[1]}" if k.startswith(bucket + "/") else k for k in keys]
                return [f"s3://{p}" if not p.startswith("s3://") else p for p in files]
            except FileNotFoundError:
                return []
            except Exception as e:
                if "404" in str(e) or "NoSuchKey" in str(e):
                    return []
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    return []
        return []

    # ── DAY FILTER ─────────────────────────────────────────────────────────────
    if day_filter_file:
        with open(day_filter_file) as f:
            dates = [datetime.strptime(l.strip(), '%Y-%m-%d').date()
                     for l in f if l.strip() and not l.startswith('#')]
        print(f"Loaded {len(dates)} days from {day_filter_file}")
    elif day_filter:
        dates = day_filter
    else:
        dates = [START_DATE + dt.timedelta(days=i)
                 for i in range((END_DATE - START_DATE).days + 1)]

    # ── COLLECT FILE LIST ──────────────────────────────────────────────────────
    all_files = []
    print(f"\nCollecting file list for {len(dates)} days...")
    for d in tqdm(dates, desc="Scanning dates"):
        all_files.extend(list_daily_files(STATION, d))

    # ── RAINY-HOUR FILTER ──────────────────────────────────────────────────────
    # Query gauge DB for hours where any station recorded measurable rain.
    # Scans outside those windows carry no training signal and can be skipped,
    # dramatically reducing the number of files to grid.
    #
    # Buffer: we keep scans from 1 hour BEFORE a rainy hour so the temporal
    # context window (12 scans / hour) has valid data at the start of rain events.
    print("\nQuerying gauge DB for rainy hours to pre-filter scan list...")
    try:
        from weather.pull_weather import get_rainy_hours_set
        rainy_hours = get_rainy_hours_set(dates, min_rainfall_mm=0.01)

        if rainy_hours:
            def _file_in_rainy_window(s3_path, rainy_hours, pre_buffer_hours=1):
                """Return True if the file's scan time falls within or just before a rainy hour."""
                m = re.search(r'(\d{8})_(\d{6})', s3_path)
                if not m:
                    return True  # can't parse — keep it to be safe
                scan_dt   = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
                scan_hour = scan_dt.replace(minute=0, second=0, microsecond=0)
                # Keep if this hour OR any of the next `pre_buffer_hours` hours is rainy
                # (i.e. this scan comes just before rain starts)
                for offset in range(pre_buffer_hours + 1):
                    if (scan_hour + timedelta(hours=offset)) in rainy_hours:
                        return True
                return False

            n_before = len(all_files)
            all_files = [f for f in all_files if _file_in_rainy_window(f, rainy_hours)]
            n_after   = len(all_files)
            pct_kept  = 100 * n_after / max(n_before, 1)
            print(f"  Rainy-hour filter: {n_before} → {n_after} files "
                  f"({pct_kept:.0f}% kept, {n_before - n_after} skipped)")
        else:
            print("  No rainy-hour data returned — processing all files.")
    except Exception as filter_err:
        print(f"  ⚠ Rainy-hour filter failed ({filter_err}) — processing all files.")

    # ── RESUMABLE LOG ──────────────────────────────────────────────────────────
    zarr_path     = f"./radar/outputs/dualpol_{int(RESOLUTION)}m_{START_DATE}_{END_DATE}.zarr"
    processed_log = f"./radar/logs/processed_files_{int(RESOLUTION)}m_{START_DATE}_{END_DATE}.txt"

    processed_files = set()
    if os.path.exists(processed_log):
        with open(processed_log) as f:
            processed_files = {l.strip() for l in f}
        print(f"Resuming — {len(processed_files)} files already done")

    files_to_process = [f for f in all_files if f not in processed_files]
    print(f"Files to process: {len(files_to_process)}  |  Already done: {len(all_files) - len(files_to_process)}")

    if not files_to_process:
        print("All files already processed!")
        return

    # ── SHARED METADATA ────────────────────────────────────────────────────────
    shared_attrs = {
        'crs'           : UTM_CRS,
        'crs_wkt'       : utm_crs.to_wkt(),
        'radar_lat'     : RADAR_LAT,
        'radar_lon'     : RADAR_LON,
        'radar_station' : STATION,
        'utm_bounds'    : f'({xmin}, {ymin}, {xmax}, {ymax})',
        'patch_size_m'  : PATCH_SIZE_M,
        'resolution_m'  : RESOLUTION,
        'description'   : (
            'NEXRAD dual-pol radar at 500m resolution, Z-collapsed to 2D (y, x). '
            'reflectivity = column maximum (CMAX) over 0-8000m (22 levels at 375m spacing). '
            'ZDR, RhoHV, PhiDP, KDP = sampled at height of max reflectivity (collocated). '
            'KDP masked where Z < 20 dBZ. Aligned to 2640m patch boundaries in UTM.'
        ),
    }

    def make_dataset(field_arrays, time_np, include_spatial=True):
        coords = {'time': [time_np]}
        if include_spatial:
            coords['y'] = np.linspace(ymin + RESOLUTION/2, ymax - RESOLUTION/2, ny)
            coords['x'] = np.linspace(xmin + RESOLUTION/2, xmax - RESOLUTION/2, nx)
        data_vars = {
            name: xr.DataArray(arr[None, ...], dims=('time', 'y', 'x'), attrs=shared_attrs)
            for name, arr in field_arrays.items()
        }
        return xr.Dataset(data_vars, coords=coords)

    # ── WORKER ARGS FACTORY ────────────────────────────────────────────────────
    def make_args(s3_path):
        return (s3_path, grid_shape, grid_limits, RADAR_LAT, RADAR_LON,
                FIELDS, APPLY_QC, QC_FIELDS)

    # ── MULTIPROCESSING MAIN LOOP ──────────────────────────────────────────────
    # Workers run grid_from_radars in parallel (no GIL).
    # The main process receives results and writes to zarr sequentially
    # (no lock needed — one writer, naturally serialized).
    successful = 0
    failed     = 0
    first_write = not os.path.exists(zarr_path)

    print(f"\nStarting multiprocessing with {MAX_WORKERS} worker processes...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(_process_one_file, make_args(p)): p
            for p in files_to_process
        }

        for future in tqdm(as_completed(future_to_path),
                           total=len(files_to_process),
                           desc="Processing files"):
            s3_path = future_to_path[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  ✗ Unexpected error for {s3_path.split('/')[-1]}: {e}")
                result = None

            if result is None:
                failed += 1
                continue

            field_arrays, scan_time = result
            time_np = np.datetime64(scan_time, 'ns')
            print(f"Writing scan at: {scan_time}")

            # ── ZARR WRITE (main process only, no lock needed) ────────────────
            try:
                if first_write:
                    ds = make_dataset(field_arrays, time_np, include_spatial=True)
                    ds.to_zarr(zarr_path, mode='w',
                               encoding={'time': {'units': 'nanoseconds since 1970-01-01',
                                                  'calendar': 'proleptic_gregorian'}})
                    first_write = False
                else:
                    ds = make_dataset(field_arrays, time_np, include_spatial=False)
                    for write_attempt in range(3):
                        try:
                            ds.to_zarr(zarr_path, append_dim='time')
                            break
                        except PermissionError:
                            time.sleep(0.2 * (write_attempt + 1))

                # Log success
                with open(processed_log, 'a') as log:
                    log.write(s3_path + "\n")
                successful += 1

            except Exception as e:
                print(f"  ✗ Zarr write failed for {s3_path.split('/')[-1]}: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Done! Successful: {successful}  |  Failed: {failed}")
    print(f"Zarr: {zarr_path}")
    print(f"To sort by time: xr.open_zarr('{zarr_path}').sortby('time')")
    print(f"{'='*60}")

    return zarr_path


# ── ENTRYPOINT ─────────────────────────────────────────────────────────────────
# The `if __name__ == '__main__':` guard is REQUIRED on Windows.
# Without it, each spawned worker process would re-run the whole script,
# causing an infinite fork-bomb of new processes.
if __name__ == '__main__':
    pull_nexrad_multi()