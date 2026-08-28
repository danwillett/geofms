"""
check_KVTX.py — Multi-radar beam quality comparison: KVBX vs KVTX

Pulls one hour of scans from both KVBX and KVTX during the top rainy day
(2025-02-13), grids each scan to the same UTM domain as the preserve, then
for every 5-minute slot compares:
  - Beam height AGL (proxy for blockage severity)
  - Lowest-gate reflectivity (what the radar can actually see)
  - Coverage (fraction of preserve pixels with valid echo)
  - Quality index (QI = 1 - beam_height_km/4, capped 0-1; simple but standard)

Output:
  - Console table per 5-min slot
  - radar/outputs/kvtx_check/kvbx_vs_kvtx_<date>_<hour>UTC.png
    * Left panel : KVBX beam height map
    * Middle panel: KVTX beam height map
    * Right panel : pixel-wise "best radar" map (lower beam height wins)
  - radar/outputs/kvtx_check/slot_comparison.csv  (all per-slot metrics)

Run from project root:
    python radar/check_KVTX.py
    python radar/check_KVTX.py --date 2025-02-13 --hour 14 --minute 0 --n-slots 12
"""

import argparse
import io
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import s3fs
import pyart
import xarray as xr
from pyproj import Transformer, CRS
from tqdm import tqdm

# ── Domain (same as KVBX Zarr) ────────────────────────────────────────────────
PRESERVE_BBOX   = (-120.5130681, 34.5775648, -120.3456914, 34.4344052)  # W,N,E,S lon/lat
UTM_CRS         = "EPSG:32610"
RESOLUTION_M    = 500.0
BUFFER_M        = 2000.0
Z_MIN, Z_MAX    = 0.0, 8000.0
Z_RES           = 375.0   # 22 levels, same as KVBX Zarr

# ── Radar sites ───────────────────────────────────────────────────────────────
RADARS = {
    'KVBX': {'lat': 34.83855,  'lon': -120.39792},
    'KVTX': {'lat': 34.41175,  'lon': -119.17958},  # Ventura/Oxnard
}

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATE   = '2025-02-13'
DEFAULT_HOUR   = 14           # UTC hour to examine (pick a wet hour on the top day)
DEFAULT_MINUTE = 0
N_SLOTS        = 12           # 5-min slots = 1 hour
SLOT_MIN       = 5
QI_BH_SCALE_KM = 4.0         # beam height at which QI drops to 0 (4 km)

S3_BUCKET = "unidata-nexrad-level2"

# ── Output dir ────────────────────────────────────────────────────────────────
OUT_DIR = Path("radar/outputs/kvtx_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Grid geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_utm_grid(radar_lat, radar_lon):
    """Compute UTM bounding box and PyART grid params for the preserve domain."""
    wgs84  = CRS.from_epsg(4326)
    utm    = CRS.from_string(UTM_CRS)
    to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)

    west_utm,  north_utm = to_utm.transform(PRESERVE_BBOX[0], PRESERVE_BBOX[1])
    east_utm,  south_utm = to_utm.transform(PRESERVE_BBOX[2], PRESERVE_BBOX[3])

    xmin = np.floor((west_utm  - BUFFER_M) / RESOLUTION_M) * RESOLUTION_M
    xmax = np.ceil( (east_utm  + BUFFER_M) / RESOLUTION_M) * RESOLUTION_M
    ymin = np.floor((south_utm - BUFFER_M) / RESOLUTION_M) * RESOLUTION_M
    ymax = np.ceil( (north_utm + BUFFER_M) / RESOLUTION_M) * RESOLUTION_M

    nx = int(round((xmax - xmin) / RESOLUTION_M))
    ny = int(round((ymax - ymin) / RESOLUTION_M))
    nz = int(round((Z_MAX - Z_MIN) / Z_RES))

    # Convert UTM corners to radar-local azimuthal-equidistant (for PyART)
    to_aeqd = Transformer.from_crs(
        utm,
        f"+proj=aeqd +lat_0={radar_lat} +lon_0={radar_lon} +units=m +datum=WGS84",
        always_xy=True,
    )
    xmin_m, ymin_m = to_aeqd.transform(xmin, ymin)
    xmax_m, ymax_m = to_aeqd.transform(xmax, ymax)

    grid_limits = ((Z_MIN, Z_MAX), (ymin_m, ymax_m), (xmin_m, xmax_m))
    grid_shape  = (nz, ny, nx)

    x_centers = np.linspace(xmin + RESOLUTION_M/2, xmax - RESOLUTION_M/2, nx)
    y_centers = np.linspace(ymin + RESOLUTION_M/2, ymax - RESOLUTION_M/2, ny)

    return grid_shape, grid_limits, x_centers, y_centers, (xmin, ymin, xmax, ymax)


def beam_height_agl(range_m, elevation_deg, radar_alt_m=0.0):
    """
    Effective beam centre height AGL using 4/3 Earth-radius model (standard refraction).
    Returns height in metres.
    """
    Re = 6371000.0 * 4.0 / 3.0
    el = np.deg2rad(elevation_deg)
    h  = np.sqrt(range_m**2 + Re**2 + 2*range_m*Re*np.sin(el)) - Re + radar_alt_m
    return h


# ══════════════════════════════════════════════════════════════════════════════
# S3 helpers
# ══════════════════════════════════════════════════════════════════════════════

def list_s3_files(fs, station, date):
    prefix = f"{date.year}/{date:%m}/{date:%d}/{station}/"
    try:
        keys = fs.ls(f"s3://{S3_BUCKET}/{prefix}")
        keys = [k for k in keys if not k.endswith("_MDM")]
        return [f"s3://{S3_BUCKET}/{k.split('/', 1)[1]}" if not k.startswith(S3_BUCKET + "/") else f"s3://{k}" for k in keys]
    except Exception:
        return []


def parse_scan_time(s3_path):
    """Extract UTC datetime from NEXRAD filename."""
    fname = s3_path.split('/')[-1]
    m = re.search(r'(\d{8})_(\d{6})', fname)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return None


def assign_slot(scan_dt, hour_start_dt, slot_minutes=SLOT_MIN, n_slots=N_SLOTS):
    """Return 0-based slot index, or -1 if outside the window."""
    offset = (scan_dt - hour_start_dt).total_seconds() / 60.0
    if offset < 0 or offset >= slot_minutes * n_slots:
        return -1
    return int(offset // slot_minutes)


# ══════════════════════════════════════════════════════════════════════════════
# Per-scan processing
# ══════════════════════════════════════════════════════════════════════════════

def process_scan(fs, s3_path, station_meta, grid_shape, grid_limits):
    """
    Download, grid, and compute quality metrics for one radar scan.

    Returns dict with:
        beam_height_2d  : (ny, nx) metres AGL at lowest valid gate
        ref_cmax_2d     : (ny, nx) column-max reflectivity dBZ
        coverage_frac   : fraction of pixels with valid Z
        mean_qi         : mean quality index over valid pixels
        scan_time       : datetime UTC
    """
    try:
        with fs.open(s3_path.replace("s3://", ""), 'rb') as f:
            radar = pyart.io.read_nexrad_archive(f)
    except Exception as e:
        print(f"  [WARN] Failed to read {s3_path.split('/')[-1]}: {e}")
        return None

    # QC: mask RhoHV < 0.85
    if 'cross_correlation_ratio' in radar.fields:
        rhohv = radar.fields['cross_correlation_ratio']['data']
        mask  = rhohv < 0.85
        if 'reflectivity' in radar.fields:
            radar.fields['reflectivity']['data'] = np.ma.masked_where(
                mask, radar.fields['reflectivity']['data'])

    try:
        grid = pyart.map.grid_from_radars(
            [radar],
            grid_shape=grid_shape,
            grid_limits=grid_limits,
            grid_origin=(station_meta['lat'], station_meta['lon']),
            fields=['reflectivity'],
            form='linear',
        )
    except Exception as e:
        print(f"  [WARN] Gridding failed for {s3_path.split('/')[-1]}: {e}")
        return None

    nz, ny, nx = grid_shape
    z_heights   = np.linspace(Z_MIN + Z_RES/2, Z_MAX - Z_RES/2, nz)  # centre of each level

    ref_3d = grid.fields['reflectivity']['data']
    if hasattr(ref_3d, 'filled'):
        ref_3d = ref_3d.filled(np.nan)
    ref_3d = np.array(ref_3d, dtype=np.float32)   # (nz, ny, nx)

    # Column-max reflectivity
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ref_cmax = np.nanmax(ref_3d, axis=0)       # (ny, nx)

    # Beam height at lowest valid gate (index of first finite Z from bottom)
    has_echo = np.isfinite(ref_3d)                 # (nz, ny, nx)
    has_any  = has_echo.any(axis=0)                # (ny, nx)

    # argmax along z of has_echo gives the index of the lowest valid gate
    first_valid_z = np.argmax(has_echo, axis=0)    # 0 where no echo (handled by has_any)
    bh_2d = np.where(has_any, z_heights[first_valid_z], np.nan)   # metres AGL

    # Quality index: 1 at ground, 0 at QI_BH_SCALE_KM km
    qi_2d = np.clip(1.0 - bh_2d / (QI_BH_SCALE_KM * 1000.0), 0.0, 1.0)

    valid_pixels  = np.sum(has_any)
    total_pixels  = ny * nx
    coverage_frac = float(valid_pixels) / total_pixels
    mean_qi       = float(np.nanmean(qi_2d)) if valid_pixels > 0 else 0.0

    # Scan time from file
    scan_time = parse_scan_time(s3_path)
    if scan_time is None:
        try:
            raw_t  = float(grid.time['data'][0])
            units  = grid.time['units']
            base   = re.search(r"since\s+([0-9T:\-\.Z]+)", units).group(1)
            base_dt = datetime.fromisoformat(base.replace("Z", "+00:00"))
            scan_time = base_dt + timedelta(seconds=raw_t)
        except Exception:
            scan_time = None

    return {
        'beam_height_2d': bh_2d,
        'ref_cmax_2d':    ref_cmax,
        'coverage_frac':  coverage_frac,
        'mean_qi':        mean_qi,
        'scan_time':      scan_time,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(date_str, hour_utc, minute_start, n_slots, out_dir):
    date       = datetime.strptime(date_str, '%Y-%m-%d').date()
    hour_start = datetime(date.year, date.month, date.day,
                          hour_utc, minute_start, 0, tzinfo=timezone.utc)
    hour_end   = hour_start + timedelta(minutes=SLOT_MIN * n_slots)

    print(f"\n{'='*70}")
    print(f"  KVBX vs KVTX — {date_str}  {hour_utc:02d}:{minute_start:02d}–"
          f"{(hour_start + timedelta(minutes=SLOT_MIN*n_slots)).strftime('%H:%M')} UTC")
    print(f"{'='*70}")

    fs = s3fs.S3FileSystem(anon=True)

    # Build grid geometry for each radar (different origin → different aeqd coords)
    grids = {}
    for stn, meta in RADARS.items():
        gs, gl, xc, yc, bbox = build_utm_grid(meta['lat'], meta['lon'])
        grids[stn] = dict(shape=gs, limits=gl, xc=xc, yc=yc, bbox=bbox)
        print(f"  {stn} grid: {gs[2]}×{gs[1]} px  "
              f"({gs[2]*RESOLUTION_M/1000:.1f}×{gs[1]*RESOLUTION_M/1000:.1f} km)")

    # List and filter S3 files to the target hour window
    slot_scans = {stn: defaultdict(list) for stn in RADARS}
    all_files  = {}
    for stn in RADARS:
        files = list_s3_files(fs, stn, date)
        # Filter to the window + 2-min pad on each end for temporal matching
        window_files = []
        for fp in files:
            t = parse_scan_time(fp)
            if t and (hour_start - timedelta(minutes=2)) <= t <= (hour_end + timedelta(minutes=2)):
                slot = assign_slot(t, hour_start)
                if slot >= 0:
                    slot_scans[stn][slot].append((t, fp))
        all_files[stn] = files
        print(f"  {stn}: {len(files)} total files on {date_str}, "
              f"{sum(len(v) for v in slot_scans[stn].values())} in target window")

    # For each station+slot, pick the scan nearest the slot centre
    best_scans = {stn: {} for stn in RADARS}   # slot → (time, s3_path)
    for stn in RADARS:
        for slot, candidates in slot_scans[stn].items():
            slot_centre = hour_start + timedelta(minutes=slot*SLOT_MIN + SLOT_MIN/2)
            best = min(candidates, key=lambda x: abs((x[0] - slot_centre).total_seconds()))
            best_scans[stn][slot] = best

    # Process scans
    results = {stn: {} for stn in RADARS}   # slot → metrics dict
    print(f"\nProcessing scans...")
    for stn, meta in RADARS.items():
        g = grids[stn]
        slots_with_data = sorted(best_scans[stn].keys())
        print(f"\n  {stn} ({len(slots_with_data)} slots):")
        for slot in slots_with_data:
            t, fp = best_scans[stn][slot]
            print(f"    slot {slot:2d} ({t.strftime('%H:%M')}) — {fp.split('/')[-1]}")
            metrics = process_scan(fs, fp, meta, g['shape'], g['limits'])
            if metrics is not None:
                results[stn][slot] = metrics

    # ── Per-slot comparison table ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'Slot':>4}  {'Time(UTC)':>9}  "
          f"{'KVBX BH(m)':>10}  {'KVBX cov':>8}  {'KVBX QI':>7}  "
          f"{'KVTX BH(m)':>10}  {'KVTX cov':>8}  {'KVTX QI':>7}  {'Best':>5}")
    print(f"{'─'*70}")

    rows = []
    all_slots = sorted(set(list(results['KVBX'].keys()) + list(results['KVTX'].keys())))
    for slot in all_slots:
        slot_time = hour_start + timedelta(minutes=slot * SLOT_MIN)
        r_bx = results['KVBX'].get(slot)
        r_tx = results['KVTX'].get(slot)
        bh_bx  = np.nanmean(r_bx['beam_height_2d']) if r_bx else np.nan
        bh_tx  = np.nanmean(r_tx['beam_height_2d']) if r_tx else np.nan
        cov_bx = r_bx['coverage_frac'] if r_bx else np.nan
        cov_tx = r_tx['coverage_frac'] if r_tx else np.nan
        qi_bx  = r_bx['mean_qi'] if r_bx else np.nan
        qi_tx  = r_tx['mean_qi'] if r_tx else np.nan

        if not np.isnan(bh_bx) and not np.isnan(bh_tx):
            best = 'KVBX' if qi_bx >= qi_tx else 'KVTX'
        elif not np.isnan(bh_bx):
            best = 'KVBX'
        elif not np.isnan(bh_tx):
            best = 'KVTX'
        else:
            best = 'none'

        print(f"  {slot:4d}  {slot_time.strftime('%H:%M UTC'):>9}  "
              f"{bh_bx:10.0f}  {cov_bx:8.3f}  {qi_bx:7.3f}  "
              f"{bh_tx:10.0f}  {cov_tx:8.3f}  {qi_tx:7.3f}  {best:>5}")
        rows.append(dict(
            slot=slot, time=slot_time.strftime('%H:%M UTC'),
            kvbx_bh_m=bh_bx, kvbx_coverage=cov_bx, kvbx_qi=qi_bx,
            kvtx_bh_m=bh_tx, kvtx_coverage=cov_tx, kvtx_qi=qi_tx,
            best_radar=best,
        ))
    print(f"{'─'*70}")

    csv_path = out_dir / f"slot_comparison_{date_str}_{hour_utc:02d}h.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved slot metrics to: {csv_path}")

    # ── Spatial maps for a representative slot ────────────────────────────────
    # Pick the slot where both radars have data and KVTX QI is highest relative to KVBX
    # (the slot most likely to show complementarity)
    best_slot = None
    best_complement = -np.inf
    for r in rows:
        if r['best_radar'] == 'KVTX' and not np.isnan(r['kvtx_qi']):
            gap = r['kvtx_qi'] - r['kvbx_qi']
            if gap > best_complement:
                best_complement = gap
                best_slot = r['slot']
    if best_slot is None and rows:
        # Fallback: just pick the first slot where both have data
        for r in rows:
            if results['KVBX'].get(r['slot']) and results['KVTX'].get(r['slot']):
                best_slot = r['slot']
                break

    if best_slot is not None:
        _make_spatial_plot(results, grids, best_slot, hour_start, date_str, hour_utc, out_dir)

    # ── Overall summary ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print(f"\n  Summary across {len(df)} slots:")
    for stn, col_bh, col_qi in [('KVBX', 'kvbx_bh_m', 'kvbx_qi'),
                                  ('KVTX', 'kvtx_bh_m', 'kvtx_qi')]:
        valid = df[col_bh].dropna()
        if len(valid):
            print(f"    {stn}  mean beam height: {valid.mean():.0f} m  "
                  f"mean QI: {df[col_qi].dropna().mean():.3f}  "
                  f"slots with data: {len(valid)}/{len(df)}")
    n_kvtx_wins = (df['best_radar'] == 'KVTX').sum()
    n_kvbx_wins = (df['best_radar'] == 'KVBX').sum()
    print(f"\n  Pixel-mean QI winner: KVBX={n_kvbx_wins}, KVTX={n_kvtx_wins} slots")
    print(f"  → {'KVTX provides complementary coverage' if n_kvtx_wins > 0 else 'KVBX dominates this domain'}")


def _make_spatial_plot(results, grids, slot, hour_start, date_str, hour_utc, out_dir):
    """Maps of beam height and best-radar for the chosen slot."""
    slot_time  = hour_start + timedelta(minutes=slot * SLOT_MIN)
    r_bx = results['KVBX'].get(slot)
    r_tx = results['KVTX'].get(slot)

    # Both grids should have the same UTM domain — use KVBX grid x/y for axes
    xc = grids['KVBX']['xc'] / 1000   # km for display
    yc = grids['KVBX']['yc'] / 1000

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"KVBX vs KVTX — {date_str}  slot {slot}  ({slot_time.strftime('%H:%M UTC')})\n"
        f"Preserve domain | beam height AGL (m)",
        fontsize=11
    )

    bh_norm = mcolors.Normalize(vmin=0, vmax=3000)
    cmap_bh  = plt.cm.RdYlGn_r

    for ax, stn, r in [(axes[0], 'KVBX', r_bx), (axes[1], 'KVTX', r_tx)]:
        if r is not None:
            im = ax.pcolormesh(xc, yc, r['beam_height_2d'],
                               cmap=cmap_bh, norm=bh_norm, shading='auto')
            plt.colorbar(im, ax=ax, label='Beam height AGL (m)')
            ax.set_title(f"{stn}  |  mean BH={np.nanmean(r['beam_height_2d']):.0f} m"
                         f"  QI={r['mean_qi']:.3f}  cov={r['coverage_frac']:.2f}")
        else:
            ax.text(0.5, 0.5, f'{stn}\nNo data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=13)
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
        ax.set_aspect('equal')

    # Best-radar map
    ax3 = axes[2]
    if r_bx is not None and r_tx is not None:
        qi_bx = np.clip(1.0 - r_bx['beam_height_2d'] / (QI_BH_SCALE_KM*1000), 0, 1)
        qi_tx = np.clip(1.0 - r_tx['beam_height_2d'] / (QI_BH_SCALE_KM*1000), 0, 1)

        # 0 = KVBX best, 1 = KVTX best, NaN = neither
        best_map = np.full_like(qi_bx, np.nan)
        both_valid = np.isfinite(qi_bx) & np.isfinite(qi_tx)
        only_bx    = np.isfinite(qi_bx) & ~np.isfinite(qi_tx)
        only_tx    = ~np.isfinite(qi_bx) & np.isfinite(qi_tx)
        best_map[both_valid] = np.where(qi_tx[both_valid] > qi_bx[both_valid], 1.0, 0.0)
        best_map[only_bx]    = 0.0
        best_map[only_tx]    = 1.0

        im3 = ax3.pcolormesh(xc, yc, best_map,
                             cmap=plt.cm.RdBu, vmin=0, vmax=1, shading='auto')
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_ticks([0, 1])
        cbar3.set_ticklabels(['KVBX better', 'KVTX better'])

        n_tx_better = int(np.nansum(best_map == 1.0))
        n_bx_better = int(np.nansum(best_map == 0.0))
        ax3.set_title(f"Best radar per pixel\nKVBX better: {n_bx_better} px | KVTX better: {n_tx_better} px")
    else:
        ax3.text(0.5, 0.5, 'Need both radars\nfor comparison',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Best radar per pixel')

    ax3.set_xlabel('Easting (km)')
    ax3.set_ylabel('Northing (km)')
    ax3.set_aspect('equal')

    plt.tight_layout()
    png_path = out_dir / f"kvbx_vs_kvtx_{date_str}_{hour_utc:02d}h_slot{slot:02d}.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved spatial map to: {png_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KVBX vs KVTX beam quality comparison')
    parser.add_argument('--date',    default=DEFAULT_DATE,
                        help='Date to analyse (YYYY-MM-DD, default: top rainy day 2025-02-13)')
    parser.add_argument('--hour',    type=int, default=DEFAULT_HOUR,
                        help='UTC hour to examine (default: 14)')
    parser.add_argument('--minute',  type=int, default=DEFAULT_MINUTE,
                        help='UTC minute offset within the hour (default: 0)')
    parser.add_argument('--n-slots', type=int, default=N_SLOTS,
                        help='Number of 5-min slots to cover (default: 12 = 1 hour)')
    args = parser.parse_args()

    run_comparison(
        date_str   = args.date,
        hour_utc   = args.hour,
        minute_start = args.minute,
        n_slots    = args.n_slots,
        out_dir    = OUT_DIR,
    )
