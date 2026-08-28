"""
volume_io.py — Shared helpers for 3D radar volume pickles and lazy zarr loading.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import zarr
from pyproj import CRS, Transformer


# Raw fields stored in dualpol_3d zarr (time, z, y, x).
RAW_FIELDS_3D = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
]


def open_3d_zarr_store(zarr_path: str):
    """Open 3D zarr read-only; return store and coordinate arrays."""
    store = zarr.open(zarr_path, mode='r')
    present = [f for f in RAW_FIELDS_3D if f in store]
    if not present:
        raise ValueError(f"No raw 3D fields in {zarr_path}. Keys: {list(store.keys())}")

    min_time = min(store[f].shape[0] for f in present)
    time_vals = store['time'][:min_time].astype('datetime64[ns]')
    x_vals = np.asarray(store['x'][:], dtype=float)
    y_vals = np.asarray(store['y'][:], dtype=float)
    z_vals = np.asarray(store['z'][:], dtype=float)
    resolution_m = float(store.attrs.get('resolution_m', 500))
    crs = store.attrs.get('crs', 'EPSG:32610')
    return store, present, min_time, time_vals, x_vals, y_vals, z_vals, resolution_m, crs


def resolve_z_indices(z_vals: np.ndarray, z_max_m: float, z_stride: int = 1) -> np.ndarray:
    """Height gate indices capped at z_max_m."""
    mask = z_vals <= z_max_m
    indices = np.where(mask)[0]
    if len(indices) == 0:
        indices = np.arange(len(z_vals))
    return indices[:: max(1, z_stride)]


def sample_radar_scans_for_hour(time_vals: np.ndarray, hour_start, n_scans: int = 12):
    """Return scan times and indices for one gauge hour."""
    import pandas as pd

    hour_end = hour_start + timedelta(hours=1)
    mask = (
        (time_vals >= np.datetime64(hour_start))
        & (time_vals < np.datetime64(hour_end))
    )
    hour_indices = np.where(mask)[0]
    if len(hour_indices) == 0:
        return [], []

    if len(hour_indices) >= n_scans:
        sample_pos = np.linspace(0, len(hour_indices) - 1, n_scans, dtype=int)
        selected = hour_indices[sample_pos]
    else:
        selected = hour_indices

    times = [pd.Timestamp(time_vals[i]).to_pydatetime() for i in selected]
    return times, selected.tolist()


def bin_scans_to_fixed_slots(radar_times, radar_indices, hour_start, n_bins: int = 12):
    """Map scans into fixed 5-minute slots (None for missing)."""
    bin_minutes = 60 / n_bins
    binned_times = [None] * n_bins
    binned_indices = [None] * n_bins

    for scan_time, scan_idx in zip(radar_times, radar_indices):
        minutes_in = (scan_time - hour_start).total_seconds() / 60
        slot = int(minutes_in / bin_minutes)
        slot = max(0, min(n_bins - 1, slot))
        if binned_indices[slot] is None:
            binned_indices[slot] = scan_idx
            binned_times[slot] = scan_time

    return binned_times, binned_indices


def compute_spatial_window(
    station_lat: float,
    station_lon: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    patch_size_m: float,
    resolution_m: float,
    crs: str = 'EPSG:32610',
) -> dict:
    """Return y/x slice bounds for a gauge-centered patch."""
    radar_crs = CRS.from_string(crs)
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, radar_crs, always_xy=True)
    station_x, station_y = transformer.transform(station_lon, station_lat)

    patch_pixels = int(patch_size_m / resolution_m)
    half_pixels = patch_pixels // 2

    x_idx = int(np.abs(x_vals - station_x).argmin())
    y_idx = int(np.abs(y_vals - station_y).argmin())

    x_start = max(0, x_idx - half_pixels)
    x_end = x_start + patch_pixels
    y_start = max(0, y_idx - half_pixels)
    y_end = y_start + patch_pixels

    if x_end > len(x_vals):
        x_end = len(x_vals)
        x_start = max(0, x_end - patch_pixels)
    if y_end > len(y_vals):
        y_end = len(y_vals)
        y_start = max(0, y_end - patch_pixels)

    center_y = y_idx - y_start
    center_x = x_idx - x_start

    return {
        'x_start': int(x_start),
        'x_end': int(x_end),
        'y_start': int(y_start),
        'y_end': int(y_end),
        'patch_pixels': int(patch_pixels),
        'center_y': int(center_y),
        'center_x': int(center_x),
    }


def read_volume_slice(
    store,
    field: str,
    time_idx: int,
    z_indices: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Read one (nz, ny, nx) volume slice from zarr."""
    if field not in store or time_idx is None:
        ny = y_end - y_start
        nx = x_end - x_start
        return np.full((len(z_indices), ny, nx), np.nan, dtype=np.float32)

    arr = np.asarray(
        store[field][time_idx, z_indices, y_start:y_end, x_start:x_end],
        dtype=np.float32,
    )
    return arr


def max_reflectivity_over_scans(
    store,
    radar_indices: list,
    z_indices: np.ndarray,
    window: dict,
) -> float:
    """Column-max reflectivity across scans (for filtering)."""
    peaks = []
    for scan_idx in radar_indices:
        if scan_idx is None:
            continue
        vol = read_volume_slice(
            store, 'reflectivity', scan_idx, z_indices,
            window['y_start'], window['y_end'],
            window['x_start'], window['x_end'],
        )
        if np.isfinite(vol).any():
            peaks.append(float(np.nanmax(vol)))
    return float(np.nanmax(peaks)) if peaks else float('nan')
