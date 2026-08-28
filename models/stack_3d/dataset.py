"""stack_3d/dataset.py — Lazy-loading 3D radar volume dataset."""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset
from dataset.volume_io import RAW_FIELDS_3D, read_volume_slice

FIELD_PRESETS = {
    'dualpol': ['reflectivity', 'differential_reflectivity', 'cross_correlation_ratio', 'specific_differential_phase'],
    'dualpol_phi': RAW_FIELDS_3D,
}
FIELD_NORMS = {
    'reflectivity': (-20.0, 70.0),
    'differential_reflectivity': (-2.0, 6.0),
    'cross_correlation_ratio': (0.0, 1.0),
    'differential_phase': (0.0, 360.0),
    'specific_differential_phase': (0.0, 1.0),
}
N_SCANS = 12

# Default bounded LRU cache for zarr chunk reads (MB). Caps memory while
# avoiding repeat disk hits for hot chunks during random-access training.
DEFAULT_ZARR_CACHE_MB = 1024

# Module-level caches shared across dataset instances in the same process
# (e.g. train + val splits). Keeps a single bounded zarr cache and a single
# in-RAM copy of the DEM instead of duplicating per split.
_STORE_CACHE = {}
_DEM_CACHE = {}


def _open_cached_store(zarr_path, cache_mb=DEFAULT_ZARR_CACHE_MB):
    """Open a zarr group backed by a bounded LRU chunk cache (shared per path)."""
    key = (zarr_path, int(cache_mb))
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]
    try:
        raw_store = zarr.storage.DirectoryStore(zarr_path)
        cached_store = zarr.storage.LRUStoreCache(raw_store, max_size=int(cache_mb) * 1024 * 1024)
        store = zarr.open(cached_store, mode='r')
    except Exception as e:
        print(f"  ⚠ LRUStoreCache unavailable ({e}); falling back to plain zarr.open")
        store = zarr.open(zarr_path, mode='r')
    _STORE_CACHE[key] = store
    return store


def _load_dem(dem_path):
    """Load DEM raster once per path; share arrays across dataset instances."""
    if dem_path in _DEM_CACHE:
        return _DEM_CACHE[dem_path]
    import rioxarray as rxr
    print(f"  Loading DEM from {dem_path} (shared cache)...")
    dem_data = rxr.open_rasterio(dem_path)
    info = {
        'dem': dem_data.values,
        'dem_x': dem_data.x.values,
        'dem_y': dem_data.y.values,
        'dem_resolution': abs(dem_data.rio.resolution()[0]),
    }
    info['dem_min'] = float(np.nanmin(info['dem']))
    info['dem_max'] = float(np.nanmax(info['dem']))
    _DEM_CACHE[dem_path] = info
    return info

def resolve_fields(fields):
    if fields is None:
        return list(FIELD_PRESETS['dualpol'])
    if isinstance(fields, str):
        return list(FIELD_PRESETS[fields]) if fields in FIELD_PRESETS else [f.strip() for f in fields.split(',')]
    return list(fields)

def compute_n_input_channels(fields, use_mask=True, use_temporal_pos=True, use_dem=True, use_feature_masks=False):
    n_fields = len(resolve_fields(fields))
    n_ch = n_fields * N_SCANS
    if use_mask: n_ch += N_SCANS
    if use_temporal_pos: n_ch += N_SCANS
    if use_dem: n_ch += 1
    if use_feature_masks: n_ch += n_fields * N_SCANS
    return n_ch

class RadarGauge3DDataset(Dataset):
    def __init__(self, pickle_path, dem_path=None, split='train', fields=None, use_dem=True,
                 use_mask=True, use_temporal_pos=True, use_feature_masks=False, log_target=True,
                 augment=False, aug_prob=0.5, patch_size_m=None, zarr_cache_mb=DEFAULT_ZARR_CACHE_MB):
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        self.samples = dataset[split]
        self.metadata = dataset['metadata']
        self.augment, self.aug_prob = augment, aug_prob
        self.use_dem, self.use_mask = use_dem, use_mask
        self.use_temporal_pos, self.use_feature_masks = use_temporal_pos, use_feature_masks
        self.log_target = log_target
        self.fields = resolve_fields(fields or self.metadata.get('fields_preset', 'dualpol'))
        self.zarr_path = self.metadata['radar_zarr_path']
        self.z_indices = np.asarray(self.metadata['z_indices'], dtype=int)
        self.patch_size_m = patch_size_m or self.metadata.get('patch_size_m', 4500)
        self._store = _open_cached_store(self.zarr_path, zarr_cache_mb)
        self.dem, self.dem_min, self.dem_max = None, 0.0, 1.0
        dem_to_load = dem_path or self.metadata.get('dem_path')
        if use_dem and dem_to_load:
            dem_info = _load_dem(dem_to_load)
            self.dem = dem_info['dem']
            self.dem_x, self.dem_y = dem_info['dem_x'], dem_info['dem_y']
            self.dem_resolution = dem_info['dem_resolution']
            self.dem_min, self.dem_max = dem_info['dem_min'], dem_info['dem_max']
        self.n_channels = compute_n_input_channels(self.fields, use_mask, use_temporal_pos, use_dem, use_feature_masks)

    def __len__(self):
        return len(self.samples)

    def _extract_dem_patch(self, station_lat, station_lon):
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
        station_x, station_y = transformer.transform(station_lon, station_lat)
        patch_pixels = int(self.patch_size_m / self.dem_resolution)
        half_pixels = patch_pixels // 2
        x_idx = int(np.abs(self.dem_x - station_x).argmin())
        y_idx = int(np.abs(self.dem_y - station_y).argmin())
        x_start, y_start = max(0, x_idx - half_pixels), max(0, y_idx - half_pixels)
        x_end, y_end = x_start + patch_pixels, y_start + patch_pixels
        if x_end > len(self.dem_x): x_end = len(self.dem_x); x_start = max(0, x_end - patch_pixels)
        if y_end > len(self.dem_y): y_end = len(self.dem_y); y_start = max(0, y_end - patch_pixels)
        patch = self.dem[:, y_start:y_end, x_start:x_end].copy()
        if patch.shape[1] != patch_pixels or patch.shape[2] != patch_pixels:
            padded = np.zeros((1, patch_pixels, patch_pixels), dtype=patch.dtype)
            padded[:, :patch.shape[1], :patch.shape[2]] = patch
            patch = padded
        return patch

    def _resize_dem_to_patch(self, dem, H, W):
        dem_2d = F.adaptive_avg_pool2d(dem.unsqueeze(0), (H, W)).squeeze(0)
        if dem_2d.dim() == 3 and dem_2d.shape[0] == 1:
            dem_2d = dem_2d.squeeze(0)
        dem_range = self.dem_max - self.dem_min
        return ((dem_2d - self.dem_min) / (dem_range if dem_range > 0 else 1.0)).clamp(0.0, 1.0)

    def _load_volumes(self, sample):
        n_scans = len(sample['radar_indices'])
        nz = len(self.z_indices)
        H, W = sample['y_end'] - sample['y_start'], sample['x_end'] - sample['x_start']
        out = np.full((n_scans, len(self.fields), nz, H, W), np.nan, dtype=np.float32)
        for t_idx, scan_idx in enumerate(sample['radar_indices']):
            if scan_idx is None: continue
            for f_idx, field in enumerate(self.fields):
                out[t_idx, f_idx] = read_volume_slice(
                    self._store, field, scan_idx, self.z_indices,
                    sample['y_start'], sample['y_end'], sample['x_start'], sample['x_end'])
        return out

    def __getitem__(self, idx):
        sample = self.samples[idx]
        radar_vol = self._load_volumes(sample)
        n_scans, _, nz, H, W = radar_vol.shape
        gauge_pixel = (sample.get('center_y', H // 2), sample.get('center_x', W // 2))
        channels = []
        for f_idx, field_name in enumerate(self.fields):
            f_min, f_max = FIELD_NORMS[field_name]
            field_arr = radar_vol[:, f_idx].copy()
            field_arr[field_arr == -9999.0] = f_min
            field_arr = np.where(np.isnan(field_arr), f_min, field_arr)
            field_norm = np.clip((field_arr - f_min) / (f_max - f_min), 0.0, 1.0)
            for i, ridx in enumerate(sample['radar_indices']):
                if ridx is None: field_norm[i] = 0.0
            for scan_idx in range(n_scans):
                channels.append(torch.from_numpy(field_norm[scan_idx]).float())
        if self.use_mask:
            mask = np.ones((n_scans, nz, H, W), dtype=np.float32)
            for i, ridx in enumerate(sample['radar_indices']):
                if ridx is None: mask[i] = 0.0
            for scan_idx in range(n_scans):
                channels.append(torch.from_numpy(mask[scan_idx]).float())
        if self.use_temporal_pos:
            for scan_idx in range(n_scans):
                channels.append(torch.from_numpy(np.full((nz, H, W), scan_idx / max(n_scans - 1, 1), np.float32)).float())
        if self.use_dem:
            dem = torch.from_numpy(self._extract_dem_patch(sample['station_lat'], sample['station_lon'])).float() if self.dem is not None else torch.zeros(1, H, W)
            dem_2d = self._resize_dem_to_patch(dem, H, W)
            channels.append(dem_2d.unsqueeze(0).expand(nz, H, W))
        radar = torch.stack(channels, dim=0)
        target = torch.tensor(sample['hourly_precip_mm'], dtype=torch.float32)
        if self.log_target: target = torch.log1p(target)
        return {'radar': radar, 'target': target, 'gauge_pixel': gauge_pixel,
                'station_id': sample['station_id'], 'station_name': sample.get('station_name', ''),
                'hour': str(sample.get('hour_start', '')), 'bias_flag': torch.tensor(sample.get('bias_flag', 0), dtype=torch.long)}
