"""
models/gfm/dataset.py — RadarDEMDataset for TerraMind/GFM precipitation model.

Key changes vs. original:
  - Field list and norms imported from unet.dataset (no duplication)
  - Field order read dynamically from pickle metadata (handles any subml pickle)
  - All filters imported from unet.train (consistent with U-Net / Stack pipelines)
  - DEM transformer hoisted to __init__; per-station DEM patches memoized
  - output_size is a configurable parameter (supports 5, 9, 13, 19 from 19×19 patches)
  - Feature-validity masks (reflectivity + low_level_ref) optionally added
  - Channel count exposed via compute_radar_channels() for model.py to consume
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning as L
import numpy as np
import pickle

from models.unet.dataset import (
    PICKLE_FIELD_ORDER, FIELD_NORMS, FIELD_FILL, FEATURE_MASK_SOURCES,
    resolve_fields,
)
from models.unet.train import (
    filter_nan_radar, filter_biased_extremes, filter_bad_samples,
    filter_suspect_station_days, filter_gauge_dumps,
)

N_SCANS = 12

# ── DEM normalization ───────────────────────────────────────────────────────────
# TerraMind v1's pretrained DEM patch-embedding was trained on DEM standardized
# with these global (TerraMesh) statistics, in metres. Feeding DEM standardized
# the same way is what lets the *pretrained* DEM embedding see in-distribution
# input. Source: terratorch terramind_register.py v1_pretraining_mean/std
# ("(un)tok_dem@224": mean 670.665, std 951.272).
TERRAMIND_DEM_MEAN = 670.665
TERRAMIND_DEM_STD  = 951.272
DEM_NORM_MODES = ('minmax', 'terramind')

# Default 17-field subml feature set (matches current daygroup+offsets pickle)
DEFAULT_FIELDS = [
    'reflectivity', 'echo_top_height', 'max_z_height', 'vil',
    'column_depth_fraction', 'vertical_reflectivity_gradient',
    'melting_layer_height', 'rhohv_min', 'bright_band_ref', 'bright_band_drop',
    'maxz_meltlayer_offset', 'bright_band_intensity',
    'subml_rhohv', 'subml_zdr', 'subml_kdp', 'subml_ref_max', 'subml_zdr_gradient',
]


def compute_radar_channels(fields=None, use_feature_masks=True):
    """Return the number of RADAR tensor channels for a given field list.

    Layout (per group, each group has N_SCANS=12 channels):
        field_values × len(fields)     — normalized [0, 1]
        validity_mask × 1              — 1=real scan, 0=padded
        temporal_pos × 1               — scan index / (N_SCANS-1)
        feature_validity × 2           — echo / low-level echo present (if enabled)
    """
    n_fields = len(fields) if fields is not None else len(DEFAULT_FIELDS)
    n = n_fields * N_SCANS + N_SCANS + N_SCANS   # fields + mask + tpos
    if use_feature_masks:
        n += len(FEATURE_MASK_SOURCES) * N_SCANS
    return n


class RadarDEMDataset(Dataset):
    """
    Dataset for TerraMind GFM precipitation model.

    Produces:
        image = {
            "DEM":   (1,   256, 256)  — DEM bilinear-upscaled to 256×256
            "RADAR": (C,   256, 256)  — radar nearest-upscaled, C=compute_radar_channels()
        }
        mask  = (output_size, output_size)  — log1p target at gauge pixel, -9999 elsewhere

    Parameters
    ----------
    samples : list of dicts
        Pre-loaded sample list (train or val split from the pickle).
    field_order : list of str
        Field order as stored in this pickle (from metadata['fields']).
    dem_path : str or None
        Path to DEM GeoTIFF. If None, uses zeros.
    patch_size_m : int
        Spatial size of the stored radar patch in metres (e.g. 9500 for 19×19 @ 500 m).
    output_size : int
        Size of the radar tile fed to the ViT after cropping (default 9).
        Must satisfy output_size <= patch_pixels = patch_size_m // 500.
        Set to patch_pixels to skip the crop entirely and use the full context window.
    augment : bool
        If True, use random crop offset during training. Centre-crop otherwise.
    use_feature_masks : bool
        If True, include two feature-validity mask channels per scan.
    fields : list of str or None
        Field names to include. Defaults to DEFAULT_FIELDS.
    """

    def __init__(
        self,
        samples,
        field_order,
        dem_path='dem/preserve_dem_10m_utm.tif',
        patch_size_m=9500,
        output_size=9,
        augment=False,
        aug_prob=0.5,
        use_feature_masks=True,
        fields=None,
        log_target=True,
        dem_norm='minmax',
    ):
        self.samples          = samples
        self.field_order      = field_order
        self.patch_size_m     = patch_size_m
        self.output_size      = output_size
        self.augment          = augment
        self.aug_prob         = aug_prob
        self.use_feature_masks = use_feature_masks
        self.log_target       = log_target
        self.fields           = resolve_fields(fields) if fields else list(DEFAULT_FIELDS)

        if dem_norm not in DEM_NORM_MODES:
            raise ValueError(f"dem_norm must be one of {DEM_NORM_MODES}, got {dem_norm!r}")
        self.dem_norm         = dem_norm

        # Validate requested fields against pickle
        missing = [f for f in self.fields if f not in self.field_order]
        if missing:
            raise ValueError(
                f"Requested field(s) {missing} not in pickle's field_order. "
                f"Available: {self.field_order}"
            )

        self.n_radar_channels = compute_radar_channels(self.fields, use_feature_masks)

        # DEM: hoist transformer + memoize per-station patch
        self.dem            = None
        self.dem_min        = 0.0
        self.dem_max        = 1.0
        self._transformer   = None
        self._dem_cache     = {}

        if dem_path:
            import rioxarray as rxr
            from pyproj import Transformer
            print(f"  Loading DEM from {dem_path}...")
            dem_data = rxr.open_rasterio(dem_path)
            self.dem             = dem_data.values
            self.dem_x           = dem_data.x.values
            self.dem_y           = dem_data.y.values
            self.dem_resolution  = abs(dem_data.rio.resolution()[0])
            self.dem_min         = float(np.nanmin(self.dem))
            self.dem_max         = float(np.nanmax(self.dem))
            self._transformer    = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
            print(f"  DEM loaded: shape={self.dem.shape}, "
                  f"res={self.dem_resolution}m, range=[{self.dem_min:.0f},{self.dem_max:.0f}]m")

    def __len__(self):
        return len(self.samples)

    def _extract_dem_patch(self, station_lat, station_lon):
        """Return a (1, patch_pixels, patch_pixels) DEM patch, memoized per station."""
        cache_key = (round(float(station_lat), 6), round(float(station_lon), 6))
        cached = self._dem_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        station_x, station_y = self._transformer.transform(station_lon, station_lat)
        patch_pixels = int(self.patch_size_m / self.dem_resolution)
        half         = patch_pixels // 2

        x_idx = int(np.abs(self.dem_x - station_x).argmin())
        y_idx = int(np.abs(self.dem_y - station_y).argmin())

        x_start = max(0, x_idx - half); x_end = x_start + patch_pixels
        y_start = max(0, y_idx - half); y_end = y_start + patch_pixels

        if x_end > len(self.dem_x): x_end = len(self.dem_x); x_start = max(0, x_end - patch_pixels)
        if y_end > len(self.dem_y): y_end = len(self.dem_y); y_start = max(0, y_end - patch_pixels)

        patch = self.dem[:, y_start:y_end, x_start:x_end].copy()
        if patch.shape[1] != patch_pixels or patch.shape[2] != patch_pixels:
            padded = np.zeros((1, patch_pixels, patch_pixels), dtype=patch.dtype)
            padded[:, :patch.shape[1], :patch.shape[2]] = patch
            patch = padded

        self._dem_cache[cache_key] = patch
        return patch.copy()

    def _process_radar(self, radar_patch, radar_indices):
        """
        Build the RADAR tensor from a (12, N_fields_pickle, H, W) array.

        Returns (C, 256, 256) nearest-neighbour upscaled tensor where
        C = compute_radar_channels(self.fields, self.use_feature_masks).
        """
        n_scans  = radar_patch.shape[0]   # 12
        H, W     = radar_patch.shape[2], radar_patch.shape[3]
        channels = []

        # ── Per-field normalised values ──────────────────────────────────────
        for field_name in self.fields:
            idx           = self.field_order.index(field_name)
            f_min, f_max  = FIELD_NORMS[field_name]
            fill          = FIELD_FILL.get(field_name, f_min)

            arr = radar_patch[:, idx, :, :].copy()
            arr[arr == -9999.0] = fill
            arr = np.where(np.isnan(arr), fill, arr)
            arr = np.clip((arr - f_min) / (f_max - f_min), 0.0, 1.0)

            for i, ridx in enumerate(radar_indices):
                if ridx is None:
                    arr[i] = 0.0
            channels.append(torch.from_numpy(arr.astype(np.float32)))

        # ── Validity mask (1 × N_SCANS) ──────────────────────────────────────
        mask = np.ones((n_scans, H, W), dtype=np.float32)
        for i, ridx in enumerate(radar_indices):
            if ridx is None:
                mask[i] = 0.0
        channels.append(torch.from_numpy(mask))

        # ── Temporal position (1 × N_SCANS) ──────────────────────────────────
        t_pos = torch.zeros((n_scans, H, W), dtype=torch.float32)
        for i in range(n_scans):
            t_pos[i] = i / max(n_scans - 1, 1)
        channels.append(t_pos)

        # ── Feature validity masks (2 × N_SCANS) ─────────────────────────────
        if self.use_feature_masks:
            for src in FEATURE_MASK_SOURCES:
                if src in self.field_order and self.field_order.index(src) < radar_patch.shape[1]:
                    raw   = radar_patch[:, self.field_order.index(src), :, :]
                    vmask = (np.isfinite(raw) & (raw != -9999.0)).astype(np.float32)
                else:
                    vmask = np.ones((n_scans, H, W), dtype=np.float32)
                for i, ridx in enumerate(radar_indices):
                    if ridx is None:
                        vmask[i] = 0.0
                channels.append(torch.from_numpy(vmask))

        radar_all = torch.cat(channels, dim=0)   # (C, H, W)
        return radar_all   # (C, out, out) — upscaled to 256×256 on GPU in UpscalingPrecipTask

    def _process_dem(self, dem_patch):
        """Normalise DEM patch; returns (1, H, W) — upscaled to 256×256 on GPU.

        Two modes (self.dem_norm):
          'minmax'    — local [0,1] scaling using this DEM's own min/max. Fine
                        for a from-scratch DEM embedding (no prior expectation).
          'terramind' — standardize to TerraMind's pretrained DEM stats
                        (dem_m - 670.665) / 951.272, in metres. Required for the
                        *pretrained* DEM embedding to see in-distribution input.
        """
        arr = dem_patch.copy()
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        arr = np.where(np.isnan(arr), 0.0, arr)
        t   = torch.from_numpy(arr).float()
        if self.dem_norm == 'terramind':
            # DEM is stored in metres; standardize with TerraMesh global stats.
            t = (t - TERRAMIND_DEM_MEAN) / TERRAMIND_DEM_STD
        else:  # 'minmax'
            dem_range = self.dem_max - self.dem_min
            t = (t - self.dem_min) / (dem_range if dem_range > 0 else 1.0)
            t = t.clamp(0.0, 1.0)
        return t   # (1, out, out)

    def __getitem__(self, idx):
        s           = self.samples[idx]
        radar_patch = s['radar_patch'].copy()   # (12, N_fields, patch_px, patch_px)
        input_size  = radar_patch.shape[-1]     # e.g. 19 for 9500m @ 500m
        out         = self.output_size

        # ── Spatial crop ─────────────────────────────────────────────────────
        if input_size > out:
            max_off = input_size - out
            center  = input_size // 2
            if self.augment and np.random.rand() < self.aug_prob:
                # Constrain offsets so gauge (always at patch center) stays
                # within the cropped output window: oy in [center-(out-1), center]
                oy_min = max(0, center - (out - 1))
                oy_max = min(max_off, center)
                ox_min = max(0, center - (out - 1))
                ox_max = min(max_off, center)
                oy = np.random.randint(oy_min, oy_max + 1)
                ox = np.random.randint(ox_min, ox_max + 1)
            else:
                oy = max_off // 2
                ox = max_off // 2
            gauge_y = center - oy
            gauge_x = center - ox
            radar_patch = radar_patch[:, :, oy:oy + out, ox:ox + out]
        else:
            oy, ox  = 0, 0
            gauge_y = out // 2
            gauge_x = out // 2

        radar_t = self._process_radar(radar_patch, s.get('radar_indices', [None]*N_SCANS))

        # ── DEM ──────────────────────────────────────────────────────────────
        if self.dem is not None:
            dem_full = self._extract_dem_patch(s['station_lat'], s['station_lon'])
            if input_size > out:
                # Resize full patch to input_size, then apply same crop
                dem_full_t   = torch.from_numpy(dem_full).float()
                dem_sized    = F.interpolate(dem_full_t.unsqueeze(0),
                                             size=(input_size, input_size),
                                             mode='bilinear', align_corners=False).squeeze(0)
                dem_patch_np = dem_sized[:, oy:oy + out, ox:ox + out].numpy()
            else:
                dem_patch_np = dem_full
        elif 'dem_patch' in s and s['dem_patch'] is not None:
            dem_patch_np = s['dem_patch']
        else:
            dem_patch_np = np.zeros((1, out, out), dtype=np.float32)

        dem_t = self._process_dem(dem_patch_np)

        # ── Target mask ───────────────────────────────────────────────────────
        precip_mm    = max(0.0, float(s['hourly_precip_mm']))
        target_val   = np.log1p(precip_mm) if self.log_target else precip_mm
        sparse_mask  = torch.full((out, out), -9999.0, dtype=torch.float32)
        sparse_mask[gauge_y, gauge_x] = target_val

        return {
            'image': {'DEM': dem_t, 'RADAR': radar_t},
            'mask':  sparse_mask,
        }


# ── Data Module ───────────────────────────────────────────────────────────────

class RadarDEMDataModule(L.LightningDataModule):
    """
    Lightning DataModule wrapping RadarDEMDataset.

    Parameters
    ----------
    pickle_path : str
    dem_path : str
    output_size : int
        Radar crop size fed to the ViT (5, 9, 13, or 19 for a 19×19 pickle).
    fields : list of str or None
    use_feature_masks : bool
    weight_sampler : callable or None
        Factory that accepts a samples list and returns a WeightedRandomSampler.
    batch_size, num_workers : int
    filter_mode : str  'blunt' | 'radar'
    """

    def __init__(
        self,
        pickle_path: str,
        dem_path: str = 'dem/preserve_dem_10m_utm.tif',
        output_size: int = 9,
        fields=None,
        use_feature_masks: bool = True,
        log_target: bool = True,
        dem_norm: str = 'minmax',
        weight_sampler=None,
        batch_size: int = 8,
        num_workers: int = 0,
        filter_mode: str = 'blunt',
    ):
        super().__init__()
        self.pickle_path       = pickle_path
        self.dem_path          = dem_path
        self.output_size       = output_size
        self.fields            = fields
        self.use_feature_masks = use_feature_masks
        self.log_target        = log_target
        self.dem_norm          = dem_norm
        self.weight_sampler    = weight_sampler
        self.batch_size        = batch_size
        self.num_workers       = num_workers
        self.filter_mode       = filter_mode
        self.train_sampler     = None

    def setup(self, stage=None):
        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)

        train_samples = data['train']
        val_samples   = data['val']
        meta          = data.get('metadata', {})
        field_order   = list(meta.get('fields', PICKLE_FIELD_ORDER))
        patch_size_m  = meta.get('patch_size_m', 9500)

        # ── Filter chain (matches U-Net / Stack) ─────────────────────────────
        from models.unet.train import filter_radar_unsupported
        for split_name, samples_ref in [('train', 'train_samples'), ('val', 'val_samples')]:
            s = train_samples if split_name == 'train' else val_samples
            s = filter_nan_radar(s)
            if self.filter_mode == 'radar':
                s = filter_radar_unsupported(s)
            else:
                s = filter_biased_extremes(s)
                s = filter_bad_samples(s)
            s = filter_suspect_station_days(s)
            s = filter_gauge_dumps(s)
            if split_name == 'train':
                train_samples = s
            else:
                val_samples = s

        ds_kwargs = dict(
            field_order       = field_order,
            dem_path          = self.dem_path,
            patch_size_m      = patch_size_m,
            output_size       = self.output_size,
            fields            = self.fields,
            use_feature_masks = self.use_feature_masks,
            log_target        = self.log_target,
            dem_norm          = self.dem_norm,
        )
        self.train_ds = RadarDEMDataset(train_samples, augment=True,  aug_prob=0.5, **ds_kwargs)
        self.val_ds   = RadarDEMDataset(val_samples,   augment=False,              **ds_kwargs)

        # Expose for external inspection
        self.train_dataset = self.train_ds
        self.val_dataset   = self.val_ds
        self.n_radar_channels = self.train_ds.n_radar_channels

        print(f"\n  GFM dataset ready:")
        print(f"    train={len(train_samples)}  val={len(val_samples)}")
        print(f"    output_size={self.output_size}  RADAR channels={self.n_radar_channels}")
        print(f"    dem_norm={self.dem_norm}")

        if self.weight_sampler is not None:
            self.train_sampler = self.weight_sampler(train_samples)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            sampler=self.train_sampler, shuffle=(self.train_sampler is None),
            num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ── Sampler ───────────────────────────────────────────────────────────────────

def create_heavy_rain_sampler(samples):
    """Oversample rare heavy-rain events during training."""
    targets = np.array([s['hourly_precip_mm'] for s in samples])
    weights = np.ones(len(targets))
    weights[targets < 0.1]                            = 0.5
    weights[(targets >= 0.1) & (targets < 2)]         = 1.0
    weights[(targets >= 2)   & (targets < 5)]         = 2.0
    weights[(targets >= 5)   & (targets < 15)]        = 5.0
    weights[targets >= 15]                            = 10.0
    weights = weights / weights.sum() * len(weights)
    print(f"  Heavy-rain sampler: {(targets>=5).mean()*100:.1f}% heavy → "
          f"effective {(weights[targets>=5]).sum()/weights.sum()*100:.1f}%")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
