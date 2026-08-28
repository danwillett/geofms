"""
models/gfm_untrained/dataset.py — Dataset for the from-scratch (untrained) GFM.

Differences vs. models/gfm/dataset.py:
  - NO upscaling anywhere. The native radar tile (clipped to `output_size`, e.g.
    18×18) is fed straight to the ViT, which uses a small `patch_size` (1 or 2)
    so a tiny tile still yields a real token grid. The DEM is downsampled to the
    same grid (co-registered) — the simple shared-grid alignment.
  - `modality_layout` controls how channels are packaged for the ViT:
      'single'  -> one combined modality "ALL" (radar + dualpol + DEM stacked as
                   channels, U-Net-like). Decoder in_channels = dim.
      'grouped' -> separate "DEM" / "RADAR" / "DUALPOL" modalities, each its own
                   token stream fused by cross-modal attention. in_channels = dim*M.
  - compute_modality_channels() exposes per-modality channel counts so model.py
    can build the matching embeddings.

Everything else (field norms, filters, DEM extraction/memoization, log target)
is shared with the U-Net pipeline via imports.
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

# Default 17-field subml feature set (matches current daygroup+offsets pickle)
DEFAULT_FIELDS = [
    'reflectivity', 'echo_top_height', 'max_z_height', 'vil',
    'column_depth_fraction', 'vertical_reflectivity_gradient',
    'melting_layer_height', 'rhohv_min', 'bright_band_ref', 'bright_band_drop',
    'maxz_meltlayer_offset', 'bright_band_intensity',
    'subml_rhohv', 'subml_zdr', 'subml_kdp', 'subml_ref_max', 'subml_zdr_gradient',
]

# Fields assigned to the polarimetric / microphysics ("DUALPOL") modality in the
# grouped layout. Everything else goes to the reflectivity/vertical ("RADAR")
# modality. Defined as a superset so it is robust to other field configs.
DUALPOL_GROUP_FIELDS = {
    'rhohv_min', 'subml_rhohv', 'subml_zdr', 'subml_kdp', 'subml_ref_max',
    'subml_zdr_gradient',
    # classic dual-pol / low-level (in case a different pickle includes them)
    'differential_reflectivity', 'cross_correlation_ratio', 'differential_phase',
    'specific_differential_phase', 'low_level_kdp', 'low_level_zdr',
    'low_level_rhohv',
}


def _aux_channels(use_feature_masks: bool) -> int:
    """Validity mask (12) + temporal position (12) + optional feature masks."""
    n = N_SCANS + N_SCANS
    if use_feature_masks:
        n += len(FEATURE_MASK_SOURCES) * N_SCANS
    return n


def compute_radar_channels(fields=None, use_feature_masks=True):
    """Total radar channels (all fields + aux), excluding DEM. Matches the
    canonical 252-channel layout for the 17-field subml set."""
    flds = resolve_fields(fields) if isinstance(fields, str) else (
        list(fields) if fields else list(DEFAULT_FIELDS))
    return len(flds) * N_SCANS + _aux_channels(use_feature_masks)


def compute_modality_channels(fields=None, use_feature_masks=True, layout='grouped',
                              use_dem=True):
    """Return an ordered dict {modality_name: num_channels} for the given layout.

    'single'  -> {'ALL': radar_channels (+1 DEM channel if use_dem)}
    'grouped' -> {'DEM': 1 (if use_dem), 'RADAR': radar_grp + aux, 'DUALPOL': dualpol_grp}
                 (DUALPOL omitted if no dual-pol fields are present)

    use_dem=False drops the DEM entirely (no extra channel in 'single', no DEM
    modality in 'grouped') — used to probe how much skill is genuine radar vs.
    terrain/location memorization.
    """
    flds = resolve_fields(fields) if isinstance(fields, str) else (
        list(fields) if fields else list(DEFAULT_FIELDS))
    aux = _aux_channels(use_feature_masks)

    if layout == 'single':
        total = len(flds) * N_SCANS + aux + (1 if use_dem else 0)
        return {'ALL': total}

    if layout == 'grouped':
        dualpol_fields = [f for f in flds if f in DUALPOL_GROUP_FIELDS]
        radar_fields   = [f for f in flds if f not in DUALPOL_GROUP_FIELDS]
        mods = {}
        if use_dem:
            mods['DEM'] = 1
        mods['RADAR'] = len(radar_fields) * N_SCANS + aux
        if dualpol_fields:
            mods['DUALPOL'] = len(dualpol_fields) * N_SCANS
        return mods

    raise ValueError(f"Unknown modality_layout: {layout!r} (use 'single' or 'grouped')")


class RadarDEMDataset(Dataset):
    """Dataset for the untrained GFM. Emits native-resolution tiles (no upscale).

    Produces:
        image = {modality_name: (C_mod, out, out), ...}   # per modality_layout
        mask  = (out, out)  — target at the gauge pixel, -9999 elsewhere
                              (log1p when log_target=True)

    Parameters
    ----------
    output_size : int
        Native tile size fed to the ViT after cropping (e.g. 18 from a 19×19
        pickle). No interpolation is applied to the radar; the DEM is downsampled
        to this grid for co-registration.
    modality_layout : str
        'single' or 'grouped' (see module docstring).
    """

    def __init__(
        self,
        samples,
        field_order,
        dem_path='dem/preserve_dem_10m_utm.tif',
        patch_size_m=9500,
        output_size=18,
        augment=False,
        aug_prob=0.5,
        use_feature_masks=True,
        fields=None,
        log_target=True,
        modality_layout='grouped',
        use_dem=True,
    ):
        self.samples           = samples
        self.field_order       = field_order
        self.patch_size_m      = patch_size_m
        self.output_size       = output_size
        self.augment           = augment
        self.aug_prob          = aug_prob
        self.use_feature_masks = use_feature_masks
        self.log_target        = log_target
        self.modality_layout   = modality_layout
        self.use_dem           = use_dem
        self.fields            = resolve_fields(fields) if fields else list(DEFAULT_FIELDS)

        missing = [f for f in self.fields if f not in self.field_order]
        if missing:
            raise ValueError(
                f"Requested field(s) {missing} not in pickle's field_order. "
                f"Available: {self.field_order}"
            )

        self.n_radar_channels   = compute_radar_channels(self.fields, use_feature_masks)
        self.modality_channels  = compute_modality_channels(
            self.fields, use_feature_masks, modality_layout, use_dem)

        # DEM: hoist transformer + memoize per-station patch
        self.dem          = None
        self.dem_min      = 0.0
        self.dem_max      = 1.0
        self._transformer = None
        self._dem_cache   = {}

        if dem_path and use_dem:
            import rioxarray as rxr
            from pyproj import Transformer
            print(f"  Loading DEM from {dem_path}...")
            dem_data = rxr.open_rasterio(dem_path)
            self.dem            = dem_data.values
            self.dem_x          = dem_data.x.values
            self.dem_y          = dem_data.y.values
            self.dem_resolution = abs(dem_data.rio.resolution()[0])
            self.dem_min        = float(np.nanmin(self.dem))
            self.dem_max        = float(np.nanmax(self.dem))
            self._transformer   = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
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

    def _radar_blocks(self, radar_patch, radar_indices):
        """Build labeled radar channel blocks.

        Returns a list of (group, tensor[N_SCANS, H, W]) where group is
        'RADAR' or 'DUALPOL'. The aux channels (validity mask, temporal position,
        feature-validity masks) are tagged 'RADAR' so they ride with the primary
        reflectivity stream in the grouped layout.
        """
        n_scans = radar_patch.shape[0]
        H, W    = radar_patch.shape[2], radar_patch.shape[3]
        blocks  = []

        # ── Per-field normalised values ─────────────────────────────────────
        for field_name in self.fields:
            idx          = self.field_order.index(field_name)
            f_min, f_max = FIELD_NORMS[field_name]
            fill         = FIELD_FILL.get(field_name, f_min)

            arr = radar_patch[:, idx, :, :].copy()
            arr[arr == -9999.0] = fill
            arr = np.where(np.isnan(arr), fill, arr)
            arr = np.clip((arr - f_min) / (f_max - f_min), 0.0, 1.0)
            for i, ridx in enumerate(radar_indices):
                if ridx is None:
                    arr[i] = 0.0

            group = 'DUALPOL' if field_name in DUALPOL_GROUP_FIELDS else 'RADAR'
            blocks.append((group, torch.from_numpy(arr.astype(np.float32))))

        # ── Validity mask ───────────────────────────────────────────────────
        mask = np.ones((n_scans, H, W), dtype=np.float32)
        for i, ridx in enumerate(radar_indices):
            if ridx is None:
                mask[i] = 0.0
        blocks.append(('RADAR', torch.from_numpy(mask)))

        # ── Temporal position ───────────────────────────────────────────────
        t_pos = torch.zeros((n_scans, H, W), dtype=torch.float32)
        for i in range(n_scans):
            t_pos[i] = i / max(n_scans - 1, 1)
        blocks.append(('RADAR', t_pos))

        # ── Feature validity masks ──────────────────────────────────────────
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
                blocks.append(('RADAR', torch.from_numpy(vmask)))

        return blocks

    def _process_dem(self, dem_patch):
        """Normalise DEM patch to [0,1]; returns (1, H, W) at native tile size."""
        arr = dem_patch.copy()
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        arr = np.where(np.isnan(arr), 0.0, arr)
        t   = torch.from_numpy(arr).float()
        dem_range = self.dem_max - self.dem_min
        t = (t - self.dem_min) / (dem_range if dem_range > 0 else 1.0)
        return t.clamp(0.0, 1.0)

    def _assemble_image(self, blocks, dem_t):
        """Package labeled blocks + DEM into the per-layout modality dict.

        dem_t is None when use_dem=False (DEM dropped entirely).
        """
        if self.modality_layout == 'single':
            radar_all = torch.cat([t for _, t in blocks], dim=0)
            if self.use_dem:
                radar_all = torch.cat([radar_all, dem_t], dim=0)
            return {'ALL': radar_all}

        # grouped
        image = {}
        if self.use_dem:
            image['DEM'] = dem_t
        image['RADAR'] = torch.cat([t for g, t in blocks if g == 'RADAR'], dim=0)
        dualpol = [t for g, t in blocks if g == 'DUALPOL']
        if dualpol:
            image['DUALPOL'] = torch.cat(dualpol, dim=0)
        return image

    def __getitem__(self, idx):
        s           = self.samples[idx]
        radar_patch = s['radar_patch'].copy()   # (12, N_fields, patch_px, patch_px)
        input_size  = radar_patch.shape[-1]      # e.g. 19 for 9500m @ 500m
        out         = self.output_size

        # ── Spatial crop (native, no upscale) ────────────────────────────────
        if input_size > out:
            max_off = input_size - out
            center  = input_size // 2
            if self.augment and np.random.rand() < self.aug_prob:
                # Keep the gauge (always at patch center) inside the cropped window.
                oy_min = max(0, center - (out - 1)); oy_max = min(max_off, center)
                ox_min = max(0, center - (out - 1)); ox_max = min(max_off, center)
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

        blocks = self._radar_blocks(radar_patch, s.get('radar_indices', [None] * N_SCANS))

        # ── DEM (downsampled to the native radar grid, same crop) ─────────────
        dem_t = None
        if self.use_dem:
            if self.dem is not None:
                dem_full = self._extract_dem_patch(s['station_lat'], s['station_lon'])
                if input_size > out:
                    dem_full_t = torch.from_numpy(dem_full).float()
                    dem_sized  = F.interpolate(dem_full_t.unsqueeze(0),
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

        image = self._assemble_image(blocks, dem_t)

        # ── Target mask ───────────────────────────────────────────────────────
        precip_mm   = max(0.0, float(s['hourly_precip_mm']))
        target_val  = np.log1p(precip_mm) if self.log_target else precip_mm
        sparse_mask = torch.full((out, out), -9999.0, dtype=torch.float32)
        sparse_mask[gauge_y, gauge_x] = target_val

        return {'image': image, 'mask': sparse_mask}


# ── Data Module ───────────────────────────────────────────────────────────────

class RadarDEMDataModule(L.LightningDataModule):
    """Lightning DataModule for the untrained GFM. Exposes `modality_channels`
    (consumed by model.py to build matching embeddings) after setup()."""

    def __init__(
        self,
        pickle_path: str,
        dem_path: str = 'dem/preserve_dem_10m_utm.tif',
        output_size: int = 18,
        fields=None,
        use_feature_masks: bool = True,
        log_target: bool = True,
        modality_layout: str = 'grouped',
        use_dem: bool = True,
        weight_sampler=None,
        batch_size: int = 32,
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
        self.modality_layout   = modality_layout
        self.use_dem           = use_dem
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
        for split_name in ('train', 'val'):
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
            modality_layout   = self.modality_layout,
            use_dem           = self.use_dem,
        )
        self.train_ds = RadarDEMDataset(train_samples, augment=True, aug_prob=0.5, **ds_kwargs)
        self.val_ds   = RadarDEMDataset(val_samples,   augment=False,            **ds_kwargs)

        self.train_dataset      = self.train_ds
        self.val_dataset        = self.val_ds
        self.n_radar_channels   = self.train_ds.n_radar_channels
        self.modality_channels  = self.train_ds.modality_channels

        # Fail fast if the emitted per-modality channel counts disagree with the
        # advertised modality_channels (what model.py uses to build embeddings).
        sample_img = self.train_ds[0]['image']
        emitted = {k: int(v.shape[0]) for k, v in sample_img.items()}
        if emitted != {k: int(v) for k, v in self.modality_channels.items()}:
            raise RuntimeError(
                f"Modality channel mismatch — emitted {emitted} but advertised "
                f"{self.modality_channels}. The model would be built with the "
                f"wrong embedding sizes."
            )

        print(f"\n  GFM-untrained dataset ready:")
        print(f"    train={len(train_samples)}  val={len(val_samples)}")
        print(f"    output_size={self.output_size}  layout={self.modality_layout}  use_dem={self.use_dem}")
        print(f"    modalities={self.modality_channels}")

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
    weights[targets < 0.1]                     = 0.5
    weights[(targets >= 0.1) & (targets < 2)]  = 1.0
    weights[(targets >= 2)   & (targets < 5)]  = 2.0
    weights[(targets >= 5)   & (targets < 15)] = 5.0
    weights[targets >= 15]                     = 10.0
    weights = weights / weights.sum() * len(weights)
    print(f"  Heavy-rain sampler: {(targets>=5).mean()*100:.1f}% heavy → "
          f"effective {(weights[targets>=5]).sum()/weights.sum()*100:.1f}%")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
