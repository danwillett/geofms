import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch.nn.functional as F


# ── Feature registry ──────────────────────────────────────────────────────────

# Order MUST match dataset/create_pickle.FIELDS and
# radar/derive_features.OUTPUT_FIELDS exactly.
PICKLE_FIELD_ORDER = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
    'echo_top_height',
    'max_z_height',
    'vil',
    'low_level_ref',
    'column_depth_fraction',
    'low_level_kdp',
    'low_level_zdr',
    'low_level_rhohv',
    'lowest_gate_reflectivity',
    'beam_height',
    'vertical_reflectivity_gradient',
    'melting_layer_height',
    'rhohv_min',
    'bright_band_ref',
    'bright_band_drop',
    'maxz_meltlayer_offset',
    'bright_band_intensity',
    'subml_rhohv',
    'subml_zdr',
    'subml_kdp',
    'subml_ref_max',
    'subml_zdr_gradient',
]

# Ranges curated from the empirical distribution audit (audit_feature_norms.py,
# train split). Several ranges were originally set to physical extremes (severe
# convection / full sensor range) that compressed this light-rain coastal data
# into a sliver of [0,1], starving the first conv of variance. Notable fixes:
#   - RhoHV family normalized over its meaningful band (~0.8-1.0) instead of
#     (0,1): the median sat at ~0.95 (top 5% of the old range) and 8-14% of
#     values clipped >1.0. This is a core warm-rain / bright-band discriminator.
#   - low_level_kdp 0-2 -> 0-0.7 (data p99 ~0.62; was 31% of range).
#   - ZDR family widened so the noisy/heavy tails stop clipping 14%.
#   - vil 0-60 -> 0-1: equation is correct, values are just genuinely tiny in
#     this regime (column-max p99 ~37 dBZ). Renormalized so it isn't dead; it is
#     low-information here and a prune candidate.
#   - height fields tightened to observed maxima.
# NOTE: the reflectivity family is left at (-20,70) on purpose — its high tail is
# the heavy-rain signal we must NOT clip. Changing norms invalidates existing
# checkpoints; retrain before evaluating.
FIELD_NORMS = {
    'reflectivity':                (-20.0, 70.0),
    'differential_reflectivity':   (-2.0,   8.0),
    'cross_correlation_ratio':     (0.80,   1.00),
    'differential_phase':          (0.0,  360.0),
    'specific_differential_phase': (0.0,    0.8),
    'echo_top_height':             (0.0, 6000.0),
    'max_z_height':                (0.0, 7000.0),
    'vil':                         (0.0,    1.0),
    'low_level_ref':               (-20.0, 70.0),
    'column_depth_fraction':       (0.0,    1.0),
    # new — low-level / warm-rain
    'low_level_kdp':               (0.0,    0.7),
    'low_level_zdr':               (-2.0,   8.0),
    'low_level_rhohv':             (0.80,   1.00),
    'lowest_gate_reflectivity':    (-20.0, 70.0),
    'beam_height':                 (0.0, 4000.0),
    'vertical_reflectivity_gradient': (-30.0, 30.0),
    # new — melting layer / bright band
    'melting_layer_height':        (0.0, 8000.0),
    'rhohv_min':                   (0.70,   1.00),
    # new — bright-band vertical structure
    'bright_band_ref':             (-20.0, 70.0),
    'bright_band_drop':            (-40.0, 20.0),
    'maxz_meltlayer_offset':       (-6000.0, 6000.0),
    'bright_band_intensity':       (-20.0, 20.0),
    # sub-melting-layer liquid column
    'subml_rhohv':                 (0.80,   1.00),
    'subml_zdr':                   (-2.0,   8.0),
    'subml_kdp':                   (0.0,    0.7),
    'subml_ref_max':               (-20.0, 70.0),
    'subml_zdr_gradient':          (-4.0,   4.0),
}

# Fill value (RAW units) for NaN / missing pixels. Defaults to each field's
# f_min when a field is not listed here. Override for fields where "no echo"
# does NOT imply a low value (ratios, gradients): a synthetic extreme like
# f_min would read as a real, misleading signal (e.g. RhoHV=0 → "perfect
# decorrelation"). For these we fill with a neutral value and let the optional
# validity masks carry the missingness signal instead.
# Fill values are RAW units and MUST land near the middle of each field's
# (possibly newly-curated) FIELD_NORMS range, so "no echo" reads as a neutral
# ~0.5 after normalization rather than as a real extreme. The RhoHV fills in
# particular were retuned: 0.5 was neutral under the old (0,1) range, but under
# the new (~0.8,1.0) band it would normalize below 0 and read as "perfect
# decorrelation" — exactly the misleading signal these fills exist to avoid.
FIELD_FILL = {
    'differential_reflectivity':      0.0,   # spherical-drop baseline (norm ~0.2)
    'low_level_zdr':                  0.0,   # spherical-drop baseline, not -2 dB
    'cross_correlation_ratio':        0.90,  # "unknown" -> mid of (0.80,1.00)
    'low_level_rhohv':                0.90,  # "unknown" -> mid of (0.80,1.00)
    'rhohv_min':                      0.85,  # "unknown" -> mid of (0.70,1.00)
    'vertical_reflectivity_gradient': 0.0,   # zero gradient (neutral)
    'bright_band_drop':               0.0,   # no drop below the band (neutral)
    'bright_band_intensity':          0.0,   # no reflectivity bump (neutral)
    'subml_rhohv':                    0.90,  # "unknown" -> mid of (0.80,1.00)
    'subml_zdr':                      0.0,   # spherical-drop baseline
    'subml_zdr_gradient':             0.0,   # no vertical ZDR gradient (neutral)
}

# Shared, cause-based validity masks (Option B). Most NaNs in the derived
# features trace to one of two physical conditions, so two masks cover them:
#   - column echo present  → readable from 'reflectivity' (column nanmax)
#   - low-level echo present → readable from 'low_level_ref'
# "low-level echo present" doubles as a warm-rain signal, not just bookkeeping.
FEATURE_MASK_SOURCES = ['reflectivity', 'low_level_ref']

FEATURE_PRESETS = {
    'base': ['reflectivity'],
    'dualpol': [
        'reflectivity',
        'differential_reflectivity',
        'cross_correlation_ratio',
        'specific_differential_phase',
    ],
    'vertical': [
        'echo_top_height',
        'max_z_height',
        'vil',
        'low_level_ref',
        'column_depth_fraction',
    ],
    'lowlevel': [
        'low_level_kdp',
        'low_level_zdr',
        'low_level_rhohv',
        'lowest_gate_reflectivity',
        'beam_height',
        'vertical_reflectivity_gradient',
    ],
    'meltinglayer': [
        'melting_layer_height',
        'rhohv_min',
    ],
    'brightband': [
        'bright_band_ref',
        'bright_band_drop',
        'maxz_meltlayer_offset',
        'bright_band_intensity',
    ],
    'subml': [
        'subml_rhohv',
        'subml_zdr',
        'subml_kdp',
        'subml_ref_max',
        'subml_zdr_gradient',
    ],
    'dualpol+vertical': [
        'reflectivity',
        'differential_reflectivity',
        'cross_correlation_ratio',
        'specific_differential_phase',
        'echo_top_height',
        'max_z_height',
        'vil',
        'low_level_ref',
        'column_depth_fraction',
    ],
    # Every field in PICKLE_FIELD_ORDER, including differential_phase (PhiDP),
    # which the 'dualpol' preset intentionally omits.
    'all': list(PICKLE_FIELD_ORDER),
}

N_SCANS = 12


def resolve_fields(fields_config):
    """
    Resolve a fields specification into a list of field names.

    Accepts:
    - A preset name (e.g. 'dualpol', 'vertical', 'all')
    - A list of field names or preset names (mixed allowed)
    - None (defaults to 'dualpol+vertical')
    """
    if fields_config is None:
        return list(FEATURE_PRESETS['dualpol+vertical'])

    if isinstance(fields_config, str):
        if fields_config in FEATURE_PRESETS:
            return list(FEATURE_PRESETS[fields_config])
        if fields_config in FIELD_NORMS:
            return [fields_config]
        raise ValueError(f"Unknown field or preset: '{fields_config}'. "
                         f"Available presets: {list(FEATURE_PRESETS.keys())}, "
                         f"Available fields: {list(FIELD_NORMS.keys())}")

    # List of fields/presets
    resolved = []
    for item in fields_config:
        if item in FEATURE_PRESETS:
            for f in FEATURE_PRESETS[item]:
                if f not in resolved:
                    resolved.append(f)
        elif item in FIELD_NORMS:
            if item not in resolved:
                resolved.append(item)
        else:
            raise ValueError(f"Unknown field or preset: '{item}'")
    return resolved


def compute_n_input_channels(fields, use_mask=True, use_temporal_pos=True,
                             use_dem=True, use_feature_masks=False):
    """Compute total input channels for a given configuration."""
    n = len(fields) * N_SCANS
    if use_mask:
        n += N_SCANS
    if use_temporal_pos:
        n += N_SCANS
    if use_feature_masks:
        n += len(FEATURE_MASK_SOURCES) * N_SCANS
    if use_dem:
        n += 1
    return n


# ── Dataset ───────────────────────────────────────────────────────────────────

class RadarGaugeDataset(Dataset):
    """
    PyTorch Dataset for multi-modal precipitation prediction.

    Configurable inputs:
    - fields: Which radar fields to include (list or preset name)
    - use_dem: Whether to include DEM elevation channel
    - use_mask: Whether to include validity mask channels
    - use_temporal_pos: Whether to include temporal position channels
    - log_target: Whether to log1p-transform the target
    """

    def __init__(self, pickle_path, dem_path=None, split='train',
                 augment=False, aug_prob=0.5, patch_size_m=4620,
                 fields=None, use_dem=True, use_mask=True,
                 use_temporal_pos=True, log_target=True,
                 use_feature_masks=False):

        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)

        self.samples = dataset[split]
        self.metadata = dataset['metadata']
        self.patch_size_m = patch_size_m
        self.augment = augment
        self.aug_prob = aug_prob

        # Field order of the stored radar_patch columns. Prefer the pickle's own
        # 'fields' metadata so the loader works with any field subset; fall back
        # to the canonical PICKLE_FIELD_ORDER for older pickles that didn't store it.
        self.field_order = list(self.metadata.get('fields', PICKLE_FIELD_ORDER))

        # Configurable feature selection
        self.fields = resolve_fields(fields)

        # Fail early (and clearly) if a requested field isn't stored in this
        # pickle, instead of a cryptic IndexError/ValueError deep in __getitem__.
        missing = [f for f in self.fields if f not in self.field_order]
        if missing:
            raise ValueError(
                f"Requested field(s) {missing} are not in this pickle's stored "
                f"fields. Available fields ({len(self.field_order)}): {self.field_order}. "
                f"Align the experiment's feature list with the pickle, or rebuild "
                f"the pickle (dataset/create_pickle.py FIELDS) to include them."
            )

        self.use_dem = use_dem
        self.use_mask = use_mask
        self.use_temporal_pos = use_temporal_pos
        self.log_target = log_target
        self.use_feature_masks = use_feature_masks

        self.n_channels = compute_n_input_channels(
            self.fields, self.use_mask, self.use_temporal_pos, self.use_dem,
            self.use_feature_masks
        )

        self.dem = None
        self.dem_min = 0.0
        self.dem_max = 1.0
        # Build the coordinate transformer once and memoize per-station DEM patches.
        # Gauge locations repeat across thousands of hourly samples, so recomputing
        # the transform + argmin + patch slice per __getitem__ is the main hot-loop cost.
        self._transformer = None
        self._dem_patch_cache = {}
        if dem_path and self.use_dem:
            import rioxarray as rxr
            from pyproj import Transformer
            print(f"  Loading DEM from {dem_path}...")
            dem_data = rxr.open_rasterio(dem_path)
            self.dem = dem_data.values
            self.dem_x = dem_data.x.values
            self.dem_y = dem_data.y.values
            self.dem_resolution = abs(dem_data.rio.resolution()[0])
            self.dem_min = float(np.nanmin(self.dem))
            self.dem_max = float(np.nanmax(self.dem))
            self._transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
            print(f"  ✓ DEM loaded: shape={self.dem.shape}, resolution={self.dem_resolution}m, "
                  f"range=[{self.dem_min:.1f}, {self.dem_max:.1f}]m")

        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Fields: {self.fields}")
        print(f"  Input channels: {self.n_channels} "
              f"({len(self.fields)} fields × {N_SCANS}"
              f"{' + mask' if self.use_mask else ''}"
              f"{' + tpos' if self.use_temporal_pos else ''}"
              f"{f' + {len(FEATURE_MASK_SOURCES)} feat_masks' if self.use_feature_masks else ''}"
              f"{' + DEM' if self.use_dem else ''})")
        print(f"  Log target: {self.log_target}")
        if self.use_feature_masks:
            print(f"  Feature validity masks: {FEATURE_MASK_SOURCES}")

    @classmethod
    def n_input_channels(cls, fields=None, use_mask=True, use_temporal_pos=True,
                         use_dem=True, use_feature_masks=False):
        """Compute input channels for given config (class-level utility)."""
        f = resolve_fields(fields)
        return compute_n_input_channels(f, use_mask, use_temporal_pos, use_dem,
                                        use_feature_masks)

    def __len__(self):
        return len(self.samples)

    def _extract_dem_patch(self, station_lat, station_lon):
        # Gauge coordinates repeat across all hours of a station, so cache the
        # extracted patch per location. Returns a fresh copy each call so callers
        # may mutate it safely without corrupting the cache.
        cache_key = (round(float(station_lat), 6), round(float(station_lon), 6))
        cached = self._dem_patch_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        station_x, station_y = self._transformer.transform(station_lon, station_lat)

        patch_pixels = int(self.patch_size_m / self.dem_resolution)
        half_pixels = patch_pixels // 2

        x_idx = np.abs(self.dem_x - station_x).argmin()
        y_idx = np.abs(self.dem_y - station_y).argmin()

        x_start = max(0, x_idx - half_pixels)
        x_end = x_start + patch_pixels
        y_start = max(0, y_idx - half_pixels)
        y_end = y_start + patch_pixels

        if x_end > len(self.dem_x):
            x_end = len(self.dem_x)
            x_start = max(0, x_end - patch_pixels)
        if y_end > len(self.dem_y):
            y_end = len(self.dem_y)
            y_start = max(0, y_end - patch_pixels)

        patch = self.dem[:, y_start:y_end, x_start:x_end].copy()

        if patch.shape[1] != patch_pixels or patch.shape[2] != patch_pixels:
            padded = np.zeros((1, patch_pixels, patch_pixels), dtype=patch.dtype)
            h, w = patch.shape[1], patch.shape[2]
            padded[:, :h, :w] = patch
            patch = padded

        self._dem_patch_cache[cache_key] = patch
        return patch.copy()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        radar_patch = sample['radar_patch'].copy()  # (12, N_fields_pickle, H, W)

        input_size = radar_patch.shape[-1]
        gauge_pixel = (input_size // 2, input_size // 2)

        n_scans = radar_patch.shape[0]
        H, W = radar_patch.shape[2], radar_patch.shape[3]

        # Per-field normalization (only selected fields)
        field_channels = []
        for field_name in self.fields:
            pickle_idx = self.field_order.index(field_name)
            f_min, f_max = FIELD_NORMS[field_name]

            field_arr = radar_patch[:, pickle_idx, :, :].copy()
            fill = FIELD_FILL.get(field_name, f_min)
            field_arr[field_arr == -9999.0] = fill
            field_arr = np.where(np.isnan(field_arr), fill, field_arr)

            field_norm = (field_arr - f_min) / (f_max - f_min)
            field_norm = np.clip(field_norm, 0.0, 1.0)

            for i, ridx in enumerate(sample['radar_indices']):
                if ridx is None:
                    field_norm[i] = 0.0

            field_channels.append(torch.from_numpy(field_norm).float())

        # Validity mask (per-scan: is this scan present at all)
        if self.use_mask:
            mask = np.ones((n_scans, H, W), dtype=np.float32)
            for i, ridx in enumerate(sample['radar_indices']):
                if ridx is None:
                    mask[i] = 0.0
            field_channels.append(torch.from_numpy(mask))

        # Feature validity masks (per-pixel: did this pixel have real echo, or
        # was it filled). Derived from the raw (pre-fill) values of the shared
        # source fields, so they work regardless of which fields are selected.
        if self.use_feature_masks:
            for src in FEATURE_MASK_SOURCES:
                if src in self.field_order and self.field_order.index(src) < radar_patch.shape[1]:
                    raw = radar_patch[:, self.field_order.index(src), :, :]
                    vmask = (np.isfinite(raw) & (raw != -9999.0)).astype(np.float32)
                else:
                    vmask = np.ones((n_scans, H, W), dtype=np.float32)
                for i, ridx in enumerate(sample['radar_indices']):
                    if ridx is None:
                        vmask[i] = 0.0
                field_channels.append(torch.from_numpy(vmask))

        # Temporal position
        if self.use_temporal_pos:
            t_pos = torch.zeros((n_scans, H, W))
            for i in range(n_scans):
                t_pos[i] = i / max(n_scans - 1, 1)
            field_channels.append(t_pos)

        # DEM channel
        if self.use_dem:
            if self.dem is not None:
                dem_patch = self._extract_dem_patch(sample['station_lat'], sample['station_lon'])
                dem = torch.from_numpy(dem_patch).float()
            elif 'dem_patch' in sample and sample['dem_patch'] is not None:
                dem = torch.from_numpy(sample['dem_patch']).float()
            else:
                dem = torch.zeros(1, 462, 462)

            dem_resized = F.adaptive_avg_pool2d(dem.unsqueeze(0), (H, W)).squeeze(0)
            dem_range = self.dem_max - self.dem_min
            dem_resized = (dem_resized - self.dem_min) / (dem_range if dem_range > 0 else 1.0)
            dem_resized = dem_resized.clamp(0.0, 1.0)
            field_channels.append(dem_resized)

        radar_with_features = torch.cat(field_channels, dim=0)

        target = torch.tensor(sample['hourly_precip_mm'], dtype=torch.float32)
        if self.log_target:
            target = torch.log1p(target)

        hour_str = str(sample['hour_start']) if 'hour_start' in sample else ''
        bias_flag = torch.tensor(sample.get('bias_flag', 0), dtype=torch.long)

        return {
            'radar': radar_with_features,
            'target': target,
            'gauge_pixel': gauge_pixel,
            'station_id': sample['station_id'],
            'station_name': sample.get('station_name', ''),
            'hour': hour_str,
            'bias_flag': bias_flag,
        }
