import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch.nn.functional as F


# ── Feature registry ──────────────────────────────────────────────────────────

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
]

FIELD_NORMS = {
    'reflectivity':                (-20.0, 70.0),
    'differential_reflectivity':   (-2.0,   6.0),
    'cross_correlation_ratio':     (0.0,    1.0),
    'differential_phase':          (0.0,  360.0),
    'specific_differential_phase': (0.0,    1.0),
    'echo_top_height':             (0.0, 8000.0),
    'max_z_height':                (0.0, 8000.0),
    'vil':                         (0.0,   60.0),
    'low_level_ref':               (-20.0, 70.0),
    'column_depth_fraction':       (0.0,    1.0),
}

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
    'all': [
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
    ],
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


def compute_n_input_channels(fields, use_mask=True, use_temporal_pos=True, use_dem=True):
    """Compute total input channels for a given configuration."""
    n = len(fields) * N_SCANS
    if use_mask:
        n += N_SCANS
    if use_temporal_pos:
        n += N_SCANS
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
                 use_temporal_pos=True, log_target=True):

        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)

        self.samples = dataset[split]
        self.metadata = dataset['metadata']
        self.patch_size_m = patch_size_m
        self.augment = augment
        self.aug_prob = aug_prob

        # Configurable feature selection
        self.fields = resolve_fields(fields)
        self.use_dem = use_dem
        self.use_mask = use_mask
        self.use_temporal_pos = use_temporal_pos
        self.log_target = log_target

        self.n_channels = compute_n_input_channels(
            self.fields, self.use_mask, self.use_temporal_pos, self.use_dem
        )

        self.dem = None
        self.dem_min = 0.0
        self.dem_max = 1.0
        if dem_path and self.use_dem:
            import rioxarray as rxr
            print(f"  Loading DEM from {dem_path}...")
            dem_data = rxr.open_rasterio(dem_path)
            self.dem = dem_data.values
            self.dem_x = dem_data.x.values
            self.dem_y = dem_data.y.values
            self.dem_resolution = abs(dem_data.rio.resolution()[0])
            self.dem_min = float(np.nanmin(self.dem))
            self.dem_max = float(np.nanmax(self.dem))
            print(f"  ✓ DEM loaded: shape={self.dem.shape}, resolution={self.dem_resolution}m, "
                  f"range=[{self.dem_min:.1f}, {self.dem_max:.1f}]m")

        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Fields: {self.fields}")
        print(f"  Input channels: {self.n_channels} "
              f"({len(self.fields)} fields × {N_SCANS}"
              f"{' + mask' if self.use_mask else ''}"
              f"{' + tpos' if self.use_temporal_pos else ''}"
              f"{' + DEM' if self.use_dem else ''})")
        print(f"  Log target: {self.log_target}")

    @classmethod
    def n_input_channels(cls, fields=None, use_mask=True, use_temporal_pos=True, use_dem=True):
        """Compute input channels for given config (class-level utility)."""
        f = resolve_fields(fields)
        return compute_n_input_channels(f, use_mask, use_temporal_pos, use_dem)

    def __len__(self):
        return len(self.samples)

    def _extract_dem_patch(self, station_lat, station_lon):
        from pyproj import Transformer

        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
        station_x, station_y = transformer.transform(station_lon, station_lat)

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

        return patch

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
            pickle_idx = PICKLE_FIELD_ORDER.index(field_name)
            f_min, f_max = FIELD_NORMS[field_name]

            field_arr = radar_patch[:, pickle_idx, :, :].copy()
            field_arr[field_arr == -9999.0] = f_min
            field_arr = np.where(np.isnan(field_arr), f_min, field_arr)

            field_norm = (field_arr - f_min) / (f_max - f_min)
            field_norm = np.clip(field_norm, 0.0, 1.0)

            for i, ridx in enumerate(sample['radar_indices']):
                if ridx is None:
                    field_norm[i] = 0.0

            field_channels.append(torch.from_numpy(field_norm).float())

        # Validity mask
        if self.use_mask:
            mask = np.ones((n_scans, H, W), dtype=np.float32)
            for i, ridx in enumerate(sample['radar_indices']):
                if ridx is None:
                    mask[i] = 0.0
            field_channels.append(torch.from_numpy(mask))

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
