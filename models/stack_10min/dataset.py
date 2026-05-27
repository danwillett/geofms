import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch.nn.functional as F


class RadarGaugeDataset10min(Dataset):
    """
    PyTorch Dataset for 10-minute precipitation prediction from a single radar scan.

    Each sample contains:
    - Radar: N dual-pol fields at 500m resolution (N_fields, H, W) — single scan
    - DEM: Elevation data downsampled to match radar patch
    - Target: 10-minute precipitation in mm (scalar, raw — no log transform)
    """

    PICKLE_FIELD_ORDER = [
        'reflectivity',
        'differential_reflectivity',
        'cross_correlation_ratio',
        'differential_phase',
        'specific_differential_phase',
    ]

    FIELDS = [
        'reflectivity',
        'differential_reflectivity',
        'cross_correlation_ratio',
        'specific_differential_phase',
    ]

    FIELD_NORMS = {
        'reflectivity':                (-20.0, 70.0),
        'differential_reflectivity':   (-2.0,   6.0),
        'cross_correlation_ratio':     (0.0,    1.0),
        'differential_phase':          (0.0,  360.0),
        'specific_differential_phase': (0.0,    1.0),
    }

    @classmethod
    def n_input_channels(cls):
        """Total input channels: N_fields + 1 DEM"""
        return len(cls.FIELDS) + 1

    def __init__(self, pickle_path, dem_path=None, split='train', augment=False, aug_prob=0.5, patch_size_m=4500):
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)

        self.samples = dataset[split]
        self.metadata = dataset['metadata']
        self.patch_size_m = patch_size_m
        self.augment = augment
        self.aug_prob = aug_prob

        self.dem = None
        self.dem_min = 0.0
        self.dem_max = 1.0
        if dem_path:
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

        print(f"Loaded {split} dataset (10-min resolution):")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Input channels: {self.n_input_channels()} "
              f"({len(self.FIELDS)} radar fields + 1 DEM)")

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
        radar_patch = sample['radar_patch'].copy()  # (N_fields_pickle, H, W)

        H, W = radar_patch.shape[1], radar_patch.shape[2]

        # Per-field normalization
        field_channels = []
        for field_name in self.FIELDS:
            pickle_idx = self.PICKLE_FIELD_ORDER.index(field_name)

            field_arr = radar_patch[pickle_idx, :, :].copy()  # (H, W)
            f_min, f_max = self.FIELD_NORMS[field_name]

            field_arr[field_arr == -9999.0] = f_min
            field_arr = np.where(np.isnan(field_arr), f_min, field_arr)

            field_norm = (field_arr - f_min) / (f_max - f_min)
            field_norm = np.clip(field_norm, 0.0, 1.0)

            field_channels.append(torch.from_numpy(field_norm).float().unsqueeze(0))  # (1, H, W)

        # DEM channel (downsampled to match radar patch size, normalized to [0,1])
        if self.dem is not None:
            dem_patch = self._extract_dem_patch(sample['station_lat'], sample['station_lon'])
            dem = torch.from_numpy(dem_patch).float()
        else:
            dem = torch.zeros(1, H, W)

        # Downsample DEM to match radar spatial size
        dem_resized = F.adaptive_avg_pool2d(dem.unsqueeze(0), (H, W)).squeeze(0)

        # Normalize DEM to [0, 1]
        dem_range = self.dem_max - self.dem_min
        dem_resized = (dem_resized - self.dem_min) / (dem_range if dem_range > 0 else 1.0)
        dem_resized = dem_resized.clamp(0.0, 1.0)

        field_channels.append(dem_resized)

        # Concatenate: (N_fields + 1, H, W)
        radar_with_features = torch.cat(field_channels, dim=0)

        # Target in raw mm (no log transform)
        target = torch.tensor(sample['precip_mm'], dtype=torch.float32)

        bin_str = str(sample.get('bin_start', ''))

        return {
            'radar': radar_with_features,
            'target': target,
            'station_id': sample['station_id'],
            'station_name': sample.get('station_name', ''),
            'bin_start': bin_str,
        }
