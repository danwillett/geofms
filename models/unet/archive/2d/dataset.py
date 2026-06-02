import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch.nn.functional as F


class RadarGaugeDataset(Dataset):
    """
    PyTorch Dataset for multi-modal precipitation prediction with dual-pol radar.

    Each sample contains:
    - Radar: 12 time slices × N dual-pol fields at 500m resolution (12, N_fields, 9, 9)
    - DEM: Elevation data at 10m resolution (extracted on-the-fly)
    - Target: Hourly precipitation in mm (scalar, log1p-transformed)
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

    N_SCANS = 12

    @classmethod
    def n_input_channels(cls):
        """Total input channels: N_fields * 12 + 12 mask + 12 tpos + 1 DEM"""
        return len(cls.FIELDS) * cls.N_SCANS + cls.N_SCANS + cls.N_SCANS + 1

    def __init__(self, pickle_path, dem_path=None, split='train', augment=False, aug_prob=0.5, patch_size_m=4620):
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

        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Input channels: {self.n_input_channels()} "
              f"({len(self.FIELDS)} fields × {self.N_SCANS} + mask + tpos + DEM)")

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
        radar_patch = sample['radar_patch'].copy()  # (12, N_fields_pickle, 9, 9)

        input_size = radar_patch.shape[-1]
        # output_size = 5

        # # ── Cropping (commented out to use full 9×9 input) ──
        # if input_size > output_size:
        #     max_offset = input_size - output_size
        #
        #     if self.augment and np.random.rand() < self.aug_prob:
        #         offset_y = np.random.randint(0, max_offset + 1)
        #         offset_x = np.random.randint(0, max_offset + 1)
        #     else:
        #         offset_y = max_offset // 2
        #         offset_x = max_offset // 2
        #
        #     center_pos = input_size // 2
        #     gauge_pixel = (center_pos - offset_y, center_pos - offset_x)
        #
        #     radar_patch = radar_patch[:, :, offset_y:offset_y+output_size, offset_x:offset_x+output_size]
        # else:
        #     offset_y, offset_x = 0, 0
        #     gauge_pixel = (2, 2)

        # Use full patch — gauge is always at center
        gauge_pixel = (input_size // 2, input_size // 2)  # (4, 4) for 9×9

        n_scans = radar_patch.shape[0]
        H, W = radar_patch.shape[2], radar_patch.shape[3]

        # Per-field normalization
        field_channels = []
        for field_name in self.FIELDS:
            pickle_idx = self.PICKLE_FIELD_ORDER.index(field_name)
            f_min, f_max = self.FIELD_NORMS[field_name]

            field_arr = radar_patch[:, pickle_idx, :, :].copy()  # (12, 5, 5)
            field_arr[field_arr == -9999.0] = f_min
            field_arr = np.where(np.isnan(field_arr), f_min, field_arr)

            field_norm = (field_arr - f_min) / (f_max - f_min)
            field_norm = np.clip(field_norm, 0.0, 1.0)

            for i, ridx in enumerate(sample['radar_indices']):
                if ridx is None:
                    field_norm[i] = 0.0

            field_channels.append(torch.from_numpy(field_norm).float())

        # Shared validity mask
        mask = np.ones((n_scans, H, W), dtype=np.float32)
        for i, ridx in enumerate(sample['radar_indices']):
            if ridx is None:
                mask[i] = 0.0
        field_channels.append(torch.from_numpy(mask))

        # Shared temporal position
        t_pos = torch.zeros((n_scans, H, W))
        for i in range(n_scans):
            t_pos[i] = i / max(n_scans - 1, 1)
        field_channels.append(t_pos)

        # DEM channel (downsampled to match radar patch size, normalized to [0,1])
        if self.dem is not None:
            dem_patch = self._extract_dem_patch(sample['station_lat'], sample['station_lon'])
            dem = torch.from_numpy(dem_patch).float()
        elif 'dem_patch' in sample and sample['dem_patch'] is not None:
            dem = torch.from_numpy(sample['dem_patch']).float()
        else:
            dem = torch.zeros(1, 462, 462)

        # Downsample DEM to match radar spatial size (9×9)
        dem_resized = F.adaptive_avg_pool2d(dem.unsqueeze(0), (H, W)).squeeze(0)

        # Normalize DEM to [0, 1] using the full DEM's min/max
        dem_range = self.dem_max - self.dem_min
        dem_resized = (dem_resized - self.dem_min) / (dem_range if dem_range > 0 else 1.0)
        dem_resized = dem_resized.clamp(0.0, 1.0)

        field_channels.append(dem_resized)

        # Concatenate: (N_fields*12 + 12 + 12 + 1, 5, 5)
        radar_with_features = torch.cat(field_channels, dim=0)

        target = torch.tensor(sample['hourly_precip_mm'], dtype=torch.float32)
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
