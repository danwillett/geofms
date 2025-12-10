import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch.nn.functional as F

class RadarGaugeDataset(Dataset):
    """
    PyTorch Dataset for multi-modal precipitation prediction
    
    Loads pre-extracted patches of radar and DEM
    aligned with rain gauge measurements.
    
    Each sample contains:
    - Radar: 6 time slices at 500m resolution (6, Z, 5, 5)
    - DEM: Elevation data at 10m resolution (1, 264, 264) [optional]
    - Target: Hourly precipitation in mm (scalar)
    - (Future: LULC at 10m resolution)
    """

    DBZ_MIN = -32.0
    DBZ_MAX = 70 #46
    
    def __init__(self, pickle_path, dem_path=None, split='train', augment=False, aug_prob=0.5, patch_size_m=4620):
        """
        Initialize the dataset by loading the pickle file
        
        Args:
            pickle_path: Path to radar_gauge_dataset.pkl
            dem_path: Path to DEM GeoTIFF (loaded once, extracted on-the-fly)
            split: 'train' or 'val'
            augment: Whether to apply data augmentation
            aug_prob: Probability of augmentation
            patch_size_m: Size of patches in meters for DEM extraction
        """
        # 1. Load the entire pickle file into memory
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # 2. Extract the split you want (train or val)
        self.samples = dataset[split]  # List of sample dicts
        
        # 3. Store metadata
        self.metadata = dataset['metadata']
        self.patch_size_m = patch_size_m
        
        self.augment = augment
        self.aug_prob = aug_prob
        
        # 4. Load DEM once into memory (if provided)
        self.dem = None
        self.dem_transform = None
        if dem_path:
            import rioxarray as rxr
            print(f"  Loading DEM from {dem_path}...")
            dem_data = rxr.open_rasterio(dem_path)
            self.dem = dem_data.values  # (1, H, W) numpy array
            self.dem_x = dem_data.x.values  # x coordinates
            self.dem_y = dem_data.y.values  # y coordinates
            self.dem_resolution = abs(dem_data.rio.resolution()[0])  # typically 10m
            print(f"  ✓ DEM loaded: shape={self.dem.shape}, resolution={self.dem_resolution}m")
        
        # 5. Print info (helpful for debugging)
        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Radar resolution: {self.metadata['radar_resolution_m']}m")
        print(f"  Patch size: {self.metadata['patch_size_m']}m")
    
    def __len__(self):
        """
        Tell PyTorch how many samples are in this dataset
        
        This is called by DataLoader to know when to stop iterating
        """
        return len(self.samples)
    
    def _extract_dem_patch(self, station_lat, station_lon):
        """
        Extract DEM patch centered on station location (on-the-fly)
        
        Returns:
            dem_patch: numpy array (1, H, W) at DEM resolution
        """
        from pyproj import Transformer
        from scipy.ndimage import zoom
        
        # Convert station lat/lon to UTM
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
        station_x, station_y = transformer.transform(station_lon, station_lat)
        
        # Calculate patch size in pixels
        patch_pixels = int(self.patch_size_m / self.dem_resolution)
        half_pixels = patch_pixels // 2
        
        # Find nearest pixel indices
        x_idx = np.abs(self.dem_x - station_x).argmin()
        y_idx = np.abs(self.dem_y - station_y).argmin()
        
        # Define pixel window
        x_start = max(0, x_idx - half_pixels)
        x_end = x_start + patch_pixels
        y_start = max(0, y_idx - half_pixels)
        y_end = y_start + patch_pixels
        
        # Handle edge cases
        if x_end > len(self.dem_x):
            x_end = len(self.dem_x)
            x_start = max(0, x_end - patch_pixels)
        if y_end > len(self.dem_y):
            y_end = len(self.dem_y)
            y_start = max(0, y_end - patch_pixels)
        
        # Extract patch
        patch = self.dem[:, y_start:y_end, x_start:x_end].copy()
        
        # Ensure correct size (pad if needed)
        if patch.shape[1] != patch_pixels or patch.shape[2] != patch_pixels:
            padded = np.zeros((1, patch_pixels, patch_pixels), dtype=patch.dtype)
            h, w = patch.shape[1], patch.shape[2]
            padded[:, :h, :w] = patch
            patch = padded
        
        return patch
    
    def __getitem__(self, idx):
        """
        Return one sample by index
        
        For 9×9 patches: randomly crop to 5×5 during training,
        center crop during validation. This teaches the model to
        predict at all pixel positions, not just the center.
        """
        # 1. Get the sample dictionary from your list
        sample = self.samples[idx]
        
        # 2. Extract radar data: shape (12, Z, 9, 9) for 9×9 patches
        radar = sample['radar_patch'].copy()  # Make a copy to avoid modifying original
        
        # 3. Determine crop offset FIRST (before any processing)
        # This ensures radar and DEM are cropped consistently
        input_size = radar.shape[-1]  # 9 for 9×9 patches, 5 for legacy 5×5
        output_size = 5
        
        if input_size > output_size:
            # We have larger patches (e.g., 9×9) - need to crop
            max_offset = input_size - output_size  # = 4 for 9×9
            
            if self.augment and np.random.rand() < self.aug_prob:
                # Random crop during training
                offset_y = np.random.randint(0, max_offset + 1)
                offset_x = np.random.randint(0, max_offset + 1)
            else:
                # Center crop during validation
                offset_y = max_offset // 2
                offset_x = max_offset // 2
            
            # Gauge position in the cropped 5×5 grid
            # When offset=0, gauge (originally at center of 9×9 = position 4) 
            # ends up at position 4 in 5×5
            # When offset=4, gauge ends up at position 0 in 5×5
            center_pos = input_size // 2  # = 4 for 9×9
            gauge_pixel = (center_pos - offset_y, center_pos - offset_x)
            
            # Crop radar from 9×9 to 5×5
            radar = radar[:, :, offset_y:offset_y+output_size, offset_x:offset_x+output_size]
        else:
            # Legacy 5×5 patches - no cropping needed
            offset_y, offset_x = 0, 0
            gauge_pixel = (2, 2)  # Center of 5×5
        
        # 4. Handle missing data
        radar[radar == -9999.0] = self.DBZ_MIN
        radar = np.where(np.isnan(radar), self.DBZ_MIN, radar)

        # Normalize to [0, 1]
        radar_norm = (radar - self.DBZ_MIN) / (self.DBZ_MAX - self.DBZ_MIN)
        radar_norm = np.clip(radar_norm, 0, 1)

        # Create binary mask
        mask = np.ones_like(radar_norm)
        for i, ridx in enumerate(sample['radar_indices']):
            if ridx is None:
                mask[i, :, :, :] = 0.0
                radar_norm[i, :, :, :] = 0.0
        
        # 5. Convert to PyTorch tensor and process
        radar_tensor = torch.from_numpy(radar_norm).float()  # (12, Z, 5, 5)
        mask_tensor = torch.from_numpy(mask).float()

        # Max over Z dimension
        radar_maxz = torch.max(radar_tensor, dim=1)[0]  # (12, 5, 5)
        mask_maxz = torch.max(mask_tensor, dim=1)[0]    # (12, 5, 5)

        # Temporal position encoding
        t_pos = torch.zeros_like(radar_maxz)
        for i in range(radar_maxz.shape[0]):
            t_pos[i] = i / 11.0

        # 6. Extract and crop DEM with SAME offset
        if self.dem is not None:
            dem_patch = self._extract_dem_patch(sample['station_lat'], sample['station_lon'])
            dem = torch.from_numpy(dem_patch).float()
        elif 'dem_patch' in sample and sample['dem_patch'] is not None:
            dem = torch.from_numpy(sample['dem_patch']).float()
        else:
            dem = torch.zeros(1, 462, 462)
        
        # Downsample DEM to match input_size, then crop with same offset
        if input_size > output_size:
            dem_full = F.adaptive_avg_pool2d(dem.unsqueeze(0), (input_size, input_size)).squeeze(0)  # (1, 9, 9)
            dem_cropped = dem_full[:, offset_y:offset_y+output_size, offset_x:offset_x+output_size]  # (1, 5, 5)
        else:
            dem_cropped = F.adaptive_avg_pool2d(dem.unsqueeze(0), (output_size, output_size)).squeeze(0)

        # 7. Concatenate all features: (12 + 12 + 12 + 1 = 37 channels, 5, 5)
        radar_with_features = torch.cat([
            radar_maxz,      # (12, 5, 5)
            mask_maxz,       # (12, 5, 5)
            t_pos,           # (12, 5, 5)
            dem_cropped      # (1, 5, 5)
        ], dim=0)

        # # 8. Random rotation augmentation (gauge stays at gauge_pixel because we rotate everything)
        # if self.augment and np.random.rand() < self.aug_prob:
        #     k = np.random.randint(0, 4)
        #     if k > 0:
        #         radar_with_features = torch.rot90(radar_with_features, k=k, dims=(-2, -1))
        #         # Rotate gauge_pixel to match
        #         for _ in range(k):
        #             gauge_pixel = (gauge_pixel[1], output_size - 1 - gauge_pixel[0])

        # 9. Extract target
        target = torch.tensor(sample['hourly_precip_mm'], dtype=torch.float32)
        target = torch.log1p(target)
        
        # 10. Get metadata
        hour_str = str(sample['hour_start']) if 'hour_start' in sample else ''
        bias_flag = torch.tensor(sample.get('bias_flag', 0), dtype=torch.long)

        return {
            'radar': radar_with_features,  
            'target': target,
            'gauge_pixel': gauge_pixel,
            'station_id': sample['station_id'],
            'hour': hour_str,
            'bias_flag': bias_flag
        }
    
    def get_sample_info(self, idx):
        """
        Optional: Get metadata about a sample without loading the data
        Useful for debugging
        """
        sample = self.samples[idx]
        return {
            'station': sample['station_name'],
            'hour': sample['hour_start'],
            'precipitation': sample['hourly_precip_mm'],
            'valid_radar_scans': sample['n_valid_radar']
        }