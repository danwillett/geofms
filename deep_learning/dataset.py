import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

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
    DBZ_MAX = 70
    
    def __init__(self, pickle_path, split='train', transform=None, use_dem=False):
        """
        Initialize the dataset by loading the pickle file
        
        Args:
            pickle_path: Path to radar_gauge_dataset.pkl
            split: 'train' or 'val'
            transform: Optional transforms (not used for now)
            use_dem: Whether to load DEM patches (default: False)
        """
        # 1. Load the entire pickle file into memory
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # 2. Extract the split you want (train or val)
        self.samples = dataset[split]  # List of sample dicts
        
        # 3. Store metadata
        self.metadata = dataset['metadata']
        self.transform = transform
        self.use_dem = use_dem
        
        # 4. Check if DEM data is available
        self.has_dem = self.metadata.get('has_dem', False)
        if self.has_dem:
            # Verify that samples actually have DEM patches
            dem_count = sum(1 for s in self.samples if s.get('dem_patch') is not None)
            if dem_count == 0:
                print(f"  ⚠️  Warning: Metadata says has_dem=True but no DEM patches found!")
                self.has_dem = False
            else:
                print(f"  ✓ DEM patches available: {dem_count}/{len(self.samples)} samples")
        
        if use_dem and not self.has_dem:
            print(f"  ⚠️  Warning: use_dem=True but no DEM data in dataset!")
            print(f"     Set use_dem=False or regenerate dataset with --dem parameter")
            self.use_dem = False
        
        # 5. Print info (helpful for debugging)
        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Radar resolution: {self.metadata['radar_resolution_m']}m")
        print(f"  Patch size: {self.metadata['patch_size_m']}m")
        print(f"  Using DEM: {self.use_dem}")
    
    def __len__(self):
        """
        Tell PyTorch how many samples are in this dataset
        
        This is called by DataLoader to know when to stop iterating
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return one sample by index
        
        This is the MOST IMPORTANT method - it defines what your model receives!
        
        PyTorch DataLoader will call this repeatedly:
        - sample_0 = dataset[0]
        - sample_1 = dataset[1]
        - ...
        
        Then it batches them together automatically.
        """
        # 1. Get the sample dictionary from your list
        sample = self.samples[idx]
        
        # 2. Extract radar data: shape (6, Z, 5, 5)
        #    6 time slices, Z altitude levels, 5×5 spatial @ 500m
        radar = sample['radar_patch']  # NumPy array
        
        # 3. Handle missing data
        # Replace sentinel (-9999) with minimum detectable reflectivity (-32 dBZ)
        # Physically: -32 dBZ represents "no significant echo" (clear sky)
        radar[radar == -9999.0] = self.DBZ_MIN
        
        # Handle remaining NaN values
        radar = np.where(np.isnan(radar), self.DBZ_MIN, radar)

        # Normalize to [0, 1]
        radar_norm = (radar - self.DBZ_MIN) / (self.DBZ_MAX - self.DBZ_MIN)
        radar_norm = np.clip(radar_norm, 0, 1)

        # Create binary mask (same shape as radar, but binary)
        # 1.0 = valid data, 0.0 = padding
        mask = np.ones_like(radar_norm)

        for i, ridx in enumerate(sample['radar_indices']):
            if ridx is None:
                # This entire scan is padding
                mask[i, :, :, :] = 0.0
                radar_norm[i, :, :, :] = 0.0  # Set to 0 (neutral value)
        
        # 4. Convert to PyTorch tensor
        radar_tensor = torch.from_numpy(radar_norm).float()  # Shape: (6, Z, 5, 5)
        mask_tensor = torch.from_numpy(mask).float()

        # Concatenate along channel dimension after max over Z
        # We'll do max over Z, then concat masks
        radar_maxz = torch.max(radar_tensor, dim=1)[0]  # (T, Y, X)
        mask_maxz = torch.max(mask_tensor, dim=1)[0]    # (T, Y, X) - 1 if any Z valid
        
        # Concatenate: (T, Y, X) + (T, Y, X) → (2*T, Y, X)
        radar_with_mask = torch.cat([radar_maxz, mask_maxz], dim=0)  # (24, 5, 5)

        # 5. Extract target (hourly precipitation)
        target = sample['hourly_precip_mm']
        target = torch.tensor(target, dtype=torch.float32)  # Scalar
        target = torch.log1p(target)
        
        # 6. Extract DEM patch (if available and requested)
        dem = None
        if self.use_dem and sample.get('dem_patch') is not None:
            dem = torch.from_numpy(sample['dem_patch']).float()
            # Shape: (1, 264, 264) @ 10m resolution
        
        # (Future) When you add LULC:
        # if 'lulc_patch' in sample:
        #     lulc = torch.from_numpy(sample['lulc_patch']).float()
        
        # 7. Determine gauge pixel location in 5×5 grid
        # Since patches are centered on gauge, gauge is typically at center pixel
        # For 5×5 grid, center is at (2, 2) using 0-based indexing
        gauge_pixel = (2, 2)  # (y, x) position in 5×5 grid
        
        # 8. Convert timestamp to string (PyTorch can't batch pandas Timestamps)
        hour_str = str(sample['hour_start']) if 'hour_start' in sample else ''


        # 9. Get bias flag
        bias_flag = sample.get('bias_flag', 0)
        bias_flag = torch.tensor(bias_flag, dtype=torch.long)  # ← Must be long for embedding!

        # 9. Return as dictionary (model will receive this structure)
        return {
            'radar': radar_with_mask,           # (6, Z, 5, 5)
            'dem': dem,               # (1, 264, 264) or None
            'target': target,         # scalar
            'gauge_pixel': gauge_pixel,  # (y, x) tuple for loss computation
            'station_id': sample['station_id'],  # metadata (optional)
            'hour': hour_str,         # metadata as string (optional)
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