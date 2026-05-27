import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
import pickle
from torch.utils.data import WeightedRandomSampler

# dataset
class RadarDEMDataset(Dataset):

    # Ordered field list — must match the field order in the zarr / pickle radar_patch dim 1
    PICKLE_FIELD_ORDER = [
        'reflectivity',
        'differential_reflectivity',
        'cross_correlation_ratio',
        'differential_phase',
        'specific_differential_phase',
    ]

    FIELDS = [
        'reflectivity',                # dBZ
        'differential_reflectivity',   # ZDR  (dB)
        'cross_correlation_ratio',     # RhoHV (unitless)
        'differential_phase',          # PhiDP (degrees)
        # 'specific_differential_phase', # KDP   (deg/km)
    ]

    # Per-field (min, max) for [0, 1] normalization
    FIELD_NORMS = {
        'reflectivity':                (-32.0, 70.0),
        'differential_reflectivity':   (-2.0,   6.0),
        'cross_correlation_ratio':     (0.0,    1.0), # maybe decrease min?
        'differential_phase':          (0.0,  180.0),
        # 'specific_differential_phase': (-1.0,   6.0),
    }

    DBZ_MIN = -32.0  # kept for backward-compat references elsewhere
    DBZ_MAX = 70.0

    def __init__(self, samples, dem_path='./preserve_dem_10m_utm.tif', patch_size_m=4620, augment=False, aug_prob=0.5):
        self.samples = samples
        self.patch_size_m = patch_size_m
        self.augment = augment
        self.aug_prob = aug_prob
        
        # Load DEM once into memory (if provided)
        self.dem = None
        if dem_path:
            import rioxarray as rxr
            print(f"  Loading DEM from {dem_path}...")
            dem_data = rxr.open_rasterio(dem_path)
            self.dem = dem_data.values  # (1, H, W) numpy array
            self.dem_x = dem_data.x.values
            self.dem_y = dem_data.y.values
            self.dem_resolution = abs(dem_data.rio.resolution()[0])
            print(f"  ✓ DEM loaded: shape={self.dem.shape}")
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_dem_patch(self, station_lat, station_lon):
        """Extract DEM patch centered on station (on-the-fly)"""
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
    
    def _process_radar(self, radar_patch, radar_indices):
        """
        Process multi-field 2D radar with temporal awareness channels.

        Data layout (post dual-pol zarr, Z already collapsed):
            radar_patch : np.ndarray  (12, N_fields, H, W)
                dim 0 — 12 scan times per hour
                dim 1 — fields in FIELDS order: Z, ZDR, RhoHV, PhiDP, KDP
                dim 2-3 — spatial (H, W)

        For each field the function produces three 12-channel groups:
            • field values  : normalized to [0, 1] via FIELD_NORMS
            • validity mask : 1.0 = real scan, 0.0 = padded / missing
            • temporal pos  : scan index / (n_scans - 1)  →  0 … 1

        Returns:
            Tensor of shape (N_fields * 36, 256, 256)
            With all 5 dual-pol fields → (72, 256, 256)
        """
        n_scans  = radar_patch.shape[0]   # 12
        n_fields = radar_patch.shape[1]   # 5
        H, W     = radar_patch.shape[2], radar_patch.shape[3]
        field_channels = []
        # ── Per-field normalized values (5 × 12 = 60 channels) ──
        for field_name in self.FIELDS:
            pickle_idx = self.PICKLE_FIELD_ORDER.index(field_name)
            f_min, f_max = self.FIELD_NORMS[field_name]
            field_arr = radar_patch[:, pickle_idx, :, :].copy()
            field_arr[field_arr == -9999.0] = f_min
            field_arr = np.where(np.isnan(field_arr), f_min, field_arr)
            field_norm = (field_arr - f_min) / (f_max - f_min)
            field_norm = np.clip(field_norm, 0.0, 1.0)
            # Zero out missing scans
            for i, ridx in enumerate(radar_indices):
                if ridx is None:
                    field_norm[i] = 0.0
            field_channels.append(torch.from_numpy(field_norm).float())
        # ── Shared validity mask (12 channels) ──
        mask = np.ones((n_scans, H, W), dtype=np.float32)
        for i, ridx in enumerate(radar_indices):
            if ridx is None:
                mask[i] = 0.0
        field_channels.append(torch.from_numpy(mask))
        # ── Shared temporal position (12 channels) ──
        t_pos = torch.zeros((n_scans, H, W))
        for i in range(n_scans):
            t_pos[i] = i / max(n_scans - 1, 1)
        field_channels.append(t_pos)
        # ── Concatenate → (84, H, W) ──
        radar_all = torch.cat(field_channels, dim=0)
        # Upscale to 256×256
        radar_up = F.interpolate(
            radar_all.unsqueeze(0),
            size=(256, 256),
            mode='nearest',
        ).squeeze(0)
        return radar_up  # (84, 256, 256)
    
    def _process_dem(self, dem_patch):
        """Process DEM: NaN → resize to 256×256"""
        dem_arr = dem_patch.copy()
        
        if dem_arr.ndim == 2:
            dem_arr = dem_arr[np.newaxis, :, :]
        
        dem_arr = np.where(np.isnan(dem_arr), 0.0, dem_arr)
        dem_t = torch.from_numpy(dem_arr).float()
        
        # Resize to 256×256 (handles any input size)
        if dem_t.shape[-1] != 256 or dem_t.shape[-2] != 256:
            dem_t = F.interpolate(
                dem_t.unsqueeze(0), 
                size=(256, 256), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        return dem_t  # (1, 256, 256)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        radar_patch = s['radar_patch'].copy()  # (12, N_fields, 9, 9) — time × fields × y × x
        
        # === Random crop from 9×9 to 5×5 ===
        input_size = radar_patch.shape[-1]  # 9 for 9x9, 5 for legacy
        output_size = 5
        
        if input_size > output_size:
            max_offset = input_size - output_size  # = 4
            
            if self.augment and np.random.rand() < self.aug_prob:
                # Random crop during training
                offset_y = np.random.randint(0, max_offset + 1)
                offset_x = np.random.randint(0, max_offset + 1)
            else:
                # Center crop during validation
                offset_y = max_offset // 2
                offset_x = max_offset // 2
            
            # Where gauge ends up in the cropped 5×5
            center_pos = input_size // 2  # = 4 for 9×9
            gauge_y = center_pos - offset_y
            gauge_x = center_pos - offset_x
            
            # Crop radar from 9×9 to 5×5
            radar_patch = radar_patch[:, :, offset_y:offset_y+output_size, offset_x:offset_x+output_size]
        else:
            offset_y, offset_x = 0, 0
            gauge_y, gauge_x = 2, 2  # Center of 5×5
        
        radar_t = self._process_radar(radar_patch, s.get("radar_indices"))
        
        # Extract DEM on-the-fly with same offset
        if self.dem is not None:
            dem_patch_full = self._extract_dem_patch(s['station_lat'], s['station_lon'])
            # Downsample to input_size, then crop with same offset
            if input_size > output_size:
                dem_t_full = torch.from_numpy(dem_patch_full).float()
                dem_t_sized = F.interpolate(dem_t_full.unsqueeze(0), size=(input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
                dem_patch = dem_t_sized[:, offset_y:offset_y+output_size, offset_x:offset_x+output_size].numpy()
            else:
                dem_patch = dem_patch_full
        elif 'dem_patch' in s and s['dem_patch'] is not None:
            dem_patch = s['dem_patch']
        else:
            dem_patch = np.zeros((1, 264, 264))
        
        dem_t = self._process_dem(dem_patch)

        image = {
            "DEM":   dem_t,    # (1, 256, 256)
            "RADAR": radar_t,  # (N_fields * 36, 256, 256) — 180 channels with 5 dual-pol fields
        }

        target_value = np.log1p(s['hourly_precip_mm'])
    
        # Create 5x5 mask with target at gauge position
        mask = torch.full((5, 5), -9999.0, dtype=torch.float32)  # Ignore value
        mask[gauge_y, gauge_x] = target_value  # Target at actual gauge position!
        
        return {
            'image': image,    # ← TerraTorch expects this key!
            'mask': mask,
        }
    
    def plot(self, sample):
        return None

# filters
# def filter_biased_extremes(samples):
#     OVERESTIMATING_STATIONS = [
#     'Dangermond_Bunker Hill', 'Dangermond_Cistern', 'Dangermond_Cojo HQ', 'Dangermond_Jalachichi','Dangermond_Repeator'  
#     ]
#     UNDERESTIMATING_STATIONS = [
#         'Dangermond_Cojo Gate', 'Dangermond_Sutter'
#     ]

#     filtered = []
#     removed = []

#     for sample in samples:
#         station_name = sample['station_name']
#         target = sample['hourly_precip_mm']
#         radar = sample['radar_patch']
#         max_dbz = np.nanmax(radar[:, 0, :, :]) 
#         # Overestimating stations: Remove extreme high values
#         # (sensor flooding - can't distinguish droplet sizes)
#         if station_name in OVERESTIMATING_STATIONS:
#             if target > 25.0:  # Very heavy rain
#                 removed.append(f"{target:.1f}mm precipitation (overestimate)")
#                 continue
#              # Filter out
#             if max_dbz > 30.0 and target < 0.3:  # Radar detects storm but gauge reads low
#                 removed.append(f"{target:.1f}mm precipitation (underestimate)")
#                 continue
    
#         # Underestimating stations: Remove cases where radar says heavy but gauge says light
#         # (sensor saturated/clogged)
#         if station_name in UNDERESTIMATING_STATIONS:
#             if max_dbz > 30.0 and target < 0.3:  # Radar detects storm but gauge reads low
#                 removed.append(f"{target:.1f}mm precipitation (underestimate)")
#                 continue
#                  # Filter out
#         filtered.append(sample)

#     print(f"✓ Filtered {len(removed)} samples")
    
#     return filtered

def filter_bad_samples(samples):
    """Filter bad samples including ground clutter"""
    filtered = []
    removed = []
    
    for sample in samples:
        target = sample['hourly_precip_mm']
        radar = sample['radar_patch']
        max_dbz = np.nanmax(radar[:, 0, :, :]) 
        if np.isnan(max_dbz):
            removed.append("all-NaN reflectivity")
            continue

        if target > 50.0:
            removed.append(f"{target:.1f}mm precipitation (sensor error)")
            continue
        
        # EXISTING: Remove radar-gauge mismatch (high rain, low radar)
        if target > 5.0 and max_dbz < 20.0:
            removed.append(f"{target:.1f}mm @ {max_dbz:.1f}dBZ (radar miss)")
            continue

        ## these are taking max_dbz but with a grid of 9x9 the max could not be hitting the sensor!!

        # If dBZ > 50 but rain < 2mm, it's likely clutter
        # if max_dbz > 50.0 and target < 2.0:
        #     removed.append(f"{target:.1f}mm @ {max_dbz:.1f}dBZ (ground clutter)")
        #     continue
        
        # # NEW: Remove extreme dBZ (likely always clutter)
        # if max_dbz > 60.0:
        #     removed.append(f"{target:.1f}mm @ {max_dbz:.1f}dBZ (extreme dBZ)")
        #     continue
            
        filtered.append(sample)
    
    print(f"✓ Filtered {len(removed)} samples")
    
    
    return filtered

def filter_suspect_station_days(samples):
    """
    Filter out samples from station-days where the station recorded ZERO
    but other stations had significant rain (likely sensor issue).
    
    Keep samples if:
    1. Station had SOME rain that day (sensor was working), OR
    2. All stations had low rain that day (genuinely dry)
    """
    # First pass: calculate daily totals per station
    daily_totals = {}
    for sample in samples:
        station = sample.get('station_name', 'Unknown')
        hour_str = str(sample.get('hour_start', ''))
        date = hour_str[:10]
        precip = sample['hourly_precip_mm']
        
        key = (station, date)
        if key not in daily_totals:
            daily_totals[key] = 0
        daily_totals[key] += precip
    
    # Calculate daily network average (excluding each station)
    date_totals = {}
    for (station, date), total in daily_totals.items():
        if date not in date_totals:
            date_totals[date] = []
        date_totals[date].append((station, total))
    
    # Identify suspect station-days
    suspect_station_days = set()
    for date, station_data in date_totals.items():
        for station, total in station_data:
            if total == 0:  # Station had zero all day
                # Calculate average of OTHER stations
                others = [t for s, t in station_data if s != station]
                others_with_rain = sum(1 for t in others if t > 2.0)
                if len(others) >= 5 and others_with_rain >= 9 and np.mean(others) > 15:  # Others had significant rain
                    suspect_station_days.add((station, date))
    
    print(f"Identified {len(suspect_station_days)} suspect station-days")

    # Filter samples
    filtered = []
    removed_count = 0
    
    for sample in samples:
        station = sample.get('station_name', 'Unknown')
        hour_str = str(sample.get('hour_start', ''))
        date = hour_str[:10]
        
        if (station, date) in suspect_station_days:
            removed_count += 1
            continue
        
        filtered.append(sample)
    
    print(f"✓ Removed {removed_count} samples from suspect station-days")
    
    # Show which station-days were removed
    station_day_counts = {}
    for station, date in suspect_station_days:
        short = station.replace('Dangermond_', '')
        if short not in station_day_counts:
            station_day_counts[short] = []
        station_day_counts[short].append(date)
    
    return filtered

# Samplers
def create_heavy_rain_sampler(samples):
    """
    Heavily oversample the rare heavy rain events.
    With only 4.5% heavy rain, we need aggressive weights!
    """
    targets = np.array([s['hourly_precip_mm'] for s in samples])
    
    # Create weights based on precipitation intensity
    weights = np.ones(len(targets))
    
    # Zero/trace rain: undersample slightly
    weights[targets < 0.1] = 0.5
    
    # Light rain (0.1-2mm): normal weight
    weights[(targets >= 0.1) & (targets < 2)] = 1.0
    
    # Moderate rain (2-5mm): slight oversample
    weights[(targets >= 2) & (targets < 5)] = 2.0
    
    # Heavy rain (5-15mm): moderate oversample
    weights[(targets >= 5) & (targets < 15)] = 5.0
    
    # Very heavy rain (15-40mm): heavier oversample
    weights[(targets >= 15) & (targets <= 40)] = 10.0
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    # Report effective distribution
    effective_pct = weights / weights.sum() * 100
    print(f"\n{'='*50}")
    print(f"WEIGHTED SAMPLER CREATED")
    print(f"{'='*50}")
    print(f"Category          | Original | Effective")
    print(f"------------------|----------|----------")
    print(f"Zero rain         | {(targets==0).mean()*100:5.1f}%   | {effective_pct[targets==0].sum():5.1f}%")
    print(f"Light (0-2mm)     | {((targets>0)&(targets<2)).mean()*100:5.1f}%   | {effective_pct[(targets>0)&(targets<2)].sum():5.1f}%")
    print(f"Moderate (2-5mm)  | {((targets>=2)&(targets<5)).mean()*100:5.1f}%   | {effective_pct[(targets>=2)&(targets<5)].sum():5.1f}%")
    print(f"Heavy (5-10mm)    | {((targets>=5)&(targets<10)).mean()*100:5.1f}%   | {effective_pct[(targets>=5)&(targets<10)].sum():5.1f}%")
    print(f"V.Heavy (>10mm)   | {(targets>=10).mean()*100:5.1f}%   | {effective_pct[targets>=10].sum():5.1f}%")
    print(f"{'='*50}")
    
    return sampler


# Data Module
class RadarDEMDataModule(L.LightningDataModule):
    def __init__(self, pickle_path: str, weight_sampler=None, batch_size=32, num_workers=0):
        super().__init__()
        self.pickle_path = pickle_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_samples = None
        self.val_samples = None
        
        self.weight_sampler = weight_sampler
        self.train_sampler = None

    def prepare_data(self):
        # no heavy downloads
        pass

    def setup(self, stage=None):

        # load data from pickle
        data = pickle.load(open(self.pickle_path, "rb"))
        train_samples, val_samples = data["train"], data["val"]
        
        # filter samples that may have sensor error or radar issues
        # train_samples = filter_biased_extremes(train_samples) 
        # val_samples = filter_biased_extremes(val_samples)

        train_samples = filter_bad_samples(train_samples) 
        val_samples = filter_bad_samples(val_samples)

        train_samples = filter_suspect_station_days(train_samples) 
        val_samples = filter_suspect_station_days(val_samples)

        self.train_ds = RadarDEMDataset(train_samples, dem_path='dem/preserve_dem_10m_utm.tif', patch_size_m=4620, augment=True, aug_prob=0.5)
        self.val_ds = RadarDEMDataset(val_samples, patch_size_m=4620, augment=False)

        self.val_dataset = self.val_ds
        self.train_dataset = self.train_ds

        # add a weight sampler to training data if configured
        if self.weight_sampler is not None:
            self.train_sampler = self.weight_sampler(train_samples)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=self.train_sampler, shuffle=self.train_sampler is None, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        