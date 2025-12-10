"""
Constants for radar data processing and ML pipeline
"""

# Radar data handling
RADAR_MISSING_VALUE = -9999.0  # Sentinel value for missing/invalid data
                                # Standard geospatial convention for no-data
                                # Real reflectivity ranges from ~-30 to +70 dBZ
                                # -9999 is clearly outside physical range
                                # Compatible with PyTorch/Terratorch ignore_index

# Reflectivity (dBZ) typical ranges
DBZ_MIN = -30.0  # Weak echoes, noise, insects
DBZ_MAX = 70.0   # Extreme precipitation (hail)
DBZ_LIGHT_RAIN = 0.0
DBZ_MODERATE_RAIN = 30.0
DBZ_HEAVY_RAIN = 45.0

# Spatial parameters
TERRAMESH_PATCH_SIZE = 264  # pixels
S2_RESOLUTION = 10.0        # meters per pixel
RADAR_RESOLUTION_500M = 500.0  # meters per pixel (coarse)
RADAR_RESOLUTION_10M = 10.0    # meters per pixel (fine, interpolated)

# Vertical levels
Z_LEVELS = 40  # Number of altitude levels in radar data
Z_MIN = 0.0    # meters
Z_MAX = 15000.0  # meters
Z_RES = 375.0  # meters

# Temporal parameters
RADAR_SCAN_INTERVAL = 300  # seconds (~5 minutes between scans)

# Data preprocessing
def is_valid_reflectivity(value):
    """Check if a reflectivity value is physically valid"""
    if value == RADAR_MISSING_VALUE:
        return False
    return DBZ_MIN <= value <= DBZ_MAX

def normalize_reflectivity(dbz, method='minmax'):
    """
    Normalize reflectivity for neural network input
    
    Parameters:
    -----------
    dbz : array-like
        Reflectivity in dBZ (may contain RADAR_MISSING_VALUE)
    method : str
        'minmax': scale to [0, 1]
        'zscore': standardize to mean=0, std=1
        'none': no normalization
    
    Returns:
    --------
    normalized : array
        Normalized values, missing data preserved
    """
    import numpy as np
    
    # Create mask for valid data
    valid_mask = (dbz != RADAR_MISSING_VALUE)
    
    if method == 'minmax':
        # Scale to [0, 1] based on physical range
        normalized = (dbz - DBZ_MIN) / (DBZ_MAX - DBZ_MIN)
        # Restore missing value indicator
        normalized[~valid_mask] = -1.0  # Use -1 as normalized missing value
        
    elif method == 'zscore':
        # Standardize based on valid data
        if valid_mask.any():
            mean = dbz[valid_mask].mean()
            std = dbz[valid_mask].std()
            normalized = (dbz - mean) / (std + 1e-8)
            normalized[~valid_mask] = RADAR_MISSING_VALUE
        else:
            normalized = dbz.copy()
            
    else:  # 'none'
        normalized = dbz.copy()
    
    return normalized


# Model hyperparameters (to be tuned)
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-4

