"""
Quality Control and Feature Extraction for NEXRAD Data

This module adds:
1. Dual-polarization QC
2. Beam blockage masks
3. Column-integrated features
"""

import numpy as np
import pyart
import xarray as xr


# =============================================================================
# 1. DUAL-POLARIZATION QUALITY CONTROL
# =============================================================================

def apply_dualpol_qc(radar, fields_to_qc=['reflectivity']):
    """
    Apply dual-polarization quality control to remove non-meteorological echoes
    
    Uses correlation coefficient (RhoHV) to identify and mask:
    - Ground clutter
    - Biological scatterers (birds, insects)
    - Anomalous propagation
    
    Parameters:
    -----------
    radar : pyart.core.Radar
        PyART radar object
    fields_to_qc : list
        List of field names to apply QC masks to
        
    Returns:
    --------
    radar : pyart.core.Radar
        Radar object with QC applied
    """
    
    # Check if dual-pol fields are available
    if 'cross_correlation_ratio' not in radar.fields:
        print("Warning: RhoHV not available - skipping dual-pol QC")
        return radar
    
    # Get correlation coefficient (RhoHV)
    rhohv = radar.fields['cross_correlation_ratio']['data']
    
    # Create QC mask: RhoHV < 0.9 is likely non-meteorological
    qc_mask = rhohv < 0.9
    
    # Apply mask to requested fields
    for field_name in fields_to_qc:
        if field_name in radar.fields:
            data = radar.fields[field_name]['data']
            # Mask out bad data
            data = np.ma.masked_where(qc_mask, data)
            radar.fields[field_name]['data'] = data
    
    n_masked = np.sum(qc_mask)
    n_total = qc_mask.size
    print(f"Dual-pol QC: Masked {n_masked}/{n_total} ({100*n_masked/n_total:.1f}%) gates")
    
    return radar


def apply_texture_qc(radar, fields_to_qc=['reflectivity'], window_size=5):
    """
    Apply texture-based QC to remove speckle and isolated echoes
    
    Calculates spatial standard deviation - meteorological echoes are smoother
    than clutter/noise.
    
    Parameters:
    -----------
    radar : pyart.core.Radar
        PyART radar object
    fields_to_qc : list
        List of field names to apply QC
    window_size : int
        Size of window for texture calculation (default: 5 gates)
    """
    # This is more advanced - would use scipy.ndimage filters
    # Placeholder for now
    print("Texture-based QC not yet implemented")
    return radar


# =============================================================================
# 2. BEAM BLOCKAGE CALCULATION
# =============================================================================

def calculate_beam_blockage(radar_lat, radar_lon, radar_alt, 
                            dem_file=None, max_range_km=250):
    """
    Calculate partial beam blockage (PBB) using terrain data
    
    Requires wradlib and a Digital Elevation Model (DEM)
    
    Parameters:
    -----------
    radar_lat : float
        Radar latitude
    radar_lon : float  
        Radar longitude
    radar_alt : float
        Radar altitude (m above sea level)
    dem_file : str
        Path to DEM file (GeoTIFF)
    max_range_km : float
        Maximum range to calculate blockage (km)
        
    Returns:
    --------
    pbb : np.ndarray
        Beam blockage fraction (0-1) for each gate
        
    Note:
    -----
    This requires:
    - pip install wradlib
    - DEM data (e.g., SRTM 30m from USGS Earth Explorer)
    """
    
    try:
        import wradlib as wrl
    except ImportError:
        print("Error: wradlib not installed. Install with: pip install wradlib")
        return None
    
    if dem_file is None:
        print("Warning: No DEM file provided - cannot calculate beam blockage")
        print("Download SRTM DEM from: https://earthexplorer.usgs.gov/")
        return None
    
    # Load DEM
    # dem_data = wrl.io.open_raster(dem_file)
    
    # Calculate beam blockage using wradlib
    # This is complex - requires setting up radar beam geometry
    # and computing intersection with terrain
    
    print("Beam blockage calculation not fully implemented yet")
    print("Requires:")
    print("  1. pip install wradlib")
    print("  2. Download SRTM DEM for your area")
    print("  3. Implement wrl.qual.beam_block_frac()")
    
    return None


# =============================================================================
# 3. COLUMN-INTEGRATED FEATURES
# =============================================================================

def calculate_column_features(zarr_path, time_idx=None):
    """
    Calculate column-integrated features from 3D reflectivity
    
    Features:
    - Column maximum reflectivity
    - Echo top height (max height with Z > threshold)
    - Vertically Integrated Liquid (VIL)
    - Mean reflectivity in column
    
    Parameters:
    -----------
    zarr_path : str
        Path to zarr file
    time_idx : int or None
        Time index to process (None = all times)
        
    Returns:
    --------
    features : xarray.Dataset
        Dataset with column features
    """
    
    import zarr
    
    print("Loading zarr data...")
    store = zarr.open(zarr_path, mode='r')
    
    ref_array = store['reflectivity']
    z_array = store['z'][:]
    
    if time_idx is not None:
        data = ref_array[time_idx, :, :, :]
        times = [time_idx]
    else:
        data = ref_array[:]
        times = range(ref_array.shape[0])
    
    print(f"Calculating column features for {len(times)} time steps...")
    
    # Initialize feature arrays
    if time_idx is not None:
        n_times, n_z, n_y, n_x = 1, *data.shape
        data = data[None, ...]
    else:
        n_times, n_z, n_y, n_x = data.shape
    
    col_max = np.zeros((n_times, n_y, n_x))
    echo_top = np.zeros((n_times, n_y, n_x))
    vil = np.zeros((n_times, n_y, n_x))
    col_mean = np.zeros((n_times, n_y, n_x))
    
    # Process each time step
    for t in range(n_times):
        time_slice = data[t, :, :, :]  # (z, y, x)
        
        # Column maximum
        col_max[t] = np.nanmax(time_slice, axis=0)
        
        # Echo top height (max height where Z > 0 dBZ)
        valid_echo = time_slice > 0  # Boolean mask
        # For each column, find highest level with valid echo
        for i in range(n_y):
            for j in range(n_x):
                if np.any(valid_echo[:, i, j]):
                    highest_idx = np.where(valid_echo[:, i, j])[0][-1]
                    echo_top[t, i, j] = z_array[highest_idx]
                else:
                    echo_top[t, i, j] = np.nan
        
        # VIL (simplified - proper calc requires density correction)
        # VIL ≈ 3.44e-6 * sum(Z^(4/7) * dz)  [kg/m²]
        z_linear = 10 ** (time_slice / 10)  # Convert dBZ to linear Z
        dz = np.diff(z_array)[0] if len(z_array) > 1 else 375  # vertical spacing
        vil[t] = 3.44e-6 * np.nansum(z_linear ** (4/7) * dz, axis=0)
        
        # Column mean
        col_mean[t] = np.nanmean(time_slice, axis=0)
    
    print("Column features calculated!")
    print(f"  Max reflectivity range: {np.nanmin(col_max):.1f} to {np.nanmax(col_max):.1f} dBZ")
    print(f"  Echo top range: {np.nanmin(echo_top):.0f} to {np.nanmax(echo_top):.0f} m")
    print(f"  VIL range: {np.nanmin(vil):.1f} to {np.nanmax(vil):.1f} kg/m²")
    
    # Return as xarray Dataset
    features = xr.Dataset({
        'column_max_reflectivity': (['time', 'y', 'x'], col_max),
        'echo_top_height': (['time', 'y', 'x'], echo_top),
        'vil': (['time', 'y', 'x'], vil),
        'column_mean_reflectivity': (['time', 'y', 'x'], col_mean),
    })
    
    return features


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("NEXRAD Quality Control and Feature Extraction")
    print("=" * 60)
    
    print("\n1. To add dual-pol QC to your pull_nexrad.py:")
    print("   - Import additional fields: 'cross_correlation_ratio', 'differential_reflectivity'")
    print("   - Call apply_dualpol_qc(radar) before gridding")
    
    print("\n2. To calculate beam blockage:")
    print("   - Install wradlib: pip install wradlib")
    print("   - Download SRTM DEM from https://earthexplorer.usgs.gov/")
    print("   - Call calculate_beam_blockage() with DEM file")
    
    print("\n3. To calculate column features:")
    print("   - Call calculate_column_features('KVBX_preserve_500m.zarr')")
    print("   - Returns dataset with VIL, echo tops, etc.")
    
    # Example: Calculate column features from existing zarr
    try:
        features = calculate_column_features('KVBX_preserve_500m.zarr', time_idx=0)
        print("\n✅ Successfully calculated column features!")
    except Exception as e:
        print(f"\n⚠️  Could not calculate features: {e}")

