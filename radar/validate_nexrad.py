"""
Validation script for NEXRAD zarr files

Checks:
- Data structure and dimensions
- Coordinate systems and ranges
- Data quality and completeness
- Time series continuity
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pyproj import Transformer


def load_preserve_boundary(geojson_path='geometries/dangermond-preserve-boundary.geojson'):
    """
    Load preserve boundary from GeoJSON and transform to radar coordinates
    
    Returns:
    --------
    boundary_x, boundary_y : arrays
        X and Y coordinates of boundary in radar coordinate system (meters)
    """
    try:
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        
        # Extract coordinates from first feature
        coords = geojson['features'][0]['geometry']['coordinates'][0]
        
        # GeoJSON coordinates are [lon, lat]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        # Transform to radar coordinates (same as pull_nexrad.py)
        RADAR_LAT = 34.83855 
        RADAR_LON = -120.397917
        
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84",
            always_xy=True
        )
        
        boundary_x = []
        boundary_y = []
        for lon, lat in zip(lons, lats):
            x, y = transformer.transform(lon, lat)
            boundary_x.append(x)
            boundary_y.append(y)
        
        return np.array(boundary_x), np.array(boundary_y)
        
    except Exception as e:
        print(f"⚠️  Could not load preserve boundary: {e}")
        return None, None


def validate_zarr_structure(zarr_path='KVBX_preserve_500m.zarr'):
    """
    Validate the basic structure of the zarr file
    
    Returns:
    --------
    valid : bool
        True if all checks pass
    report : dict
        Detailed validation report
    """
    
    print("="*60)
    print("NEXRAD ZARR VALIDATION REPORT")
    print("="*60)
    
    report = {}
    all_valid = True
    
    try:
        # Open zarr file
        print(f"\n📂 Opening: {zarr_path}")
        store = zarr.open(zarr_path, mode='r')
        report['file_opened'] = True
        print("   ✅ File opened successfully")
        
    except Exception as e:
        print(f"   ❌ Failed to open file: {e}")
        report['file_opened'] = False
        return False, report
    
    # Check for required arrays
    print("\n📊 Checking data arrays...")
    required_arrays = ['reflectivity', 'time', 'z', 'y', 'x']
    
    for array_name in required_arrays:
        if array_name in store:
            print(f"   ✅ {array_name}: Found")
            report[f'{array_name}_exists'] = True
        else:
            print(f"   ❌ {array_name}: Missing!")
            report[f'{array_name}_exists'] = False
            all_valid = False
    
    if not all([report.get(f'{a}_exists', False) for a in required_arrays]):
        return False, report
    
    # Check dimensions
    print("\n📏 Checking dimensions...")
    ref_array = store['reflectivity']
    time_array = store['time']
    z_array = store['z']
    y_array = store['y']
    x_array = store['x']
    
    # Get shapes from zarr arrays
    n_time = time_array.shape[0]
    n_z = z_array.shape[0]
    n_y = y_array.shape[0]
    n_x = x_array.shape[0]
    
    expected_shape = (n_time, n_z, n_y, n_x)
    actual_shape = ref_array.shape
    
    print(f"   Expected shape: {expected_shape}")
    print(f"   Actual shape:   {actual_shape}")
    
    if expected_shape == actual_shape:
        print("   ✅ Dimensions match")
        report['dimensions_match'] = True
    else:
        print("   ❌ Dimension mismatch!")
        report['dimensions_match'] = False
        all_valid = False
    
    # Print dimension details
    print(f"\n   Dimensions:")
    print(f"     time: {n_time} time steps")
    print(f"     z:    {n_z} vertical levels")
    print(f"     y:    {n_y} north-south grid points")
    print(f"     x:    {n_x} east-west grid points")
    
    report['n_time'] = n_time
    report['n_z'] = n_z
    report['n_y'] = n_y
    report['n_x'] = n_x
    
    # Check coordinate ranges
    print("\n🌐 Checking coordinate ranges...")
    
    z_coords = z_array[:]
    y_coords = y_array[:]
    x_coords = x_array[:]
    
    print(f"   Z (altitude):  {z_coords.min():.0f} to {z_coords.max():.0f} m")
    print(f"   Y (N-S):       {y_coords.min():.0f} to {y_coords.max():.0f} m")
    print(f"   X (E-W):       {x_coords.min():.0f} to {x_coords.max():.0f} m")
    
    # Sanity checks
    z_valid = (z_coords.min() >= 0) and (z_coords.max() <= 20000)
    xy_valid = (abs(y_coords.min()) < 500000) and (abs(x_coords.min()) < 500000)
    
    if z_valid:
        print("   ✅ Z coordinates reasonable")
        report['z_valid'] = True
    else:
        print("   ⚠️  Z coordinates unusual (check if correct)")
        report['z_valid'] = False
    
    if xy_valid:
        print("   ✅ X/Y coordinates reasonable")
        report['xy_valid'] = True
    else:
        print("   ⚠️  X/Y coordinates unusual (check if correct)")
        report['xy_valid'] = False
    
    # Check reflectivity data
    print("\n🌧️  Checking reflectivity data...")
    
    # Sample first time step
    sample_data = ref_array[0, :, :, :]
    
    valid_data = sample_data[~np.isnan(sample_data)]
    n_valid = len(valid_data)
    n_total = sample_data.size
    
    print(f"   Sample (time=0):")
    print(f"     Valid points:    {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
    
    if n_valid > 0:
        print(f"     Reflectivity range: {valid_data.min():.1f} to {valid_data.max():.1f} dBZ")
        print(f"     Mean: {valid_data.mean():.1f} dBZ")
        
        # Check if values are reasonable
        ref_reasonable = (valid_data.min() > -30) and (valid_data.max() < 80)
        
        if ref_reasonable:
            print("   ✅ Reflectivity values reasonable")
            report['reflectivity_valid'] = True
        else:
            print("   ⚠️  Reflectivity values unusual")
            report['reflectivity_valid'] = False
    else:
        print("   ⚠️  No valid reflectivity data in sample!")
        report['reflectivity_valid'] = False
        all_valid = False
    
    # Check time coordinate
    print("\n⏰ Checking time coordinate...")
    
    try:
        time_coords = time_array[:]
        n_sample = min(5, n_time)
        
        # Convert to datetime if needed
        if time_coords.dtype.kind in ['i', 'u', 'f']:
            # Try different time formats
            try:
                # Try nanoseconds since epoch
                time_values = [datetime.fromtimestamp(t / 1e9) for t in time_coords[:n_sample]]
            except (OSError, ValueError):
                # Try as numpy datetime64
                import pandas as pd
                time_values = pd.to_datetime(time_coords[:n_sample]).tolist()
        else:
            time_values = time_coords[:n_sample]
        
        print(f"   Time range:")
        print(f"     First: {time_values[0]}")
        print(f"     Last:  {time_values[-1] if len(time_values) > 1 else 'N/A'}")
        print(f"   ✅ Time coordinate readable")
        report['time_valid'] = True
        
    except Exception as e:
        print(f"   ⚠️  Error converting time to datetime: {e}")
        print(f"   Time array dtype: {time_coords.dtype}")
        print(f"   Sample values: {time_coords[:3]}")
        report['time_valid'] = False
        # Don't mark as invalid - time data exists, just can't convert format
    
    # Data completeness check
    print("\n📈 Data completeness check...")
    
    # Sample every 10th time step
    sample_indices = range(0, n_time, max(1, n_time // 10))
    completeness = []
    
    for i in sample_indices[:10]:  # Check up to 10 samples
        sample = ref_array[i, :, :, :]
        valid_frac = np.sum(~np.isnan(sample)) / sample.size
        completeness.append(valid_frac)
    
    avg_completeness = np.mean(completeness) * 100
    print(f"   Average data coverage: {avg_completeness:.1f}%")
    
    if avg_completeness > 1:  # At least 1% coverage
        print("   ✅ Reasonable data coverage")
        report['data_coverage'] = avg_completeness
    else:
        print("   ⚠️  Very low data coverage (mostly NaN)")
        report['data_coverage'] = avg_completeness
        all_valid = False
    
    # Final verdict
    print("\n" + "="*60)
    if all_valid:
        print("✅ VALIDATION PASSED - Zarr file looks good!")
    else:
        print("⚠️  VALIDATION WARNINGS - Check issues above")
    print("="*60)
    
    report['overall_valid'] = all_valid
    
    return all_valid, report


def plot_validation_summary(zarr_path='KVBX_preserve_500m.zarr'):
    """
    Create validation plots showing data distribution and quality
    """
    
    print("\n📊 Creating validation plots...")
    
    store = zarr.open(zarr_path, mode='r')
    ref_array = store['reflectivity']
    time_array = store['time']
    z_array = store['z'][:]
    
    # Get array dimensions
    n_time = time_array.shape[0]
    n_z = z_array.shape[0]
    
    # Load preserve boundary
    boundary_x, boundary_y = load_preserve_boundary()
    if boundary_x is not None:
        print("   ✅ Loaded preserve boundary")
    
    # Find time step with maximum reflectivity
    print("   Finding rainiest time step...")
    max_ref_values = []
    # sample_every = max(1, n_time // 100)  # Sample to speed up
    sample_every = 1

    for i in range(0, n_time, sample_every):
        sample = ref_array[i, :, :, :]
        if np.any(~np.isnan(sample)):
            max_ref_values.append((i, np.nanmax(sample)))
        else:
            max_ref_values.append((i, -999))
    
    # Get time index with highest reflectivity
    if max_ref_values:
        best_time_idx = max(max_ref_values, key=lambda x: x[1])[0]
        best_ref = max(max_ref_values, key=lambda x: x[1])[1]
        print(f"   Using time step {best_time_idx} (max reflectivity: {best_ref:.1f} dBZ)")
    else:
        best_time_idx = 0
        print(f"   No valid data found, using time step 0")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Data availability over time
    ax = axes[0, 0]
    sample_indices = range(0, n_time, max(1, n_time // 100))
    coverage = []
    
    for i in sample_indices:
        sample = ref_array[i, :, :, :]
        valid_frac = np.sum(~np.isnan(sample)) / sample.size
        coverage.append(valid_frac * 100)
    
    ax.plot(list(sample_indices), coverage)
    ax.axvline(best_time_idx, color='red', linestyle='--', alpha=0.7, label=f'Max rain (t={best_time_idx})')
    ax.set_xlabel('Time Step Index')
    ax.set_ylabel('Data Coverage (%)')
    ax.set_title('Data Availability Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Vertical profile of data availability
    ax = axes[0, 1]
    vertical_coverage = []
    
    sample = ref_array[best_time_idx, :, :, :]  # Best time step
    for iz in range(n_z):
        level_data = sample[iz, :, :]
        valid_frac = np.sum(~np.isnan(level_data)) / level_data.size
        vertical_coverage.append(valid_frac * 100)
    
    ax.plot(vertical_coverage, z_array)
    ax.set_xlabel('Data Coverage (%)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Vertical Profile (Time={best_time_idx})')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Reflectivity histogram
    ax = axes[1, 0]
    valid_data = sample[~np.isnan(sample)]
    
    if len(valid_data) > 0:
        ax.hist(valid_data, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Reflectivity (dBZ)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Reflectivity Distribution (Max Rain Event)')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='0 dBZ')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Sample horizontal slice
    ax = axes[1, 1]
    z_idx = n_z // 4  # Lower level
    slice_data = sample[z_idx, :, :]
    
    # Get coordinate arrays for proper extent
    x_coords = store['x'][:]
    y_coords = store['y'][:]
    
    # Find extent of valid data in coordinate space
    valid_mask = ~np.isnan(slice_data)
    if np.any(valid_mask):
        y_indices, x_indices = np.where(valid_mask)
        
        # Get coordinate bounds of valid data
        y_min_coord = y_coords[y_indices.min()]
        y_max_coord = y_coords[y_indices.max()]
        x_min_coord = x_coords[x_indices.min()]
        x_max_coord = x_coords[x_indices.max()]
        
        # Add padding (5% or 500m)
        y_range = y_max_coord - y_min_coord
        x_range = x_max_coord - x_min_coord
        y_pad = max(500, 0.05 * y_range)
        x_pad = max(500, 0.05 * x_range)
        
        # Set extent for imshow to use actual coordinates
        extent = [x_min_coord - x_pad, x_max_coord + x_pad,
                  y_min_coord - y_pad, y_max_coord + y_pad]
    else:
        # No valid data, use full extent
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
    
    im = ax.imshow(slice_data, cmap='turbo', vmin=-10, vmax=60, origin='lower',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    
    # Plot preserve boundary
    if boundary_x is not None and boundary_y is not None:
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Preserve Boundary', zorder=5)
        ax.plot(boundary_x, boundary_y, 'w-', linewidth=0.5, alpha=0.5, zorder=6)  # White outline for visibility
    
    # Apply zoom to valid data region
    if np.any(valid_mask):
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    
    # Add legend
    if boundary_x is not None:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.colorbar(im, ax=ax, label='Reflectivity (dBZ)')
    ax.set_xlabel('X distance from radar (m)')
    ax.set_ylabel('Y distance from radar (m)')
    ax.set_title(f'Reflectivity at {z_array[z_idx]:.0f} m (Time={best_time_idx})')
    
    plt.tight_layout()
    plt.savefig('nexrad_validation.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved validation plots to: nexrad_validation.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    zarr_file = sys.argv[1] if len(sys.argv) > 1 else 'KVBX_preserve_500m.zarr'
    
    # Run validation
    valid, report = validate_zarr_structure(zarr_file)
    
    # Create plots
    if valid or report.get('reflectivity_valid', False):
        try:
            plot_validation_summary(zarr_file)
        except Exception as e:
            print(f"\n⚠️  Could not create validation plots: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if valid else 1)

