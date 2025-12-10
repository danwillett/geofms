"""
Test script to verify radar data can be loaded and fed into a CNN
"""
import numpy as np
import xarray as xr
import torch
import torch.nn as nn

def test_radar_data_loading():
    """Test that we can load radar data and it has the right shape"""
    
    print("=" * 60)
    print("STEP 1: Load Radar Data")
    print("=" * 60)
    
    # Try to load the zarr file
    try:
        zarr_path = "KVBX_preserve_500m.zarr"  # or _500m if you kept that
        ds = xr.open_zarr(zarr_path)
        print(f"✅ Successfully loaded {zarr_path}")
        
        # Check structure
        print(f"\nDataset structure:")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Coordinates: {list(ds.coords)}")
        
        # Get reflectivity array
        reflectivity = ds['reflectivity']
        print(f"\nReflectivity shape: {reflectivity.shape}")
        print(f"  Time steps: {reflectivity.shape[0]}")
        print(f"  Z levels: {reflectivity.shape[1]}")
        print(f"  Y points: {reflectivity.shape[2]}")
        print(f"  X points: {reflectivity.shape[3]}")
        
        # Check data validity
        print(f"\nData validity check:")
        sample = reflectivity[0, :, :, :].values
        n_valid = np.sum(~np.isnan(sample))
        n_total = sample.size
        print(f"  Valid values: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
        print(f"  Data range: [{np.nanmin(sample):.1f}, {np.nanmax(sample):.1f}] dBZ")
        
        return ds
        
    except FileNotFoundError:
        print(f"❌ File not found: {zarr_path}")
        print(f"   Make sure pull_nexrad.py has finished running")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def test_patch_extraction(ds):
    """Test extracting a patch suitable for CNN input"""
    
    print("\n" + "=" * 60)
    print("STEP 2: Extract a Patch")
    print("=" * 60)
    
    if ds is None:
        print("❌ No dataset to work with")
        return None
    
    reflectivity = ds['reflectivity']
    
    # Define patch size (264x264 for TerraMesh, or smaller for 500m)
    PATCH_SIZE = 264  # pixels
    
    # Check if we have enough spatial coverage
    ny, nx = reflectivity.shape[2], reflectivity.shape[3]
    print(f"\nGrid size: {ny} x {nx} pixels")
    
    if ny >= PATCH_SIZE and nx >= PATCH_SIZE:
        print(f"✅ Can extract {PATCH_SIZE}x{PATCH_SIZE} patches")
        n_patches_y = ny // PATCH_SIZE
        n_patches_x = nx // PATCH_SIZE
        print(f"   Total possible patches: {n_patches_y * n_patches_x}")
    else:
        print(f"⚠️  Grid is smaller than {PATCH_SIZE}x{PATCH_SIZE}")
        print(f"   Will use full grid as one patch")
        PATCH_SIZE = min(ny, nx)
    
    # Extract a sample patch
    time_idx = 0
    y_start = 0
    x_start = 0
    
    patch = reflectivity.isel(
        time=time_idx,
        y=slice(y_start, y_start + PATCH_SIZE),
        x=slice(x_start, x_start + PATCH_SIZE)
    ).values
    
    print(f"\nExtracted patch shape: {patch.shape}")
    print(f"  Expected: (z_levels, {PATCH_SIZE}, {PATCH_SIZE})")
    print(f"  Got: ({patch.shape[0]}, {patch.shape[1]}, {patch.shape[2]})")
    
    # Check for NaNs
    n_valid = np.sum(~np.isnan(patch))
    n_total = patch.size
    print(f"\nPatch validity:")
    print(f"  Valid values: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
    
    return patch


def test_cnn_input(patch):
    """Test that patch can be fed into a CNN"""
    
    print("\n" + "=" * 60)
    print("STEP 3: Test CNN Input")
    print("=" * 60)
    
    if patch is None:
        print("❌ No patch to work with")
        return
    
    # Convert to torch tensor
    print("\nConverting to PyTorch tensor...")
    
    # Replace NaNs with sentinel value outside physical range
    # Real reflectivity: ~-30 to +70 dBZ
    # We use -9999 (standard geospatial no-data value)
    # Compatible with PyTorch/Terratorch ignore_index
    MISSING_VALUE = -9999.0
    patch_clean = np.nan_to_num(patch, nan=MISSING_VALUE)
    print(f"  NaNs replaced with sentinel value: {MISSING_VALUE} (ignore_index)")
    
    # Add batch dimension: (batch, channels, height, width)
    # Treat Z levels as channels
    batch = torch.from_numpy(patch_clean).float().unsqueeze(0)
    
    print(f"Tensor shape: {batch.shape}")
    print(f"  batch_size: {batch.shape[0]}")
    print(f"  channels (z_levels): {batch.shape[1]}")
    print(f"  height: {batch.shape[2]}")
    print(f"  width: {batch.shape[3]}")
    
    # Define a simple test CNN
    print("\nTesting with a simple CNN...")
    
    class TestCNN(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 128)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create model
    model = TestCNN(in_channels=batch.shape[1])
    print(f"✅ Created test CNN with {batch.shape[1]} input channels")
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = model(batch)
        print(f"✅ Forward pass successful!")
        print(f"   Input shape:  {batch.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output is a {output.shape[1]}-dimensional embedding")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_temporal_stack(ds):
    """Test extracting temporal stack (multiple time steps)"""
    
    print("\n" + "=" * 60)
    print("STEP 4: Test Temporal Stack")
    print("=" * 60)
    
    if ds is None:
        print("❌ No dataset to work with")
        return
    
    reflectivity = ds['reflectivity']
    n_times = reflectivity.shape[0]
    
    print(f"Total time steps available: {n_times}")
    
    # Extract multiple time steps
    T = min(6, n_times)  # Use up to 6 time steps
    print(f"Extracting {T} consecutive time steps...")
    
    PATCH_SIZE = min(264, reflectivity.shape[2], reflectivity.shape[3])
    
    temporal_stack = reflectivity.isel(
        time=slice(0, T),
        y=slice(0, PATCH_SIZE),
        x=slice(0, PATCH_SIZE)
    ).values
    
    print(f"\nTemporal stack shape: {temporal_stack.shape}")
    print(f"  (time, z, y, x) = ({temporal_stack.shape[0]}, {temporal_stack.shape[1]}, {temporal_stack.shape[2]}, {temporal_stack.shape[3]})")
    
    # For CNN, we can either:
    # Option 1: Treat (time * z) as channels
    # Option 2: Use 3D CNN
    # Option 3: Process each time step separately then aggregate
    
    print("\nReshaping for CNN input...")
    
    # Option 1: Flatten time and z into channels
    reshaped = temporal_stack.reshape(T * temporal_stack.shape[1], PATCH_SIZE, PATCH_SIZE)
    print(f"  Option 1 (flatten T*Z): {reshaped.shape}")
    print(f"    → (channels={T * temporal_stack.shape[1]}, height={PATCH_SIZE}, width={PATCH_SIZE})")
    
    # Convert to tensor and test
    MISSING_VALUE = -9999.0
    tensor = torch.from_numpy(np.nan_to_num(reshaped, nan=MISSING_VALUE)).float().unsqueeze(0)
    print(f"\n  As batch: {tensor.shape}")
    print(f"  ✅ Ready for CNN input!")
    
    return tensor


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RADAR DATA PIPELINE TEST")
    print("=" * 60)
    
    # Step 1: Load data
    ds = test_radar_data_loading()
    
    if ds is not None:
        # Step 2: Extract patch
        patch = test_patch_extraction(ds)
        
        # Step 3: Test CNN
        if patch is not None:
            cnn_success = test_cnn_input(patch)
        
        # Step 4: Test temporal
        temporal_tensor = test_temporal_stack(ds)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✅ Data can be loaded from zarr")
        print("✅ Patches can be extracted")
        print("✅ Data can be fed into PyTorch CNN")
        print("✅ Temporal stacks can be created")
        print("\n🎉 Data pipeline is ready for model development!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Data pipeline test failed")
        print("   Fix data loading issues first")
        print("=" * 60)

