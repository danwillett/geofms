"""
Quick diagnostic script to check zarr file contents
"""
import zarr
import numpy as np

def diagnose_zarr(zarr_path='KVBX_preserve_500m.zarr'):
    """Check what's in the zarr file"""
    
    print(f"Opening {zarr_path}...")
    store = zarr.open(zarr_path, mode='r')
    
    print(f"\nZarr structure:")
    print(f"  Groups/Arrays: {list(store.keys())}")
    
    # Check reflectivity
    if 'reflectivity' in store:
        ref = store['reflectivity']
        print(f"\nReflectivity array:")
        print(f"  Shape: {ref.shape}")
        print(f"  Dtype: {ref.dtype}")
        print(f"  Chunks: {ref.chunks}")
        
        # Check metadata
        if hasattr(ref, 'attrs'):
            print(f"\n  Attributes:")
            for key, val in ref.attrs.items():
                print(f"    {key}: {val}")
        
        # Check a slice of data
        print(f"\nSampling data...")
        
        # Try to load first time step
        try:
            first_slice = ref[0, :, :, :]
            n_valid = np.sum(~np.isnan(first_slice))
            n_total = first_slice.size
            
            print(f"  Time step 0:")
            print(f"    Valid values: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
            if n_valid > 0:
                print(f"    Data range: [{np.nanmin(first_slice):.2f}, {np.nanmax(first_slice):.2f}]")
            else:
                print(f"    ⚠️  All NaN!")
                
        except Exception as e:
            print(f"  Error reading data: {e}")
    
    # Check coordinates
    for coord in ['time', 'z', 'y', 'x']:
        if coord in store:
            arr = store[coord]
            data = arr[:]
            print(f"\n{coord} coordinate:")
            print(f"  Shape: {arr.shape}")
            print(f"  Range: [{data.min()}, {data.max()}]")
            if coord == 'time':
                print(f"  First: {data[0]}")
                print(f"  Last: {data[-1]}")

if __name__ == "__main__":
    diagnose_zarr()

