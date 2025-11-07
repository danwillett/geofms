"""
Diagnostic tool to investigate reflectivity values and QC effectiveness
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt


def diagnose_reflectivity_values(zarr_path='KVBX_preserve_500m.zarr'):
    """
    Investigate reflectivity values to understand if QC is working correctly
    or if there's just no rain in the data
    """
    
    print("="*60)
    print("REFLECTIVITY DIAGNOSTIC REPORT")
    print("="*60)
    
    store = zarr.open(zarr_path, mode='r')
    ref_array = store['reflectivity']
    
    n_time = ref_array.shape[0]
    
    print(f"\nAnalyzing {n_time} time steps...")
    
    # Statistics for each time step
    max_values = []
    mean_values = []
    percent_positive = []
    percent_valid = []
    
    for i in range(n_time):
        data = ref_array[i, :, :, :]
        valid = data[~np.isnan(data)]
        
        if len(valid) > 0:
            max_values.append(np.max(valid))
            mean_values.append(np.mean(valid))
            percent_positive.append(100 * np.sum(valid > 0) / len(valid))
            percent_valid.append(100 * len(valid) / data.size)
        else:
            max_values.append(np.nan)
            mean_values.append(np.nan)
            percent_positive.append(0)
            percent_valid.append(0)
    
    # Overall statistics
    max_values = np.array(max_values)
    mean_values = np.array(mean_values)
    percent_positive = np.array(percent_positive)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Maximum reflectivity ever: {np.nanmax(max_values):.1f} dBZ")
    print(f"   Time steps with valid data: {np.sum(~np.isnan(max_values))}/{n_time}")
    print(f"   Time steps with positive dBZ: {np.sum(max_values > 0)}/{n_time}")
    print(f"   Time steps with rain (>10 dBZ): {np.sum(max_values > 10)}/{n_time}")
    print(f"   Time steps with strong rain (>30 dBZ): {np.sum(max_values > 30)}/{n_time}")
    
    if np.nanmax(max_values) < 0:
        print(f"\n⚠️  WARNING: All reflectivity values are negative!")
        print(f"   This suggests:")
        print(f"   1. Quality control might be removing all real echoes")
        print(f"   2. Or there's genuinely no precipitation in these time steps")
        print(f"   3. Or the data processing has an issue")
        print(f"\n   Recommendation:")
        print(f"   - Try re-running pull_nexrad() with apply_qc=False")
        print(f"   - Verify you're processing days with actual rainfall")
        print(f"   - Check your rainy days filter file")
    
    elif np.nanmax(max_values) < 10:
        print(f"\n⚠️  WARNING: Maximum reflectivity is only {np.nanmax(max_values):.1f} dBZ")
        print(f"   This is very weak - possibly just noise or light drizzle")
        print(f"   Real rain events typically have 20-50 dBZ")
    
    elif np.sum(max_values > 30) < 5:
        print(f"\n⚠️  Low rain intensity detected")
        print(f"   Only {np.sum(max_values > 30)} time steps have strong echoes (>30 dBZ)")
        print(f"   You may want to process more rainy days")
    
    else:
        print(f"\n✅ Good range of reflectivity values detected!")
        print(f"   Data includes real precipitation events")
    
    # Find best examples
    if np.sum(max_values > 0) > 0:
        positive_indices = np.where(max_values > 0)[0]
        best_idx = positive_indices[np.argmax(max_values[positive_indices])]
        print(f"\n🌧️  Best rain event:")
        print(f"   Time step: {best_idx}")
        print(f"   Max reflectivity: {max_values[best_idx]:.1f} dBZ")
        print(f"   Mean reflectivity: {mean_values[best_idx]:.1f} dBZ")
        print(f"   % positive values: {percent_positive[best_idx]:.1f}%")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Max reflectivity over time
    ax = axes[0, 0]
    ax.plot(max_values, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', label='0 dBZ')
    ax.axhline(10, color='orange', linestyle='--', label='10 dBZ (light rain)')
    ax.axhline(30, color='green', linestyle='--', label='30 dBZ (moderate rain)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Maximum Reflectivity (dBZ)')
    ax.set_title('Maximum Reflectivity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean reflectivity over time
    ax = axes[0, 1]
    ax.plot(mean_values, alpha=0.7, color='purple')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Reflectivity (dBZ)')
    ax.set_title('Mean Reflectivity Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of max values
    ax = axes[1, 0]
    valid_max = max_values[~np.isnan(max_values)]
    if len(valid_max) > 0:
        ax.hist(valid_max, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='0 dBZ')
        ax.set_xlabel('Maximum Reflectivity (dBZ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Maximum Reflectivity')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Percent of positive values over time
    ax = axes[1, 1]
    ax.plot(percent_positive, alpha=0.7, color='green')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('% Positive Reflectivity')
    ax.set_title('Percentage of Positive dBZ Values')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reflectivity_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved diagnostic plots to: reflectivity_diagnostic.png")
    plt.show()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    zarr_file = sys.argv[1] if len(sys.argv) > 1 else 'KVBX_preserve_500m.zarr'
    diagnose_reflectivity_values(zarr_file)


