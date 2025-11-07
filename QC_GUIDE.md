# Quality Control & Advanced Processing Guide

## Current Status ✅❌

| Feature | Status | What It Does |
|---------|--------|--------------|
| Pull NEXRAD data | ✅ Done | Download Level-II volumes from AWS |
| Build zarr store | ✅ Done | Store gridded data locally |
| Grid to 500m | ✅ Done | Cartesian grid at 500m resolution |
| Dual-pol QC | ❌ Missing | Remove non-meteorological echoes |
| PBB masks | ❌ Missing | Account for terrain blocking radar beam |
| Column features | ❌ Missing | VIL, echo tops, max reflectivity |

---

## 1. Dual-Polarization QC (Quality Control)

### What Is It?
Uses additional radar variables to filter out "junk" echoes.

### The Problem:
Your current data includes:
- ✅ Real precipitation
- ❌ Ground clutter (buildings, mountains)
- ❌ Biological scatterers (birds, insects)  
- ❌ Anomalous propagation (weird atmospheric conditions)

### The Solution:
Use **correlation coefficient (RhoHV)** to identify real weather:
- RhoHV > 0.95: Definitely rain ✅
- RhoHV < 0.90: Probably not rain ❌
- 0.90-0.95: Maybe rain 🤔

### To Implement:
```python
# In your pull_nexrad.py, change:
FIELD = 'reflectivity'

# To:
FIELDS = ['reflectivity', 'cross_correlation_ratio']

# Then in grid_radar_file_from_s3():
from radar.qc_and_features import apply_dualpol_qc
radar = apply_dualpol_qc(radar)  # Add this before gridding
```

---

## 2. PBB Masks (Partial Beam Blockage)

### What Is It?
Identifies where terrain partially blocks the radar beam.

### The Problem:
```
         Radar beam →  ╱╲  ← Mountain
                      ╱  ╲
Your Preserve  →  🌧️      ← Underestimated rainfall!
```

The radar beam hits the mountain before reaching your preserve, causing **underestimation** of rainfall.

### The Solution:
Calculate how much the beam is blocked using:
- Radar location (you have this)
- Terrain elevation data (DEM - you need to download)
- Beam geometry

### To Implement:
```bash
# 1. Install wradlib
pip install wradlib

# 2. Download DEM for your area
# Go to: https://earthexplorer.usgs.gov/
# Search: Vandenberg area (34.8°N, 120.4°W)
# Download: SRTM 1 Arc-Second Global elevation data

# 3. Use in your code
from radar.qc_and_features import calculate_beam_blockage
pbb = calculate_beam_blockage(
    radar_lat=34.83855, 
    radar_lon=-120.397917,
    radar_alt=376,  # meters above sea level (look this up!)
    dem_file='path/to/your/dem.tif'
)
```

**For Vandenberg:** The radar is on the coast with relatively flat terrain, so beam blockage is probably **minimal** for your preserve area. You might not need this!

---

## 3. Column Features

### What Is It?
Summarizes 3D radar data into useful 2D metrics.

### Features You Can Calculate:

**Column Maximum Reflectivity:**
- Max dBZ in the vertical column
- Useful for severe weather detection

**Echo Top Height:**
- Highest altitude with precipitation
- Taller = more intense convection

**VIL (Vertically Integrated Liquid):**
- Total liquid water in the column (kg/m²)
- Better rain rate estimator than reflectivity alone
- VIL > 30 kg/m² = heavy rain

**Column Mean:**
- Average reflectivity through the column

### To Use:
```python
from radar.qc_and_features import calculate_column_features

# Calculate features for all time steps
features = calculate_column_features('KVBX_preserve_500m.zarr')

# Or just one time step
features = calculate_column_features('KVBX_preserve_500m.zarr', time_idx=0)

# Access features
vil = features['vil']  # Vertically integrated liquid
echo_tops = features['echo_top_height']
max_ref = features['column_max_reflectivity']
```

---

## Priority Recommendations

### **High Priority:**
1. **Dual-pol QC** - Easy to add, big quality improvement
2. **Column features** - Already have the code, just run it!

### **Medium Priority:**
3. **PBB masks** - Only if terrain is an issue (check if mountains block your line of sight)

### **Next Steps:**
1. Re-run your pull script with dual-pol QC enabled
2. Calculate column features from your existing zarr
3. Compare QC'd vs. non-QC'd data to see the difference

---

## Testing QC Impact

Create a simple comparison script to see what QC removes:

```python
# Compare before/after QC
radar_raw = pyart.io.read_nexrad_archive(file)
radar_qc = apply_dualpol_qc(radar_raw.copy())

# Plot difference
fig, (ax1, ax2) = plt.subplots(1, 2)
# Plot raw on ax1
# Plot QC'd on ax2
# See what got removed!
```

You'll likely see QC removes ground clutter near the radar and weird echoes at range edges.

