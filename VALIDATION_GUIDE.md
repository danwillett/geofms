# NEXRAD Zarr Validation & Visualization Guide

## Overview

This guide covers two new tools:
1. **Validation Script** - Verify your zarr file structure and data quality
2. **Enhanced Visualization** - View radar data with weather station overlays

---

## 1. Validation Script

### What It Does

The validation script (`radar/validate_nexrad.py`) performs comprehensive checks:

✅ **Structure Checks:**
- Verifies all required arrays exist (reflectivity, time, z, y, x)
- Validates dimensions match expectations
- Checks coordinate ranges are reasonable

✅ **Data Quality Checks:**
- Reflectivity values within reasonable range (-30 to 80 dBZ)
- Data completeness (% of non-NaN values)
- Time coordinate format and range

✅ **Visual Diagnostics:**
- Data availability over time
- Vertical profile of coverage
- Reflectivity distribution histogram
- Sample horizontal slice

### How To Use

**Basic validation:**
```bash
python radar/validate_nexrad.py
```

**Specify different zarr file:**
```bash
python radar/validate_nexrad.py path/to/my_radar.zarr
```

**From Python:**
```python
from radar.validate_nexrad import validate_zarr_structure, plot_validation_summary

# Run validation checks
valid, report = validate_zarr_structure('KVBX_preserve_500m.zarr')

# Create diagnostic plots
plot_validation_summary('KVBX_preserve_500m.zarr')
```

### Expected Output

```
============================================================
NEXRAD ZARR VALIDATION REPORT
============================================================

📂 Opening: KVBX_preserve_500m.zarr
   ✅ File opened successfully

📊 Checking data arrays...
   ✅ reflectivity: Found
   ✅ time: Found
   ✅ z: Found
   ✅ y: Found
   ✅ x: Found

📏 Checking dimensions...
   Expected shape: (34064, 40, 27, 32)
   Actual shape:   (34064, 40, 27, 32)
   ✅ Dimensions match

   Dimensions:
     time: 34064 time steps
     z:    40 vertical levels
     y:    27 north-south grid points
     x:    32 east-west grid points

🌐 Checking coordinate ranges...
   Z (altitude):  188 to 14812 m
   Y (N-S):       -13250 to 12250 m
   X (E-W):       -15750 to 14750 m
   ✅ Z coordinates reasonable
   ✅ X/Y coordinates reasonable

🌧️  Checking reflectivity data...
   Sample (time=0):
     Valid points:    1,247 / 34,560 (3.6%)
     Reflectivity range: -5.5 to 45.2 dBZ
     Mean: 15.3 dBZ
   ✅ Reflectivity values reasonable

⏰ Checking time coordinate...
   Time range:
     First: 2022-10-01 00:05:10
     Last:  2025-05-12 23:51:51
   ✅ Time coordinate readable

📈 Data completeness check...
   Average data coverage: 3.2%
   ✅ Reasonable data coverage

============================================================
✅ VALIDATION PASSED - Zarr file looks good!
============================================================
```

### Validation Plots

The script generates `nexrad_validation.png` with 4 diagnostic plots:

1. **Data Availability Over Time** - Shows if any time periods are missing
2. **Vertical Profile** - Data coverage vs. altitude
3. **Reflectivity Distribution** - Histogram of dBZ values
4. **Sample Horizontal Slice** - Visual check of gridded data

---

## 2. Enhanced Visualization with Weather Stations

### What's New

The updated `visualize_nexrad.py` now:
- Queries your database for weather station locations
- Transforms lat/lon to radar coordinates
- Overlays rain gauge locations on radar imagery
- Labels stations on the first subplot

### How To Use

**Default - show stations:**
```python
from radar.visualize_nexrad import show_nexrad

# Shows 4 altitude slices with station overlay
show_nexrad()
```

**Custom altitudes with stations:**
```python
# Low-level slices
show_nexrad(altitudes=[500, 1000, 1500, 2000])

# Mid-level slices
show_nexrad(altitudes=[3000, 6000, 9000, 12000])
```

**Different time steps:**
```python
# First scan with stations
show_nexrad(time_idx=0)

# 100th scan
show_nexrad(time_idx=100)
```

**Without station overlay:**
```python
# Just radar data
show_nexrad(show_stations=False)
```

### Visual Elements

**Symbology:**
- 🔴 **Red Star** = KVBX Radar location (0,0)
- 🟢 **Green Triangles** = Rain Gauge stations
- 📝 **Labels** = Station names (first subplot only)

**Color Scale:**
- -10 to 60 dBZ
- Blue = Light rain
- Yellow/Orange = Moderate rain
- Red = Heavy rain
- Purple = Very heavy rain/hail

### Example Output

The visualization shows:
1. **Multiple altitude slices** (2x2 grid by default)
2. **Radar at center** (red star at 0,0)
3. **Rain gauges** (green triangles showing your weather stations)
4. **Station names** (labeled on first subplot)
5. **Legend** (showing what symbols mean)

---

## 3. Complete Workflow Example

### Step 1: Generate rainy days list
```python
from weather.pull_weather import save_rainy_days_list

save_rainy_days_list(
    filename='top_100_storms.txt',
    top_n=100,
    start_date='2022-10-01',
    end_date='2025-05-12',
    metric='max_hourly'
)
```

### Step 2: Pull NEXRAD data with QC
```python
from radar.pull_nexrad import pull_nexrad

pull_nexrad(
    day_filter_file='top_100_storms.txt',
    apply_qc=True  # Enable quality control
)
```

### Step 3: Validate the zarr
```bash
python radar/validate_nexrad.py KVBX_preserve_500m.zarr
```

### Step 4: Visualize with stations
```python
from radar.visualize_nexrad import show_nexrad

# Show a storm event
show_nexrad(time_idx=500, altitudes=[500, 1500, 3000, 6000])
```

---

## 4. Troubleshooting

### "No weather stations found"
**Cause:** Database connection issue or no stations with rainfall data

**Fix:**
```python
# Check database connection
from database.config import connect, create_session
from database.models import DendraStation

engine = connect()
session = create_session(engine)
stations = session.query(DendraStation).all()
print(f"Found {len(stations)} stations")
```

### "Validation failed - dimension mismatch"
**Cause:** Corrupted zarr file or interrupted write

**Fix:** Delete zarr and processed_files.txt, rerun pull_nexrad

### "Low data coverage warning"
**Possible causes:**
1. **Normal** - Clear weather days have mostly NaN (no rain)
2. **QC removed too much** - Try `apply_qc=False` to check
3. **Coverage issue** - Station too far from radar

**To investigate:**
```python
# Check a specific time step
from radar.visualize_nexrad import show_nexrad
show_nexrad(time_idx=100)  # Should see data if it's a rainy day
```

---

## 5. Data Quality Indicators

### Good Quality Signs:
✅ Reflectivity mostly -5 to 50 dBZ  
✅ Data coverage 2-10% (normal for selective storms)  
✅ Smooth patterns in horizontal slices  
✅ Weather stations within radar beam  

### Warning Signs:
⚠️ All reflectivity > 60 dBZ (check for hail or bright band)  
⚠️ Data coverage < 0.1% (very little data)  
⚠️ Blocky/pixelated patterns (gridding artifacts)  
⚠️ Stations far from radar (> 100 km)  

---

## 6. Quick Reference

| Task | Command |
|------|---------|
| Validate zarr | `python radar/validate_nexrad.py` |
| View with stations | `show_nexrad()` |
| View without stations | `show_nexrad(show_stations=False)` |
| Check specific time | `show_nexrad(time_idx=500)` |
| Custom altitudes | `show_nexrad(altitudes=[1000, 3000])` |
| Generate validation plots | `plot_validation_summary('file.zarr')` |

---

## 7. Next Steps

After validation:
1. ✅ Zarr looks good? → Proceed with analysis
2. ⚠️ Issues found? → Check validation plots for details
3. 📊 Ready for science? → Extract features using `qc_and_features.py`
4. 🎯 Want more? → Add beam blockage analysis (see QC_GUIDE.md)


