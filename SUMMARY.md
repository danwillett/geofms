# GeoFMS Model Development Summary

## Overview

This project develops precipitation prediction models that map NEXRAD dual-polarization radar data to rain gauge measurements at the Dangermond Preserve (central California coast). Multiple architectures have been explored, with the primary bottleneck being **data quality** rather than model capacity.

## Models

### GFM (TerraMind Foundation Model)
- Pre-trained vision transformer fine-tuned for precipitation regression
- Operates in log-space with pixelwise regression
- Limited by frozen backbone capacity and small effective receptive field

### Stack CNN (Hourly)
- 3-block CNN encoder (192→384→512) with adaptive pooling
- Processes 12 radar scans per hour (73 input channels: 4 fields×12 + masks + temporal position + DEM)
- Scalar or spatial map output; trained with various loss functions
- Best suited for structured experiment comparison

### Stack 10-min (Single Scan)
- Lightweight CNN (64→128→256, ~700K params) operating on a single radar snapshot
- 5 input channels (4 radar fields + DEM)
- Confirmed that temporal context helps: hourly R²≈0.28 vs 10-min R²≈0.20
- Valuable as a diagnostic tool for gauge quality analysis

### U-Net (Hourly) — Current Best
- Encoder-decoder with skip connections, 64 base filters
- Same 73-channel input as Stack CNN
- Trained in raw mm-space with Huber/weighted MAE loss
- Best daily test R² = 0.58 with optimized filtering

## Key Findings

### 1. Data Quality > Architecture

The single largest performance gain came from **filtering improvements**, not architecture changes:

| Change | Val R² | Daily Test R² |
|--------|--------|---------------|
| Baseline (blunt filters) | 0.289 | 0.406 |
| + dump_ratio filter (0.95, cross-station) | 0.301 | **0.580** |
| Radar-based filter (no caps) | 0.236 | -0.436 |

Removing ~145 gauge dump artifacts improved daily test R² by 17 percentage points. Conversely, removing all caps and letting the model see extreme (noisy) samples degraded performance significantly.

### 2. Gauge Dump Artifacts

Tipping bucket gauges can produce spurious readings when the mechanism sticks and releases. These manifest as:
- High precipitation in a single 10-min bin with zeros on either side
- High reported rainfall with low/moderate radar reflectivity
- Isolated spikes that no other nearby station corroborates

**Detection**: `dump_ratio` = max_10min_bin / hourly_total. Values ≥0.95 with no cross-station corroboration are likely artifacts.

### 3. Filter Strategy

The optimal approach uses **layered filtering**:
1. `filter_nan_radar` — remove samples with no valid radar data
2. `filter_biased_extremes` — station-specific caps for known-problematic gauges (blunt but effective regularization)
3. `filter_bad_samples` — remove physically implausible combinations (high rain + very low dBZ)
4. `filter_suspect_station_days` — remove station-days where gauge read zero while all neighbors had rain
5. `filter_gauge_dumps` — remove samples with dump_ratio≥0.95 AND no cross-station corroboration

A pure physics-based filter (`filter_radar_unsupported`) that replaces station-name caps was tested but performed worse — the blunt caps act as useful regularization preventing the model from chasing noisy extreme targets.

### 4. The Prediction Cap Problem

All models exhibit a prediction cap (~14-18 mm/hr max prediction). Root causes:
- **Loss function**: MAE/Huber optimizes toward the conditional median, which is conservative for heavy-tailed distributions
- **Information content**: 2D max reflectivity cannot distinguish storm intensity above ~45 dBZ (both 15 mm/hr and 40 mm/hr can produce similar 2D fields)
- **Sample rarity**: Extreme rainfall events are <1% of training data

### 5. Temporal Resolution

The 10-min single-scan experiment confirmed:
- Temporal context genuinely helps (12 scans > 1 scan)
- The prediction cap is NOT caused by temporal integration complexity
- 10-min gauge data is noisier (more dump artifacts, timing mismatches)
- The bottleneck is input information content, not model architecture

### 6. Feature Importance (Ablation)

From ablation studies, feature importance ranking:
1. **Temporal position** — most critical (encodes scan ordering)
2. **ZDR** (Differential Reflectivity) — strong precipitation signal
3. **RhoHV** (Cross-correlation) — rain/hail discrimination
4. **Validity mask** — tells model which scans are present
5. **Reflectivity** — base signal
6. **DEM** — minor contribution
7. **KDP** — minimal impact with current normalization

## Next Steps

### Vertical Structure Features (In Progress)
New zarr generation includes 5 derived 3D features computed before collapsing to 2D:
- **Echo Top Height** — highest altitude with Z≥18 dBZ
- **VIL** (Vertically Integrated Liquid) — total liquid content
- **Max Z Height** — altitude of peak reflectivity
- **Low-level Mean Reflectivity** — 0-2 km average
- **Column Depth Fraction** — fraction of levels with Z>10 dBZ

These should directly address the prediction cap by providing the physical information needed to distinguish shallow moderate rain from deep heavy rain at the same max dBZ.

## File Structure

```
dataset/
  create_pickle.py          # Hourly pickle with dump_ratio fields
  create_pickle_10min.py    # 10-min single-scan pickle

models/
  unet/                     # Current best model
    train.py                # --filter-mode blunt|radar
    evaluate.py             # Respects filter_mode from checkpoint
    ablation.py
    diagnose_outliers.py    # Investigate extreme samples
  stack/                    # Hourly CNN baseline
  stack_10min/              # 10-min temporal experiment
    diagnose_outliers.py    # Cross-station outlier analysis
  gfm/                      # Foundation model approach

weather/
  pull_weather.py           # Now returns dump_ratio, max_bin_mm, n_active_bins
```

## Reproducing Best Result

```bash
# Generate pickle with dump metrics
python -m dataset.create_pickle \
  --radar "radar/outputs/2d/dualpol_500m_2022-01-01_2026-04-04.zarr" \
  --days "weather/days/top_100_days_2022-01-01_2026-04-04.txt" \
  --dem "dem/preserve_dem_10m_utm.tif" \
  --train-years 2022 2024 2026 --val-years 2023 2025 \
  --patch-size 9500 --half-hour-offsets --include-test \
  --output "dataset/outputs/radar_gauge_dataset_with_offsets_9500.pkl"

# Train U-Net with blunt filters + gauge dump filter (best config)
python -m models.unet.run_unet \
  --mode all --loss weighted_mae --no-sampler \
  --run-name wmae_no_sampler_filter_95_cross
```
