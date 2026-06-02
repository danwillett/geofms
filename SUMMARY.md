# GeoFMS Model Development Summary

## Overview

This project develops precipitation prediction models that map NEXRAD dual-polarization radar data to rain gauge measurements at the Dangermond Preserve (central California coast). Multiple architectures have been explored, with the primary bottleneck being **data quality** rather than model capacity.

## Models

### GFM (TerraMind Foundation Model)
- Pre-trained vision transformer fine-tuned for precipitation regression
- Operates in log-space with pixelwise regression
- Limited by frozen backbone capacity and small effective receptive field

### Stack CNN (Hourly)
- CNN encoder (192→384→512, configurable depth/width) processing 12 radar scans per hour
- Input channels scale with feature selection (e.g. 133 channels for 9 fields×12 + mask + temporal position + DEM)
- Originally used `AdaptiveAvgPool2d` → MLP decoder, which collapsed spatial structure and **hard-capped predictions**
- A `SpatialConvHead` option (convolutional decoder, no global pooling) was added and **resolved the cap**, allowing the model to predict higher values; wide + spatial head gave the strongest Stack results
- Best suited for structured experiment comparison

### Stack 10-min (Single Scan)
- Lightweight CNN (64→128→256, ~700K params) operating on a single radar snapshot
- 5 input channels (4 radar fields + DEM)
- Confirmed that temporal context helps: hourly R²≈0.28 vs 10-min R²≈0.20
- Valuable as a diagnostic tool for gauge quality analysis

### U-Net (Hourly) — Current Best
- Encoder-decoder with skip connections; configurable depth (`n_encoder_blocks`) and width presets (`shallow`/`normal`/`wide`/`extra_wide`)
- Input channels scale with feature selection (133 channels for the 9-field dual-pol + vertical set)
- Trainable in raw mm-space or log-space; weighted MAE / Huber loss
- Best daily test R² = 0.58–0.62 with optimized filtering
- Now ingests the 9-field dual-pol + vertical-structure set; adding vertical features helped modestly

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

With the 9-field dual-pol + vertical set, ablation (ΔR² when removed) ranks:
1. **Temporal position** — by far most critical (encodes scan ordering); removing it is catastrophic
2. **Max Z Height** — now one of the strongest single features (vertical structure matters)
3. **All dual-pol** (ZDR + RhoHV + KDP together) — large combined contribution
4. **RhoHV** (Cross-correlation) — rain/phase discrimination
5. **Validity mask** — tells model which scans are present
6. **ZDR**, **Reflectivity**, **Low-level Ref** — moderate
7. **DEM**, **VIL**, **Echo Top**, **Column Depth**, **KDP** — minor / near-zero individually

The vertical features (esp. `max_z_height`) earned real importance, validating the decision to add them.

### 7. Log-space vs. Raw mm-space

- **Stack** trains noticeably better in **log-space**; the same architecture in raw mm-space underperformed.
- **U-Net** differences between log and no-log are **modest** on validation R² (≈0.29–0.32 across normal/wide/xwide).
- ⚠️ **Metric caveat**: `val_R²` printed during training is computed in whatever space the model trains in. A log-space `val_R²` is **not** directly comparable to the mm-space R² in the evaluation charts (log-space R² looks higher because the compression tames the heavy tail). Always compare models in the same space — the evaluation/test charts (mm-space) are the apples-to-apples number.

### 8. Model Width

- Width presets (`normal`/`wide`/`extra_wide`) were wired in for both architectures.
- Wider was **not reliably better** for the U-Net — `extra_wide` sometimes scored *below* `wide` (e.g. val R² xwide 0.29 < wide 0.32), i.e. diminishing/negative returns and overfitting risk.
- For the Stack, **wide + spatial head** was the most promising combination.

### 9. Error Diagnostics: Two Distinct Failure Modes

Detailed per-sample diagnostics (`diagnose_overestimates.py`, `diagnose_underestimates.py`, `diagnose_overpredict_weather.py`) revealed the model fails in two physically opposite regimes:

**Overprediction (cold / bright-band) — small but clear:**
- ~54 validation samples where pred ≥8 mm but actual <5 mm
- Occur at **cold** surface temps (~6 °C), with **depressed RhoHV** (87% <0.97 vs 56% for correct), **low max-Z height** (~736 m), slightly elevated ZDR
- Combined flag (RhoHV<0.97 AND max-Z<2500 m) → **4.5× higher** overprediction rate
- Interpretation: **melting-layer / bright-band contamination** — on cold days the freezing level (and bright band) sits low, the radar over-reads reflectivity from melting hydrometeors, and the gauge catches less at the surface

**Underprediction (warm / atmospheric-river) — the bigger problem:**
- **68% of all heavy-rain hours (271/398, actual >5 mm) underpredicted by >25%**; median pred/actual ratio ≈ **0.40**
- Pervasive across all 15 stations and 46 storm days, but with coherent multi-station AR events (e.g. confirmed atmospheric river on 2025-11-14)
- Warm-rain signatures: **low reflectivity-per-rainrate** (3.97 vs 5.57 dBZ per mm/hr), **shallower echo tops** (3.8 vs 4.5 km), **lower ZDR** (small drops), **higher RhoHV** (pure liquid), slightly **warmer** surface temps (12.4 vs 11.6 °C)
- The current KDP — collocated at the height of max reflectivity (often aloft) — pointed the "wrong" way, a strong hint that we are **measuring dual-pol at the wrong height** for surface QPE
- Interpretation: **warm-rain / orographic (AR) underestimation** — shallow, small-drop, high-liquid rain that S-band Z–R systematically under-reads

### 10. The Loss Function Is Still an Open Problem

The heavy-rain low bias is too large and too broad (46 storm days, 68% of heavy rain) to be explained by physics/features alone — it is partly a **loss/optimization** issue:
- `weighted_mae` up-weights large targets but the model still regresses toward the conditional mean on the rare heavy tail
- Even perfect input features won't fix a loss that drives predictions to the middle of a heavy-tailed distribution
- **This needs continued experimentation** independent of feature engineering: e.g. stronger target weighting, focal/quantile-style losses, importance sampling of heavy events, or a two-stage (detect-then-regress) approach. Sampler strategy and log-vs-mm interaction with the loss should be revisited too.

## Next Steps

### Vertical Structure Features — Phase 1 (Done)
The first batch of 5 derived 3D features is already computed and in the model:
- **Echo Top Height** — highest altitude with Z≥18 dBZ
- **VIL** (Vertically Integrated Liquid) — total liquid content
- **Max Z Height** — altitude of peak reflectivity (became a top-ranked feature in ablation)
- **Low-level Mean Reflectivity** — 0-2 km average
- **Column Depth Fraction** — fraction of levels with Z>10 dBZ

### Full 3D Volume + New Features — Phase 2 (In Progress)
Driven directly by the diagnostics above, the radar pull (`radar/pull_nexrad_multi.py`) now stores the **full 3D volume** (22 Z levels × y × x for all 5 raw fields) instead of collapsing to 2D, and the **KDP `Z<20 dBZ` mask was removed** (it was deleting warm-rain signal at low reflectivity). A new offline step (`radar/derive_features.py`) collapses the 3D archive into the 2D feature zarr, so features can be iterated **without re-pulling from S3** (~2-day job).

New features being derived to target the two failure modes:
- *Warm-rain / orographic (underprediction):* **low-level KDP, low-level ZDR, low-level RhoHV** (0–2 km, near the surface rather than at max-Z), **lowest-gate reflectivity**, **beam height** (overshoot indicator), **vertical reflectivity gradient** (seeder-feeder)
- *Bright band (overprediction):* **melting-layer height** (RhoHV-minimum height) and **RhoHV-min** value

The 3D archive also future-proofs feature engineering: any new vertical metric can be derived offline without another S3 pull.

> Note: features address *information content*, but the heavy-rain shortfall is also a **loss-function** problem (see Finding #10) — both fronts need work.

## File Structure

```
dataset/
  create_pickle.py          # Hourly pickle with dump_ratio fields
  create_pickle_10min.py    # 10-min single-scan pickle

models/
  run_experiments.py        # Model-agnostic YAML-driven batch runner
  unet/                     # Current best model
    train.py                # --filter-mode blunt|radar, width presets, log/mm
    evaluate.py             # Respects filter_mode/log_target from checkpoint
    ablation.py
    diagnose_overestimates.py        # Pred high, actual low
    diagnose_overpredict_weather.py  # RH/temp/ZDR/elevation/bright-band
    diagnose_underestimates.py       # Heavy rain underprediction (warm-rain/AR)
    experiments.yaml
  stack/                    # Hourly CNN baseline (+ SpatialConvHead option)
    experiments.yaml
  stack_10min/              # 10-min temporal experiment
    diagnose_outliers.py    # Cross-station outlier analysis
  gfm/                      # Foundation model approach

radar/
  pull_nexrad_multi.py      # Now stores FULL 3D volume (no KDP mask)
  derive_features.py        # Offline 3D → 2D feature derivation

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
