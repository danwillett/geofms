# Precipitation Maps via Stack (CNN)

## Overview

A multi-modal CNN model for hourly precipitation prediction, trained on NEXRAD radar + DEM. Designed for small 5x5 spatial patches (2.5 km x 2.5 km at 500m resolution) centered on rain gauge stations. A lightweight `RadarEncoder` CNN encodes the radar stack into a latent embedding, which is decoded into a 5x5 precipitation map by an MLP decoder. An optional bias embedding corrects for known per-station systematic errors.

---

## Architecture

```
Radar (37, 5, 5) ──→ RadarEncoder (CNN) ──→ (256,) embedding ──┐
                                                                  ├──→ PrecipitationDecoder (MLP) ──→ (5, 5) precip map
[optional] bias_flag ──→ BiasEmbedding ──→ (32,) ──────────────┘
```

### Components

| Component | Description |
|---|---|
| **RadarEncoder** | 3-block CNN (Conv-BN-Dropout x 3, MaxPool after block 2) → Flatten → FC(512 → 256) → 256-dim embedding |
| **PrecipitationDecoder** | MLP (256 → 512 → 128 → 25) reshaped to (5, 5) precipitation map |
| **BiasEmbedding** | `nn.Embedding(3, 32)` for per-station bias correction (-1 = underestimating, 0 = neutral, +1 = overestimating). Concatenated to radar embedding before decoding. Optional. |

---

## Input

### Input Tensor Shape: `(batch, 37, 5, 5)`

The 37 channels come from stacking three groups across 12 radar scans per hour, plus one DEM channel:

| Channels | Content |
|---|---|
| 0-11 | Reflectivity CMAX (normalized to [0, 1]) |
| 12-23 | Validity mask (1.0 = valid scan, 0.0 = padded/missing) |
| 24-35 | Temporal position encoding (scan index normalized 0 to 1) |
| 36 | DEM elevation (downsampled to 5x5, same spatial offset as radar crop) |

The temporal position encoding encodes *when* in the hour each scan falls, giving the model sequence awareness without a recurrent structure.

> **Dual-pol update (pending):** With the new zarr storing ZDR, RhoHV, PhiDP, and KDP alongside reflectivity, input channels will expand beyond 37. Each dual-pol field adds its own radar + mask + t_pos triple (12 channels each), with field-specific normalization ranges.

### Spatial Patch

- **Radar**: 9x9 patch (4.5 km x 4.5 km at 500m) stored in the pickle. Randomly cropped to 5x5 during training, center-cropped during validation.
- **DEM**: Extracted on-the-fly from a 10m GeoTIFF centered on the gauge station, downsampled to match the radar patch, then cropped with the **same spatial offset** as the radar crop.

### Normalization

- **Reflectivity**: `(dBZ - (-32)) / (70 - (-32))`, clipped to [0, 1]
- **DEM**: Raw elevation values passed through directly (no normalization currently)

---

## Training

| Setting | Value |
|---|---|
| Framework | PyTorch (manual training loop) |
| Loss | MSE on `log1p(mm/hr)` at gauge pixel only |
| Optimizer | Adam |
| Dropout | 0.25 (spatial Dropout2d in CNN blocks, 0.05 in MLP) |
| Batch norm | Yes (CNN blocks) / Layer norm (MLP blocks) |

### Target

- **Label**: Hourly accumulated precipitation (mm/hr), `log1p`-transformed
- **Supervision**: Loss computed only at the gauge pixel location within the 5x5 grid
- **Gauge pixel**: Varies per sample due to random crop augmentation (tracked as `gauge_pixel` in the batch dict)

### Data Augmentation

- **Random 9x9 → 5x5 crop** during training (with probability `aug_prob=0.5`), applied consistently to radar and DEM
- Center crop at validation for reproducibility

### Bias Correction

Known biased stations are flagged at dataset load time:

- **Overestimating** (`bias_flag = +1`): Dangermond Bunker Hill, Cistern, Cojo HQ, Jalachichi, Repeater
- **Underestimating** (`bias_flag = -1`): Dangermond Cojo Gate, Sutter
- **Neutral** (`bias_flag = 0`): All others

When `add_bias=True`, the bias embedding is concatenated to the radar embedding, giving the decoder a signal to adjust predictions per station type.

### Data Filtering

Two filters applied before training:

1. **`filter_biased_extremes`** — Removes implausible radar-gauge combinations from known biased stations (e.g., heavy radar echo but near-zero gauge)
2. **`filter_bad_samples`** — Removes ground clutter (dBZ > 50 with < 2 mm rain), sensor errors (> 40 mm/hr), and radar misses (> 5 mm rain with dBZ < 20)

---

## Files

```
models/stack/
└── STACK.md                              <- You are here

deep_learning/
├── train_precip_model_stack.ipynb        <- Training notebook
├── dataset_stack.py                      <- PyTorch Dataset (RadarGaugeDataset)
└── prepare_radar_gauge_data.py           <- Builds pickle from zarr + gauge DB

radar/
└── pull_nexrad.py                        <- Zarr data pipeline (NEXRAD -> dual-pol zarr)
```

---

## Pending Changes

- [ ] Expand input channels for dual-pol features (ZDR, RhoHV, PhiDP, KDP)
- [ ] Add per-field normalization (ZDR: -2 to 6 dB, RhoHV: 0-1, KDP: -1 to 6 deg/km, PhiDP: 0-360 deg)
- [ ] Remove Z-dimension max-collapse from `dataset_stack.py` (now handled in zarr)
- [ ] Move training notebook into `models/stack/`
