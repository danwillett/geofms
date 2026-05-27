# Precipitation Maps via GFM (TerraMind)

## Overview

A precipitation prediction model built on IBM/ESA's **TerraMind v1 tiny** — a multi-modal Vision Transformer (ViT) pre-trained on satellite imagery (DEM, optical, SAR, etc.). The backbone is extended with a custom RADAR embedding and a lightweight CNN decoder to predict 5×5 precipitation maps from NEXRAD radar + DEM inputs.

---

## Architecture

```
DEM   (1, 256, 256)  ──→ [DEM Embedding  | frozen]  ──┐
                                                        ├──→ TerraMind ViT Backbone ──→ SpatialPrecipitationDecoder ──→ (1, 5, 5) precip map
RADAR (36, 256, 256) ──→ [RADAR Embedding | unfrozen] ─┘
```

### Components

| Component | Description |
|---|---|
| **TerraMind v1 tiny** | Multi-modal ViT backbone, loaded from HuggingFace. Backbone is **frozen** during training. |
| **DEM Embedding** | Pre-trained modality embedding for elevation data. Frozen. |
| **RADAR Embedding** | Custom modality embedding for radar (randomly initialized — TerraMind has no radar pre-training). **Only this is unfrozen.** |
| **SpatialPrecipitationDecoder** | CNN (Conv-BN × 3 → AdaptiveAvgPool) that reshapes ViT feature maps `(batch, patches, channels)` → `(batch, 1, 5, 5)`. |

---

## Input

### Modalities

Two modalities are passed to the backbone as an `image` dict:

| Key | Shape | Description |
|---|---|---|
| `"DEM"` | `(1, 256, 256)` | Digital Elevation Model, bilinear-resized to 256×256 |
| `"RADAR"` | `(36, 256, 256)` | NEXRAD radar stack, nearest-neighbor upscaled from 5×5 to 256×256 |

### RADAR Channel Layout (36 = 12 + 12 + 12)

The 5×5 radar patch covers 12 consecutive scans per hour (~every 5 min) at 500m resolution. After max-collapsing the Z dimension, three channel groups are stacked:

| Channels | Content |
|---|---|
| 0–11 | Reflectivity (CMAX, normalized to [0, 1]) |
| 12–23 | Validity mask (1.0 = valid scan, 0.0 = padded/missing) |
| 24–35 | Temporal position encoding (scan index normalized 0 → 1) |

> **Dual-pol update (pending):** With the new zarr storing ZDR, RhoHV, PhiDP, and KDP alongside reflectivity, RADAR channels will expand beyond 36. Each dual-pol field will add its own radar + mask + t_pos triple, with field-specific normalization ranges.

### Spatial Patch

- **Radar**: 9×9 patch (4.5 km × 4.5 km at 500m) randomly cropped to 5×5 during training, center-cropped during validation.
- **DEM**: Extracted on-the-fly centered on the gauge station, with the same spatial offset as the radar crop.

---

## Training

| Setting | Value |
|---|---|
| Framework | PyTorch Lightning + TerraTorch `PixelwiseRegressionTask` |
| Loss | MSE (`ignore_index=-9999`) |
| LR | 1e-5 |
| LR Scheduler | `ReduceLROnPlateau` |
| Precision | 16-bit mixed (CUDA) |
| Freeze strategy | Backbone frozen; RADAR embedding + decoder trained |

### Target

- **Label**: Hourly accumulated precipitation (mm/hr), `log1p`-transformed
- **Supervision**: Sparse — loss computed only at the gauge pixel location within the 5×5 grid
- **Format**: 5×5 mask tensor filled with `−9999` (ignore) everywhere except the gauge pixel

### Data Filtering

Three filters applied at dataset setup:

1. **`filter_biased_extremes`** — Removes implausible radar-gauge combinations from known over/underestimating stations
2. **`filter_bad_samples`** — Removes ground clutter (high dBZ, low rain) and sensor errors (>40 mm/hr)
3. **`filter_suspect_station_days`** — Removes station-days where a gauge reads zero while ≥9 other stations recorded significant rain

### Class Imbalance

A `WeightedRandomSampler` oversamples rare heavy-rain events during training:

| Category | Weight |
|---|---|
| Zero / trace (< 0.1 mm) | 0.5× |
| Light (0.1–2 mm) | 1.0× |
| Moderate (2–5 mm) | 2.0× |
| Heavy (5–15 mm) | 5.0× |
| Very heavy (> 15 mm) | 10.0× |

---

## Files

```
models/gfm/
└── README.md                         ← You are here

deep_learning/
├── train_precip_model_terramind.ipynb  ← Training notebook
└── precipitation_terrmind.ipynb        ← Inference / evaluation

radar/
└── pull_nexrad.py                    ← Zarr data pipeline (NEXRAD → dual-pol zarr)
```

---

## Pending Changes

- [ ] Expand RADAR channels for dual-pol features (ZDR, RhoHV, PhiDP, KDP)
- [ ] Add per-field normalization (ZDR: −2 to 6 dB, RhoHV: 0–1, KDP: −1 to 6 °/km, PhiDP: 0–360°)
- [ ] Remove Z-dimension max-collapse from dataset code (now handled in zarr)
- [ ] Move training notebook into `models/gfm/`