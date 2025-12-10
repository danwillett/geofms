# Deep Learning Workflow: NEXRAD → TerraMind Integration

## Overview
Train a precipitation prediction model using NEXRAD radar + TerraMind foundation model.

## Pipeline Steps

### ✅ Step 1: Data Collection (COMPLETE)
**Files:**
- `radar/pull_nexrad.py` - Pull NEXRAD radar data
- `weather/pull_weather.py` - Pull rain gauge measurements

**What you have:**
- NEXRAD radar at 500m resolution (or 10m)
- Rain gauge measurements every 10 minutes
- Spatial alignment to TerraMesh patch grid

**Run:**
```bash
# Pull radar for rainy days
python main.py

# Verify gauge data exists
python -c "from weather.pull_weather import get_10min_gauge_data; df = get_10min_gauge_data('2022-10-01', '2022-10-31'); print(df.head())"
```

---

### ✅ Step 2: Data Preparation (READY TO RUN)
**File:** `deep_learning/prepare_radar_gauge_data.py`

**What it does:**
1. Load radar zarr and gauge database
2. Get hourly accumulated precipitation for each station
3. Sample ~6 NEXRAD scans from each hour
4. Extract radar patches around each station
5. Create train/val split (80/20)
6. Save as pickle file for training

**Run:**
```bash
# From project root (RECOMMENDED - handles .env properly)
python main.py

# Or manually from deep_learning/ (if you prefer):
cd deep_learning
python prepare_radar_gauge_data.py \
    --radar ../KVBX_preserve_500m.zarr \
    --days ../my_rainy_days.txt \
    --output radar_gauge_dataset.pkl

# Inspect the dataset
python prepare_radar_gauge_data.py \
    --output radar_gauge_dataset.pkl \
    --inspect
```

**Note:** Running from `main.py` in project root is recommended because:
- ✅ Proper `.env` file loading (database connection)
- ✅ Consistent working directory
- ✅ All steps in one place

**Output:**
- `radar_gauge_dataset.pkl` containing aligned radar-gauge pairs

---

### 🔨 Step 3: Build Radar CNN (NEXT STEP)
**File to create:** `deep_learning/models.py`

**What you'll build:**
```python
class RadarEncoder(nn.Module):
    """
    Input: (batch, T=6, Z=40, H, W) - temporal radar sequence
    Output: (batch, embedding_dim) - feature vector
    """
    # Handle temporal + vertical + spatial dimensions
    # Extract features for precipitation prediction

class PrecipitationModel(nn.Module):
    """
    Input: 
        - radar_patch: (batch, 6, Z, H, W)
        - context: DEM/LULC (batch, C, H, W)
    Output:
        - precipitation: (batch, H, W) - mm/hr at each pixel
    """
    def __init__(self):
        self.radar_encoder = RadarEncoder()
        self.terramind = load_terramind(frozen=True)
        self.fusion = FusionLayer()
        self.decoder = PrecipDecoder()
```

---

### 🔨 Step 4: Create PyTorch Dataset (NEXT STEP)
**File to create:** `deep_learning/dataset.py`

**What you'll build:**
```python
class RadarGaugeDataset(Dataset):
    """Load prepared samples and convert to tensors"""
    def __init__(self, pickle_path, split='train'):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        self.samples = data[split]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Radar: (6, Z, H, W) - temporal sequence from the hour
        radar = torch.from_numpy(
            np.nan_to_num(sample['radar_patch'], nan=-9999.0)
        ).float()
        
        # Target: Single value - hourly accumulated precipitation
        target = torch.tensor(sample['hourly_precip_mm']).float()
        
        return {'radar': radar, 'target': target}
```

---

### 🔨 Step 5: Training with Terratorch (FUTURE)
**File to create:** `deep_learning/train.py`

**What you'll do:**
```python
from terratorch import TaskFactory
from pytorch_lightning import Trainer

# Create task
task = TaskFactory.build(
    task_type="pixel_wise_regression",
    model_args={
        "backbone": "terramind",
        "decoder": "simple_regression_head",
        "auxiliary_head": "radar_encoder"  # Your custom radar CNN
    },
    loss="mse",
    ignore_index=-9999
)

# Train
trainer = Trainer(max_epochs=50, gpus=1)
trainer.fit(task, datamodule)
```

---

## Current Status

```
✅ radar/pull_nexrad.py           - Pull NEXRAD data
✅ radar/visualize_nexrad.py      - Visualize radar
✅ radar/constants.py             - Data handling constants
✅ radar/test_data_pipeline.py    - Test data loading
✅ weather/pull_weather.py        - Gauge data functions
✅ deep_learning/prepare_radar_gauge_data.py - Data alignment

🔨 NEXT: Run data preparation
🔨 NEXT: Build models.py
🔨 NEXT: Build dataset.py
🔨 NEXT: Build train.py
```

---

## Quick Start

```bash
# Run from project root
python main.py
```

That's it! `main.py` will:
1. ✅ Load your rainy days from `my_rainy_days.txt`
2. ✅ Pull gauge data from database (using `.env` for connection)
3. ✅ Align NEXRAD radar with gauge measurements
4. ✅ Create train/val split
5. ✅ Save to `deep_learning/radar_gauge_dataset.pkl`
6. ✅ Display summary statistics

**Next:** Build and train the model (coming soon)

---

## Architecture Decisions

### Resolution: 500m vs 10m
**Using 500m** because:
- ✅ Matches radar's actual information content (~250m beam)
- ✅ 2500× smaller than 10m
- ✅ Weather patterns are coarse-scale
- ✅ Sufficient for precipitation prediction

### Data Alignment
**Sample ~6 radar scans per hour** because:
- ✅ Captures temporal evolution of precipitation
- ✅ NEXRAD scans every 5-6 minutes (10-12 per hour available)
- ✅ 6 scans = good balance (not too many parameters, enough temporal info)
- ✅ Target is single hourly accumulation (mm/hr)

### Missing Data
**Use -9999 as no-data marker** because:
- ✅ Standard geospatial convention
- ✅ Compatible with PyTorch ignore_index
- ✅ Terratorch automatically handles it
- ✅ Outside physical range of reflectivity

---

## Next Steps (In Order)

1. ✅ **Verify gauge data exists** for your date range
2. **Run data preparation** script
3. **Inspect prepared dataset** to verify alignment
4. **Build RadarEncoder CNN** architecture
5. **Create PyTorch Dataset** class
6. **Integrate with TerraMind** using Terratorch
7. **Train model** on your data
8. **Evaluate** against held-out gauges
9. **Visualize** predictions vs ground truth

---

## Key Files Reference

```
geofms/
├── radar/
│   ├── pull_nexrad.py           # Pull NEXRAD data
│   ├── visualize_nexrad.py      # Visualize radar
│   ├── constants.py             # Constants & utilities
│   └── test_data_pipeline.py    # Test data loading
├── weather/
│   └── pull_weather.py          # Gauge data functions
├── deep_learning/
│   ├── prepare_radar_gauge_data.py  # ← Data alignment (READY)
│   ├── models.py                # ← Build next
│   ├── dataset.py               # ← Then this
│   ├── train.py                 # ← Then train
│   └── WORKFLOW.md              # ← You are here
└── grid/
    └── B02.tif                   # Reference S2 image
```

