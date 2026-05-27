# Stack 10-min Model

Single-scan CNN model for 10-minute precipitation prediction.

## Key Differences from Hourly Stack Model

| Aspect | Hourly Stack | 10-min Stack |
|--------|-------------|--------------|
| Temporal window | 1 hour (12 scans) | 10 minutes (1 scan) |
| Input shape | (73, H, W) = 4 fields×12 + masks + tpos + DEM | (5, H, W) = 4 fields + DEM |
| Target | Hourly precip (log1p mm) | 10-min precip (raw mm) |
| Loss | MAE in log-space | Huber in mm-space |
| Samples/day | ~24 per station | ~144 per station |

## Architecture

- **Encoder**: 3-block CNN (64→128→256) with adaptive pooling
- **Decoder**: 2-layer MLP → scalar output (mm/10min)
- **Parameters**: ~700K (vs ~5M for hourly model)

## Usage

```bash
# Generate pickle
python -m dataset.create_pickle_10min \
    --radar radar/outputs/dualpol_500m_2022-01-01_2026-04-04.zarr \
    --days weather/days/top_100_days_2022-01-01_2026-04-04.txt \
    --patch-size 4500 \
    --train-years 2022 2024 2026 --val-years 2023 2025 \
    --output dataset/outputs/10min/radar_gauge_10min.pkl

# Train + evaluate
python -m models.stack_10min.run_stack --mode all --run-name baseline

# Train only
python -m models.stack_10min.run_stack --mode train --no-sampler --run-name huber_no_sampler

# Evaluate existing run
python -m models.stack_10min.run_stack --mode eval --run-dir models/checkpoints/stack_10min/...
```

## Hypothesis

By removing temporal stacking, the model's task becomes simpler: predict 10-min rainfall from a single radar snapshot. This removes the need for the model to learn temporal integration, potentially reducing noise from misaligned scan-to-accumulation mappings.
