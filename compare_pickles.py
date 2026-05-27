import pickle
import numpy as np
from collections import Counter

# Load both
with open('dataset/outputs/2d/radar_gauge_dataset_with_offsets_9500.pkl', 'rb') as f:
    old = pickle.load(f)
with open('dataset/outputs/radar_gauge_dataset_with_offsets_9500.pkl', 'rb') as f:
    new = pickle.load(f)

print("=== OLD PICKLE ===")
print(f"  Train: {len(old['train'])}, Val: {len(old['val'])}")
print(f"  Test:  {len(old.get('test', []))}")
print(f"  Metadata: {old['metadata']}")

print("\n=== NEW PICKLE ===")
print(f"  Train: {len(new['train'])}, Val: {len(new['val'])}")
print(f"  Test:  {len(new.get('test', []))}")
print(f"  Metadata: {new['metadata']}")

# Year distribution
print("\n=== YEAR DISTRIBUTION ===")
for name, data in [("OLD", old), ("NEW", new)]:
    for split in ['train', 'val']:
        years = [s['hour_start'].year for s in data[split]]
        print(f"  {name} {split}: {Counter(years)}")

# Station distribution
print("\n=== STATION COUNTS ===")
for name, data in [("OLD", old), ("NEW", new)]:
    for split in ['train', 'val']:
        stations = Counter(s.get('station_name', '') for s in data[split])
        print(f"  {name} {split}: {len(stations)} stations, {dict(stations)}")

# Rainfall distribution
print("\n=== RAINFALL STATS ===")
for name, data in [("OLD", old), ("NEW", new)]:
    for split in ['train', 'val']:
        precip = [s['hourly_precip_mm'] for s in data[split]]
        print(f"  {name} {split}: mean={np.mean(precip):.3f}, max={np.max(precip):.2f}, "
              f"median={np.median(precip):.3f}, >5mm={sum(1 for p in precip if p > 5)}")

# Patch shape
print("\n=== PATCH SHAPES ===")
print(f"  OLD train[0]: {old['train'][0]['radar_patch'].shape}")
print(f"  NEW train[0]: {new['train'][0]['radar_patch'].shape}")