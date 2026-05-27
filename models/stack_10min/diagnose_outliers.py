"""
diagnose_outliers.py — Investigate extreme 10-min precipitation samples.

Checks for gauge artifacts by examining:
1. Radar reflectivity at the time of the reading
2. What other stations reported for the same timestamp
3. Whether neighboring time windows are zero (dump signature)

Run from project root:
    python -m models.stack_10min.diagnose_outliers --pickle dataset/outputs/10min/radar_gauge_10min.pkl --threshold 6
"""

import argparse
import pickle
import numpy as np
from collections import defaultdict


def diagnose_outliers(pickle_path, threshold_mm=6.0):
    print(f"\nLoading pickle: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)

    all_samples = dataset['train'] + dataset['val']
    print(f"Total samples: {len(all_samples)} (train={len(dataset['train'])}, val={len(dataset['val'])})")

    # Find outliers
    outliers = [s for s in all_samples if s['precip_mm'] >= threshold_mm]
    print(f"\nSamples with precip >= {threshold_mm} mm/10min: {len(outliers)}")

    if not outliers:
        print("No outliers found.")
        return

    # Group all samples by timestamp for cross-station comparison
    by_timestamp = defaultdict(list)
    for s in all_samples:
        by_timestamp[s['bin_start']].append(s)

    # Analyze each outlier
    print(f"\n{'='*80}")
    print(f"  OUTLIER ANALYSIS (threshold = {threshold_mm} mm/10min)")
    print(f"{'='*80}")

    for i, s in enumerate(sorted(outliers, key=lambda x: -x['precip_mm'])):
        precip = s['precip_mm']
        station = s.get('station_name', 'Unknown')
        bin_start = s['bin_start']
        scan_time = s.get('scan_time', 'N/A')

        # Radar stats
        ref_data = s['radar_patch'][0, :, :]  # reflectivity channel
        ref_valid = ref_data[~np.isnan(ref_data) & (ref_data != -9999.0)]
        max_dbz = np.nanmax(ref_valid) if len(ref_valid) > 0 else float('nan')
        mean_dbz = np.nanmean(ref_valid) if len(ref_valid) > 0 else float('nan')
        center_dbz = ref_data[ref_data.shape[0]//2, ref_data.shape[1]//2]
        if center_dbz == -9999.0:
            center_dbz = float('nan')

        # What did other stations report at the same timestamp?
        same_time = by_timestamp[bin_start]
        other_readings = [(x['station_name'], x['precip_mm']) for x in same_time if x['station_id'] != s['station_id']]
        other_precip = [r[1] for r in other_readings]

        print(f"\n{'─'*80}")
        print(f"  Outlier #{i+1}: {precip:.2f} mm/10min")
        print(f"{'─'*80}")
        print(f"  Station:       {station}")
        print(f"  Timestamp:     {bin_start}")
        print(f"  Scan time:     {scan_time}")
        print(f"  Rate equiv:    {precip * 6:.1f} mm/hr")
        print(f"\n  Radar (9×9 patch):")
        print(f"    Max dBZ:     {max_dbz:.1f}")
        print(f"    Mean dBZ:    {mean_dbz:.1f}")
        print(f"    Center dBZ:  {center_dbz:.1f}")

        print(f"\n  Other stations at same timestamp ({len(other_readings)} stations):")
        if other_readings:
            other_sorted = sorted(other_readings, key=lambda x: -x[1])[:10]
            for name, val in other_sorted:
                short_name = name.replace('Dangermond_', '')
                print(f"    {short_name:<20s} {val:.2f} mm")
            if other_precip:
                print(f"    ───")
                print(f"    Mean:  {np.mean(other_precip):.3f} mm")
                print(f"    Max:   {np.max(other_precip):.3f} mm")
                print(f"    Zeros: {sum(1 for p in other_precip if p < 0.1)}/{len(other_precip)}")
        else:
            print("    (no other stations reporting at this time)")

        # Check neighboring windows for this station (dump signature)
        from datetime import timedelta
        neighbors = []
        for offset_min in [-30, -20, -10, 10, 20, 30]:
            neighbor_time = bin_start + timedelta(minutes=offset_min)
            neighbor_samples = [x for x in by_timestamp.get(neighbor_time, [])
                                if x['station_id'] == s['station_id']]
            if neighbor_samples:
                neighbors.append((offset_min, neighbor_samples[0]['precip_mm']))
            else:
                neighbors.append((offset_min, None))

        print(f"\n  Neighboring windows (same station):")
        for offset, val in neighbors:
            label = f"    t{offset:+d}min:"
            if val is not None:
                print(f"{label}  {val:.3f} mm")
            else:
                print(f"{label}  (no sample)")

        # Verdict
        radar_supports = max_dbz > 50.0
        others_also_high = len(other_precip) > 0 and np.max(other_precip) > threshold_mm * 0.3
        neighbor_vals = [v for _, v in neighbors if v is not None]
        neighbors_zero = len(neighbor_vals) > 0 and np.max(neighbor_vals) < 0.5

        print(f"\n  Diagnosis:")
        print(f"    Radar supports heavy rain (max>{50} dBZ)?  {'YES' if radar_supports else 'NO'}")
        print(f"    Other stations also elevated?              {'YES' if others_also_high else 'NO'}")
        print(f"    Neighbors near zero (dump signature)?      {'YES' if neighbors_zero else 'NO'}")

        if not radar_supports and neighbors_zero:
            print(f"    → LIKELY GAUGE ARTIFACT (stuck bucket dump)")
        elif not radar_supports and not others_also_high:
            print(f"    → SUSPECT (radar doesn't support, isolated reading)")
        elif radar_supports and others_also_high:
            print(f"    → LIKELY REAL (radar + other stations confirm)")
        else:
            print(f"    → INCONCLUSIVE")

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"\n  {'Station':<25s} {'Precip':<12s} {'Max dBZ':<10s} {'Rate (mm/hr)':<14s} {'Verdict'}")
    print(f"  {'─'*25} {'─'*12} {'─'*10} {'─'*14} {'─'*20}")

    for s in sorted(outliers, key=lambda x: -x['precip_mm']):
        station = s.get('station_name', 'Unknown').replace('Dangermond_', '')
        precip = s['precip_mm']
        ref_data = s['radar_patch'][0, :, :]
        ref_valid = ref_data[~np.isnan(ref_data) & (ref_data != -9999.0)]
        max_dbz = np.nanmax(ref_valid) if len(ref_valid) > 0 else float('nan')

        radar_ok = max_dbz > 50.0
        verdict = "Real" if radar_ok else "Suspect"
        print(f"  {station:<25s} {precip:<12.2f} {max_dbz:<10.1f} {precip*6:<14.1f} {verdict}")

    print(f"\n  Total outliers: {len(outliers)}")
    print(f"  Suspect (max dBZ < 50): {sum(1 for s in outliers if np.nanmax(s['radar_patch'][0][~np.isnan(s['radar_patch'][0]) & (s['radar_patch'][0] != -9999.0)]) < 50)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose extreme 10-min precipitation outliers")
    parser.add_argument('--pickle', default='dataset/outputs/10min/radar_gauge_10min.pkl')
    parser.add_argument('--threshold', type=float, default=6.0,
                        help='Precipitation threshold in mm/10min (default: 6.0)')
    args = parser.parse_args()

    diagnose_outliers(args.pickle, threshold_mm=args.threshold)
