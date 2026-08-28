"""
diagnose_catastrophic_misses.py — Analyze the WORST underpredictions: hours
where a real, heavy gauge total (default 5-25 mm) was predicted as essentially
zero (default < 1 mm).

Unlike diagnose_underestimates.py (which captures the broad warm-rain
under-read), these are total misses — predicting ~nothing when 5-25 mm fell.
Those almost always have a different root cause, so the diagnostic splits each
miss by WHAT THE RADAR SAW at/around the gauge pixel:

  A1  radar blind everywhere   center≈0 AND patch≈0   -> gauge error / dump,
                                                          beam overshoot, or
                                                          missing scan
  A2  rain nearby, not at pixel center≈0 BUT patch hot -> spatial offset /
                                                          parallax / advection
  B   radar saw it, model 0    center hot              -> genuine model failure

To separate "gauge error" from "real rain the radar missed", it cross-checks
neighbouring gauges at the SAME hour: if neighbours were also wet the gauge is
probably right (radar/model missed); if neighbours were dry the lone wet gauge
is likely an instrument error.

Run from project root:
    python -m models.unet.diagnostics.diagnose_catastrophic_misses --run-dir models/checkpoints/unet_dualpol/<run_name>
"""

import argparse
import math
import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from torch.utils.data import DataLoader

from models.unet.evaluate import load_model, find_checkpoint
from models.unet.dataset import RadarGaugeDataset, resolve_fields
from models.unet.train import (
    filter_nan_radar, filter_biased_extremes, filter_bad_samples,
    filter_suspect_station_days, filter_radar_unsupported, filter_gauge_dumps
)
from models.unet.diagnostics.common import extract_radar_features, _valid_center

try:
    from models.unet.diagnostics.diagnose_overpredict_weather import query_hourly_avg
except Exception:
    query_hourly_avg = None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_neighbor_index(data):
    """Map hour_start -> list of (station_id, name, lat, lon, precip_mm) across
    all hourly splits, so we can look up what other gauges saw at the same hour."""
    index = defaultdict(list)
    for split in ('train', 'val'):
        for s in data.get(split, []):
            hs = s.get('hour_start')
            if hs is None or 'hourly_precip_mm' not in s:
                continue
            index[hs].append((
                s.get('station_id'),
                s.get('station_name', ''),
                s.get('station_lat'),
                s.get('station_lon'),
                float(s.get('hourly_precip_mm', 0.0)),
            ))
    return index


def compute_train_heavy_counts(data, actual_min, filter_mode):
    """How many CLEAN (post train-filter) train samples each station has, and how
    many of those are heavy (>= actual_min). Tests whether the stations the model
    zeroes simply lack valid heavy-rain training examples."""
    train = list(data.get('train', []))
    if not train:
        return {}, {}
    print("\n  Computing clean train heavy-rain availability per station...")
    train = filter_nan_radar(train)
    if filter_mode == 'radar':
        train = filter_radar_unsupported(train)
    else:
        train = filter_biased_extremes(train)
        train = filter_bad_samples(train)
    train = filter_suspect_station_days(train)
    train = filter_gauge_dumps(train)
    total = Counter(s.get('station_name', '') for s in train)
    heavy = Counter(s.get('station_name', '') for s in train
                    if float(s.get('hourly_precip_mm', 0.0)) >= actual_min)
    return total, heavy


def neighbor_stats(sample, neighbor_index, wet_threshold=1.0):
    """For one miss sample, summarise what other gauges recorded that hour."""
    hs = sample.get('hour_start')
    sid = sample.get('station_id')
    slat, slon = sample.get('station_lat'), sample.get('station_lon')
    out = dict(n_neighbors=0, n_wet=0, neighbor_max=np.nan,
               nearest_wet_km=np.nan, nearest_wet_mm=np.nan)
    if hs is None or hs not in neighbor_index:
        return out
    wet = []
    others = [n for n in neighbor_index[hs] if n[0] != sid]
    if not others:
        return out
    precs = [n[4] for n in others]
    out['n_neighbors'] = len(others)
    out['n_wet'] = sum(1 for p in precs if p >= wet_threshold)
    out['neighbor_max'] = max(precs) if precs else np.nan
    if slat is not None and slon is not None:
        for (nid, nname, nlat, nlon, p) in others:
            if p >= wet_threshold and nlat is not None and nlon is not None:
                d = haversine_km(slat, slon, nlat, nlon)
                wet.append((d, p))
        if wet:
            wet.sort(key=lambda x: x[0])
            out['nearest_wet_km'] = wet[0][0]
            out['nearest_wet_mm'] = wet[0][1]
    return out


def run_diagnostic(run_dir, pred_max=1.0, actual_min=5.0, actual_max=25.0,
                   echo_thresh=20.0, patch_thresh=25.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = find_checkpoint(run_dir=run_dir)
    model, cfg = load_model(checkpoint_path, device)

    pickle_path = cfg.get('pickle_path', 'dataset/outputs/3d/radar_gauge_dataset_vertical_9500.pkl')
    dem_path = cfg.get('dem_path', 'dem/preserve_dem_10m_utm.tif')
    fields = resolve_fields(cfg.get('fields'))
    log_target = cfg.get('log_target', True)

    ds_kwargs = dict(
        dem_path=dem_path,
        fields=fields,
        use_dem=cfg.get('use_dem', True),
        use_mask=cfg.get('use_mask', True),
        use_temporal_pos=cfg.get('use_temporal_pos', True),
        use_feature_masks=cfg.get('use_feature_masks', False),
        log_target=log_target,
    )

    val_ds = RadarGaugeDataset(pickle_path, split='val', augment=False, **ds_kwargs)
    val_ds.samples = filter_nan_radar(val_ds.samples)
    filter_mode = cfg.get('filter_mode', 'blunt')
    if filter_mode == 'radar':
        val_ds.samples = filter_radar_unsupported(val_ds.samples)
    else:
        val_ds.samples = filter_biased_extremes(val_ds.samples)
        val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_suspect_station_days(val_ds.samples)

    print(f"\n  Validation samples: {len(val_ds.samples)}")
    print(f"  Catastrophic-miss criteria: {actual_min} <= actual <= {actual_max}mm "
          f"AND pred < {pred_max}mm")

    print("  Building neighbor index (other gauges per hour)...")
    with open(pickle_path, 'rb') as f:
        full_data = pkl.load(f)
    neighbor_index = build_neighbor_index(full_data)
    train_total, train_heavy = compute_train_heavy_counts(full_data, actual_min, filter_mode)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            radar = batch['radar'].to(device)
            gauge_pixel = batch['gauge_pixel']
            pred_map = model(radar).cpu()

            if pred_map.dim() == 1:
                pred_at_gauge = pred_map
            else:
                batch_idx = torch.arange(pred_map.shape[0])
                if isinstance(gauge_pixel, torch.Tensor) and gauge_pixel.dim() == 2:
                    y = gauge_pixel[:, 0].long()
                    x = gauge_pixel[:, 1].long()
                    pred_at_gauge = pred_map[batch_idx, y, x]
                elif isinstance(gauge_pixel, (tuple, list)):
                    y, x = gauge_pixel
                    if isinstance(y, torch.Tensor):
                        pred_at_gauge = pred_map[batch_idx, y, x]
                    else:
                        pred_at_gauge = pred_map[:, y, x]
                else:
                    center = pred_map.shape[-1] // 2
                    pred_at_gauge = pred_map[:, center, center]

            preds = pred_at_gauge.numpy()
            targets = batch['target'].numpy()
            if log_target:
                preds = np.expm1(preds)
                targets = np.expm1(targets)
            all_preds.extend(preds.tolist())
            all_actuals.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    miss_mask = (all_actuals >= actual_min) & (all_actuals <= actual_max) & (all_preds < pred_max)
    miss_indices = np.where(miss_mask)[0]
    n_miss = len(miss_indices)
    n_heavy = int(((all_actuals >= actual_min) & (all_actuals <= actual_max)).sum())
    print(f"\n  Heavy samples ({actual_min}-{actual_max}mm): {n_heavy}")
    print(f"  Catastrophic misses (pred<{pred_max}mm): {n_miss} "
          f"({100*n_miss/max(n_heavy,1):.1f}% of them)")

    if n_miss == 0:
        print("  No catastrophic misses found. Try raising --pred-max or widening the range.")
        return

    # ── Extract per-sample diagnostics + assign a bucket ──
    rows = []
    for i in miss_indices:
        s = val_ds.samples[i]
        feat = extract_radar_features(s, fields)
        center_dbz = feat.get('reflectivity_center_max', np.nan)
        patch_dbz = feat.get('reflectivity_patch_max', np.nan)
        beam_h = feat.get('beam_height_center_max', np.nan)
        echo_top = feat.get('echo_top_height_center_max', np.nan)
        ridx = s.get('radar_indices', []) or []
        n_missing = sum(1 for r in ridx if r is None)

        c = center_dbz if not np.isnan(center_dbz) else -99.0
        p = patch_dbz if not np.isnan(patch_dbz) else -99.0
        if c >= echo_thresh:
            bucket = 'B_model_failure'
        elif p >= patch_thresh:
            bucket = 'A2_rain_nearby'
        else:
            bucket = 'A1_radar_blind'

        nb = neighbor_stats(s, neighbor_index)
        rows.append(dict(
            idx=int(i),
            station=s.get('station_name', ''),
            date=_date(s), hour=str(s.get('hour_start', '')),
            actual=float(all_actuals[i]), pred=float(all_preds[i]),
            center_dbz=center_dbz, patch_dbz=patch_dbz,
            beam_h=beam_h, echo_top=echo_top, n_missing=n_missing,
            dump_ratio=s.get('dump_ratio'),
            bucket=bucket, **nb,
        ))

    bucket_counts = Counter(r['bucket'] for r in rows)
    print(f"\n{'='*70}")
    print("  CATASTROPHIC-MISS ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Bucket breakdown (what the radar saw):")
    labels = {
        'A1_radar_blind':  'A1  radar blind (center & patch dry)  -> gauge error / overshoot / missing scan',
        'A2_rain_nearby':  'A2  rain nearby, not at pixel         -> spatial offset / advection',
        'B_model_failure': 'B   radar saw it, model said ~0       -> model failure',
    }
    for b in ('A1_radar_blind', 'A2_rain_nearby', 'B_model_failure'):
        n = bucket_counts.get(b, 0)
        print(f"    {n:>4} ({100*n/n_miss:>5.1f}%)  {labels[b]}")

    # ── Per-bucket signal summary ──
    def _bstats(bucket):
        br = [r for r in rows if r['bucket'] == bucket]
        if not br:
            return None
        def med(key):
            vals = [r[key] for r in br if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
            return np.median(vals) if vals else np.nan
        dumps = [r['dump_ratio'] for r in br if r['dump_ratio'] is not None]
        return dict(
            n=len(br),
            center=med('center_dbz'), patch=med('patch_dbz'),
            beam=med('beam_h'), missing=med('n_missing'),
            nwet=med('n_wet'), nmax=med('neighbor_max'),
            near_km=med('nearest_wet_km'),
            dump=np.median(dumps) if dumps else np.nan,
        )

    print(f"\n  Per-bucket medians:")
    print(f"  {'Bucket':<18} {'n':>4} {'cen dBZ':>8} {'pch dBZ':>8} {'beam m':>8} "
          f"{'miss/12':>8} {'#wet nbr':>9} {'nbr max':>8} {'near km':>8} {'dump':>6}")
    print(f"  {'-'*18} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*6}")
    for b in ('A1_radar_blind', 'A2_rain_nearby', 'B_model_failure'):
        st = _bstats(b)
        if not st:
            continue
        def f(v, suff=''):
            return f"{v:.1f}{suff}" if v is not None and not np.isnan(v) else '—'
        print(f"  {b:<18} {st['n']:>4} {f(st['center']):>8} {f(st['patch']):>8} "
              f"{f(st['beam']):>8} {f(st['missing']):>8} {f(st['nwet']):>9} "
              f"{f(st['nmax']):>8} {f(st['near_km']):>8} {f(st['dump']):>6}")

    # ── Gauge-error vs real-rain read from neighbors ──
    wet_neighbor = [r for r in rows if r['n_wet'] and r['n_wet'] > 0]
    dry_neighbor = [r for r in rows if r['n_neighbors'] and r['n_wet'] == 0]
    print(f"\n  Neighbour cross-check (real rain vs gauge error):")
    print(f"    Misses with >=1 wet neighbour same hour: {len(wet_neighbor)} "
          f"({100*len(wet_neighbor)/n_miss:.1f}%) -> likely REAL rain radar/model missed")
    print(f"    Misses with ALL neighbours dry:          {len(dry_neighbor)} "
          f"({100*len(dry_neighbor)/n_miss:.1f}%) -> likely GAUGE ERROR / isolated spike")
    a1 = [r for r in rows if r['bucket'] == 'A1_radar_blind']
    if a1:
        a1_dry = sum(1 for r in a1 if r['n_neighbors'] and r['n_wet'] == 0)
        print(f"    Of the A1 (radar-blind) misses, {a1_dry}/{len(a1)} also had dry "
              f"neighbours -> strongest gauge-error candidates")

    # ── Per-station breakdown + clean train heavy-rain availability ──
    # If a station with many B (model-failure) misses also has few clean heavy
    # training samples, the model likely never learned heavy rain there and
    # defaults low (a data/coverage problem, not a loss problem).
    by_station = defaultdict(Counter)
    for r in rows:
        by_station[r['station']][r['bucket']] += 1
    print(f"\n  Per-station miss buckets vs. clean train heavy samples:")
    print(f"  {'Station':<22} {'miss':>5} {'A1':>4} {'A2':>4} {'B':>4} "
          f"{'train_heavy':>12} {'train_total':>12}")
    print(f"  {'-'*22} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*12} {'-'*12}")
    for station, bc in sorted(by_station.items(), key=lambda kv: -kv[1]['B_model_failure']):
        short = station.replace('Dangermond_', '')
        th = train_heavy.get(station, 0)
        tt = train_total.get(station, 0)
        print(f"  {short:<22} {sum(bc.values()):>5} {bc['A1_radar_blind']:>4} "
              f"{bc['A2_rain_nearby']:>4} {bc['B_model_failure']:>4} {th:>12} {tt:>12}")

    # ── Weather context (low RH => virga/evaporation) ──
    rh_vals = {}
    if query_hourly_avg is not None:
        try:
            from database.config import connect, create_session
            engine = connect()
            session = create_session(engine)
            hours_by_station = defaultdict(list)
            for r in rows:
                hours_by_station[r['station']].append(r['hour'])
            for station, hours in hours_by_station.items():
                rh = query_hourly_avg(session, station, '%Relative Humidity Avg%', hours)
                for h in hours:
                    v = rh.get(str(h), np.nan)
                    if not np.isnan(v):
                        rh_vals[(station, h)] = v
            session.close()
        except Exception as e:
            print(f"\n  ⚠ Humidity query skipped ({e})")
    if rh_vals:
        arr = np.array(list(rh_vals.values()))
        print(f"\n  Surface relative humidity at miss hours:")
        print(f"    Median: {np.median(arr):.1f}%   < 90%: {(arr<90).mean()*100:.1f}%   "
              f"< 80%: {(arr<80).mean()*100:.1f}%  (low RH supports virga/evaporation)")

    # ── Plots ──
    output_dir = Path(run_dir) if run_dir else Path('evaluation_figures/unet_dualpol')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Catastrophic Misses ({actual_min}-{actual_max}mm actual, pred<{pred_max}mm)\n'
                 f'n={n_miss}', fontsize=14, fontweight='bold')
    bcolors = {'A1_radar_blind': '#e74c3c', 'A2_rain_nearby': '#f39c12', 'B_model_failure': '#8e44ad'}

    # 1. Bucket counts
    ax = axes[0, 0]
    bs = ['A1_radar_blind', 'A2_rain_nearby', 'B_model_failure']
    ax.bar([b.split('_')[0] for b in bs], [bucket_counts.get(b, 0) for b in bs],
           color=[bcolors[b] for b in bs])
    ax.set_ylabel('Count')
    ax.set_title('Miss buckets')

    # 2. center vs patch dBZ, colored by bucket
    ax = axes[0, 1]
    for b in bs:
        br = [r for r in rows if r['bucket'] == b]
        if br:
            ax.scatter([r['center_dbz'] for r in br], [r['patch_dbz'] for r in br],
                       c=bcolors[b], label=b.split('_')[0], alpha=0.6, s=25)
    ax.axvline(echo_thresh, color='gray', ls='--', alpha=0.6)
    ax.axhline(patch_thresh, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('Center max dBZ')
    ax.set_ylabel('Patch max dBZ')
    ax.set_title('What the radar saw')
    ax.legend(fontsize=8)

    # 3. Neighbour max precip
    ax = axes[0, 2]
    nm = [r['neighbor_max'] for r in rows if not np.isnan(r['neighbor_max'])]
    if nm:
        ax.hist(nm, bins=20, color='#3498db', alpha=0.8)
    ax.axvline(1.0, color='red', ls='--', alpha=0.7, label='1mm wet threshold')
    ax.set_xlabel('Max neighbour precip same hour (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Did neighbours see rain?')
    ax.legend(fontsize=8)

    # 4. Beam height (overshoot check for A1)
    ax = axes[1, 0]
    bh = [r['beam_h'] for r in rows if not np.isnan(r['beam_h'])]
    if bh:
        ax.hist(bh, bins=20, color='#16a085', alpha=0.8)
    ax.set_xlabel('Beam height of lowest gate (m)')
    ax.set_ylabel('Count')
    ax.set_title('Beam overshoot check')

    # 5. Missing scans
    ax = axes[1, 1]
    ms = [r['n_missing'] for r in rows]
    ax.hist(ms, bins=np.arange(-0.5, 13.5, 1), color='#95a5a6', alpha=0.8)
    ax.set_xlabel('Missing radar sub-scans (of 12)')
    ax.set_ylabel('Count')
    ax.set_title('Radar coverage gaps')

    # 6. Actual vs neighbour-max (gauge-error scatter)
    ax = axes[1, 2]
    for b in bs:
        br = [r for r in rows if r['bucket'] == b and not np.isnan(r['neighbor_max'])]
        if br:
            ax.scatter([r['actual'] for r in br], [r['neighbor_max'] for r in br],
                       c=bcolors[b], label=b.split('_')[0], alpha=0.6, s=25)
    ax.axhline(1.0, color='red', ls='--', alpha=0.7)
    ax.set_xlabel('This gauge actual (mm)')
    ax.set_ylabel('Max neighbour precip (mm)')
    ax.set_title('Gauge vs neighbours\n(low y = isolated = likely error)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / 'catastrophic_miss_diagnostics.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved diagnostic plot to: {out_path}")
    plt.close()

    # ── CSV ──
    csv_path = output_dir / 'catastrophic_miss_samples.csv'
    with open(csv_path, 'w') as f:
        f.write('station,date,hour,actual_mm,pred_mm,bucket,center_dbz,patch_dbz,'
                'beam_height_m,echo_top_m,missing_scans,dump_ratio,'
                'n_neighbors,n_wet_neighbors,neighbor_max_mm,nearest_wet_km\n')
        for r in sorted(rows, key=lambda x: -x['actual']):
            def g(v, fmt='.2f'):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return ''
                return format(v, fmt)
            f.write(f"{r['station']},{r['date']},{r['hour']},{r['actual']:.2f},{r['pred']:.2f},"
                    f"{r['bucket']},{g(r['center_dbz'],'.1f')},{g(r['patch_dbz'],'.1f')},"
                    f"{g(r['beam_h'],'.0f')},{g(r['echo_top'],'.0f')},{r['n_missing']},"
                    f"{g(r['dump_ratio'],'.2f')},{r['n_neighbors']},{r['n_wet']},"
                    f"{g(r['neighbor_max'],'.2f')},{g(r['nearest_wet_km'],'.1f')}\n")
    print(f"  ✓ Saved sample details to: {csv_path}")

    station_csv = output_dir / 'catastrophic_miss_by_station.csv'
    with open(station_csv, 'w') as f:
        f.write('station,total_misses,A1_radar_blind,A2_rain_nearby,B_model_failure,'
                'train_heavy_clean,train_total_clean\n')
        for station, bc in sorted(by_station.items(), key=lambda kv: -kv[1]['B_model_failure']):
            f.write(f"{station},{sum(bc.values())},{bc['A1_radar_blind']},"
                    f"{bc['A2_rain_nearby']},{bc['B_model_failure']},"
                    f"{train_heavy.get(station, 0)},{train_total.get(station, 0)}\n")
    print(f"  ✓ Saved per-station breakdown to: {station_csv}")


def _date(s):
    h = s.get('hour_start', None)
    if h is None:
        return ''
    if hasattr(h, 'date'):
        return h.date()
    return str(h).split(' ')[0]


def main():
    parser = argparse.ArgumentParser(description='Diagnose catastrophic U-Net misses (heavy actual, ~0 prediction)')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to the model run directory')
    parser.add_argument('--pred-max', type=float, default=1.0,
                        help='Flag a miss when prediction < this (mm/hr)')
    parser.add_argument('--actual-min', type=float, default=5.0,
                        help='Minimum actual precip to consider (mm/hr)')
    parser.add_argument('--actual-max', type=float, default=25.0,
                        help='Maximum actual precip to consider (mm/hr)')
    parser.add_argument('--echo-thresh', type=float, default=20.0,
                        help='Center dBZ above which the radar "saw" rain at the pixel (bucket B)')
    parser.add_argument('--patch-thresh', type=float, default=25.0,
                        help='Patch dBZ above which there is rain nearby (bucket A2)')
    args = parser.parse_args()
    run_diagnostic(args.run_dir, args.pred_max, args.actual_min, args.actual_max,
                   args.echo_thresh, args.patch_thresh)


if __name__ == '__main__':
    main()
