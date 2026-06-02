"""
diagnose_overpredict_weather.py — Correlate surface weather conditions
(relative humidity, air temperature) with model overpredictions to test
the sub-cloud evaporation hypothesis.

If overpredictions correlate with low RH and high temperature, it suggests
rain is evaporating before reaching the gauge (virga).

Run from project root:
    python -m models.unet.diagnose_overpredict_weather --run-dir models/checkpoints/unet_dualpol/<run_name>
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from sqlalchemy import func

from database.config import connect, create_session
from database.models.DendraStations import DendraStation, DendraDatastream, DendraDatapoint


def query_hourly_avg(session, station_name, datastream_pattern, hours):
    """
    Query hourly average for a datastream at a station for specific hours.
    Uses ilike pattern matching for datastream name.
    """
    station = session.query(DendraStation).filter(
        DendraStation.name == station_name
    ).first()
    if station is None:
        return {}

    ds = session.query(DendraDatastream).filter(
        DendraDatastream.station_id == station.id,
        DendraDatastream.name.ilike(datastream_pattern)
    ).first()
    if ds is None:
        return {}

    results = {}
    for hour_str in hours:
        hour_start = pd.to_datetime(hour_str)
        hour_end = hour_start + timedelta(hours=1)

        val = session.query(
            func.avg(DendraDatapoint.value)
        ).filter(
            DendraDatapoint.datastream_id == ds.id,
            DendraDatapoint.timestamp_utc >= hour_start,
            DendraDatapoint.timestamp_utc < hour_end,
        ).scalar()

        if val is not None:
            results[hour_str] = float(val)

    return results


def query_station_elevations(session, station_names):
    """Return {station_name: elevation_m} from the DendraStation table."""
    elevations = {}
    for name in station_names:
        station = session.query(DendraStation).filter(
            DendraStation.name == name
        ).first()
        if station is not None and station.elevation is not None:
            elevations[name] = float(station.elevation)
        else:
            elevations[name] = np.nan
    return elevations


def run_weather_correlation(run_dir, pred_threshold=8.0, actual_threshold=5.0):
    """Correlate RH and temperature with overpredictions."""
    import torch
    from torch.utils.data import DataLoader
    from models.unet.evaluate import load_model, find_checkpoint
    from models.unet.dataset import RadarGaugeDataset, resolve_fields
    from models.unet.train import (
        filter_nan_radar, filter_biased_extremes, filter_bad_samples,
        filter_suspect_station_days, filter_radar_unsupported
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = find_checkpoint(run_dir=run_dir)
    model, cfg = load_model(checkpoint_path, device)

    pickle_path = cfg.get('pickle_path', 'dataset/outputs/3d/radar_gauge_dataset_vertical_9500.pkl')
    dem_path = cfg.get('dem_path', 'dem/preserve_dem_10m_utm.tif')
    fields = resolve_fields(cfg.get('fields'))
    log_target = cfg.get('log_target', True)

    ds_kwargs = dict(
        dem_path=dem_path, fields=fields,
        use_dem=cfg.get('use_dem', True),
        use_mask=cfg.get('use_mask', True),
        use_temporal_pos=cfg.get('use_temporal_pos', True),
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

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Run inference
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

    # Categorize
    overpredict_mask = (all_preds >= pred_threshold) & (all_actuals < actual_threshold)
    correct_mask = (np.abs(all_preds - all_actuals) < 2.0) & (all_actuals >= 2.0) & (all_actuals < 15.0)

    samples_list = []
    for i in range(len(all_preds)):
        s = val_ds.samples[i]
        if overpredict_mask[i]:
            cat = 'overpredict'
        elif correct_mask[i]:
            cat = 'correct'
        else:
            continue
        samples_list.append({
            'station': s.get('station_name', ''),
            'hour': str(s.get('hour_start', '')),
            'actual_mm': all_actuals[i],
            'predicted_mm': all_preds[i],
            'residual': all_preds[i] - all_actuals[i],
            'category': cat,
        })

    samples_df = pd.DataFrame(samples_list)
    n_over = (samples_df['category'] == 'overpredict').sum()
    n_correct = (samples_df['category'] == 'correct').sum()
    print(f"\n  Samples to query weather for:")
    print(f"    Overpredictions: {n_over}")
    print(f"    Correct predictions: {n_correct}")

    if n_correct > 300:
        correct_idx = samples_df[samples_df['category'] == 'correct'].sample(300, random_state=42).index
        overpredict_idx = samples_df[samples_df['category'] == 'overpredict'].index
        samples_df = samples_df.loc[correct_idx.union(overpredict_idx)]
        print(f"    (Subsampled correct to 300)")

    # Query weather from database
    print(f"\n  Querying relative humidity and temperature from database...")
    engine = connect()
    session = create_session(engine)

    rh_data = {}
    temp_data = {}
    stations = samples_df['station'].unique()

    for station in stations:
        station_hours = samples_df[samples_df['station'] == station]['hour'].tolist()

        rh_dict = query_hourly_avg(session, station, '%Relative Humidity Avg%', station_hours)
        temp_dict = query_hourly_avg(session, station, '%Air Temp Avg%', station_hours)

        for hour in station_hours:
            key = (station, hour)
            rh_data[key] = rh_dict.get(hour, np.nan)
            temp_data[key] = temp_dict.get(hour, np.nan)

    station_elevations = query_station_elevations(session, stations)

    session.close()

    samples_df['rh_pct'] = samples_df.apply(
        lambda r: rh_data.get((r['station'], r['hour']), np.nan), axis=1)
    samples_df['temp_c'] = samples_df.apply(
        lambda r: temp_data.get((r['station'], r['hour']), np.nan), axis=1)
    samples_df['elevation_m'] = samples_df['station'].map(station_elevations)

    rh_valid = samples_df.dropna(subset=['rh_pct'])
    temp_valid = samples_df.dropna(subset=['temp_c'])
    print(f"  ✓ Got RH data for {len(rh_valid)}/{len(samples_df)} samples")
    print(f"  ✓ Got temperature data for {len(temp_valid)}/{len(samples_df)} samples")

    over_rh = rh_valid[rh_valid['category'] == 'overpredict']
    correct_rh = rh_valid[rh_valid['category'] == 'correct']
    over_temp = temp_valid[temp_valid['category'] == 'overpredict']
    correct_temp = temp_valid[temp_valid['category'] == 'correct']

    print(f"\n{'='*70}")
    print("  WEATHER CONDITIONS vs OVERPREDICTION")
    print(f"{'='*70}")

    # RH summary
    if len(over_rh) > 0 and len(correct_rh) > 0:
        print(f"\n  Relative Humidity (%):")
        print(f"  {'Metric':<20} {'Overpredictions':>15} {'Correct preds':>15} {'Difference':>12}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*12}")
        over_rh_mean = over_rh['rh_pct'].mean()
        correct_rh_mean = correct_rh['rh_pct'].mean()
        print(f"  {'Mean':<20} {over_rh_mean:>14.1f}% {correct_rh_mean:>14.1f}% {over_rh_mean-correct_rh_mean:>+11.1f}%")
        print(f"  {'Median':<20} {over_rh['rh_pct'].median():>14.1f}% {correct_rh['rh_pct'].median():>14.1f}% {over_rh['rh_pct'].median()-correct_rh['rh_pct'].median():>+11.1f}%")
        print(f"  {'Min':<20} {over_rh['rh_pct'].min():>14.1f}% {correct_rh['rh_pct'].min():>14.1f}%")
        print(f"  {'N samples':<20} {len(over_rh):>15} {len(correct_rh):>15}")

        # RH bins
        print(f"\n  RH distribution:")
        rh_bins = [(0, 50), (50, 70), (70, 85), (85, 95), (95, 101)]
        rh_labels = ['Dry (<50%)', 'Low (50-70%)', 'Moderate (70-85%)', 'High (85-95%)', 'Saturated (95%+)']
        print(f"  {'Category':<22} {'Overpredict':>12} {'Correct':>12}")
        print(f"  {'-'*22} {'-'*12} {'-'*12}")
        for (lo, hi), label in zip(rh_bins, rh_labels):
            n_o = len(over_rh[(over_rh['rh_pct'] >= lo) & (over_rh['rh_pct'] < hi)])
            n_c = len(correct_rh[(correct_rh['rh_pct'] >= lo) & (correct_rh['rh_pct'] < hi)])
            pct_o = 100 * n_o / max(len(over_rh), 1)
            pct_c = 100 * n_c / max(len(correct_rh), 1)
            print(f"  {label:<22} {pct_o:>10.1f}% {pct_c:>10.1f}%")

        corr_rh = rh_valid[['rh_pct', 'residual']].dropna()
        if len(corr_rh) > 10:
            r = corr_rh['rh_pct'].corr(corr_rh['residual'])
            print(f"\n  Correlation (RH vs. residual): r = {r:.3f}")
            if r < -0.15:
                print(f"    → Negative correlation: lower humidity = more overprediction (supports evaporation)")
            elif r > 0.15:
                print(f"    → Positive correlation: unexpected direction")
            else:
                print(f"    → Weak correlation")

    # Temperature summary
    if len(over_temp) > 0 and len(correct_temp) > 0:
        print(f"\n  Air Temperature (°C):")
        print(f"  {'Metric':<20} {'Overpredictions':>15} {'Correct preds':>15} {'Difference':>12}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*12}")
        over_t_mean = over_temp['temp_c'].mean()
        correct_t_mean = correct_temp['temp_c'].mean()
        print(f"  {'Mean':<20} {over_t_mean:>14.1f}° {correct_t_mean:>14.1f}° {over_t_mean-correct_t_mean:>+11.1f}°")
        print(f"  {'Median':<20} {over_temp['temp_c'].median():>14.1f}° {correct_temp['temp_c'].median():>14.1f}° {over_temp['temp_c'].median()-correct_temp['temp_c'].median():>+11.1f}°")
        print(f"  {'N samples':<20} {len(over_temp):>15} {len(correct_temp):>15}")

        corr_temp = temp_valid[['temp_c', 'residual']].dropna()
        if len(corr_temp) > 10:
            r = corr_temp['temp_c'].corr(corr_temp['residual'])
            print(f"\n  Correlation (temp vs. residual): r = {r:.3f}")
            if r > 0.15:
                print(f"    → Positive correlation: warmer = more overprediction (supports evaporation)")
            elif r < -0.15:
                print(f"    → Negative correlation: unexpected direction")
            else:
                print(f"    → Weak correlation")

    # Elevation summary
    elev_valid = samples_df.dropna(subset=['elevation_m'])
    over_elev = elev_valid[elev_valid['category'] == 'overpredict']
    correct_elev = elev_valid[elev_valid['category'] == 'correct']
    if len(over_elev) > 0 and len(correct_elev) > 0:
        print(f"\n  Station Elevation (m):")
        print(f"  {'Metric':<20} {'Overpredictions':>15} {'Correct preds':>15} {'Difference':>12}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*12}")
        over_e_mean = over_elev['elevation_m'].mean()
        correct_e_mean = correct_elev['elevation_m'].mean()
        print(f"  {'Mean':<20} {over_e_mean:>14.1f}m {correct_e_mean:>14.1f}m {over_e_mean-correct_e_mean:>+11.1f}m")
        print(f"  {'Median':<20} {over_elev['elevation_m'].median():>14.1f}m {correct_elev['elevation_m'].median():>14.1f}m {over_elev['elevation_m'].median()-correct_elev['elevation_m'].median():>+11.1f}m")
        print(f"  {'Min':<20} {over_elev['elevation_m'].min():>14.1f}m {correct_elev['elevation_m'].min():>14.1f}m")
        print(f"  {'Max':<20} {over_elev['elevation_m'].max():>14.1f}m {correct_elev['elevation_m'].max():>14.1f}m")
        print(f"  {'N samples':<20} {len(over_elev):>15} {len(correct_elev):>15}")

        # Per-station elevation and overprediction rate
        print(f"\n  Per-station elevation vs. overprediction count:")
        print(f"  {'Station':<28} {'Elev (m)':>10} {'Overpr':>8} {'Correct':>8}")
        print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*8}")
        for station in sorted(stations, key=lambda s: station_elevations.get(s, 0) or 0, reverse=True):
            elev = station_elevations.get(station, np.nan)
            n_o = len(over_elev[over_elev['station'] == station])
            n_c = len(correct_elev[correct_elev['station'] == station])
            station_short = station.replace('Dangermond_', '')
            elev_str = f"{elev:.0f}" if not np.isnan(elev) else "N/A"
            print(f"  {station_short:<28} {elev_str:>10} {n_o:>8} {n_c:>8}")

        corr_elev = elev_valid[['elevation_m', 'residual']].dropna()
        if len(corr_elev) > 10:
            r = corr_elev['elevation_m'].corr(corr_elev['residual'])
            print(f"\n  Correlation (elevation vs. residual): r = {r:.3f}")
            if r > 0.15:
                print(f"    → Positive correlation: higher elevation = more overprediction "
                      f"(supports bright band / melting layer hypothesis)")
            elif r < -0.15:
                print(f"    → Negative correlation: unexpected direction")
            else:
                print(f"    → Weak correlation")

    # Elevation × Temperature interaction (bright band: high elevation + cold)
    interaction = samples_df.dropna(subset=['elevation_m', 'temp_c']).copy()
    if len(interaction) > 20:
        elev_thresh = 200.0
        temp_thresh = float(interaction['temp_c'].median())
        interaction['date'] = interaction['hour'].apply(
            lambda h: h.date() if hasattr(h, 'date') else str(h).split(' ')[0])
        print(f"\n  Elevation × Temperature interaction:")
        print(f"    (elevation split @ {elev_thresh:.0f}m, temperature split @ median {temp_thresh:.1f}°C)")
        print(f"  {'Quadrant':<22} {'N':>6} {'StnDays':>8} {'OverpRate':>11} {'MeanResid':>11} {'MedResid':>10}")
        print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*11} {'-'*11} {'-'*10}")
        quadrants = [
            ('high elev + cold', interaction['elevation_m'] >= elev_thresh, interaction['temp_c'] < temp_thresh),
            ('high elev + warm', interaction['elevation_m'] >= elev_thresh, interaction['temp_c'] >= temp_thresh),
            ('low elev + cold',  interaction['elevation_m'] < elev_thresh,  interaction['temp_c'] < temp_thresh),
            ('low elev + warm',  interaction['elevation_m'] < elev_thresh,  interaction['temp_c'] >= temp_thresh),
        ]
        for label, em, tm in quadrants:
            q = interaction[em & tm]
            if len(q) == 0:
                print(f"  {label:<22} {0:>6} {0:>8} {'—':>11} {'—':>11} {'—':>10}")
                continue
            n_stn_days = q.groupby(['station', 'date']).ngroups
            over_rate = 100 * (q['category'] == 'overpredict').mean()
            print(f"  {label:<22} {len(q):>6} {n_stn_days:>8} {over_rate:>10.1f}% "
                  f"{q['residual'].mean():>+10.2f} {q['residual'].median():>+9.2f}")
        print(f"    → If 'high elev + cold' has the largest residuals/overprediction rate, "
              f"that's the bright-band signature.")
        print(f"    → StnDays = distinct station-days in each cell (small counts = treat with caution).")

    # Also check ZDR from the radar data for overpredictions
    print(f"\n  Differential Reflectivity (ZDR) from radar patch:")
    from models.unet.dataset import PICKLE_FIELD_ORDER
    if 'differential_reflectivity' in PICKLE_FIELD_ORDER:
        zdr_idx = PICKLE_FIELD_ORDER.index('differential_reflectivity')
        over_zdr = []
        correct_zdr_vals = []

        over_indices = samples_df[samples_df['category'] == 'overpredict'].index
        correct_indices = samples_df[samples_df['category'] == 'correct'].index

        for orig_i, row in samples_df.iterrows():
            match_idx = None
            for j, s in enumerate(val_ds.samples):
                if (s.get('station_name') == row['station'] and
                    str(s.get('hour_start', '')) == row['hour']):
                    match_idx = j
                    break
            if match_idx is None:
                continue

            rp = val_ds.samples[match_idx]['radar_patch']
            if zdr_idx < rp.shape[1]:
                cy, cx = rp.shape[2] // 2, rp.shape[3] // 2
                zdr_center = rp[:, zdr_idx, cy, cx]
                valid = zdr_center[(zdr_center != -9999.0) & ~np.isnan(zdr_center)]
                if len(valid) > 0:
                    mean_zdr = np.mean(valid)
                    if row['category'] == 'overpredict':
                        over_zdr.append(mean_zdr)
                    else:
                        correct_zdr_vals.append(mean_zdr)

        if over_zdr and correct_zdr_vals:
            print(f"  {'Metric':<20} {'Overpredictions':>15} {'Correct preds':>15}")
            print(f"  {'-'*20} {'-'*15} {'-'*15}")
            print(f"  {'Mean ZDR (dB)':<20} {np.mean(over_zdr):>15.3f} {np.mean(correct_zdr_vals):>15.3f}")
            print(f"  {'Median ZDR (dB)':<20} {np.median(over_zdr):>15.3f} {np.median(correct_zdr_vals):>15.3f}")
            print(f"  {'N samples':<20} {len(over_zdr):>15} {len(correct_zdr_vals):>15}")

    # ── Bright-band proxy: depressed RhoHV + low max-Z height at gauge pixel ──
    # RhoHV is stored collocated at the height of max reflectivity. A depressed
    # value there (mixed-phase melting hydrometeors) combined with a low max-Z
    # height (low freezing level) is the classic bright-band signature.
    print(f"\n  Bright-band proxy (RhoHV + max-Z height at gauge pixel):")
    bb_idx = {f: PICKLE_FIELD_ORDER.index(f)
              for f in ['cross_correlation_ratio', 'max_z_height']
              if f in PICKLE_FIELD_ORDER}
    if bb_idx:
        # Build (station, hour) -> sample index lookup once
        sample_lookup = {}
        for j, s in enumerate(val_ds.samples):
            key = (s.get('station_name'), str(s.get('hour_start', '')))
            sample_lookup.setdefault(key, j)

        rhohv_vals, maxz_vals = {}, {}
        for orig_i, row in samples_df.iterrows():
            j = sample_lookup.get((row['station'], row['hour']))
            if j is None:
                continue
            rp = val_ds.samples[j]['radar_patch']
            cy, cx = rp.shape[2] // 2, rp.shape[3] // 2
            for fname, store in (('cross_correlation_ratio', rhohv_vals),
                                 ('max_z_height', maxz_vals)):
                idx = bb_idx.get(fname)
                if idx is None or idx >= rp.shape[1]:
                    continue
                v = rp[:, idx, cy, cx]
                v = v[(v != -9999.0) & ~np.isnan(v) & (v > 0)]
                if len(v) > 0:
                    store[orig_i] = float(np.nanmean(v))

        samples_df['rhohv'] = samples_df.index.map(rhohv_vals)
        samples_df['max_z_height'] = samples_df.index.map(maxz_vals)

        over_i = samples_df['category'] == 'overpredict'
        correct_i = samples_df['category'] == 'correct'

        rh = samples_df['rhohv']
        o_rh, c_rh = rh[over_i].dropna(), rh[correct_i].dropna()
        if len(o_rh) and len(c_rh):
            print(f"  {'Metric':<22} {'Overpredictions':>15} {'Correct preds':>15} {'Difference':>12}")
            print(f"  {'-'*22} {'-'*15} {'-'*15} {'-'*12}")
            print(f"  {'Mean RhoHV':<22} {o_rh.mean():>15.4f} {c_rh.mean():>15.4f} {o_rh.mean()-c_rh.mean():>+12.4f}")
            print(f"  {'Median RhoHV':<22} {o_rh.median():>15.4f} {c_rh.median():>15.4f} {o_rh.median()-c_rh.median():>+12.4f}")
            print(f"  {'% RhoHV < 0.97':<22} {100*(o_rh<0.97).mean():>14.1f}% {100*(c_rh<0.97).mean():>14.1f}%")
            print(f"  {'N samples':<22} {len(o_rh):>15} {len(c_rh):>15}")

        mz = samples_df['max_z_height']
        o_mz, c_mz = mz[over_i].dropna(), mz[correct_i].dropna()
        if len(o_mz) and len(c_mz):
            print(f"\n  {'Metric':<22} {'Overpredictions':>15} {'Correct preds':>15} {'Difference':>12}")
            print(f"  {'-'*22} {'-'*15} {'-'*15} {'-'*12}")
            print(f"  {'Mean max-Z ht (m)':<22} {o_mz.mean():>15.0f} {c_mz.mean():>15.0f} {o_mz.mean()-c_mz.mean():>+12.0f}")
            print(f"  {'Median max-Z ht (m)':<22} {o_mz.median():>15.0f} {c_mz.median():>15.0f} {o_mz.median()-c_mz.median():>+12.0f}")
            print(f"  {'N samples':<22} {len(o_mz):>15} {len(c_mz):>15}")

        # Combined bright-band flag: depressed RhoHV AND low max-Z height
        bb_flag = (samples_df['rhohv'] < 0.97) & (samples_df['max_z_height'] < 2500)
        labeled = samples_df[samples_df['category'].isin(['overpredict', 'correct'])]
        flagged = labeled[bb_flag.reindex(labeled.index, fill_value=False)]
        unflagged = labeled[~bb_flag.reindex(labeled.index, fill_value=False)]
        if len(flagged) > 0 and len(unflagged) > 0:
            print(f"\n  Combined flag (RhoHV<0.97 AND max-Z<2500m):")
            print(f"    Overprediction rate WHEN flagged:   "
                  f"{100*(flagged['category']=='overpredict').mean():.1f}% (n={len(flagged)})")
            print(f"    Overprediction rate when NOT flagged: "
                  f"{100*(unflagged['category']=='overpredict').mean():.1f}% (n={len(unflagged)})")
            print(f"    → A higher overprediction rate when flagged supports the bright-band hypothesis.")

        corr_rh = samples_df[['rhohv', 'residual']].dropna()
        if len(corr_rh) > 10:
            r = corr_rh['rhohv'].corr(corr_rh['residual'])
            print(f"\n  Correlation (RhoHV vs. residual): r = {r:.3f}")
            if r < -0.15:
                print(f"    → Negative: lower RhoHV (melting layer) = more overprediction (supports bright band)")
            elif r > 0.15:
                print(f"    → Positive: unexpected direction")
            else:
                print(f"    → Weak correlation")

    # Generate plots
    output_dir = Path(run_dir) if run_dir else Path('evaluation_figures/unet_dualpol')
    fig, axes = plt.subplots(4, 3, figsize=(18, 21))
    fig.suptitle(f'Weather + Elevation + Bright-band vs. Overprediction\n'
                 f'(pred≥{pred_threshold}mm, actual<{actual_threshold}mm, n={n_over})',
                 fontsize=14, fontweight='bold')

    # 1. RH distribution
    ax = axes[0, 0]
    if len(over_rh) > 0 and len(correct_rh) > 0:
        bins_rh = np.linspace(0, 100, 25)
        ax.hist(correct_rh['rh_pct'], bins=bins_rh, alpha=0.5, density=True,
                label=f'Correct (n={len(correct_rh)})', color='#2ecc71')
        ax.hist(over_rh['rh_pct'], bins=bins_rh, alpha=0.7, density=True,
                label=f'Overpredictions (n={len(over_rh)})', color='#e74c3c')
        ax.axvline(over_rh['rh_pct'].median(), color='red', linestyle='--', alpha=0.7)
        ax.axvline(correct_rh['rh_pct'].median(), color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_ylabel('Density')
    ax.set_title('RH Distribution')
    ax.legend(fontsize=9)

    # 2. Temperature distribution
    ax = axes[0, 1]
    if len(over_temp) > 0 and len(correct_temp) > 0:
        t_min = min(over_temp['temp_c'].min(), correct_temp['temp_c'].min())
        t_max = max(over_temp['temp_c'].max(), correct_temp['temp_c'].max())
        bins_t = np.linspace(t_min, t_max, 25)
        ax.hist(correct_temp['temp_c'], bins=bins_t, alpha=0.5, density=True,
                label=f'Correct (n={len(correct_temp)})', color='#2ecc71')
        ax.hist(over_temp['temp_c'], bins=bins_t, alpha=0.7, density=True,
                label=f'Overpredictions (n={len(over_temp)})', color='#e74c3c')
        ax.axvline(over_temp['temp_c'].median(), color='red', linestyle='--', alpha=0.7)
        ax.axvline(correct_temp['temp_c'].median(), color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Air Temperature (°C)')
    ax.set_ylabel('Density')
    ax.set_title('Temperature Distribution')
    ax.legend(fontsize=9)

    # 3. RH vs residual scatter
    ax = axes[0, 2]
    if len(rh_valid) > 0:
        over_v = rh_valid[rh_valid['category'] == 'overpredict']
        correct_v = rh_valid[rh_valid['category'] == 'correct']
        ax.scatter(correct_v['rh_pct'], correct_v['residual'],
                   alpha=0.3, s=20, c='#2ecc71', label='Correct')
        ax.scatter(over_v['rh_pct'], over_v['residual'],
                   alpha=0.8, s=40, c='#e74c3c', label='Overpredictions')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_ylabel('Residual (pred - actual, mm/hr)')
    ax.set_title('RH vs. Prediction Error')
    ax.legend(fontsize=9)

    # 4. Temperature vs residual scatter
    ax = axes[1, 0]
    if len(temp_valid) > 0:
        over_v = temp_valid[temp_valid['category'] == 'overpredict']
        correct_v = temp_valid[temp_valid['category'] == 'correct']
        ax.scatter(correct_v['temp_c'], correct_v['residual'],
                   alpha=0.3, s=20, c='#2ecc71', label='Correct')
        ax.scatter(over_v['temp_c'], over_v['residual'],
                   alpha=0.8, s=40, c='#e74c3c', label='Overpredictions')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Air Temperature (°C)')
    ax.set_ylabel('Residual (pred - actual, mm/hr)')
    ax.set_title('Temperature vs. Prediction Error')
    ax.legend(fontsize=9)

    # 5. RH vs Temp colored by category
    ax = axes[1, 1]
    both_valid = samples_df.dropna(subset=['rh_pct', 'temp_c'])
    if len(both_valid) > 0:
        correct_b = both_valid[both_valid['category'] == 'correct']
        over_b = both_valid[both_valid['category'] == 'overpredict']
        ax.scatter(correct_b['temp_c'], correct_b['rh_pct'],
                   alpha=0.3, s=20, c='#2ecc71', label='Correct')
        ax.scatter(over_b['temp_c'], over_b['rh_pct'],
                   alpha=0.8, s=40, c='#e74c3c', label='Overpredictions')
    ax.set_xlabel('Air Temperature (°C)')
    ax.set_ylabel('Relative Humidity (%)')
    ax.set_title('Temp vs. RH (weather space)')
    ax.legend(fontsize=9)

    # 6. Residual vs RH colored by temperature
    ax = axes[1, 2]
    if len(both_valid) > 0:
        sc = ax.scatter(both_valid['rh_pct'], both_valid['residual'],
                        c=both_valid['temp_c'], cmap='coolwarm',
                        alpha=0.6, s=30)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Relative Humidity (%)')
        ax.set_ylabel('Residual (pred - actual, mm/hr)')
        ax.set_title('Residual vs. RH\n(colored by temperature)')
        plt.colorbar(sc, ax=ax, label='Temperature (°C)')

    # 7. Elevation distribution
    ax = axes[2, 0]
    if len(over_elev) > 0 and len(correct_elev) > 0:
        e_min = min(over_elev['elevation_m'].min(), correct_elev['elevation_m'].min())
        e_max = max(over_elev['elevation_m'].max(), correct_elev['elevation_m'].max())
        bins_e = np.linspace(e_min, e_max, 20)
        ax.hist(correct_elev['elevation_m'], bins=bins_e, alpha=0.5, density=True,
                label=f'Correct (n={len(correct_elev)})', color='#2ecc71')
        ax.hist(over_elev['elevation_m'], bins=bins_e, alpha=0.7, density=True,
                label=f'Overpredictions (n={len(over_elev)})', color='#e74c3c')
        ax.axvline(over_elev['elevation_m'].median(), color='red', linestyle='--', alpha=0.7)
        ax.axvline(correct_elev['elevation_m'].median(), color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Station Elevation (m)')
    ax.set_ylabel('Density')
    ax.set_title('Elevation Distribution')
    ax.legend(fontsize=9)

    # 8. Elevation vs residual scatter
    ax = axes[2, 1]
    if len(elev_valid) > 0:
        over_v = elev_valid[elev_valid['category'] == 'overpredict']
        correct_v = elev_valid[elev_valid['category'] == 'correct']
        ax.scatter(correct_v['elevation_m'], correct_v['residual'],
                   alpha=0.3, s=20, c='#2ecc71', label='Correct')
        ax.scatter(over_v['elevation_m'], over_v['residual'],
                   alpha=0.8, s=40, c='#e74c3c', label='Overpredictions')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Station Elevation (m)')
    ax.set_ylabel('Residual (pred - actual, mm/hr)')
    ax.set_title('Elevation vs. Prediction Error')
    ax.legend(fontsize=9)

    # 9. Elevation vs temperature colored by residual
    ax = axes[2, 2]
    elev_temp_valid = samples_df.dropna(subset=['elevation_m', 'temp_c'])
    if len(elev_temp_valid) > 0:
        sc = ax.scatter(elev_temp_valid['elevation_m'], elev_temp_valid['temp_c'],
                        c=elev_temp_valid['residual'], cmap='RdYlBu_r',
                        alpha=0.6, s=30, vmin=-8, vmax=8)
        ax.set_xlabel('Station Elevation (m)')
        ax.set_ylabel('Air Temperature (°C)')
        ax.set_title('Elevation vs. Temperature\n(colored by residual)')
        plt.colorbar(sc, ax=ax, label='Residual (mm/hr)')

    # 10. RhoHV distribution
    ax = axes[3, 0]
    if 'rhohv' in samples_df.columns:
        o_rh = samples_df.loc[samples_df['category'] == 'overpredict', 'rhohv'].dropna()
        c_rh = samples_df.loc[samples_df['category'] == 'correct', 'rhohv'].dropna()
        if len(o_rh) > 0 and len(c_rh) > 0:
            bins_rh = np.linspace(0.7, 1.0, 25)
            ax.hist(c_rh, bins=bins_rh, alpha=0.5, density=True,
                    label=f'Correct (n={len(c_rh)})', color='#2ecc71')
            ax.hist(o_rh, bins=bins_rh, alpha=0.7, density=True,
                    label=f'Overpredictions (n={len(o_rh)})', color='#e74c3c')
            ax.axvline(o_rh.median(), color='red', linestyle='--', alpha=0.7)
            ax.axvline(c_rh.median(), color='green', linestyle='--', alpha=0.7)
            ax.axvline(0.97, color='black', linestyle=':', alpha=0.6, label='0.97 (melting)')
    ax.set_xlabel('RhoHV at max-Z height')
    ax.set_ylabel('Density')
    ax.set_title('RhoHV Distribution')
    ax.legend(fontsize=8)

    # 11. max-Z height distribution
    ax = axes[3, 1]
    if 'max_z_height' in samples_df.columns:
        o_mz = samples_df.loc[samples_df['category'] == 'overpredict', 'max_z_height'].dropna()
        c_mz = samples_df.loc[samples_df['category'] == 'correct', 'max_z_height'].dropna()
        if len(o_mz) > 0 and len(c_mz) > 0:
            mz_max = max(o_mz.max(), c_mz.max())
            bins_mz = np.linspace(0, mz_max, 25)
            ax.hist(c_mz, bins=bins_mz, alpha=0.5, density=True,
                    label=f'Correct (n={len(c_mz)})', color='#2ecc71')
            ax.hist(o_mz, bins=bins_mz, alpha=0.7, density=True,
                    label=f'Overpredictions (n={len(o_mz)})', color='#e74c3c')
            ax.axvline(o_mz.median(), color='red', linestyle='--', alpha=0.7)
            ax.axvline(c_mz.median(), color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Height of max reflectivity (m)')
    ax.set_ylabel('Density')
    ax.set_title('Max-Z Height Distribution')
    ax.legend(fontsize=8)

    # 12. RhoHV vs max-Z height colored by residual (bright-band space)
    ax = axes[3, 2]
    if 'rhohv' in samples_df.columns and 'max_z_height' in samples_df.columns:
        bb_valid = samples_df.dropna(subset=['rhohv', 'max_z_height'])
        if len(bb_valid) > 0:
            sc = ax.scatter(bb_valid['rhohv'], bb_valid['max_z_height'],
                            c=bb_valid['residual'], cmap='RdYlBu_r',
                            alpha=0.6, s=30, vmin=-8, vmax=8)
            ax.axvline(0.97, color='black', linestyle=':', alpha=0.6)
            ax.axhline(2500, color='black', linestyle=':', alpha=0.6)
            ax.set_xlabel('RhoHV at max-Z height')
            ax.set_ylabel('Height of max reflectivity (m)')
            ax.set_title('Bright-band space\n(colored by residual)')
            plt.colorbar(sc, ax=ax, label='Residual (mm/hr)')

    plt.tight_layout()
    out_path = output_dir / 'weather_overpredict_correlation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved weather correlation plot to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Correlate weather with U-Net overpredictions')
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--pred-threshold', type=float, default=8.0)
    parser.add_argument('--actual-threshold', type=float, default=5.0)
    args = parser.parse_args()
    run_weather_correlation(args.run_dir, args.pred_threshold, args.actual_threshold)


if __name__ == '__main__':
    main()
