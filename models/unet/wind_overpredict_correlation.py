"""
wind_overpredict_correlation.py — Correlate wind speed with model overpredictions.

Queries Dendra database for "Wind Speed Avg" at stations where the U-Net
overpredicts precipitation, to test the hypothesis that high wind causes
gauge undercatch.

Run from project root:
    python -m models.unet.wind_overpredict_correlation --run-dir models/checkpoints/unet_dualpol/<run_name>
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import func, cast, Date

from database.config import connect, create_session
from database.models.DendraStations import DendraStation, DendraDatastream, DendraDatapoint


def get_wind_speed_for_hours(session, station_name, hours):
    """
    Query hourly average wind speed for a station at specific hours.

    Parameters
    ----------
    session : SQLAlchemy session
    station_name : str (e.g., 'Dangermond_Oaks')
    hours : list of datetime-like strings (e.g., '2023-01-05 04:00:00')

    Returns
    -------
    dict : {hour_str: wind_speed_m_s} for hours where data exists
    """
    station = session.query(DendraStation).filter(
        DendraStation.name == station_name
    ).first()

    if station is None:
        return {}

    wind_ds = session.query(DendraDatastream).filter(
        DendraDatastream.station_id == station.id,
        DendraDatastream.name.ilike('%Wind Speed%')
    ).first()

    if wind_ds is None:
        return {}

    results = {}
    for hour_str in hours:
        hour_start = pd.to_datetime(hour_str)
        hour_end = hour_start + timedelta(hours=1)

        wind_data = session.query(
            func.avg(DendraDatapoint.value).label('avg_wind')
        ).filter(
            DendraDatapoint.datastream_id == wind_ds.id,
            DendraDatapoint.timestamp_utc >= hour_start,
            DendraDatapoint.timestamp_utc < hour_end,
        ).scalar()

        if wind_data is not None:
            results[hour_str] = float(wind_data)

    return results


def get_wind_for_all_samples(session, samples_df):
    """Query wind speed for all overprediction and comparison samples."""
    wind_data = []

    stations = samples_df['station'].unique()
    for station in stations:
        station_samples = samples_df[samples_df['station'] == station]
        hours = station_samples['hour'].tolist()
        wind_dict = get_wind_speed_for_hours(session, station, hours)

        for _, row in station_samples.iterrows():
            wind_speed = wind_dict.get(row['hour'], np.nan)
            wind_data.append({
                'station': row['station'],
                'hour': row['hour'],
                'actual_mm': row['actual_mm'],
                'predicted_mm': row['predicted_mm'],
                'residual': row['predicted_mm'] - row['actual_mm'],
                'wind_speed_ms': wind_speed,
                'category': row['category'],
            })

    return pd.DataFrame(wind_data)


def run_wind_correlation(run_dir, pred_threshold=8.0, actual_threshold=5.0):
    """Main analysis: correlate wind speed with overpredictions."""
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

    # Categorize samples
    overpredict_mask = (all_preds >= pred_threshold) & (all_actuals < actual_threshold)
    correct_mask = (np.abs(all_preds - all_actuals) < 2.0) & (all_actuals >= 2.0) & (all_actuals < 15.0)

    # Build sample dataframes
    samples_list = []
    for i in range(len(all_preds)):
        s = val_ds.samples[i]
        hour_str = str(s.get('hour_start', ''))
        if overpredict_mask[i]:
            cat = 'overpredict'
        elif correct_mask[i]:
            cat = 'correct'
        else:
            continue
        samples_list.append({
            'station': s.get('station_name', ''),
            'hour': hour_str,
            'actual_mm': all_actuals[i],
            'predicted_mm': all_preds[i],
            'category': cat,
        })

    samples_df = pd.DataFrame(samples_list)
    n_over = (samples_df['category'] == 'overpredict').sum()
    n_correct = (samples_df['category'] == 'correct').sum()
    print(f"\n  Samples to query wind for:")
    print(f"    Overpredictions (pred>={pred_threshold}, actual<{actual_threshold}): {n_over}")
    print(f"    Correct predictions (|residual|<2, actual 2-15mm): {n_correct}")

    # Limit correct samples to keep DB queries reasonable
    if n_correct > 300:
        correct_idx = samples_df[samples_df['category'] == 'correct'].sample(300, random_state=42).index
        overpredict_idx = samples_df[samples_df['category'] == 'overpredict'].index
        samples_df = samples_df.loc[correct_idx.union(overpredict_idx)]
        print(f"    (Subsampled correct to 300 for DB efficiency)")

    # Query wind speed from database
    print(f"\n  Querying wind speed from database...")
    engine = connect()
    session = create_session(engine)

    wind_df = get_wind_for_all_samples(session, samples_df)
    session.close()

    wind_valid = wind_df.dropna(subset=['wind_speed_ms'])
    n_with_wind = len(wind_valid)
    print(f"  ✓ Got wind data for {n_with_wind}/{len(wind_df)} samples")

    over_wind = wind_valid[wind_valid['category'] == 'overpredict']
    correct_wind = wind_valid[wind_valid['category'] == 'correct']

    print(f"\n{'='*70}")
    print("  WIND SPEED vs OVERPREDICTION ANALYSIS")
    print(f"{'='*70}")

    if len(over_wind) > 0:
        print(f"\n  Wind speed summary (m/s):")
        print(f"  {'Metric':<25} {'Overpredictions':>15} {'Correct preds':>15}")
        print(f"  {'-'*25} {'-'*15} {'-'*15}")
        print(f"  {'Mean':<25} {over_wind['wind_speed_ms'].mean():>15.2f} {correct_wind['wind_speed_ms'].mean():>15.2f}")
        print(f"  {'Median':<25} {over_wind['wind_speed_ms'].median():>15.2f} {correct_wind['wind_speed_ms'].median():>15.2f}")
        print(f"  {'Std':<25} {over_wind['wind_speed_ms'].std():>15.2f} {correct_wind['wind_speed_ms'].std():>15.2f}")
        print(f"  {'Max':<25} {over_wind['wind_speed_ms'].max():>15.2f} {correct_wind['wind_speed_ms'].max():>15.2f}")
        print(f"  {'N samples':<25} {len(over_wind):>15} {len(correct_wind):>15}")

        # Wind speed bins
        print(f"\n  Wind speed distribution:")
        bins = [(0, 2), (2, 5), (5, 8), (8, 12), (12, float('inf'))]
        labels = ['Calm (0-2)', 'Light (2-5)', 'Moderate (5-8)', 'Strong (8-12)', 'Very strong (12+)']
        print(f"  {'Category':<20} {'Overpredict':>12} {'Correct':>12} {'Over % higher':>15}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*15}")
        for (lo, hi), label in zip(bins, labels):
            n_over_bin = len(over_wind[(over_wind['wind_speed_ms'] >= lo) & (over_wind['wind_speed_ms'] < hi)])
            n_correct_bin = len(correct_wind[(correct_wind['wind_speed_ms'] >= lo) & (correct_wind['wind_speed_ms'] < hi)])
            pct_over = 100 * n_over_bin / max(len(over_wind), 1)
            pct_correct = 100 * n_correct_bin / max(len(correct_wind), 1)
            diff = pct_over - pct_correct
            print(f"  {label:<20} {pct_over:>10.1f}% {pct_correct:>10.1f}% {diff:>+13.1f}%")

        # Correlation between wind speed and residual
        wind_all = wind_valid[['wind_speed_ms', 'residual']].dropna()
        if len(wind_all) > 10:
            corr = wind_all['wind_speed_ms'].corr(wind_all['residual'])
            print(f"\n  Correlation (wind speed vs. residual): r = {corr:.3f}")
            if abs(corr) > 0.2:
                print(f"    → {'Positive' if corr > 0 else 'Negative'} correlation: "
                      f"{'higher wind = more overprediction (gauge undercatch likely)' if corr > 0 else 'unexpected direction'}")
            else:
                print(f"    → Weak correlation: wind alone doesn't explain overpredictions")

    # Plot
    output_dir = Path(run_dir) if run_dir else Path('evaluation_figures/unet_dualpol')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Wind Speed vs. Model Overprediction\n'
                 f'(pred≥{pred_threshold}mm, actual<{actual_threshold}mm)',
                 fontsize=14, fontweight='bold')

    # 1. Wind speed distributions
    ax = axes[0, 0]
    if len(over_wind) > 0 and len(correct_wind) > 0:
        bins_wind = np.linspace(0, max(wind_valid['wind_speed_ms'].max(), 15), 20)
        ax.hist(correct_wind['wind_speed_ms'], bins=bins_wind, alpha=0.5, density=True,
                label=f'Correct (n={len(correct_wind)})', color='#2ecc71')
        ax.hist(over_wind['wind_speed_ms'], bins=bins_wind, alpha=0.7, density=True,
                label=f'Overpredictions (n={len(over_wind)})', color='#e74c3c')
        ax.axvline(over_wind['wind_speed_ms'].median(), color='red', linestyle='--', alpha=0.7)
        ax.axvline(correct_wind['wind_speed_ms'].median(), color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('Wind Speed Distribution')
    ax.legend()

    # 2. Scatter: wind speed vs residual
    ax = axes[0, 1]
    if len(wind_valid) > 0:
        over_w = wind_valid[wind_valid['category'] == 'overpredict']
        correct_w = wind_valid[wind_valid['category'] == 'correct']
        ax.scatter(correct_w['wind_speed_ms'], correct_w['residual'],
                   alpha=0.3, s=20, c='#2ecc71', label='Correct')
        ax.scatter(over_w['wind_speed_ms'], over_w['residual'],
                   alpha=0.8, s=40, c='#e74c3c', label='Overpredictions')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Residual (pred - actual, mm/hr)')
        ax.set_title('Wind Speed vs. Prediction Error')
        ax.legend()

    # 3. Box plot by wind category
    ax = axes[1, 0]
    if len(wind_valid) > 0:
        wind_valid_copy = wind_valid.copy()
        wind_valid_copy['wind_cat'] = pd.cut(
            wind_valid_copy['wind_speed_ms'],
            bins=[0, 2, 5, 8, 12, 50],
            labels=['0-2', '2-5', '5-8', '8-12', '12+']
        )
        wind_valid_copy.boxplot(column='residual', by='wind_cat', ax=ax)
        ax.set_xlabel('Wind Speed Category (m/s)')
        ax.set_ylabel('Residual (pred - actual, mm/hr)')
        ax.set_title('Prediction Error by Wind Speed')
        ax.get_figure().suptitle('')

    # 4. Scatter: wind speed vs prediction colored by actual
    ax = axes[1, 1]
    if len(wind_valid) > 0:
        sc = ax.scatter(wind_valid['wind_speed_ms'], wind_valid['predicted_mm'],
                        c=wind_valid['actual_mm'], cmap='YlOrRd',
                        alpha=0.6, s=30, vmin=0, vmax=15)
        ax.axhline(pred_threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'Pred threshold ({pred_threshold}mm)')
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Predicted Precipitation (mm/hr)')
        ax.set_title('Wind Speed vs. Prediction\n(colored by actual precip)')
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label='Actual (mm/hr)')

    plt.tight_layout()
    out_path = output_dir / 'wind_overpredict_correlation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved wind correlation plot to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Correlate wind speed with U-Net overpredictions')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to the model run directory')
    parser.add_argument('--pred-threshold', type=float, default=8.0,
                        help='Minimum prediction to flag as overprediction (mm/hr)')
    parser.add_argument('--actual-threshold', type=float, default=5.0,
                        help='Maximum actual precip for flagging (mm/hr)')
    args = parser.parse_args()

    run_wind_correlation(args.run_dir, args.pred_threshold, args.actual_threshold)


if __name__ == '__main__':
    main()
