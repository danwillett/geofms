"""
compare_split_regimes.py — Compare rain-event regimes between train and val splits.

Tags each sample with a coarse storm/regime proxy from center-pixel radar structure
and gauge rain, then compares distributions across splits. Useful for detecting
temporal train/val imbalance (e.g. val years richer in warm-rain / AR events).

Run from project root:
    python -m models.unet.diagnostics.compare_split_regimes \\
        --pickle dataset/outputs/3d/radar_gauge_dataset_subml_9500.pkl

    python -m models.unet.diagnostics.compare_split_regimes \\
        --pickle dataset/outputs/3d/radar_gauge_dataset_vertlowmeltbb_9500.pkl \\
        --output-dir analysis/split_regimes_vertlowmeltbb \\
        --apply-filters
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.unet.diagnose_underestimates import extract_radar_features
from models.unet.train import (
    filter_bad_samples,
    filter_biased_extremes,
    filter_gauge_dumps,
    filter_nan_radar,
    filter_stations,
    filter_suspect_station_days,
)

RAIN_TIERS = [
    ('dry_light', 0.0, 1.0),
    ('moderate', 1.0, 5.0),
    ('heavy', 5.0, float('inf')),
]

REGIME_ORDER = [
    'light_dry',
    'bright_band',
    'warm_shallow_heavy',
    'shallow',
    'stratiform_mid',
    'convective_deep',
    'other',
]


def _ensure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


def _sample_date(sample) -> str:
    h = sample.get('hour_start')
    if h is None:
        return ''
    if hasattr(h, 'date'):
        return str(h.date())
    return str(h).split(' ')[0]


def _sample_year(sample) -> int:
    h = sample.get('hour_start')
    if hasattr(h, 'year'):
        return int(h.year)
    return int(str(h)[:4])


def _sample_month(sample) -> int:
    h = sample.get('hour_start')
    if hasattr(h, 'month'):
        return int(h.month)
    return int(str(h)[5:7])


def apply_blunt_filters(samples, exclude_stations=None):
    """Same filter chain as U-Net train/eval (blunt mode)."""
    out = list(samples)
    out = filter_stations(out, exclude_stations or [])
    out = filter_nan_radar(out)
    out = filter_biased_extremes(out)
    out = filter_bad_samples(out)
    out = filter_suspect_station_days(out)
    out = filter_gauge_dumps(out)
    return out


def rain_tier(actual_mm: float) -> str:
    for name, lo, hi in RAIN_TIERS:
        if lo <= actual_mm < hi:
            return name
    return RAIN_TIERS[-1][0]


def classify_regime(actual_mm: float, feats: dict) -> str:
    """
    Coarse regime tag from gauge rain + center-pixel radar structure.

    Priority order matters — first match wins.
    """
    eth = feats.get('echo_top_height_center_max', np.nan)
    max_z = feats.get('max_z_height_center_max', np.nan)
    rhohv_min = feats.get('rhohv_min_center_mean', np.nan)
    refl = feats.get('reflectivity_center_max', np.nan)

    if actual_mm < 1.0:
        return 'light_dry'

    # Bright-band / cold stratiform over-read signature (matches diagnose scripts).
    if np.isfinite(rhohv_min) and rhohv_min < 0.95:
        if np.isfinite(max_z) and max_z < 2500:
            return 'bright_band'

    # Warm-rain / shallow heavy — primary underpredict regime in diagnostics.
    if actual_mm >= 5.0 and np.isfinite(eth) and eth < 3000:
        if np.isfinite(rhohv_min) and rhohv_min > 0.97:
            return 'warm_shallow_heavy'
        if np.isfinite(refl) and refl < 35 and actual_mm >= 8.0:
            return 'warm_shallow_heavy'

    if np.isfinite(eth):
        if eth >= 6000:
            return 'convective_deep'
        if eth >= 3000:
            return 'stratiform_mid'
        return 'shallow'

    return 'other'


def z_over_rain_efficiency(feats: dict, actual_mm: float) -> float:
    refl = feats.get('reflectivity_center_max', np.nan)
    if not np.isfinite(refl) or actual_mm <= 0:
        return np.nan
    return refl / actual_mm


def mark_ar_like_hours(samples, heavy_mm: float = 5.0, min_stations: int = 4) -> set:
    """
    Return set of (date_str, hour_str) keys for widespread heavy-rain hours.

    AR proxy: >= min_stations distinct stations report >= heavy_mm in the same hour.
    """
    hour_stations: dict[tuple[str, str], set] = defaultdict(set)
    for s in samples:
        actual = s.get('hourly_precip_mm', 0.0)
        if actual < heavy_mm:
            continue
        h = s.get('hour_start')
        hour_key = str(h)
        date_key = _sample_date(s)
        station = s.get('station_name', s.get('station_id', ''))
        hour_stations[(date_key, hour_key)].add(station)

    ar_hours = {
        key for key, stations in hour_stations.items()
        if len(stations) >= min_stations
    }
    return ar_hours


def ar_like_days(ar_hours: set[tuple[str, str]]) -> set[str]:
    return {date for date, _ in ar_hours}


def analyze_split(samples, split_name: str, ar_hours: set[tuple[str, str]]) -> list[dict]:
    rows = []
    for s in samples:
        actual = float(s.get('hourly_precip_mm', np.nan))
        feats = extract_radar_features(s, [])
        regime = classify_regime(actual, feats)
        date_str = _sample_date(s)
        hour_str = str(s.get('hour_start', ''))
        is_ar_hour = (date_str, hour_str) in ar_hours
        rows.append({
            'split': split_name,
            'year': _sample_year(s),
            'month': _sample_month(s),
            'date': date_str,
            'hour': hour_str,
            'station': s.get('station_name', ''),
            'actual_mm': actual,
            'rain_tier': rain_tier(actual),
            'regime': regime,
            'ar_like_hour': int(is_ar_hour),
            'ar_like_day': int(is_ar_hour),  # updated below per-day
            'echo_top_m': feats.get('echo_top_height_center_max', np.nan),
            'max_z_height_m': feats.get('max_z_height_center_max', np.nan),
            'rhohv_min': feats.get('rhohv_min_center_mean', np.nan),
            'reflectivity_dbz': feats.get('reflectivity_center_max', np.nan),
            'z_per_mm': z_over_rain_efficiency(feats, actual),
        })

    ar_days = ar_like_days(ar_hours)
    for row in rows:
        row['ar_like_day'] = int(row['date'] in ar_days)
    return rows


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def summarize_regimes(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split in ('train', 'val'):
        split_rows = [r for r in rows if r['split'] == split]
        n = len(split_rows)
        counts = Counter(r['regime'] for r in split_rows)
        out[split] = {
            'n': n,
            **{f'pct_{reg}': _pct(counts.get(reg, 0), n) for reg in REGIME_ORDER},
            **{f'n_{reg}': counts.get(reg, 0) for reg in REGIME_ORDER},
            'pct_heavy': _pct(sum(1 for r in split_rows if r['actual_mm'] >= 5.0), n),
            'pct_ar_hour': _pct(sum(r['ar_like_hour'] for r in split_rows), n),
            'pct_ar_day': _pct(sum(r['ar_like_day'] for r in split_rows), n),
        }
    return out


def print_summary(metadata: dict, regime_summary: dict[str, dict], rows: list[dict]):
    print("\n" + "=" * 72)
    print("  TRAIN vs VAL REGIME COMPARISON")
    print("=" * 72)
    print(f"  Pickle train years: {metadata.get('train_years', 'N/A')}")
    print(f"  Pickle val years:   {metadata.get('val_years', 'N/A')}")
    print(f"  Split type:         {metadata.get('split_type', 'N/A')}")

    for split in ('train', 'val'):
        split_rows = [r for r in rows if r['split'] == split]
        actuals = [r['actual_mm'] for r in split_rows]
        years = Counter(r['year'] for r in split_rows)
        print(f"\n  [{split.upper()}] n={len(split_rows)}")
        print(f"    Years: {dict(sorted(years.items()))}")
        if actuals:
            print(
                f"    Rain (mm/hr): mean={np.mean(actuals):.2f}  "
                f"median={np.median(actuals):.2f}  max={np.max(actuals):.1f}"
            )
            print(
                f"    >5 mm/hr: {sum(1 for a in actuals if a >= 5)} "
                f"({_pct(sum(1 for a in actuals if a >= 5), len(actuals)):.1f}%)"
            )

    print("\n  Regime mix (% of split):")
    print(f"  {'Regime':<22} {'Train %':>10} {'Val %':>10} {'Delta (V-T)':>12}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 12}")
    for regime in REGIME_ORDER:
        t = regime_summary['train'].get(f'pct_{regime}', 0.0)
        v = regime_summary['val'].get(f'pct_{regime}', 0.0)
        print(f"  {regime:<22} {t:>9.1f}% {v:>9.1f}% {v - t:>+11.1f}pp")

    print("\n  Rain tiers (% of split):")
    for tier_name, _, _ in RAIN_TIERS:
        t = _pct(sum(1 for r in rows if r['split'] == 'train' and r['rain_tier'] == tier_name),
                 regime_summary['train']['n'])
        v = _pct(sum(1 for r in rows if r['split'] == 'val' and r['rain_tier'] == tier_name),
                 regime_summary['val']['n'])
        print(f"    {tier_name:<12} train={t:5.1f}%  val={v:5.1f}%  delta={v - t:+.1f}pp")

    print("\n  AR-like widespread heavy rain (>=4 stations, >=5 mm/hr same hour):")
    for split in ('train', 'val'):
        s = regime_summary[split]
        print(
            f"    {split}: {s['pct_ar_hour']:.1f}% of hours AR-like  |  "
            f"{s['pct_ar_day']:.1f}% of samples on AR-like days"
        )

    # Unique storm days
    for split in ('train', 'val'):
        split_rows = [r for r in rows if r['split'] == split]
        days_by_year: dict[int, set[str]] = defaultdict(set)
        ar_days_by_year: dict[int, set[str]] = defaultdict(set)
        for r in split_rows:
            if r['actual_mm'] >= 5.0:
                days_by_year[r['year']].add(r['date'])
            if r['ar_like_day']:
                ar_days_by_year[r['year']].add(r['date'])
        heavy_total = sum(len(v) for v in days_by_year.values())
        ar_total = sum(len(v) for v in ar_days_by_year.values())
        print(f"\n  [{split}] heavy-rain days (any station >=5 mm/hr): {heavy_total}")
        print(f"    by year: {dict(sorted((y, len(d)) for y, d in days_by_year.items()))}")
        print(f"    AR-like days: {ar_total}  by year: "
              f"{dict(sorted((y, len(d)) for y, d in ar_days_by_year.items()))}")


def plot_regime_bars(regime_summary: dict, out_path: Path):
    x = np.arange(len(REGIME_ORDER))
    width = 0.35
    train_p = [regime_summary['train'].get(f'pct_{r}', 0.0) for r in REGIME_ORDER]
    val_p = [regime_summary['val'].get(f'pct_{r}', 0.0) for r in REGIME_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, train_p, width, label='Train', alpha=0.85)
    ax.bar(x + width / 2, val_p, width, label='Val', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', '\n') for r in REGIME_ORDER], fontsize=8)
    ax.set_ylabel('% of split samples')
    ax.set_title('Storm regime mix: train vs val')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_monthly_heatmap(rows: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, ('train', 'val')):
        split_rows = [r for r in rows if r['split'] == split]
        matrix = np.zeros((len(REGIME_ORDER), 12))
        for r in split_rows:
            mi = r['month'] - 1
            if 0 <= mi < 12:
                ri = REGIME_ORDER.index(r['regime']) if r['regime'] in REGIME_ORDER else -1
                if ri >= 0:
                    matrix[ri, mi] += 1
        col_sum = matrix.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1
        pct = 100.0 * matrix / col_sum
        im = ax.imshow(pct, aspect='auto', cmap='YlOrRd', vmin=0, vmax=40)
        ax.set_title(split)
        ax.set_xlabel('Month')
        ax.set_xticks(range(12))
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.set_yticks(range(len(REGIME_ORDER)))
        ax.set_yticklabels(REGIME_ORDER, fontsize=7)
    fig.colorbar(im, ax=axes, label='% within month', shrink=0.8)
    fig.suptitle('Regime fraction by calendar month', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_heavy_cdf(rows: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for split, color in [('train', 'C0'), ('val', 'C3')]:
        actuals = sorted(r['actual_mm'] for r in rows if r['split'] == split)
        if not actuals:
            continue
        y = np.arange(1, len(actuals) + 1) / len(actuals)
        ax.plot(actuals, y, label=split, color=color, alpha=0.8)
    ax.axvline(5.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Actual rain (mm/hr)')
    ax.set_ylabel('ECDF')
    ax.set_xlim(0, min(50, max(r['actual_mm'] for r in rows) * 1.05))
    ax.set_title('Rainfall ECDF (all samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_ar_days_by_year(rows: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    years = sorted({r['year'] for r in rows})
    x = np.arange(len(years))

    for offset, split, color in [(-width / 2, 'train', 'C0'), (width / 2, 'val', 'C3')]:
        heavy_days = []
        ar_days = []
        for yr in years:
            split_rows = [r for r in rows if r['split'] == split and r['year'] == yr]
            heavy_days.append(len({r['date'] for r in split_rows if r['actual_mm'] >= 5.0}))
            ar_days.append(len({r['date'] for r in split_rows if r['ar_like_day']}))
        ax.bar(x + offset, heavy_days, width, label=f'{split} heavy days', color=color, alpha=0.5)
        ax.bar(x + offset, ar_days, width, label=f'{split} AR-like days', color=color, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel('Year')
    ax.set_ylabel('Distinct storm days')
    ax.set_title('Heavy-rain and AR-like days by year')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def write_csv(rows: list[dict], regime_summary: dict, out_dir: Path):
    summary_path = out_dir / 'split_regime_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['split', 'n', 'pct_heavy', 'pct_ar_hour', 'pct_ar_day']
        fields += [f'pct_{r}' for r in REGIME_ORDER]
        fields += [f'n_{r}' for r in REGIME_ORDER]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in ('train', 'val'):
            row = {'split': split, **regime_summary[split]}
            writer.writerow({k: row.get(k, '') for k in fields})

    samples_path = out_dir / 'split_regime_samples.csv'
    if rows:
        with open(samples_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return summary_path, samples_path


def run_compare(
    pickle_path: str,
    output_dir: str | None = None,
    apply_filters: bool = False,
    ar_heavy_mm: float = 5.0,
    ar_min_stations: int = 4,
    exclude_stations: list[str] | None = None,
):
    _ensure_utf8_stdio()
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")

    out_dir = Path(output_dir) if output_dir else pickle_path.parent / 'split_regime_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pickle: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)

    metadata = dataset.get('metadata', {})
    train = list(dataset['train'])
    val = list(dataset['val'])
    print(f"  Raw: train={len(train)}  val={len(val)}")

    if apply_filters:
        print("  Applying blunt training filters to both splits...")
        train = apply_blunt_filters(train, exclude_stations)
        val = apply_blunt_filters(val, exclude_stations)
        print(f"  Filtered: train={len(train)}  val={len(val)}")

    all_samples = train + val
    ar_hours = mark_ar_like_hours(all_samples, heavy_mm=ar_heavy_mm, min_stations=ar_min_stations)

    train_rows = analyze_split(train, 'train', ar_hours)
    val_rows = analyze_split(val, 'val', ar_hours)
    rows = train_rows + val_rows

    regime_summary = summarize_regimes(rows)
    print_summary(metadata, regime_summary, rows)

    plot_regime_bars(regime_summary, out_dir / 'regime_mix_train_vs_val.png')
    plot_monthly_heatmap(rows, out_dir / 'regime_by_month_heatmap.png')
    plot_heavy_cdf(rows, out_dir / 'rainfall_ecdf.png')
    plot_ar_days_by_year(rows, out_dir / 'ar_days_by_year.png')

    summary_csv, samples_csv = write_csv(rows, regime_summary, out_dir)

    report_path = out_dir / 'summary.txt'
    report_path.write_text(
        f"Pickle: {pickle_path}\n"
        f"Train years: {metadata.get('train_years')}\n"
        f"Val years: {metadata.get('val_years')}\n"
        f"Filtered: {apply_filters}\n"
        f"Train n: {regime_summary['train']['n']}\n"
        f"Val n: {regime_summary['val']['n']}\n"
        f"Val - Train warm_shallow_heavy: "
        f"{regime_summary['val']['pct_warm_shallow_heavy'] - regime_summary['train']['pct_warm_shallow_heavy']:+.1f}pp\n"
        f"Val - Train bright_band: "
        f"{regime_summary['val']['pct_bright_band'] - regime_summary['train']['pct_bright_band']:+.1f}pp\n"
        f"Val - Train heavy (>5mm): "
        f"{regime_summary['val']['pct_heavy'] - regime_summary['train']['pct_heavy']:+.1f}pp\n",
        encoding='utf-8',
    )

    print(f"\n  Wrote: {out_dir}/")
    print(f"    {summary_csv.name}")
    print(f"    {samples_csv.name}")
    print(f"    summary.txt + 4 plots")
    return rows, regime_summary


def main():
    parser = argparse.ArgumentParser(
        description='Compare storm/regime distributions between train and val splits',
    )
    parser.add_argument(
        '--pickle',
        default='dataset/outputs/3d/radar_gauge_dataset_subml_9500.pkl',
        help='Path to radar-gauge pickle',
    )
    parser.add_argument('--output-dir', default=None, help='Output directory for CSV/plots')
    parser.add_argument(
        '--apply-filters', action='store_true',
        help='Apply blunt train/eval sample filters before analysis',
    )
    parser.add_argument('--ar-heavy-mm', type=float, default=5.0,
                        help='Min mm/hr for AR-like hour detection')
    parser.add_argument('--ar-min-stations', type=int, default=4,
                        help='Min stations in same hour for AR-like flag')
    parser.add_argument('--exclude-stations', nargs='*', default=[])
    args = parser.parse_args()

    run_compare(
        pickle_path=args.pickle,
        output_dir=args.output_dir,
        apply_filters=args.apply_filters,
        ar_heavy_mm=args.ar_heavy_mm,
        ar_min_stations=args.ar_min_stations,
        exclude_stations=args.exclude_stations,
    )


if __name__ == '__main__':
    main()
