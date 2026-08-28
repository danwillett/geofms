"""
audit_feature_norms.py — Distribution / normalization audit for the radar
feature channels.

The model normalizes every field with (x - f_min) / (f_max - f_min) clipped to
[0, 1], using the hand-set ranges in models/unet/dataset.FIELD_NORMS. If a range
is set to physical extremes instead of the data's actual spread, the real signal
gets squeezed into a narrow sub-interval of [0, 1]. A compressed input has tiny
variance, so the first conv layer effectively down-weights it — which can make a
genuinely useful feature look dead in an ablation.

This script measures, per field, the ACTUAL distribution across the training
patches (all scans, all pixels) and compares it to the configured range:

  used_frac  (p1..p99 spread) / (f_max - f_min)   -> <0.40 = COMPRESSED
  clip_lo/hi fraction of valid values outside the range -> >1% = CLIPPED
  norm p1/p50/p99   where the data lands in [0,1] after normalization
  missing%   fraction of sentinel / NaN pixels (filled at train time)

It prints a table, writes a CSV, and plots per-field histograms with the
configured range and suggested p1..p99 range overlaid.

Run from project root:
    python -m models.unet.audit_feature_norms \
        --pickle dataset/outputs/3d/radar_gauge_dataset_vertlowmeltbb_9500.pkl
"""

import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.unet.dataset import PICKLE_FIELD_ORDER, FIELD_NORMS, FIELD_FILL

SENTINEL = -9999.0


def collect_values(samples, max_samples=6000, per_sample=400, seed=0):
    """Gather valid (non-sentinel, non-NaN) values per field across patches.

    To bound memory we subsample: up to ``max_samples`` patches, and within each
    patch up to ``per_sample`` valid values per field (random). Also tracks the
    total/valid pixel counts per field so we can report a missing fraction.
    """
    rng = np.random.default_rng(seed)
    n_fields = len(PICKLE_FIELD_ORDER)
    values = [[] for _ in range(n_fields)]
    total_px = np.zeros(n_fields, dtype=np.int64)
    valid_px = np.zeros(n_fields, dtype=np.int64)

    if len(samples) > max_samples:
        idx = rng.choice(len(samples), size=max_samples, replace=False)
    else:
        idx = np.arange(len(samples))

    for si in idx:
        patch = samples[int(si)].get('radar_patch')
        if patch is None:
            continue
        nf = patch.shape[1]
        for f in range(min(n_fields, nf)):
            arr = patch[:, f, :, :].astype(np.float64).ravel()
            total_px[f] += arr.size
            valid = arr[(arr != SENTINEL) & np.isfinite(arr)]
            valid_px[f] += valid.size
            if valid.size == 0:
                continue
            if valid.size > per_sample:
                valid = rng.choice(valid, size=per_sample, replace=False)
            values[f].append(valid)

    out = []
    for f in range(n_fields):
        out.append(np.concatenate(values[f]) if values[f] else np.array([]))
    return out, total_px, valid_px


def audit(pickle_path, split='train', max_samples=6000, per_sample=400):
    print(f"Loading {pickle_path} (split={split})...")
    with open(pickle_path, 'rb') as f:
        data = pkl.load(f)

    if split == 'all':
        samples = list(data.get('train', [])) + list(data.get('val', []))
    else:
        samples = list(data.get(split, []))
    if not samples:
        print(f"  No samples in split '{split}'.")
        return
    print(f"  {len(samples)} samples (auditing up to {max_samples}, "
          f"<= {per_sample} px/field/patch)\n")

    values, total_px, valid_px = collect_values(samples, max_samples, per_sample)

    header = (f"  {'field':<30} {'miss%':>6} {'p1':>9} {'p50':>9} {'p99':>9} "
              f"{'cfg_min':>9} {'cfg_max':>9} {'used':>6} {'clipLo':>7} "
              f"{'clipHi':>7} {'nP50':>6}  flags")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for f, field in enumerate(PICKLE_FIELD_ORDER):
        vals = values[f]
        miss = (1.0 - valid_px[f] / total_px[f]) * 100 if total_px[f] else np.nan
        if field not in FIELD_NORMS:
            continue
        nmin, nmax = FIELD_NORMS[field]
        span = (nmax - nmin) if nmax > nmin else np.nan

        if vals.size == 0:
            print(f"  {field:<30} {miss:>6.1f}  (no valid values)")
            rows.append(dict(field=field, miss_pct=miss, n_valid=0))
            continue

        p1, p50, p99 = np.percentile(vals, [1, 50, 99])
        dmin, dmax = float(vals.min()), float(vals.max())
        used = (p99 - p1) / span if span and np.isfinite(span) else np.nan
        clip_lo = float(np.mean(vals < nmin)) * 100
        clip_hi = float(np.mean(vals > nmax)) * 100
        norm = lambda v: (v - nmin) / span if span else np.nan
        nP1, nP50, nP99 = norm(p1), norm(p50), norm(p99)

        flags = []
        if np.isfinite(used) and used < 0.40:
            flags.append('COMPRESSED')
        if (clip_lo + clip_hi) > 1.0:
            flags.append('CLIPPED')
        if np.isfinite(nP50) and (nP50 < 0.15 or nP50 > 0.85):
            flags.append('OFFCENTER')
        flag_str = ','.join(flags) if flags else 'ok'

        print(f"  {field:<30} {miss:>6.1f} {p1:>9.2f} {p50:>9.2f} {p99:>9.2f} "
              f"{nmin:>9.1f} {nmax:>9.1f} {used:>6.2f} {clip_lo:>6.1f}% "
              f"{clip_hi:>6.1f}% {nP50:>6.2f}  {flag_str}")

        rows.append(dict(
            field=field, miss_pct=miss, n_valid=int(vals.size),
            p1=p1, p50=p50, p99=p99, data_min=dmin, data_max=dmax,
            cfg_min=nmin, cfg_max=nmax, used_frac=used,
            clip_lo_pct=clip_lo, clip_hi_pct=clip_hi,
            norm_p1=nP1, norm_p50=nP50, norm_p99=nP99,
            suggest_min=float(p1), suggest_max=float(p99),
            flags=flag_str, fill=FIELD_FILL.get(field, nmin),
        ))

    # ── Suggestions for flagged fields ──
    flagged = [r for r in rows if r.get('flags') and r['flags'] != 'ok']
    if flagged:
        print(f"\n  Suggested FIELD_NORMS updates (robust p1..p99):")
        for r in flagged:
            print(f"    '{r['field']}': ({r['suggest_min']:.2f}, "
                  f"{r['suggest_max']:.2f}),   # was "
                  f"({r['cfg_min']}, {r['cfg_max']})  [{r['flags']}]")
    else:
        print("\n  No compression/clip/offset flags — ranges look reasonable.")

    # ── Plots ──
    out_dir = Path('evaluation_figures/unet_dualpol')
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_rows = [r for r in rows if r.get('n_valid', 0) > 0]
    n = len(plot_rows)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, plot_rows):
        f = PICKLE_FIELD_ORDER.index(r['field'])
        vals = values[f]
        ax.hist(vals, bins=60, color='#34495e', alpha=0.85)
        ax.axvline(r['cfg_min'], color='red', ls='--', lw=1, label='cfg range')
        ax.axvline(r['cfg_max'], color='red', ls='--', lw=1)
        ax.axvline(r['suggest_min'], color='lime', ls=':', lw=1.2, label='p1..p99')
        ax.axvline(r['suggest_max'], color='lime', ls=':', lw=1.2)
        title = r['field']
        if r['flags'] != 'ok':
            title += f"  [{r['flags']}]"
        ax.set_title(title, fontsize=9,
                     color='darkred' if r['flags'] != 'ok' else 'black')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for ax in axes[len(plot_rows):]:
        ax.set_axis_off()
    plt.tight_layout()
    plot_path = out_dir / 'feature_norm_audit.png'
    plt.savefig(plot_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved histograms to: {plot_path}")

    # ── CSV ──
    csv_path = out_dir / 'feature_norm_audit.csv'
    cols = ['field', 'n_valid', 'miss_pct', 'p1', 'p50', 'p99',
            'data_min', 'data_max', 'cfg_min', 'cfg_max', 'used_frac',
            'clip_lo_pct', 'clip_hi_pct', 'norm_p1', 'norm_p50', 'norm_p99',
            'suggest_min', 'suggest_max', 'flags']
    with open(csv_path, 'w') as fcsv:
        fcsv.write(','.join(cols) + '\n')
        for r in rows:
            def g(k):
                v = r.get(k, '')
                if isinstance(v, float):
                    return '' if not np.isfinite(v) else f"{v:.4f}"
                return str(v)
            fcsv.write(','.join(g(c) for c in cols) + '\n')
    print(f"  Saved table to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Audit feature distributions vs FIELD_NORMS ranges')
    parser.add_argument('--pickle', type=str,
                        default='dataset/outputs/3d/radar_gauge_dataset_vertlowmeltbb_9500.pkl',
                        help='Path to the dataset pickle')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'all'])
    parser.add_argument('--max-samples', type=int, default=6000,
                        help='Max patches to sample for the audit')
    parser.add_argument('--per-sample', type=int, default=400,
                        help='Max valid pixels per field per patch to keep')
    args = parser.parse_args()
    audit(args.pickle, args.split, args.max_samples, args.per_sample)


if __name__ == '__main__':
    main()
