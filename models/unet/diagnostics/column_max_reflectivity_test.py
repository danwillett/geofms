"""
column_max_reflectivity_test.py — Three diagnostics for the column-max Z hypothesis.

  Test 1 (center-ig):     Full-patch vs center-pixel IG for reflectivity-related fields.
  Test 2 (vertical):      On underpredictions, compare column-max Z vs gate-level Z vs actual rain.
  Test 4 (swap-lowlevel): Val eval with reflectivity channels replaced by low_level_ref (or gate-ref).

Run from project root:
    python -m models.unet.diagnostics.column_max_reflectivity_test --run-dir models/checkpoints/unet_dualpol/<run>
    python -m models.unet.diagnostics.column_max_reflectivity_test --run-dir ... --test center-ig vertical swap
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None  # only required for center-ig test

from models.unet.ablation import build_feature_groups
from models.unet.dataset import N_SCANS, resolve_fields
from models.unet.evaluate import find_checkpoint, load_model
from models.unet.diagnose_underestimates import extract_radar_features
from models.unet.diagnostics.gradient_test import (
    IGSummary,
    SampleRecord,
    aggregate_field_attributions,
    build_fill_baseline,
    collect_predictions,
    compute_integrated_gradients,
    select_cohorts,
)
from models.unet.diagnostics.permutation_test import (
    DEFAULT_CKPT_DIR,
    DEFAULT_DEM,
    DEFAULT_PICKLE,
    _ensure_utf8_stdio,
    _pred_at_gauge,
    build_val_loader,
)

# Fields of interest for the column-max hypothesis.
IG_FIELDS = [
    'reflectivity',
    'low_level_ref',
    'lowest_gate_reflectivity',
    'vil',
    'max_z_height',
]

UNDER_MIN_ACTUAL = 5.0
UNDER_MAX_PRED = 2.0


def _field_channels(fields: list[str], field_name: str) -> list[int]:
    if field_name not in fields:
        return []
    idx = fields.index(field_name)
    start = idx * N_SCANS
    return list(range(start, start + N_SCANS))


def _swap_field_channels(radar: torch.Tensor, src_ch: list[int], dst_ch: list[int]) -> torch.Tensor:
    """Replace dst channels with src channels (same spatial layout)."""
    if not src_ch or not dst_ch or len(src_ch) != len(dst_ch):
        return radar
    out = radar.clone()
    out[:, dst_ch, :, :] = radar[:, src_ch, :, :]
    return out


def _compute_metrics(preds_mm: np.ndarray, targets_mm: np.ndarray) -> dict:
    valid = targets_mm >= 0
    preds = preds_mm[valid]
    targets = targets_mm[valid]
    if len(targets) == 0:
        return {'n': 0, 'r2': float('nan'), 'mae': float('nan'), 'rmse': float('nan')}

    mae = float(np.mean(np.abs(targets - preds)))
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'n': int(len(targets)), 'r2': r2, 'mae': mae, 'rmse': rmse}


def _eval_loader(
    model,
    loader,
    device,
    log_target: bool,
    swap_src: list[int] | None = None,
    swap_dst: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference; optionally swap reflectivity channels before forward pass."""
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            radar = batch['radar'].to(device)
            if swap_src and swap_dst:
                radar = _swap_field_channels(radar, swap_src, swap_dst)
            gauge_pixel = batch['gauge_pixel']
            pred_map = model(radar)
            pred_at_gauge = _pred_at_gauge(pred_map, gauge_pixel).cpu()
            target = batch['target'].cpu()
            if log_target:
                pred_at_gauge = torch.expm1(pred_at_gauge)
                target = torch.expm1(target)
            all_preds.extend(pred_at_gauge.numpy().tolist())
            all_targets.extend(target.numpy().tolist())
    return np.asarray(all_preds), np.asarray(all_targets)


def _slice_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    if mask is not None:
        preds = preds[mask]
        targets = targets[mask]
    return _compute_metrics(preds, targets)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    x, y = x[mask], y[mask]
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _mean_ig_table(
    summaries: list[IGSummary],
    field_names: list[str],
    use_gauge: bool,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in field_names:
        key = 'gauge_signed' if use_gauge else 'signed'
        vals = [getattr(s, key).get(name, 0.0) for s in summaries]
        out[name] = float(np.mean(vals)) if vals else float('nan')
    return out


def run_center_ig_test(
    model,
    loader,
    val_ds,
    device,
    cfg,
    fields,
    out_dir: Path,
    top_k: int,
    n_steps: int,
    baseline_mode: str = 'fill',
) -> list[str]:
    if IntegratedGradients is None:
        raise ImportError("captum is required for center-ig test. Install with: pip install captum")

    log_target = cfg.get('log_target', True)
    use_dem = cfg.get('use_dem', True)
    use_mask = cfg.get('use_mask', True)
    use_temporal_pos = cfg.get('use_temporal_pos', True)
    use_feature_masks = cfg.get('use_feature_masks', False)

    feature_groups = build_feature_groups(
        fields, use_mask, use_temporal_pos, use_dem, use_feature_masks,
    )
    ig_fields = [f for f in IG_FIELDS if f in feature_groups]

    lines = [
        "",
        "=" * 70,
        "  TEST 1 — CENTER-PIXEL vs FULL-PATCH INTEGRATED GRADIENTS",
        "=" * 70,
        "",
        "  Full-patch IG sums over 19×19 × 12 scans per field.",
        "  Center IG sums over gauge pixel only × 12 scans.",
        "  Ratio > 1 => attribution concentrated at gauge (not diluted by off-patch echo).",
        "",
    ]

    records = collect_predictions(model, loader, device, log_target)
    cohort_pairs = select_cohorts(
        records, top_k, ['under', 'over'],
        UNDER_MIN_ACTUAL, UNDER_MAX_PRED, 5.0, 1.0, 100,
    )

    sample0 = val_ds[0]['radar']
    h, w = sample0.shape[1], sample0.shape[2]
    baseline = build_fill_baseline(
        fields, use_mask, use_temporal_pos, use_dem, use_feature_masks, h, w,
    )

    csv_rows: list[dict] = []
    plot_data: dict[str, dict[str, dict[str, float]]] = {}

    for cohort_name, pairs in cohort_pairs.items():
        outlier_records = [o for o, _ in pairs]
        if not outlier_records:
            lines.append(f"  [{cohort_name}] No samples — skipped.")
            continue

        summaries: list[IGSummary] = []
        for rec in outlier_records:
            radar = val_ds[rec.idx]['radar']
            attr = compute_integrated_gradients(
                model, radar, baseline, rec.gauge_y, rec.gauge_x,
                log_target, device, n_steps,
            )
            summary = aggregate_field_attributions(
                attr, feature_groups, ig_fields, rec.gauge_y, rec.gauge_x,
            )
            summaries.append(summary)
            for fname in ig_fields:
                csv_rows.append({
                    'cohort': cohort_name,
                    'sample_idx': rec.idx,
                    'station': rec.station_name,
                    'actual_mm': rec.actual_mm,
                    'pred_mm': rec.pred_mm,
                    'field': fname,
                    'full_patch_ig': summary.signed.get(fname, 0.0),
                    'center_ig': summary.gauge_signed.get(fname, 0.0),
                })

        full_means = _mean_ig_table(summaries, ig_fields, use_gauge=False)
        center_means = _mean_ig_table(summaries, ig_fields, use_gauge=True)
        plot_data[cohort_name] = {'full': full_means, 'center': center_means}

        lines.append(f"  [{cohort_name}] n={len(outlier_records)} outliers")
        lines.append(f"  {'Field':<28} {'Full-patch':>12} {'Center':>12} {'Center/Full':>12}")
        lines.append(f"  {'-' * 28} {'-' * 12} {'-' * 12} {'-' * 12}")
        for fname in ig_fields:
            full_v = full_means[fname]
            cen_v = center_means[fname]
            ratio = cen_v / full_v if abs(full_v) > 1e-9 else float('nan')
            ratio_s = f"{ratio:+.2f}" if np.isfinite(ratio) else "n/a"
            lines.append(f"  {fname:<28} {full_v:>+12.4f} {cen_v:>+12.4f} {ratio_s:>12}")
        lines.append("")

    if csv_rows:
        csv_path = out_dir / 'test1_center_vs_full_ig.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        lines.append(f"  Wrote: {csv_path}")

    if plot_data:
        fig, axes = plt.subplots(1, len(plot_data), figsize=(5 * len(plot_data), 4), squeeze=False)
        for ax_idx, (cohort_name, data) in enumerate(plot_data.items()):
            ax = axes[0, ax_idx]
            names = ig_fields
            x = np.arange(len(names))
            width = 0.35
            full_vals = [data['full'].get(n, 0.0) for n in names]
            cen_vals = [data['center'].get(n, 0.0) for n in names]
            ax.bar(x - width / 2, full_vals, width, label='Full patch', alpha=0.85)
            ax.bar(x + width / 2, cen_vals, width, label='Center pixel', alpha=0.85)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
            ax.set_title(f'{cohort_name} cohort (mean IG)')
            ax.set_ylabel('Signed IG (mm/hr)')
            ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = out_dir / 'test1_center_vs_full_ig.png'
        fig.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        lines.append(f"  Wrote: {plot_path}")

    return lines


def run_vertical_alignment_test(
    model,
    loader,
    val_ds,
    device,
    out_dir: Path,
    log_target: bool,
    under_min_actual: float = UNDER_MIN_ACTUAL,
    under_max_pred: float = UNDER_MAX_PRED,
) -> list[str]:
    lines = [
        "",
        "=" * 70,
        "  TEST 2 — COLUMN-MAX vs GATE-LEVEL Z vs ACTUAL RAIN",
        "=" * 70,
        "",
        f"  Underpredictions: actual >= {under_min_actual} mm/hr AND pred <= {under_max_pred} mm/hr",
        "  Control: heavy rain (actual >= 5) with |error| <= 1 mm/hr",
        "",
    ]

    records = collect_predictions(model, loader, device, log_target)
    preds = np.array([r.pred_mm for r in records])
    actuals = np.array([r.actual_mm for r in records])

    under_mask = np.array([
        r.actual_mm >= under_min_actual and r.pred_mm <= under_max_pred for r in records
    ])
    control_mask = np.array([
        r.actual_mm >= under_min_actual and r.abs_error <= 1.0 for r in records
    ])

    rows: list[dict] = []
    for i, rec in enumerate(records):
        if not (under_mask[i] or control_mask[i]):
            continue
        feats = extract_radar_features(val_ds.samples[rec.idx], [])
        colmax = feats.get('reflectivity_center_max', np.nan)
        gate = feats.get('lowest_gate_reflectivity_center_max', np.nan)
        lowlevel = feats.get('low_level_ref_center_max', np.nan)
        max_z = feats.get('max_z_height_center_max', np.nan)
        beam_h = feats.get('beam_height_center_max', np.nan)
        rows.append({
            'sample_idx': rec.idx,
            'cohort': 'under' if under_mask[i] else 'control',
            'station': rec.station_name,
            'actual_mm': rec.actual_mm,
            'pred_mm': rec.pred_mm,
            'shortfall_mm': rec.actual_mm - rec.pred_mm,
            'colmax_dbz': colmax,
            'gate_dbz': gate,
            'lowlevel_dbz': lowlevel,
            'gate_minus_colmax': gate - colmax if np.isfinite(gate) and np.isfinite(colmax) else np.nan,
            'lowlevel_minus_colmax': lowlevel - colmax if np.isfinite(lowlevel) and np.isfinite(colmax) else np.nan,
            'max_z_height_m': max_z,
            'beam_height_m': beam_h,
            'maxz_minus_beam': max_z - beam_h if np.isfinite(max_z) and np.isfinite(beam_h) else np.nan,
        })

    if not rows:
        lines.append("  No underpredict or control samples found.")
        return lines

    csv_path = out_dir / 'test2_vertical_alignment.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines.append(f"  Wrote: {csv_path} ({len(rows)} samples)")

    def _cohort_rows(cohort: str) -> list[dict]:
        return [r for r in rows if r['cohort'] == cohort]

    for cohort, label in [('under', 'Underpredictions'), ('control', 'Accurate heavy')]:
        cr = _cohort_rows(cohort)
        if not cr:
            continue
        actual = np.array([r['actual_mm'] for r in cr])
        colmax = np.array([r['colmax_dbz'] for r in cr])
        gate = np.array([r['gate_dbz'] for r in cr])
        lowlevel = np.array([r['lowlevel_dbz'] for r in cr])
        gate_minus = np.array([r['gate_minus_colmax'] for r in cr])
        maxz_minus_beam = np.array([r['maxz_minus_beam'] for r in cr])

        lines.append(f"\n  {label} (n={len(cr)}):")
        lines.append(f"    corr(actual, colmax Z):    {_pearson(actual, colmax):+.3f}")
        lines.append(f"    corr(actual, gate Z):      {_pearson(actual, gate):+.3f}")
        lines.append(f"    corr(actual, low_level Z): {_pearson(actual, lowlevel):+.3f}")
        lines.append(f"    median gate - colmax (dBZ): {np.nanmedian(gate_minus):+.1f}")
        lines.append(f"    median max_z - beam (m):    {np.nanmedian(maxz_minus_beam):+.0f}")
        gate_stronger = np.sum(gate_minus > 3.0) / max(np.isfinite(gate_minus).sum(), 1)
        lines.append(f"    fraction gate > colmax+3dB: {100 * gate_stronger:.1f}%")

    # Scatter plots
    under_rows = _cohort_rows('under')
    ctrl_rows = _cohort_rows('control')
    if under_rows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (xkey, xlabel) in zip(axes, [
            ('colmax_dbz', 'Column-max Z (dBZ)'),
            ('gate_dbz', 'Lowest-gate Z (dBZ)'),
            ('lowlevel_dbz', 'Low-level mean Z (dBZ)'),
        ]):
            x_under = np.array([r[xkey] for r in under_rows])
            y_under = np.array([r['actual_mm'] for r in under_rows])
            ax.scatter(x_under, y_under, alpha=0.6, s=25, label='Under', c='C3')
            if ctrl_rows:
                x_ctrl = np.array([r[xkey] for r in ctrl_rows])
                y_ctrl = np.array([r['actual_mm'] for r in ctrl_rows])
                ax.scatter(x_ctrl, y_ctrl, alpha=0.35, s=15, label='Accurate heavy', c='C0')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Actual rain (mm/hr)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle('Test 2: Radar Z vs gauge rain at center pixel', fontsize=11)
        fig.tight_layout()
        plot_path = out_dir / 'test2_z_vs_actual_scatter.png'
        fig.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        lines.append(f"  Wrote: {plot_path}")

        fig, ax = plt.subplots(figsize=(5, 4))
        gate_m = np.array([r['gate_minus_colmax'] for r in under_rows])
        shortfall = np.array([r['shortfall_mm'] for r in under_rows])
        ax.scatter(gate_m, shortfall, alpha=0.6, s=25, c='C3')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel('Gate Z - column-max Z (dBZ)')
        ax.set_ylabel('Shortfall (actual - pred, mm/hr)')
        ax.set_title('Underpredictions: vertical Z mismatch vs miss size')
        ax.grid(True, alpha=0.3)
        plot_path2 = out_dir / 'test2_gate_minus_colmax_vs_shortfall.png'
        fig.savefig(plot_path2, dpi=120, bbox_inches='tight')
        plt.close(fig)
        lines.append(f"  Wrote: {plot_path2}")

    return lines


def run_swap_lowlevel_test(
    model,
    loader,
    device,
    fields,
    out_dir: Path,
    log_target: bool,
    under_min_actual: float = UNDER_MIN_ACTUAL,
    under_max_pred: float = UNDER_MAX_PRED,
) -> list[str]:
    lines = [
        "",
        "=" * 70,
        "  TEST 4 — SWAP COLUMN-MAX REFLECTIVITY FOR LOW-LEVEL Z",
        "=" * 70,
        "",
        "  Replaces all 12 reflectivity channels with low_level_ref (or gate-ref) values.",
        "  No retrain — tests whether column-max Z is a noisy proxy at inference time.",
        "",
    ]

    refl_ch = _field_channels(fields, 'reflectivity')
    ll_ch = _field_channels(fields, 'low_level_ref')
    gate_ch = _field_channels(fields, 'lowest_gate_reflectivity')

    if not refl_ch:
        lines.append("  ERROR: reflectivity not in model fields.")
        return lines

    baseline_preds, targets = _eval_loader(model, loader, device, log_target)
    ll_preds, _ = _eval_loader(model, loader, device, log_target, ll_ch, refl_ch)
    gate_preds = None
    if gate_ch:
        gate_preds, _ = _eval_loader(model, loader, device, log_target, gate_ch, refl_ch)

    heavy = targets > 5.0
    under = (targets >= under_min_actual) & (baseline_preds <= under_max_pred)

    scenarios = [
        ('baseline', baseline_preds),
        ('swap_low_level_ref', ll_preds),
    ]
    if gate_preds is not None:
        scenarios.append(('swap_lowest_gate_refl', gate_preds))

    slices = [
        ('all_val', np.ones(len(targets), dtype=bool)),
        ('heavy_gt5', heavy),
        ('underpredict', under),
    ]

    lines.append(f"  {'Scenario':<24} {'Slice':<16} {'n':>6} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
    lines.append(f"  {'-' * 24} {'-' * 16} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8}")

    csv_rows: list[dict] = []
    for scen_name, preds in scenarios:
        for slice_name, mask in slices:
            m = _slice_metrics(preds, targets, mask)
            lines.append(
                f"  {scen_name:<24} {slice_name:<16} {m['n']:>6} "
                f"{m['r2']:>8.4f} {m['mae']:>8.3f} {m['rmse']:>8.3f}"
            )
            csv_rows.append({
                'scenario': scen_name,
                'slice': slice_name,
                **m,
            })

    # Delta vs baseline
    lines.append("")
    lines.append("  Delta vs baseline (positive dR2 = swap helped):")
    base_all = _slice_metrics(baseline_preds, targets)
    for scen_name, preds in scenarios[1:]:
        for slice_name, mask in slices:
            m = _slice_metrics(preds, targets, mask)
            base_m = _slice_metrics(baseline_preds, targets, mask)
            dr2 = m['r2'] - base_m['r2'] if np.isfinite(m['r2']) and np.isfinite(base_m['r2']) else float('nan')
            dmae = m['mae'] - base_m['mae']
            lines.append(
                f"    {scen_name} / {slice_name}: dR2={dr2:+.4f}  dMAE={dmae:+.3f} mm/hr"
            )

    csv_path = out_dir / 'test4_swap_eval.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines.append(f"\n  Wrote: {csv_path}")

    return lines


def run_column_max_tests(
    run_dir: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_dir: str | None = None,
    pickle_path: str | None = None,
    dem_path: str | None = None,
    tests: list[str] | None = None,
    top_k: int = 15,
    n_steps: int = 50,
    batch_size: int = 64,
    under_min_actual: float = UNDER_MIN_ACTUAL,
    under_max_pred: float = UNDER_MAX_PRED,
) -> str:
    _ensure_utf8_stdio()
    tests = tests or ['center-ig', 'vertical', 'swap']
    test_set = set(tests)

    pickle_path = pickle_path or DEFAULT_PICKLE
    dem_path = dem_path or DEFAULT_DEM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = find_checkpoint(checkpoint_path, checkpoint_dir or DEFAULT_CKPT_DIR, run_dir=run_dir)

    if not run_dir:
        ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
        run_dir = ckpt_data.get('run_dir')

    model, cfg = load_model(ckpt, device)
    pickle_path = cfg.get('pickle_path') or pickle_path
    dem_path = cfg.get('dem_path') or dem_path
    fields = resolve_fields(cfg.get('fields'))
    log_target = cfg.get('log_target', True)

    loader, n_samples = build_val_loader(cfg, pickle_path, dem_path, batch_size)
    val_ds = loader.dataset

    out_dir = Path(run_dir) / 'column_max_reflectivity_tests'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "=" * 70,
        "  COLUMN-MAX REFLECTIVITY DIAGNOSTICS",
        "=" * 70,
        f"  Checkpoint: {ckpt}",
        f"  Val samples: {n_samples}",
        f"  Tests: {', '.join(sorted(test_set))}",
    ]

    if 'center-ig' in test_set:
        print("\n  Running Test 1 (center vs full-patch IG)...")
        summary_lines.extend(run_center_ig_test(
            model, loader, val_ds, device, cfg, fields, out_dir,
            top_k=top_k, n_steps=n_steps,
        ))

    if 'vertical' in test_set:
        print("\n  Running Test 2 (vertical alignment)...")
        summary_lines.extend(run_vertical_alignment_test(
            model, loader, val_ds, device, out_dir, log_target,
            under_min_actual=under_min_actual,
            under_max_pred=under_max_pred,
        ))

    if 'swap' in test_set:
        print("\n  Running Test 4 (low-level reflectivity swap)...")
        summary_lines.extend(run_swap_lowlevel_test(
            model, loader, device, fields, out_dir, log_target,
            under_min_actual=under_min_actual,
            under_max_pred=under_max_pred,
        ))

    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / 'summary.txt'
    summary_path.write_text(summary_text, encoding='utf-8')
    print(summary_text)
    print(f"\n  OK: Results in {out_dir}")
    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description="Column-max reflectivity hypothesis diagnostics (tests 1, 2, 4)",
    )
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--pickle', default=DEFAULT_PICKLE)
    parser.add_argument('--dem', default=DEFAULT_DEM)
    parser.add_argument(
        '--test', nargs='+', default=['all'],
        choices=['all', 'center-ig', 'vertical', 'swap'],
        help='Which tests to run (default: all)',
    )
    parser.add_argument('--top-k', type=int, default=15,
                        help='Outlier samples per cohort for IG test')
    parser.add_argument('--n-steps', type=int, default=50,
                        help='Integrated Gradients steps')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--under-min-actual', type=float, default=UNDER_MIN_ACTUAL)
    parser.add_argument('--under-max-pred', type=float, default=UNDER_MAX_PRED)
    args = parser.parse_args()

    tests = ['center-ig', 'vertical', 'swap'] if 'all' in args.test else args.test

    run_column_max_tests(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        pickle_path=args.pickle,
        dem_path=args.dem,
        tests=tests,
        top_k=args.top_k,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        under_min_actual=args.under_min_actual,
        under_max_pred=args.under_max_pred,
    )


if __name__ == '__main__':
    main()
