"""
gradient_test.py — Integrated Gradients attribution for outlier vs control samples.

Compares IG profiles on large errors (under/overpredictions) to rain-matched
accurate controls and to stratified accurate baselines (low / medium / heavy
rain tiers). Attributes the gauge-pixel prediction in mm/hr (not the
asymmetric training loss).

Run from the project root:
    python -m models.unet.diagnostics.gradient_test --run-dir models/checkpoints/unet_dualpol/<run_name>
    python -m models.unet.diagnostics.gradient_test --run-dir ... --top-k 15 --cohorts under over
    python -m models.unet.diagnostics.gradient_test --run-dir ... --accurate-per-tier 15 --stratified-error-max 0.5

Requires: pip install captum
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from captum.attr import IntegratedGradients
except ImportError as exc:
    raise ImportError(
        "captum is required for gradient_test.py. Install with: pip install captum"
    ) from exc

from models.unet.ablation import build_feature_groups
from models.unet.dataset import (
    FEATURE_MASK_SOURCES,
    FIELD_FILL,
    FIELD_NORMS,
    N_SCANS,
    RadarGaugeDataset,
    resolve_fields,
)
from models.unet.evaluate import find_checkpoint, load_model
from models.unet.diagnostics.permutation_test import (
    DEFAULT_CKPT_DIR,
    DEFAULT_DEM,
    DEFAULT_PICKLE,
    _ensure_utf8_stdio,
    _pred_at_gauge,
    build_val_loader,
    select_groups,
)

RAIN_BINS = [(0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, float('inf'))]

# Coarse tiers for stratified accurate baselines and tier-wise outlier comparison.
RAIN_TIERS = [
    ('low', 0.0, 1.0),
    ('medium', 1.0, 5.0),
    ('heavy', 5.0, float('inf')),
]


def _gauge_coords(gauge_pixel, height: int, width: int) -> tuple[int, int]:
    """Resolve gauge (y, x); dataset uses patch center for all samples."""
    if isinstance(gauge_pixel, torch.Tensor) and gauge_pixel.dim() == 2:
        return int(gauge_pixel[0, 0].item()), int(gauge_pixel[0, 1].item())
    if isinstance(gauge_pixel, (tuple, list)):
        y, x = gauge_pixel
        if isinstance(y, torch.Tensor):
            return int(y.flatten()[0].item()), int(x.flatten()[0].item())
        return int(y), int(x)
    return height // 2, width // 2


@dataclass
class SampleRecord:
    idx: int
    actual_mm: float
    pred_mm: float
    error_mm: float
    abs_error: float
    station_name: str
    hour: str
    gauge_y: int
    gauge_x: int
    rain_bin: int = 0
    rain_tier: str = 'low'


@dataclass
class IGSummary:
    signed: dict[str, float] = field(default_factory=dict)
    abs_mean: dict[str, float] = field(default_factory=dict)
    gauge_signed: dict[str, float] = field(default_factory=dict)


class GaugePixelPredictor(nn.Module):
    """Captum wrapper: radar (B,C,H,W) -> gauge prediction in mm/hr (B,1)."""

    def __init__(self, model, gauge_y: int, gauge_x: int, log_target: bool):
        super().__init__()
        self.model = model
        self.gauge_y = gauge_y
        self.gauge_x = gauge_x
        self.log_target = log_target

    def forward(self, radar: torch.Tensor) -> torch.Tensor:
        pred_map = self.model(radar)
        pred = _pred_at_gauge(pred_map, (self.gauge_y, self.gauge_x))
        if self.log_target:
            pred = torch.expm1(pred)
        return pred.unsqueeze(1)


def rain_bin_index(actual_mm: float) -> int:
    for i, (lo, hi) in enumerate(RAIN_BINS):
        if lo <= actual_mm < hi:
            return i
    return len(RAIN_BINS) - 1


def rain_tier_name(actual_mm: float) -> str:
    for name, lo, hi in RAIN_TIERS:
        if lo <= actual_mm < hi:
            return name
    return RAIN_TIERS[-1][0]


def collect_predictions(model, loader, device, log_target) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    idx = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            radar = batch['radar'].to(device)
            target = batch['target']
            gauge_pixel = batch['gauge_pixel']
            station_names = batch.get('station_name', [''] * radar.shape[0])
            hours = batch.get('hour', [''] * radar.shape[0])

            pred_map = model(radar).cpu()
            pred_at_gauge = _pred_at_gauge(pred_map, gauge_pixel)
            target_cpu = target.cpu()
            if log_target:
                pred_at_gauge = torch.expm1(pred_at_gauge)
                target_cpu = torch.expm1(target_cpu)

            gy, gx = _gauge_coords(gauge_pixel, radar.shape[2], radar.shape[3])

            for b in range(radar.shape[0]):
                actual = float(target_cpu[b].item())
                pred = float(pred_at_gauge[b].item())
                if actual < 0:
                    idx += 1
                    continue
                records.append(SampleRecord(
                    idx=idx,
                    actual_mm=actual,
                    pred_mm=pred,
                    error_mm=pred - actual,
                    abs_error=abs(pred - actual),
                    station_name=str(station_names[b] if b < len(station_names) else ''),
                    hour=str(hours[b] if b < len(hours) else ''),
                    gauge_y=gy,
                    gauge_x=gx,
                    rain_bin=rain_bin_index(actual),
                    rain_tier=rain_tier_name(actual),
                ))
                idx += 1
    return records


def _match_control(outlier: SampleRecord, pool: list[SampleRecord], match_error_max: float) -> SampleRecord | None:
    candidates = [
        r for r in pool
        if r.rain_bin == outlier.rain_bin
        and r.abs_error <= match_error_max
        and r.idx != outlier.idx
    ]
    if not candidates:
        candidates = [r for r in pool if r.abs_error <= match_error_max and r.idx != outlier.idx]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r.actual_mm - outlier.actual_mm))


def select_cohorts(
    records: list[SampleRecord],
    top_k: int,
    cohorts: list[str],
    under_min_actual: float,
    under_max_pred: float,
    over_min_error: float,
    match_error_max: float,
    accurate_k: int,
) -> dict[str, list[tuple[SampleRecord, SampleRecord | None]]]:
    """Return {cohort_name: [(outlier_or_focus, matched_control), ...]}."""
    pool = list(records)
    accurate_pool = sorted(records, key=lambda r: r.abs_error)[: max(accurate_k, top_k * 3)]

    result: dict[str, list[tuple[SampleRecord, SampleRecord | None]]] = {}

    if 'under' in cohorts:
        unders = [
            r for r in records
            if r.actual_mm >= under_min_actual and r.pred_mm <= under_max_pred
        ]
        unders = sorted(unders, key=lambda r: r.actual_mm - r.pred_mm, reverse=True)[:top_k]
        result['under'] = [(o, _match_control(o, accurate_pool, match_error_max)) for o in unders]

    if 'over' in cohorts:
        overs = [r for r in records if (r.pred_mm - r.actual_mm) >= over_min_error]
        overs = sorted(overs, key=lambda r: r.pred_mm - r.actual_mm, reverse=True)[:top_k]
        result['over'] = [(o, _match_control(o, accurate_pool, match_error_max)) for o in overs]

    if 'abs' in cohorts:
        worst = sorted(records, key=lambda r: r.abs_error, reverse=True)[:top_k]
        result['abs'] = [(o, _match_control(o, accurate_pool, match_error_max)) for o in worst]

    if 'accurate' in cohorts:
        acc = sorted(records, key=lambda r: r.abs_error)[:top_k]
        result['accurate'] = [(a, None) for a in acc]

    return result


def select_accurate_stratified(
    records: list[SampleRecord],
    per_tier: int,
    max_abs_error: float,
) -> dict[str, list[SampleRecord]]:
    """Best accurate predictions within each rain tier (low / medium / heavy)."""
    pools: dict[str, list[SampleRecord]] = {name: [] for name, _, _ in RAIN_TIERS}
    for r in records:
        if r.abs_error <= max_abs_error:
            pools[rain_tier_name(r.actual_mm)].append(r)
    selected: dict[str, list[SampleRecord]] = {}
    for name in pools:
        tier = sorted(pools[name], key=lambda r: (r.abs_error, abs(r.error_mm)))
        selected[name] = tier[:per_tier]
    return selected


def _norm_fill_value(field_name: str) -> float:
    f_min, f_max = FIELD_NORMS[field_name]
    fill_raw = FIELD_FILL.get(field_name, f_min)
    val = (fill_raw - f_min) / (f_max - f_min)
    return float(np.clip(val, 0.0, 1.0))


def build_fill_baseline(
    fields,
    use_mask,
    use_temporal_pos,
    use_dem,
    use_feature_masks,
    height: int,
    width: int,
) -> torch.Tensor:
    """Channel layout matches RadarGaugeDataset."""
    planes = []
    for field_name in fields:
        v = _norm_fill_value(field_name)
        for _ in range(N_SCANS):
            planes.append(torch.full((height, width), v))
    if use_mask:
        for _ in range(N_SCANS):
            planes.append(torch.zeros((height, width)))
    if use_feature_masks:
        for _ in FEATURE_MASK_SOURCES:
            for _ in range(N_SCANS):
                planes.append(torch.zeros((height, width)))
    if use_temporal_pos:
        for i in range(N_SCANS):
            planes.append(torch.full((height, width), i / max(N_SCANS - 1, 1)))
    if use_dem:
        planes.append(torch.full((height, width), 0.5))
    return torch.stack(planes, dim=0)


def build_mean_baseline(model, loader, device, max_batches: int) -> torch.Tensor:
    total = None
    count = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            radar = batch['radar'].to(device)
            if total is None:
                total = torch.zeros_like(radar[0], dtype=torch.float64)
            for b in range(radar.shape[0]):
                total += radar[b].double().cpu()
                count += 1
    if count == 0:
        raise RuntimeError("No batches available to compute mean baseline")
    return (total / count).float()


def aggregate_field_attributions(
    attr: torch.Tensor,
    feature_groups: dict[str, list[int]],
    group_names: list[str],
    gauge_y: int,
    gauge_x: int,
) -> IGSummary:
    """attr shape (C, H, W)."""
    attr_np = attr.detach().cpu().numpy()
    summary = IGSummary()
    for name in group_names:
        ch = feature_groups[name]
        if not ch:
            continue
        block = attr_np[ch]
        summary.signed[name] = float(block.sum())
        summary.abs_mean[name] = float(np.abs(block).mean())
        summary.gauge_signed[name] = float(block[:, gauge_y, gauge_x].sum())
    return summary


def compute_integrated_gradients(
    model,
    radar: torch.Tensor,
    baseline: torch.Tensor,
    gauge_y: int,
    gauge_x: int,
    log_target: bool,
    device,
    n_steps: int,
) -> torch.Tensor:
    """Return attributions (C, H, W) for a single sample."""
    wrapper = GaugePixelPredictor(model, gauge_y, gauge_x, log_target).to(device)
    wrapper.eval()
    ig = IntegratedGradients(wrapper)

    x = radar.unsqueeze(0).to(device)
    base = baseline.unsqueeze(0).to(device)
    attrs = ig.attribute(x, baselines=base, n_steps=n_steps, internal_batch_size=1)
    return attrs.squeeze(0)


def get_sample_ig_summary(
    model,
    val_ds,
    record: SampleRecord,
    baseline: torch.Tensor,
    feature_groups: dict,
    group_names: list[str],
    log_target: bool,
    device,
    n_steps: int,
    cache: dict[int, IGSummary],
) -> IGSummary:
    if record.idx in cache:
        return cache[record.idx]
    radar = val_ds[record.idx]['radar']
    attr = compute_integrated_gradients(
        model, radar, baseline, record.gauge_y, record.gauge_x,
        log_target, device, n_steps,
    )
    summary = aggregate_field_attributions(
        attr, feature_groups, group_names, record.gauge_y, record.gauge_x,
    )
    cache[record.idx] = summary
    return summary


def append_ig_csv_rows(
    csv_rows: list[dict],
    cohort: str,
    role: str,
    pair_idx: int,
    record: SampleRecord,
    summary: IGSummary,
) -> None:
    for fname, val in summary.signed.items():
        csv_rows.append({
            'cohort': cohort,
            'role': role,
            'rain_tier': record.rain_tier,
            'pair_idx': pair_idx,
            'sample_idx': record.idx,
            'station': record.station_name,
            'hour': record.hour,
            'actual_mm': record.actual_mm,
            'pred_mm': record.pred_mm,
            'error_mm': record.error_mm,
            'field': fname,
            'signed_ig': val,
            'abs_ig_mean': summary.abs_mean.get(fname, 0.0),
            'gauge_signed_ig': summary.gauge_signed.get(fname, 0.0),
        })


def run_stratified_analysis(
    cohort_name: str,
    outlier_records: list[SampleRecord],
    accurate_by_tier: dict[str, list[SampleRecord]],
    ig_cache: dict[int, IGSummary],
    group_names: list[str],
    summary_lines: list[str],
    plot_dir: Path,
) -> None:
    """Compare outlier IG to accurate IG within each rain tier."""
    if not outlier_records:
        return

    accurate_stats: dict[str, dict] = {}
    for tier_name, acc_records in accurate_by_tier.items():
        if acc_records:
            accurate_stats[tier_name] = summarize_cohort(
                [ig_cache[r.idx] for r in acc_records], group_names,
            )

    outliers_by_tier: dict[str, list[SampleRecord]] = {name: [] for name, _, _ in RAIN_TIERS}
    for record in outlier_records:
        outliers_by_tier[record.rain_tier].append(record)

    summary_lines.append("")
    summary_lines.append(f"  Stratified: {cohort_name} outliers vs accurate (low / medium / heavy)")
    summary_lines.append(
        f"  {'Tier':<8} {'N_out':>6} {'N_acc':>6} {'Field':<24} "
        f"{'Outlier IG':>12} {'Accurate IG':>12} {'Delta':>12}"
    )
    summary_lines.append(
        f"  {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 24} {'-' * 12} {'-' * 12} {'-' * 12}"
    )

    for tier_name, lo, hi in RAIN_TIERS:
        outs = outliers_by_tier.get(tier_name, [])
        acc = accurate_by_tier.get(tier_name, [])
        if not outs or not acc or tier_name not in accurate_stats:
            summary_lines.append(
                f"  {tier_name:<8} {len(outs):>6} {len(acc):>6}  (skipped — need outliers and accurate in tier)"
            )
            continue

        o_stats = summarize_cohort([ig_cache[r.idx] for r in outs], group_names)
        a_stats = accurate_stats[tier_name]
        ranked = sorted(
            group_names,
            key=lambda n: abs(o_stats['signed'].get(n, 0.0) - a_stats['signed'].get(n, 0.0)),
            reverse=True,
        )[:12]

        hi_label = 'inf' if hi == float('inf') else f'{hi:g}'
        summary_lines.append(
            f"  --- tier {tier_name} (actual {lo:g}-{hi_label} mm/hr) ---"
        )
        for name in ranked:
            o = o_stats['signed'].get(name, 0.0)
            a = a_stats['signed'].get(name, 0.0)
            summary_lines.append(
                f"  {tier_name:<8} {len(outs):>6} {len(acc):>6} {name:<24} "
                f"{o:>+12.6f} {a:>+12.6f} {o - a:>+12.6f}"
            )

        plot_field_comparison(
            o_stats, a_stats, ranked,
            f'{cohort_name}_{tier_name}_vs_accurate',
            plot_dir / f'stratified_{cohort_name}_{tier_name}.png',
        )


def plot_accurate_tier_profiles(
    accurate_by_tier: dict[str, list[SampleRecord]],
    ig_cache: dict[int, IGSummary],
    group_names: list[str],
    plot_dir: Path,
) -> None:
    """Bar chart of mean signed IG on accurate samples, one panel per rain tier."""
    tier_stats = {}
    for tier_name, records in accurate_by_tier.items():
        if records:
            tier_stats[tier_name] = summarize_cohort(
                [ig_cache[r.idx] for r in records], group_names,
            )

    if not tier_stats:
        return

    ranked = sorted(
        group_names,
        key=lambda n: max(abs(tier_stats[t]['signed'].get(n, 0.0)) for t in tier_stats),
        reverse=True,
    )[:14]
    x = np.arange(len(ranked))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(ranked) * 0.5), 5))
    for i, tier_name in enumerate([t[0] for t in RAIN_TIERS if t[0] in tier_stats]):
        vals = [tier_stats[tier_name]['signed'].get(n, 0.0) for n in ranked]
        ax.bar(x + (i - 1) * width, vals, width, label=f'accurate {tier_name}')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ranked, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Mean signed IG (mm/hr)')
    ax.set_title('Accurate predictions by rain tier')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / 'accurate_tier_profiles.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def summarize_cohort(
    summaries: list[IGSummary],
    group_names: list[str],
) -> dict[str, dict[str, float]]:
    if not summaries:
        return {'signed': {}, 'abs_mean': {}, 'gauge_signed': {}}
    out = {'signed': {}, 'abs_mean': {}, 'gauge_signed': {}}
    for name in group_names:
        out['signed'][name] = float(np.mean([s.signed.get(name, 0.0) for s in summaries]))
        out['abs_mean'][name] = float(np.mean([s.abs_mean.get(name, 0.0) for s in summaries]))
        out['gauge_signed'][name] = float(np.mean([s.gauge_signed.get(name, 0.0) for s in summaries]))
    return out


def flag_anomalous_fields(
    outlier_stats: dict[str, dict[str, float]],
    control_stats: dict[str, dict[str, float]],
    group_names: list[str],
    sign_flip_threshold: float = 0.001,
) -> list[tuple[str, str]]:
    flags = []
    for name in group_names:
        o = outlier_stats['signed'].get(name, 0.0)
        c = control_stats['signed'].get(name, 0.0)
        if o * c < 0 and abs(o - c) >= sign_flip_threshold:
            flags.append((name, 'sign_flip'))
        elif abs(o) > 3 * max(abs(c), 1e-6) and abs(o) >= sign_flip_threshold:
            flags.append((name, 'outlier_dominant'))
        elif o < 0 and abs(o) >= sign_flip_threshold:
            flags.append((name, 'negative_push'))
    return flags


def plot_field_comparison(
    outlier_stats,
    control_stats,
    group_names,
    cohort_name,
    out_path,
):
    names = list(group_names)
    o_vals = [outlier_stats['signed'].get(n, 0.0) for n in names]
    c_vals = [control_stats['signed'].get(n, 0.0) for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 5))
    ax.bar(x - width / 2, o_vals, width, label=f'{cohort_name} outliers', color='coral')
    ax.bar(x + width / 2, c_vals, width, label='matched controls', color='steelblue')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Mean signed IG (mm/hr attribution)')
    ax.set_title(f'Integrated Gradients: {cohort_name} outliers vs controls')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_spatial_field(
    attr: torch.Tensor,
    channel_indices: list[int],
    field_name: str,
    gauge_y: int,
    gauge_x: int,
    title: str,
    out_path,
):
    block = attr[channel_indices].detach().cpu().numpy()
    spatial = block.sum(axis=0)
    vmax = max(np.abs(spatial).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(spatial, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.plot(gauge_x, gauge_y, 'k+', markersize=10, markeredgewidth=2)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def run_gradient_test(
    checkpoint_path=None,
    checkpoint_dir=None,
    pickle_path=None,
    dem_path=None,
    run_dir=None,
    top_k=20,
    cohorts=None,
    under_min_actual=5.0,
    under_max_pred=2.0,
    over_min_error=5.0,
    match_error_max=1.0,
    accurate_k=100,
    baseline_mode='fill',
    baseline_batches=30,
    n_steps=50,
    batch_size=64,
    fields_only=True,
    plot_spatial=True,
    spatial_fields=None,
    seed=0,
    stratified=True,
    accurate_per_tier=10,
    stratified_error_max=0.5,
):
    _ensure_utf8_stdio()
    cohorts = cohorts or ['under', 'over']
    spatial_fields = spatial_fields or ['reflectivity', 'vil', 'low_level_kdp']

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
    use_dem = cfg.get('use_dem', True)
    use_mask = cfg.get('use_mask', True)
    use_temporal_pos = cfg.get('use_temporal_pos', True)
    use_feature_masks = cfg.get('use_feature_masks', False)
    log_target = cfg.get('log_target', True)

    feature_groups = build_feature_groups(
        fields, use_mask, use_temporal_pos, use_dem, use_feature_masks,
    )
    group_names = select_groups(feature_groups, fields_only=fields_only, skip_composite=True)

    loader, n_samples = build_val_loader(cfg, pickle_path, dem_path, batch_size)
    val_ds = loader.dataset

    print(f"\n{'=' * 78}")
    print("  INTEGRATED GRADIENTS — OUTLIER ATTRIBUTION")
    print(f"{'=' * 78}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Samples:    {n_samples}")
    print(f"  Baseline:   {baseline_mode}  IG steps: {n_steps}")
    print(f"  Cohorts:    {', '.join(cohorts)}  top_k={top_k}")
    if stratified:
        print(
            f"  Stratified: {accurate_per_tier} accurate samples/tier "
            f"(|error| <= {stratified_error_max} mm/hr)"
        )
    print(f"{'=' * 78}")

    print("\n  Pass 1: collecting validation predictions...")
    records = collect_predictions(model, loader, device, log_target)
    print(f"  Collected {len(records)} samples")

    cohort_pairs = select_cohorts(
        records, top_k, cohorts,
        under_min_actual, under_max_pred, over_min_error,
        match_error_max, accurate_k,
    )

    # Baseline tensor
    sample0 = val_ds[0]['radar']
    h, w = sample0.shape[1], sample0.shape[2]
    if baseline_mode == 'mean':
        print(f"  Computing mean baseline over {baseline_batches} batches...")
        baseline = build_mean_baseline(model, loader, device, baseline_batches)
    else:
        baseline = build_fill_baseline(
            fields, use_mask, use_temporal_pos, use_dem, use_feature_masks, h, w,
        )

    out_root = Path(run_dir) / 'gradient_attribution'
    plot_dir = out_root / 'plots'
    out_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / 'sample_attributions.csv'
    csv_rows = []
    summary_lines = [
        "=" * 60,
        "  INTEGRATED GRADIENTS ATTRIBUTION",
        "=" * 60,
        "",
        f"  Checkpoint: {ckpt}",
        f"  Baseline:   {baseline_mode}",
        f"  IG steps:   {n_steps}",
        "",
    ]

    torch.manual_seed(seed)
    ig_cache: dict[int, IGSummary] = {}

    # Pre-compute stratified accurate baselines (low / medium / heavy).
    accurate_by_tier: dict[str, list[SampleRecord]] = {}
    if stratified:
        accurate_by_tier = select_accurate_stratified(
            records, accurate_per_tier, stratified_error_max,
        )
        print("\n  Stratified accurate samples per tier:")
        for tier_name, lo, hi in RAIN_TIERS:
            hi_label = 'inf' if hi == float('inf') else f'{hi:g}'
            n = len(accurate_by_tier.get(tier_name, []))
            print(f"    {tier_name:8s} ({lo:g}-{hi_label} mm/hr): {n} samples")
        for tier_records in accurate_by_tier.values():
            for record in tier_records:
                get_sample_ig_summary(
                    model, val_ds, record, baseline, feature_groups, group_names,
                    log_target, device, n_steps, ig_cache,
                )
                append_ig_csv_rows(csv_rows, 'accurate_tier', 'accurate', -1, record,
                                   ig_cache[record.idx])

    for cohort_name, pairs in cohort_pairs.items():
        print(f"\n  Cohort: {cohort_name} ({len(pairs)} samples)")
        outlier_summaries = []
        control_summaries = []
        outlier_records: list[SampleRecord] = []

        for i, (focus, control) in enumerate(pairs):
            outlier_records.append(focus)
            o_sum = get_sample_ig_summary(
                model, val_ds, focus, baseline, feature_groups, group_names,
                log_target, device, n_steps, ig_cache,
            )
            outlier_summaries.append(o_sum)
            append_ig_csv_rows(csv_rows, cohort_name, 'outlier', i, focus, o_sum)

            if plot_spatial and i < 3 and cohort_name in ('under', 'over', 'abs'):
                radar = val_ds[focus.idx]['radar']
                attr = compute_integrated_gradients(
                    model, radar, baseline, focus.gauge_y, focus.gauge_x,
                    log_target, device, n_steps,
                )
                for fname in spatial_fields:
                    if fname not in feature_groups:
                        continue
                    plot_spatial_field(
                        attr, feature_groups[fname], fname, focus.gauge_y, focus.gauge_x,
                        f'{cohort_name} #{i} {fname}\nactual={focus.actual_mm:.1f} pred={focus.pred_mm:.1f}',
                        plot_dir / f'{cohort_name}_{i:02d}_{fname}_spatial.png',
                    )

            if control is not None:
                c_sum = get_sample_ig_summary(
                    model, val_ds, control, baseline, feature_groups, group_names,
                    log_target, device, n_steps, ig_cache,
                )
                control_summaries.append(c_sum)
                append_ig_csv_rows(csv_rows, cohort_name, 'control', i, control, c_sum)
            else:
                print(
                    f"    [{i}] outlier idx={focus.idx} tier={focus.rain_tier} "
                    f"actual={focus.actual_mm:.1f} pred={focus.pred_mm:.1f} (no control)"
                )

        o_stats = summarize_cohort(outlier_summaries, group_names)
        c_stats = summarize_cohort(control_summaries, group_names) if control_summaries else o_stats

        summary_lines.append(f"  Cohort: {cohort_name} (paired controls)")
        summary_lines.append(
            f"  {'Field':<28} {'Outlier IG':>12} {'Control IG':>12} {'Delta':>12} {'Flag':>16}"
        )
        summary_lines.append(f"  {'-' * 28} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 16}")

        ranked = sorted(
            group_names,
            key=lambda n: abs(o_stats['signed'].get(n, 0.0)),
            reverse=True,
        )
        flags = flag_anomalous_fields(o_stats, c_stats, group_names)
        flag_map = {n: tag for n, tag in flags}

        for name in ranked:
            o = o_stats['signed'].get(name, 0.0)
            c = c_stats['signed'].get(name, 0.0)
            delta = o - c
            flag = flag_map.get(name, '')
            summary_lines.append(
                f"  {name:<28} {o:>+12.6f} {c:>+12.6f} {delta:>+12.6f} {flag:>16}"
            )
        summary_lines.append("")

        if control_summaries:
            plot_field_comparison(
                o_stats, c_stats, ranked[: min(20, len(ranked))],
                cohort_name, plot_dir / f'{cohort_name}_field_comparison.png',
            )

        if stratified and cohort_name in ('under', 'over', 'abs'):
            run_stratified_analysis(
                cohort_name, outlier_records, accurate_by_tier,
                ig_cache, group_names, summary_lines, plot_dir,
            )

    if stratified and accurate_by_tier:
        summary_lines.append("")
        summary_lines.append("  Accurate baseline profiles by rain tier (reference):")
        for tier_name, lo, hi in RAIN_TIERS:
            acc = accurate_by_tier.get(tier_name, [])
            if not acc:
                continue
            stats = summarize_cohort([ig_cache[r.idx] for r in acc], group_names)
            hi_label = 'inf' if hi == float('inf') else f'{hi:g}'
            top = sorted(group_names, key=lambda n: abs(stats['signed'].get(n, 0.0)), reverse=True)[:5]
            tops = ', '.join(f"{n}={stats['signed'].get(n, 0.0):+.3f}" for n in top)
            summary_lines.append(
                f"    {tier_name} ({lo:g}-{hi_label} mm/hr, n={len(acc)}): top |IG| -> {tops}"
            )
        plot_accurate_tier_profiles(accurate_by_tier, ig_cache, group_names, plot_dir)

    summary_lines.extend([
        "  Flags:",
        "  - sign_flip: outlier vs control attribution opposite sign",
        "  - outlier_dominant: |outlier IG| >> |control IG|",
        "  - negative_push: signed IG negative on outliers (feature pushes pred down)",
        "",
        "  Note: signed IG sums over 12 scans + full spatial patch per field.",
        "  Positive = increasing this field (from baseline) increases predicted mm/hr.",
        "",
    ])

    summary_text = "\n".join(summary_lines)
    (out_root / 'summary.txt').write_text(summary_text, encoding='utf-8')
    print(summary_text)

    if csv_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  OK: Wrote CSV to: {csv_path}")
    print(f"  OK: Wrote plots to: {plot_dir}")

    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Gradients outlier attribution for U-Net QPE models",
    )
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--pickle', default=DEFAULT_PICKLE)
    parser.add_argument('--dem', default=DEFAULT_DEM)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--cohorts', nargs='+', default=['under', 'over'],
                        choices=['under', 'over', 'abs', 'accurate'],
                        help='Cohorts to analyze (default: under over)')
    parser.add_argument('--under-min-actual', type=float, default=5.0)
    parser.add_argument('--under-max-pred', type=float, default=2.0)
    parser.add_argument('--over-min-error', type=float, default=5.0,
                        help='Min (pred - actual) mm/hr for overprediction cohort')
    parser.add_argument('--match-error-max', type=float, default=1.0,
                        help='Max |error| for matched accurate controls')
    parser.add_argument('--accurate-k', type=int, default=100,
                        help='Pool size of low-error samples for per-outlier matching')
    parser.add_argument('--accurate-per-tier', type=int, default=10,
                        help='Accurate samples per rain tier for stratified comparison')
    parser.add_argument('--stratified-error-max', type=float, default=0.5,
                        help='Max |error| mm/hr for stratified accurate baselines')
    parser.add_argument('--no-stratified', action='store_true',
                        help='Skip low/medium/heavy accurate tier comparison')
    parser.add_argument('--baseline', choices=['fill', 'mean'], default='fill')
    parser.add_argument('--baseline-batches', type=int, default=30)
    parser.add_argument('--n-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--fields-only', action='store_true', default=True,
                        help='Attribute radar fields only (default: True)')
    parser.add_argument('--include-aux', action='store_true',
                        help='Include mask, temporal_pos, dem, feature_masks')
    parser.add_argument('--no-spatial-plots', action='store_true')
    parser.add_argument('--spatial-fields', nargs='+', default=['reflectivity', 'vil', 'low_level_kdp'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_gradient_test(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        pickle_path=args.pickle,
        dem_path=args.dem,
        run_dir=args.run_dir,
        top_k=args.top_k,
        cohorts=args.cohorts,
        under_min_actual=args.under_min_actual,
        under_max_pred=args.under_max_pred,
        over_min_error=args.over_min_error,
        match_error_max=args.match_error_max,
        accurate_k=args.accurate_k,
        baseline_mode=args.baseline,
        baseline_batches=args.baseline_batches,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        fields_only=not args.include_aux,
        plot_spatial=not args.no_spatial_plots,
        spatial_fields=args.spatial_fields,
        seed=args.seed,
        stratified=not args.no_stratified,
        accurate_per_tier=args.accurate_per_tier,
        stratified_error_max=args.stratified_error_max,
    )


if __name__ == '__main__':
    main()
