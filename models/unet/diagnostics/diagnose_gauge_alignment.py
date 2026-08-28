"""
diagnose_gauge_alignment.py — Test whether catastrophic misses are a
GAUGE-PIXEL GEOMETRY problem rather than a model/feature problem.

The training/eval pipeline assumes the gauge sits at the geometric center of the
radar patch (models/unet/dataset.py: gauge_pixel = (input_size//2, input_size//2)),
while create_pickle.py snaps the gauge to its NEAREST 500 m grid cell and may
shift the window at the grid edge. So the value the loss reads at pixel (C, C)
can be up to ~half a diagonal pixel (~350 m) away from the true gauge, and a
narrow convective core can land in an adjacent cell entirely.

This script quantifies that for the catastrophic-miss set, with four checks:

  A  Center-vs-patch gap   how much stronger is the patch-max reflectivity than
                           the center-pixel reflectivity? (rain in patch, not at
                           the gauge pixel)
  B  Best-match offset     where is the strongest echo (and the model's own
                           prediction peak) relative to the center pixel? A
                           consistent nonzero offset vector per station =>
                           systematic misalignment (bad coords / advection).
  C  Sub-pixel residual    distance from the gauge to its snapped grid-cell
                           center (stations near a cell boundary straddle two
                           cells).  [needs radar zarr grid; degrades gracefully]
  E  Edge-clip flag        did the boundary guard move the gauge off-center?

It compares the MISS group against a HIT control group (heavy hours the model
got right) so a real signal stands out from background scatter.

Run from project root:
    python -m models.unet.diagnostics.diagnose_gauge_alignment --run-dir models/checkpoints/unet_dualpol/<run_name>
"""

import argparse
import warnings
import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from torch.utils.data import DataLoader

from models.unet.evaluate import load_model, find_checkpoint
from models.unet.dataset import RadarGaugeDataset, resolve_fields, PICKLE_FIELD_ORDER
from models.unet.train import (
    filter_nan_radar, filter_biased_extremes, filter_bad_samples,
    filter_suspect_station_days, filter_radar_unsupported,
)


# ── Reflectivity / offset helpers ─────────────────────────────────────────────

def refl_score_map(radar_patch):
    """Per-pixel max-over-time reflectivity, sentinel/NaN-safe.

    Returns (H, W) float array; np.nan where a pixel never had valid data.
    This is what the radar "saw" at each pixel of the patch over the hour.
    """
    if 'reflectivity' not in PICKLE_FIELD_ORDER:
        return None
    idx = PICKLE_FIELD_ORDER.index('reflectivity')
    if idx >= radar_patch.shape[1]:
        return None
    arr = radar_patch[:, idx, :, :].astype(np.float32)          # (T, H, W)
    arr = np.where((arr == -9999.0) | np.isnan(arr), np.nan, arr)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        score = np.nanmax(arr, axis=0)                           # (H, W)
    return score


def best_offset(score_map, cy, cx, radius=None):
    """Offset (dy, dx) of the argmax pixel relative to (cy, cx), plus peak value.

    If ``radius`` is given, the argmax is restricted to the (2*radius+1)^2
    window centered on (cy, cx) — this probes genuine 1-2 pixel misalignment
    rather than the global reflectivity gradient across the whole patch.

    Returns None if every (in-window) pixel is NaN.
    """
    if score_map is None:
        return None
    finite = np.isfinite(score_map)
    if radius is not None:
        window = np.zeros_like(finite)
        y0, y1 = max(0, cy - radius), min(score_map.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(score_map.shape[1], cx + radius + 1)
        window[y0:y1, x0:x1] = True
        finite = finite & window
    if not finite.any():
        return None
    masked = np.where(finite, score_map, -np.inf)
    by, bx = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return dict(
        dy=int(by - cy), dx=int(bx - cx),
        dist_px=float(np.hypot(by - cy, bx - cx)),
        peak=float(score_map[by, bx]),
        center=float(score_map[cy, cx]) if finite[cy, cx] else np.nan,
    )


# ── Radar grid (for sub-pixel residual + edge-clip) ───────────────────────────

def load_radar_grid(metadata):
    """Read just the x/y coordinate arrays + CRS from the radar zarr referenced
    in the pickle metadata. Returns (x, y, crs, resolution_m) or None on failure.
    """
    zarr_path = metadata.get('radar_zarr')
    if not zarr_path:
        return None
    try:
        import zarr as _zarr
        store = _zarr.open(zarr_path, mode='r')
        x = np.asarray(store['x'][:], dtype=np.float64)
        y = np.asarray(store['y'][:], dtype=np.float64)
        attrs = dict(store.attrs)
        crs = attrs.get('crs')
        if crs is None and 'reflectivity' in store:
            crs = dict(store['reflectivity'].attrs).get('crs')
        crs = crs or 'EPSG:32610'
        res = float(attrs.get('resolution_m', 500))
        return dict(x=x, y=y, crs=crs, res=res)
    except Exception as e:
        print(f"  ⚠ Could not load radar grid for sub-pixel residual ({e}). "
              f"Skipping checks C/E.")
        return None


def station_grid_geometry(stations, grid, patch_pixels):
    """For each unique station compute the snap residual (distance to nearest
    grid-cell center) and whether the patch window would clip the grid edge.

    stations: dict station_name -> (lat, lon)
    Returns dict station_name -> dict(residual_m, edge_clipped, x_idx, y_idx).
    """
    if grid is None:
        return {}
    from pyproj import Transformer, CRS
    transformer = Transformer.from_crs(CRS.from_epsg(4326),
                                       CRS.from_string(grid['crs']),
                                       always_xy=True)
    x, y = grid['x'], grid['y']
    half = patch_pixels // 2
    out = {}
    for name, (lat, lon) in stations.items():
        if lat is None or lon is None:
            continue
        sx, sy = transformer.transform(lon, lat)
        x_idx = int(np.abs(x - sx).argmin())
        y_idx = int(np.abs(y - sy).argmin())
        residual = float(np.hypot(x[x_idx] - sx, y[y_idx] - sy))
        edge = (x_idx - half < 0) or (x_idx + half >= len(x)) or \
               (y_idx - half < 0) or (y_idx + half >= len(y))
        out[name] = dict(residual_m=residual, edge_clipped=bool(edge),
                         x_idx=x_idx, y_idx=y_idx)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_diagnostic(run_dir, pred_max=1.0, actual_min=5.0, actual_max=25.0,
                   search_radius=2):
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

    win_px = 2 * search_radius + 1
    print(f"\n  Validation samples: {len(val_ds.samples)}")
    print(f"  Heavy band: {actual_min}-{actual_max}mm | miss := pred < {pred_max}mm")
    print(f"  Offset search window: {win_px}x{win_px} px "
          f"(+/-{search_radius} px = +/-{search_radius*500} m around the gauge pixel)")

    # ── Inference: center prediction + full prediction map per sample ──
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    all_center_pred, all_actual = [], []
    all_pred_maps = []   # list of (H, W) arrays in mm space (model peak location)
    have_maps = True
    with torch.no_grad():
        for batch in val_loader:
            radar = batch['radar'].to(device)
            gauge_pixel = batch['gauge_pixel']
            pred_map = model(radar).cpu()

            if pred_map.dim() == 4:           # (B, 1, H, W) -> (B, H, W)
                pred_map = pred_map.squeeze(1)

            if pred_map.dim() == 1:           # scalar head: no spatial map
                have_maps = False
                center_pred = pred_map
            else:
                cyx = pred_map.shape[-1] // 2
                batch_idx = torch.arange(pred_map.shape[0])
                if isinstance(gauge_pixel, torch.Tensor) and gauge_pixel.dim() == 2:
                    y = gauge_pixel[:, 0].long(); x = gauge_pixel[:, 1].long()
                    center_pred = pred_map[batch_idx, y, x]
                elif isinstance(gauge_pixel, (tuple, list)):
                    y, x = gauge_pixel
                    center_pred = (pred_map[batch_idx, y, x]
                                   if isinstance(y, torch.Tensor) else pred_map[:, y, x])
                else:
                    center_pred = pred_map[:, cyx, cyx]

            targets = batch['target'].numpy()
            cp = center_pred.numpy()
            maps = pred_map.numpy() if have_maps else None
            if log_target:
                cp = np.expm1(cp)
                targets = np.expm1(targets)
                if maps is not None:
                    maps = np.expm1(maps)
            all_center_pred.extend(cp.tolist())
            all_actual.extend(targets.tolist())
            if maps is not None:
                all_pred_maps.extend(list(maps))

    all_center_pred = np.array(all_center_pred)
    all_actual = np.array(all_actual)

    heavy_mask = (all_actual >= actual_min) & (all_actual <= actual_max)
    miss_mask = heavy_mask & (all_center_pred < pred_max)
    hit_mask = heavy_mask & (all_center_pred >= actual_min)   # model also called it heavy
    heavy_idx = np.where(heavy_mask)[0]
    miss_idx = np.where(miss_mask)[0]
    hit_idx = np.where(hit_mask)[0]
    print(f"\n  Heavy samples: {len(heavy_idx)} | misses: {len(miss_idx)} | "
          f"hits (pred>= {actual_min}mm): {len(hit_idx)}")
    if len(miss_idx) == 0:
        print("  No catastrophic misses found; nothing to align. Try --pred-max higher.")
        return

    # ── Radar grid for sub-pixel residual / edge-clip ──
    with open(pickle_path, 'rb') as f:
        full_data = pkl.load(f)
    metadata = full_data.get('metadata', {})
    patch_pixels = int(val_ds.samples[0]['radar_patch'].shape[-1])
    grid = load_radar_grid(metadata)
    stations = {}
    for s in val_ds.samples:
        nm = s.get('station_name', '')
        if nm not in stations:
            stations[nm] = (s.get('station_lat'), s.get('station_lon'))
    station_geo = station_grid_geometry(stations, grid, patch_pixels)

    # ── Per-sample offset rows for a given index set ──
    def build_rows(indices):
        rows = []
        for i in indices:
            i = int(i)
            s = val_ds.samples[i]
            patch = s['radar_patch']
            cy, cx = patch.shape[2] // 2, patch.shape[3] // 2
            ref = best_offset(refl_score_map(patch), cy, cx, radius=search_radius)
            mod = (best_offset(all_pred_maps[i], cy, cx, radius=search_radius)
                   if all_pred_maps else None)
            rows.append(dict(
                idx=i,
                station=s.get('station_name', ''),
                actual=float(all_actual[i]),
                pred=float(all_center_pred[i]),
                refl=ref,
                model=mod,
            ))
        return rows

    miss_rows = build_rows(miss_idx)
    hit_rows = build_rows(hit_idx)

    def _arr(rows, src, key):
        out = []
        for r in rows:
            d = r.get(src)
            if d and np.isfinite(d.get(key, np.nan)):
                out.append(d[key])
        return np.array(out, dtype=float)

    def _gap(rows):
        """Per-row (patch-max minus center) reflectivity, only where both valid."""
        out = []
        for r in rows:
            d = r.get('refl')
            if d and np.isfinite(d.get('peak', np.nan)) and np.isfinite(d.get('center', np.nan)):
                out.append(d['peak'] - d['center'])
        return np.array(out, dtype=float)

    # ── Report: A + B globally (miss vs hit control) ──
    print(f"\n{'='*72}")
    print("  GAUGE-PIXEL ALIGNMENT DIAGNOSTIC")
    print(f"{'='*72}")

    def _summarize(rows, label):
        ref_dist = _arr(rows, 'refl', 'dist_px')
        ref_peak = _arr(rows, 'refl', 'peak')
        ref_cen = _arr(rows, 'refl', 'center')
        gap = _gap(rows)
        off_center = (ref_dist > 0).mean() * 100 if len(ref_dist) else np.nan
        print(f"\n  [{label}]  n={len(rows)}")
        if len(ref_dist):
            print(f"    Refl argmax distance from gauge pixel : "
                  f"median {np.median(ref_dist):.2f} px ({np.median(ref_dist)*500:.0f} m), "
                  f"off-center {off_center:.0f}%")
        else:
            print("    (no valid reflectivity)")
        if len(gap):
            print(f"    Patch-max minus center reflectivity   : "
                  f"median {np.median(gap):+.1f} dBZ  (center {np.median(ref_cen):.1f} -> "
                  f"peak {np.median(ref_peak):.1f})")
        if all_pred_maps:
            mod_dist = _arr(rows, 'model', 'dist_px')
            if len(mod_dist):
                print(f"    Model prediction-peak distance        : "
                      f"median {np.median(mod_dist):.2f} px ({np.median(mod_dist)*500:.0f} m)")

    _summarize(miss_rows, 'MISS (pred~0, heavy actual)')
    _summarize(hit_rows, 'HIT control (model called it heavy)')

    # Mean reflectivity-offset VECTOR for misses (systematic direction?)
    mdy = _arr(miss_rows, 'refl', 'dy'); mdx = _arr(miss_rows, 'refl', 'dx')
    if len(mdy):
        print(f"\n  Mean refl-offset vector (misses): "
              f"drow={mdy.mean():+.2f}, dcol={mdx.mean():+.2f} px  "
              f"(|mean|={np.hypot(mdy.mean(), mdx.mean()):.2f} px). "
              f"A large |mean| => consistent DIRECTION => coordinate/advection bias; "
              f"~0 with high spread => random sub-pixel scatter.")

    # ── Per-station alignment table ──
    by_station = defaultdict(list)
    for r in miss_rows:
        by_station[r['station']].append(r)
    print(f"\n  Per-station alignment (misses):")
    print(f"  {'Station':<22} {'miss':>4} {'refl_off_px':>11} {'mean(drow,dcol)':>16} "
          f"{'gap_dBZ':>8} {'resid_m':>8} {'edge':>5}")
    print(f"  {'-'*22} {'-'*4} {'-'*11} {'-'*16} {'-'*8} {'-'*8} {'-'*5}")
    station_table = []
    for station, rs in sorted(by_station.items(), key=lambda kv: -len(kv[1])):
        rd = _arr(rs, 'refl', 'dist_px')
        dy = _arr(rs, 'refl', 'dy'); dx = _arr(rs, 'refl', 'dx')
        gap_arr = _gap(rs)
        gap = np.median(gap_arr) if len(gap_arr) else np.nan
        geo = station_geo.get(station, {})
        resid = geo.get('residual_m', np.nan)
        edge = 'Y' if geo.get('edge_clipped') else ('-' if geo else '?')
        short = station.replace('Dangermond_', '')
        vec = f"({dy.mean():+.1f},{dx.mean():+.1f})" if len(dy) else "—"
        print(f"  {short:<22} {len(rs):>4} "
              f"{(f'{np.median(rd):.2f}' if len(rd) else '—'):>11} {vec:>16} "
              f"{(f'{gap:+.1f}' if np.isfinite(gap) else '—'):>8} "
              f"{(f'{resid:.0f}' if np.isfinite(resid) else '—'):>8} {edge:>5}")
        station_table.append(dict(
            station=station, n_miss=len(rs),
            refl_off_px=float(np.median(rd)) if len(rd) else np.nan,
            mean_drow=float(dy.mean()) if len(dy) else np.nan,
            mean_dcol=float(dx.mean()) if len(dx) else np.nan,
            gap_dbz=float(gap) if np.isfinite(gap) else np.nan,
            residual_m=float(resid) if np.isfinite(resid) else np.nan,
            edge_clipped=bool(geo.get('edge_clipped')) if geo else None,
        ))

    print(f"\n  How to read this table:")
    print(f"    refl_off_px ~0           -> radar core IS over the gauge pixel; "
          f"miss is a MODEL/feature failure, not geometry.")
    print(f"    refl_off_px large + tight mean vector -> echo consistently lands "
          f"off the gauge pixel in one direction -> coordinate error or advection.")
    print(f"    resid_m near {500//2} (==half pixel) -> gauge straddles a cell boundary.")
    print(f"    edge=Y                   -> boundary guard de-centered the gauge.")

    # ── Plots ──
    output_dir = Path(run_dir) if run_dir else Path('evaluation_figures/unet_dualpol')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Gauge-pixel alignment — misses vs hits '
                 f'({actual_min}-{actual_max}mm, miss pred<{pred_max}mm)',
                 fontsize=14, fontweight='bold')
    half = patch_pixels // 2
    obound = search_radius
    extent = [-obound - 0.5, obound + 0.5, obound + 0.5, -obound - 0.5]

    def offset_hist2d(ax, rows, title):
        dx = _arr(rows, 'refl', 'dx'); dy = _arr(rows, 'refl', 'dy')
        if len(dx):
            bins = np.arange(-obound - 0.5, obound + 1.5, 1)
            ax.hist2d(dx, dy, bins=[bins, bins], cmap='magma')
            ax.scatter([dx.mean()], [dy.mean()], c='cyan', marker='x', s=120,
                       label=f'mean ({dx.mean():+.1f},{dy.mean():+.1f})')
            ax.legend(fontsize=8)
        ax.scatter([0], [0], c='lime', marker='+', s=160)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel('dcol (px)'); ax.set_ylabel('drow (px)')
        ax.set_title(title)

    offset_hist2d(axes[0, 0], miss_rows, 'Refl-argmax offset — MISSES\n(green + = gauge pixel)')
    offset_hist2d(axes[0, 1], hit_rows, 'Refl-argmax offset — HITS (control)')

    # offset-distance distribution miss vs hit
    ax = axes[0, 2]
    md = _arr(miss_rows, 'refl', 'dist_px'); hd = _arr(hit_rows, 'refl', 'dist_px')
    bins = np.arange(0, obound + 1.5, 0.5)
    if len(md):
        ax.hist(md, bins=bins, alpha=0.6, density=True, label='miss', color='#8e44ad')
    if len(hd):
        ax.hist(hd, bins=bins, alpha=0.6, density=True, label='hit', color='#16a085')
    ax.set_xlabel('Refl argmax distance from gauge pixel (px)')
    ax.set_ylabel('density'); ax.set_title('Is the echo off-center more for misses?')
    ax.legend(fontsize=9)

    # center vs patch reflectivity (misses)
    ax = axes[1, 0]
    cen = _arr(miss_rows, 'refl', 'center'); peak = _arr(miss_rows, 'refl', 'peak')
    if len(cen) and len(cen) == len(peak):
        ax.scatter(cen, peak, alpha=0.6, s=25, color='#8e44ad')
        lim = [min(cen.min(), peak.min()) - 2, peak.max() + 2]
        ax.plot(lim, lim, 'k--', alpha=0.5)
    ax.set_xlabel('Center-pixel max dBZ'); ax.set_ylabel('Patch max dBZ')
    ax.set_title('Misses: rain in patch but not at gauge pixel?\n(points above 1:1 = off-center echo)')

    # per-station median offset distance (top misses)
    ax = axes[1, 1]
    top = sorted(station_table, key=lambda d: -d['n_miss'])[:12]
    if top:
        names = [d['station'].replace('Dangermond_', '')[:14] for d in top]
        vals = [d['refl_off_px'] if np.isfinite(d['refl_off_px']) else 0 for d in top]
        ax.barh(names[::-1], vals[::-1], color='#e67e22')
        ax.set_xlabel('Median refl offset (px)')
        ax.set_title('Per-station echo offset (top misses)')

    # per-station sub-pixel residual (if grid available)
    ax = axes[1, 2]
    geo_items = [(n.replace('Dangermond_', '')[:14], g['residual_m'])
                 for n, g in station_geo.items()]
    if geo_items:
        geo_items.sort(key=lambda kv: -kv[1])
        geo_items = geo_items[:14]
        ax.barh([n for n, _ in geo_items][::-1], [v for _, v in geo_items][::-1],
                color='#2980b9')
        ax.axvline((grid['res'] if grid else 500) / 2, color='red', ls='--',
                   label='half-pixel (max)')
        ax.set_xlabel('Gauge-to-cell-center residual (m)')
        ax.set_title('Sub-pixel snap residual per station')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'radar grid unavailable\n(checks C/E skipped)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    plt.tight_layout()
    out_path = output_dir / 'gauge_alignment_diagnostics.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved alignment plot to: {out_path}")
    plt.close()

    # ── CSVs ──
    station_csv = output_dir / 'gauge_alignment_by_station.csv'
    with open(station_csv, 'w') as f:
        f.write('station,n_miss,refl_offset_px_median,mean_drow,mean_dcol,'
                'patchmax_minus_center_dbz,subpixel_residual_m,edge_clipped\n')
        for d in sorted(station_table, key=lambda x: -x['n_miss']):
            def g(v, fmt='.2f'):
                return '' if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)
            f.write(f"{d['station']},{d['n_miss']},{g(d['refl_off_px'])},"
                    f"{g(d['mean_drow'])},{g(d['mean_dcol'])},{g(d['gap_dbz'],'.1f')},"
                    f"{g(d['residual_m'],'.0f')},"
                    f"{'' if d['edge_clipped'] is None else int(d['edge_clipped'])}\n")
    print(f"  ✓ Saved per-station alignment table to: {station_csv}")

    sample_csv = output_dir / 'gauge_alignment_samples.csv'
    with open(sample_csv, 'w') as f:
        f.write('station,actual_mm,center_pred_mm,refl_center_dbz,refl_peak_dbz,'
                'refl_drow,refl_dcol,refl_dist_px,model_drow,model_dcol,model_dist_px\n')
        for r in sorted(miss_rows, key=lambda x: -x['actual']):
            ref = r['refl'] or {}
            mod = r['model'] or {}
            def g(d, k, fmt='.2f'):
                v = d.get(k, np.nan)
                return '' if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
            f.write(f"{r['station']},{r['actual']:.2f},{r['pred']:.2f},"
                    f"{g(ref,'center','.1f')},{g(ref,'peak','.1f')},"
                    f"{g(ref,'dy','.0f')},{g(ref,'dx','.0f')},{g(ref,'dist_px')},"
                    f"{g(mod,'dy','.0f')},{g(mod,'dx','.0f')},{g(mod,'dist_px')}\n")
    print(f"  ✓ Saved per-sample offsets to: {sample_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose whether catastrophic misses are a gauge-pixel '
                    'geometry/alignment problem')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to the model run directory')
    parser.add_argument('--pred-max', type=float, default=1.0,
                        help='Flag a miss when center prediction < this (mm/hr)')
    parser.add_argument('--actual-min', type=float, default=5.0,
                        help='Minimum actual precip for the heavy band (mm/hr)')
    parser.add_argument('--actual-max', type=float, default=25.0,
                        help='Maximum actual precip for the heavy band (mm/hr)')
    parser.add_argument('--search-radius', type=int, default=2,
                        help='Half-width (px) of the window around the gauge pixel '
                             'to search for the echo/prediction peak (2 => 5x5 = '
                             '+/-1km). Probes true alignment, not global gradient.')
    args = parser.parse_args()
    run_diagnostic(args.run_dir, args.pred_max, args.actual_min, args.actual_max,
                   args.search_radius)


if __name__ == '__main__':
    main()
