"""
derive_features.py — Derive 2D radar features from the full 3D volume zarr.

The radar pull (radar/pull_nexrad_multi.py) now stores the FULL 3D volume
(z, y, x) for the 5 raw fields. This script collapses that volume into the
model-ready 2D feature zarr (time, y, x), so feature engineering can be
iterated WITHOUT re-pulling from S3 (which takes ~2 days).

It reproduces the legacy 10 features (drop-in replacement for the old 2D zarr),
with one deliberate change: height features (echo_top_height, max_z_height,
melting_layer_height) now fill no-echo pixels with NaN instead of 0.0, so that
"no echo" is distinguishable from "low height" and can drive a validity mask
downstream. It also adds new vertical-structure features motivated by the
over/under-prediction diagnostics:

  Low-level (warm-rain / orographic) — fixed 0–2 km (legacy, unchanged):
    low_level_kdp, low_level_zdr, low_level_rhohv,
    lowest_gate_reflectivity, beam_height, vertical_reflectivity_gradient
  Sub-melting-layer (dynamic liquid column):
    subml_rhohv, subml_zdr, subml_kdp, subml_ref_max, subml_zdr_gradient
    Uses gates below the melting layer when a credible bright band is detected;
    otherwise falls back to beam_height → min(2 km, echo top).
  Melting layer / bright band (cold over-read):
    melting_layer_height, rhohv_min,
    bright_band_ref, bright_band_drop, maxz_meltlayer_offset,
    bright_band_intensity

Output schema matches what dataset/create_pickle.py expects: data_vars with
dims (time, y, x), coords time/x/y, and per-field 'crs' attrs.

Run from the project root:
    python -m radar.derive_features \
        --input  radar/outputs/dualpol_3d_500m_2022-01-01_2026-04-04.zarr \
        --output radar/outputs/dualpol_feat_500m_2022-01-01_2026-04-04.zarr
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm


# Raw 3D fields expected in the input zarr.
RAW_FIELDS = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
]

# Output field order. First 21 entries are the legacy schema (drop-in compatible);
# subml_* fields are appended so existing pickles/checkpoints keep channel alignment
# when trained without them.
OUTPUT_FIELDS = [
    # legacy
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
    'echo_top_height',
    'max_z_height',
    'vil',
    'low_level_ref',
    'column_depth_fraction',
    # fixed 0–2 km low-level / warm-rain (legacy)
    'low_level_kdp',
    'low_level_zdr',
    'low_level_rhohv',
    'lowest_gate_reflectivity',
    'beam_height',
    'vertical_reflectivity_gradient',
    # melting layer / bright band
    'melting_layer_height',
    'rhohv_min',
    'bright_band_ref',
    'bright_band_drop',
    'maxz_meltlayer_offset',
    'bright_band_intensity',
    # sub-melting-layer liquid column (dynamic or warm-rain fallback)
    'subml_rhohv',
    'subml_zdr',
    'subml_kdp',
    'subml_ref_max',
    'subml_zdr_gradient',
]

# Bright-band credibility thresholds for sub-ML dynamic sampling.
SUBML_RHOHV_MAX = 0.95
SUBML_ML_MIN_HEIGHT_M = 500.0
WARM_RAIN_CAP_M = 2000.0


def _layer_stat(arr3d, mask3d, stat='mean'):
    """Reduce arr3d (nz, ny, nx) over axis 0 where mask3d is True."""
    masked = np.where(mask3d, arr3d, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if stat == 'mean':
            return np.nanmean(masked, axis=0)
        if stat == 'max':
            return np.nanmax(masked, axis=0)
        raise ValueError(f"Unknown stat: {stat}")


def _fixed_low_mean(arr3d, low_mask):
    """Legacy fixed 0–2 km column mean."""
    if not low_mask.any():
        return np.full(arr3d.shape[1:], np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(arr3d[low_mask, :, :], axis=0)


def _build_liquid_layer_mask(
    z_heights,
    valid,
    beam_height,
    echo_top_height,
    melting_layer_height,
    rhohv_min,
    has_any,
    z_res,
):
    """
    Per-column 3D mask for the liquid rain layer.

    When a credible bright band is present: beam_height <= z < melting_layer_height.
    Otherwise (warm rain / shallow columns): beam_height <= z <= min(2 km, echo top).
    """
    z3 = z_heights[:, None, None]
    beam = beam_height[None, :, :]
    ml_h = melting_layer_height[None, :, :]
    cap = np.minimum(WARM_RAIN_CAP_M, echo_top_height)[None, :, :]

    below_ml = valid & (z3 < ml_h) & (z3 >= beam)
    fallback = valid & (z3 >= beam) & (z3 <= cap)

    bb_present = (
        has_any
        & np.isfinite(rhohv_min) & (rhohv_min < SUBML_RHOHV_MAX)
        & np.isfinite(melting_layer_height) & (melting_layer_height > SUBML_ML_MIN_HEIGHT_M)
        & np.isfinite(beam_height) & (melting_layer_height > beam_height + z_res)
    )
    liquid_mask = np.where(bb_present[None, :, :], below_ml, fallback)
    return liquid_mask, bb_present


def compute_features(vol, z_heights):
    """
    Compute all 2D features for a single scan.

    vol        : dict field_name -> (nz, ny, nx) array (NaN where no data)
    z_heights  : (nz,) heights in metres above radar
    returns    : dict field_name -> (ny, nx) array
    """
    ref_3d = vol['reflectivity']
    nz_grid, ny, nx = ref_3d.shape
    z_res = float(z_heights[1] - z_heights[0]) if nz_grid > 1 else 375.0

    out = {}

    # ── Legacy features (identical to the old in-worker derivation) ──
    out['reflectivity'] = np.nanmax(ref_3d, axis=0)

    ref_safe = np.where(np.isfinite(ref_3d), ref_3d, -np.inf)
    z_idx = np.argmax(ref_safe, axis=0)

    echo_mask = ref_3d >= 18.0
    has_echo = np.any(echo_mask, axis=0)
    echo_top_idx = nz_grid - 1 - np.argmax(echo_mask[::-1, :, :], axis=0)
    out['echo_top_height'] = np.where(has_echo, z_heights[echo_top_idx], np.nan)

    max_z_height = z_heights[z_idx]
    out['max_z_height'] = np.where(np.isfinite(out['reflectivity']), max_z_height, np.nan)

    ref_linear = 10.0 ** (ref_3d / 10.0)
    ref_linear = np.where(np.isfinite(ref_linear), ref_linear, 0.0)
    out['vil'] = 3.44e-6 * np.nansum(ref_linear ** (4.0 / 7.0) * z_res, axis=0)

    low_mask = z_heights <= WARM_RAIN_CAP_M
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if low_mask.any():
            out['low_level_ref'] = np.nanmean(ref_3d[low_mask, :, :], axis=0)
        else:
            out['low_level_ref'] = out['reflectivity'].copy()

    precip_levels = np.sum(ref_3d > 10.0, axis=0).astype(np.float32)
    out['column_depth_fraction'] = precip_levels / nz_grid

    # Dual-pol fields collocated at the height of max reflectivity (legacy).
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    for field_name in RAW_FIELDS:
        if field_name == 'reflectivity':
            continue
        out[field_name] = vol[field_name][z_idx, yy, xx]

    # ── Shared validity / beam geometry ──
    valid = np.isfinite(ref_3d)
    has_any = valid.any(axis=0)

    first_idx = np.argmax(valid, axis=0)
    out['lowest_gate_reflectivity'] = np.where(has_any, ref_3d[first_idx, yy, xx], np.nan)
    out['beam_height'] = np.where(has_any, z_heights[first_idx], np.nan)

    # Low-level (0–2 km) dual-pol means — legacy warm-rain discriminators (unchanged).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        out['low_level_kdp'] = _fixed_low_mean(vol['specific_differential_phase'], low_mask)
        out['low_level_zdr'] = _fixed_low_mean(vol['differential_reflectivity'], low_mask)
        out['low_level_rhohv'] = _fixed_low_mean(vol['cross_correlation_ratio'], low_mask)

        mid_mask = (z_heights > WARM_RAIN_CAP_M) & (z_heights <= 4000.0)
        if mid_mask.any():
            mid_ref = np.nanmean(ref_3d[mid_mask, :, :], axis=0)
        else:
            mid_ref = np.full((ny, nx), np.nan, dtype=float)
        out['vertical_reflectivity_gradient'] = out['low_level_ref'] - mid_ref

    # ── Melting layer / bright band ──
    rhohv_3d = vol['cross_correlation_ratio']
    rh_filled = np.where(valid & np.isfinite(rhohv_3d), rhohv_3d, np.inf)
    ml_idx = np.argmin(rh_filled, axis=0)
    rh_min = np.min(rh_filled, axis=0)
    out['melting_layer_height'] = np.where(has_any, z_heights[ml_idx], np.nan)
    out['rhohv_min'] = np.where(np.isfinite(rh_min) & has_any, rh_min, np.nan)

    # ── Sub-melting-layer liquid column (new; legacy fields above unchanged) ──
    liquid_mask, _bb_present = _build_liquid_layer_mask(
        z_heights, valid, out['beam_height'], out['echo_top_height'],
        out['melting_layer_height'], out['rhohv_min'], has_any, z_res,
    )
    has_liquid = liquid_mask.any(axis=0)

    zdr_3d = vol['differential_reflectivity']
    kdp_3d = vol['specific_differential_phase']

    subml_rhohv = _layer_stat(rhohv_3d, liquid_mask, 'mean')
    subml_zdr = _layer_stat(zdr_3d, liquid_mask, 'mean')
    subml_kdp = _layer_stat(kdp_3d, liquid_mask, 'mean')
    subml_ref_max = _layer_stat(ref_3d, liquid_mask, 'max')

    idx_below_ml = np.maximum(ml_idx - 1, 0)
    zdr_below_ml = zdr_3d[idx_below_ml, yy, xx]
    subml_zdr_gradient = zdr_below_ml - subml_zdr

    out['subml_rhohv'] = np.where(has_liquid, subml_rhohv, np.nan)
    out['subml_zdr'] = np.where(has_liquid, subml_zdr, np.nan)
    out['subml_kdp'] = np.where(has_liquid, subml_kdp, np.nan)
    out['subml_ref_max'] = np.where(has_liquid, subml_ref_max, np.nan)
    out['subml_zdr_gradient'] = np.where(has_liquid, subml_zdr_gradient, np.nan)

    # ── Bright-band vertical structure (legacy) ──
    bb_ref = ref_3d[ml_idx, yy, xx]
    out['bright_band_ref'] = np.where(has_any, bb_ref, np.nan)

    out['bright_band_drop'] = np.where(
        has_any, bb_ref - out['lowest_gate_reflectivity'], np.nan)

    out['maxz_meltlayer_offset'] = np.where(
        has_any, z_heights[z_idx] - z_heights[ml_idx], np.nan)

    idx_above = np.minimum(ml_idx + 1, nz_grid - 1)
    ref_above = ref_3d[idx_above, yy, xx]
    out['bright_band_intensity'] = np.where(
        has_any & np.isfinite(ref_above), bb_ref - ref_above, np.nan)

    return out


def derive_features(input_zarr, output_zarr, batch_size=200):
    print(f"Opening 3D volume zarr: {input_zarr}")
    ds3 = xr.open_zarr(input_zarr)

    missing = [f for f in RAW_FIELDS if f not in ds3]
    if missing:
        raise ValueError(f"Input zarr missing raw fields: {missing}")

    n_time = ds3.sizes['time']
    z_heights = np.asarray(ds3['z'].values, dtype=float)
    x_vals = ds3['x'].values
    y_vals = ds3['y'].values
    src_attrs = dict(ds3['reflectivity'].attrs)  # carries crs, units, etc.

    print(f"  Scans: {n_time}  |  Z levels: {len(z_heights)}  |  grid: {len(y_vals)}×{len(x_vals)}")
    print(f"  Output features ({len(OUTPUT_FIELDS)}): {OUTPUT_FIELDS}")
    print(f"  Writing to: {output_zarr}\n")

    first_write = not Path(output_zarr).exists()

    for b0 in tqdm(range(0, n_time, batch_size), desc="Deriving features"):
        b1 = min(b0 + batch_size, n_time)
        batch = ds3[RAW_FIELDS].isel(time=slice(b0, b1)).load()
        nb = b1 - b0

        acc = {f: np.empty((nb, len(y_vals), len(x_vals)), dtype=np.float32)
               for f in OUTPUT_FIELDS}

        for k in range(nb):
            vol = {f: batch[f].isel(time=k).values for f in RAW_FIELDS}
            feats = compute_features(vol, z_heights)
            for f in OUTPUT_FIELDS:
                acc[f][k] = feats[f].astype(np.float32)

        time_slice = ds3['time'].isel(time=slice(b0, b1)).values
        coords = {'time': time_slice}
        if first_write:
            coords['y'] = y_vals
            coords['x'] = x_vals
        data_vars = {
            f: xr.DataArray(acc[f], dims=('time', 'y', 'x'), attrs=src_attrs)
            for f in OUTPUT_FIELDS
        }
        out_ds = xr.Dataset(data_vars, coords=coords)

        if first_write:
            out_ds.to_zarr(output_zarr, mode='w',
                           encoding={'time': {'units': 'nanoseconds since 1970-01-01',
                                              'calendar': 'proleptic_gregorian'}})
            first_write = False
        else:
            out_ds.to_zarr(output_zarr, append_dim='time')

    print(f"\n✓ Done. Feature zarr written to: {output_zarr}")
    print(f"  Point dataset/create_pickle.py --radar at this path.")
    print(f"  To sort by time: xr.open_zarr('{output_zarr}').sortby('time')")


def main():
    parser = argparse.ArgumentParser(description='Derive 2D radar features from the 3D volume zarr')
    parser.add_argument('--input', required=True, help='Path to the 3D volume zarr')
    parser.add_argument('--output', default=None, help='Path for the derived 2D feature zarr')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Number of scans to process per batch (memory/speed trade-off)')
    args = parser.parse_args()

    output = args.output
    if output is None:
        inp = Path(args.input)
        output = str(inp.with_name(inp.name.replace('dualpol_3d', 'dualpol_feat')))
        if output == args.input:
            output = str(inp.with_name('features_' + inp.name))

    derive_features(args.input, output, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
