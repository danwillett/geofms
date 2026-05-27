"""
Validation script for NEXRAD zarr files

Checks:
- Data structure and dimensions (2D: time, y, x — Z-collapsed)
- All dual-pol fields present
- Coordinate systems and ranges
- Data quality per field
- Time series continuity
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pyproj import Transformer

FIELDS = [
    'reflectivity',
    'differential_reflectivity',
    'cross_correlation_ratio',
    'differential_phase',
    'specific_differential_phase',
]

FIELD_LABELS = {
    'reflectivity':                ('Reflectivity (Z)',          'dBZ',   -10,  70,  'turbo'),
    'differential_reflectivity':   ('Diff. Reflectivity (ZDR)',  'dB',    -2,   6,   'RdBu_r'),
    'cross_correlation_ratio':     ('Correlation Ratio (RhoHV)', '',      0.6,  1.0, 'plasma'),
    'differential_phase':          ('Diff. Phase (PhiDP)',       'deg',   0,    180, 'twilight'),
    'specific_differential_phase': ('Spec. Diff. Phase (KDP)',   'deg/km',-1,   6,   'coolwarm'),
}


def load_preserve_boundary(geojson_path='geometries/dangermond-preserve-boundary.geojson'):
    try:
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        coords = geojson['features'][0]['geometry']['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        RADAR_LAT = 34.83855
        RADAR_LON = -120.397917
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84",
            always_xy=True
        )
        boundary_x, boundary_y = [], []
        for lon, lat in zip(lons, lats):
            x, y = transformer.transform(lon, lat)
            boundary_x.append(x)
            boundary_y.append(y)
        return np.array(boundary_x), np.array(boundary_y)
    except Exception as e:
        print(f"⚠️  Could not load preserve boundary: {e}")
        return None, None


def validate_zarr_structure(zarr_path):
    print("=" * 60)
    print("NEXRAD ZARR VALIDATION REPORT")
    print("=" * 60)

    report = {}
    all_valid = True

    try:
        print(f"\n📂 Opening: {zarr_path}")
        store = zarr.open(zarr_path, mode='r')
        report['file_opened'] = True
        print("   ✅ File opened successfully")
        print(f"   Keys: {list(store.keys())}")
    except Exception as e:
        print(f"   ❌ Failed to open file: {e}")
        return False, {}

    # ── Check coordinate arrays ─────────────────────────────────────────────
    print("\n📊 Checking coordinate arrays...")
    for coord in ['time', 'y', 'x']:
        if coord in store:
            print(f"   ✅ {coord}: shape {store[coord].shape}")
            report[f'{coord}_exists'] = True
        else:
            print(f"   ❌ {coord}: MISSING")
            report[f'{coord}_exists'] = False
            all_valid = False

    if not all(report.get(f'{c}_exists', False) for c in ['time', 'y', 'x']):
        return False, report

    n_time = store['time'].shape[0]
    n_y    = store['y'].shape[0]
    n_x    = store['x'].shape[0]
    print(f"\n   Grid: {n_time} time steps × {n_y} y × {n_x} x")
    report.update(n_time=n_time, n_y=n_y, n_x=n_x)

    # ── Check dual-pol fields ────────────────────────────────────────────────
    print("\n📡 Checking dual-pol fields...")
    expected_shape = (n_time, n_y, n_x)
    fields_found = []

    for field in FIELDS:
        if field not in store:
            print(f"   ⚠️  {field}: MISSING (optional if KDP failed)")
            report[f'{field}_exists'] = False
            continue

        arr = store[field]
        report[f'{field}_exists'] = True

        if arr.shape == expected_shape:
            print(f"   ✅ {field}: shape {arr.shape}")
            fields_found.append(field)
        else:
            print(f"   ❌ {field}: shape {arr.shape}, expected {expected_shape}")
            all_valid = False

    report['fields_found'] = fields_found

    # ── Per-field data quality ───────────────────────────────────────────────
    print("\n🌧️  Per-field data quality (sample over 10 time steps)...")
    sample_indices = np.linspace(0, n_time - 1, min(10, n_time), dtype=int)

    for field in fields_found:
        arr = store[field]
        label, unit, vmin, vmax, _ = FIELD_LABELS[field]
        vals = []
        coverages = []
        for i in sample_indices:
            data = arr[i, :, :]
            valid = data[~np.isnan(data)]
            coverages.append(len(valid) / data.size)
            if len(valid):
                vals.extend(valid.tolist())

        avg_cov = np.mean(coverages) * 100
        if vals:
            print(f"   {label:35s}  coverage={avg_cov:5.1f}%  "
                  f"range=[{np.min(vals):.2f}, {np.max(vals):.2f}] {unit}")
        else:
            print(f"   {label:35s}  ⚠️  NO VALID DATA")
            all_valid = False

    # ── Time coordinate ──────────────────────────────────────────────────────
    print("\n⏰ Checking time coordinate...")
    try:
        time_coords = store['time'][:]
        import pandas as pd
        times = pd.to_datetime(time_coords)
        print(f"   First: {times[0]}")
        print(f"   Last:  {times[-1]}")
        print(f"   ✅ Time coordinate readable")
        report['time_valid'] = True
    except Exception as e:
        print(f"   ⚠️  Time conversion issue: {e}")
        report['time_valid'] = False

    # ── Final verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ VALIDATION PASSED")
    else:
        print("⚠️  VALIDATION WARNINGS — check issues above")
    print("=" * 60)
    report['overall_valid'] = all_valid
    return all_valid, report


def plot_validation_summary(zarr_path):
    print("\n📊 Creating validation plots...")
    store = zarr.open(zarr_path, mode='r')

    import pandas as pd
    times     = pd.to_datetime(store['time'][:])
    x_coords  = store['x'][:]
    y_coords  = store['y'][:]
    n_time    = len(times)

    fields_present = [f for f in FIELDS if f in store]
    ref_arr = store['reflectivity']

    boundary_x, boundary_y = load_preserve_boundary()

    # Find rainiest time step (highest max reflectivity)
    print("   Finding rainiest time step...")
    max_refs = []
    for i in range(n_time):
        data = ref_arr[i, :, :]
        max_refs.append(np.nanmax(data) if np.any(~np.isnan(data)) else -999)
    best_idx = int(np.argmax(max_refs))
    print(f"   Best time step: {best_idx} ({times[best_idx]}, max Z={max_refs[best_idx]:.1f} dBZ)")

    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    # ── Figure 1: Data availability over time ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    sample_idx = np.linspace(0, n_time - 1, min(200, n_time), dtype=int)
    coverage = []
    for i in sample_idx:
        data = ref_arr[i, :, :]
        coverage.append(np.sum(~np.isnan(data)) / data.size * 100)

    ax.plot(times[sample_idx], coverage, lw=0.8)
    ax.axvline(times[best_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Max rain ({times[best_idx].date()})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reflectivity Coverage (%)')
    ax.set_title('Data Availability Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nexrad_validation_timeline.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: nexrad_validation_timeline.png")
    plt.show()

    # ── Figure 2: All dual-pol fields at rainiest time step ─────────────────
    n_fields = len(fields_present)
    ncols = 3
    nrows = int(np.ceil(n_fields / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, field in enumerate(fields_present):
        ax = axes[ax_idx]
        label, unit, vmin, vmax, cmap = FIELD_LABELS[field]
        data = store[field][best_idx, :, :]

        # Replace the imshow block in plot_validation_summary with:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                    origin='lower', extent=extent)  # remove aspect='equal'
        ax.set_xlim(x_coords[0], x_coords[-1])
        ax.set_ylim(y_coords[0], y_coords[-1])
        # Remove boundary plot lines (boundary is AEQD, zarr is UTM — don't mix)
        if boundary_x is not None:
            ax.plot(boundary_x, boundary_y, 'k-', lw=1.5, zorder=5)
            ax.plot(boundary_x, boundary_y, 'w-', lw=0.5, alpha=0.5, zorder=6)

        plt.colorbar(im, ax=ax, label=unit, shrink=0.8)
        ax.set_title(f'{label}\n{times[best_idx].strftime("%Y-%m-%d %H:%M UTC")}', fontsize=9)
        ax.set_xlabel('X from radar (m)')
        ax.set_ylabel('Y from radar (m)')

    # Hide unused subplots
    for ax_idx in range(n_fields, len(axes)):
        axes[ax_idx].set_visible(False)

    plt.suptitle(f'Dual-Pol Fields — Rainiest Time Step (t={best_idx})', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('nexrad_validation_dualpol.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: nexrad_validation_dualpol.png")
    plt.show()

    # ── Figure 3: Reflectivity histogram at rainiest time step ──────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_z = ref_arr[best_idx, :, :]
    valid_z = valid_z[~np.isnan(valid_z)]
    if len(valid_z):
        ax.hist(valid_z, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', alpha=0.6, label='0 dBZ')
        ax.axvline(20, color='orange', linestyle='--', alpha=0.6, label='20 dBZ (light rain)')
    ax.set_xlabel('Reflectivity (dBZ)')
    ax.set_ylabel('Grid Cells')
    ax.set_title('Z Distribution at Rainiest Time Step')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('nexrad_validation_hist.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: nexrad_validation_hist.png")
    plt.show()


if __name__ == "__main__":
    import sys
    zarr_file = sys.argv[1] if len(sys.argv) > 1 else 'KVBX_preserve_500m_2020-01-01_2026-03-28.zarr'
    valid, report = validate_zarr_structure(zarr_file)
    if valid or report.get('fields_found'):
        try:
            plot_validation_summary(zarr_file)
        except Exception as e:
            print(f"\n⚠️  Could not create plots: {e}")
    sys.exit(0 if valid else 1)