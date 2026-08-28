"""
run_experiments.py — Batch experiment runner for precipitation models.

Define experiments in a YAML file and run them sequentially.
Each experiment gets its own timestamped run directory with full config tracking.

Usage:
    python -m models.run_experiments experiments.yaml
    python -m models.run_experiments experiments.yaml --dry-run
    python -m models.run_experiments experiments.yaml --only dualpol_baseline vertical_all

Example YAML:
    base:
      pickle: dataset/outputs/3d/radar_gauge_dataset_vertical_9500.pkl
      dem: dem/preserve_dem_10m_utm.tif
      filter_mode: radar
      patience: 20
      epochs: 100
      batch_size: 32
      lr: 5e-5
      base_filters: 64

    experiments:
      - name: dualpol_only
        features: dualpol
        loss: mae

      - name: dualpol_vertical_mae
        features: dualpol+vertical
        loss: mae

      - name: vertical_only_huber
        features: vertical
        loss: huber

      - name: all_features_no_log
        features: all
        loss: mae
        log_target: false

      - name: reflectivity_vil_only
        features: [reflectivity, vil]
        loss: mae
        no_sampler: true
"""

import argparse
import json
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

from models.unet.dataset import FEATURE_PRESETS, FIELD_NORMS, resolve_fields, compute_n_input_channels


# Width presets for model capacity scaling
WIDTH_PRESETS = {
    'shallow':          32,
    'normal':           64,
    'wide':             96,
    'extra_wide':       128,
    'extra_extra_wide': 160,
}


def load_experiment_config(yaml_path):
    """Load and validate experiment configuration from YAML."""
    with open(yaml_path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    base = raw.get('base', {})
    experiments = raw.get('experiments', [])

    if not experiments:
        raise ValueError("No experiments defined in YAML file")

    return base, experiments


def build_cfg(base, experiment):
    """Merge base config with experiment-specific overrides."""
    from models.unet.run_unet import CONFIG as DEFAULT_CONFIG

    model_type = experiment.get('model_type', base.get('model_type', 'unet'))

    cfg = dict(DEFAULT_CONFIG)

    # Inject Stack-specific defaults if model_type is stack
    if model_type == 'stack':
        from models.stack.train import DEFAULT_CONFIG as STACK_DEFAULTS
        for k, v in STACK_DEFAULTS.items():
            cfg.setdefault(k, v)
    elif model_type == 'stack_3d':
        from models.stack_3d.train import DEFAULT_CONFIG as STACK3D_DEFAULTS
        for k, v in STACK3D_DEFAULTS.items():
            cfg.setdefault(k, v)
    elif model_type == 'unet_3d':
        from models.unet_3d.train import DEFAULT_CONFIG as UNET3D_DEFAULTS
        for k, v in UNET3D_DEFAULTS.items():
            cfg.setdefault(k, v)

    # Apply base overrides
    field_map = {
        'pickle': 'pickle_path',
        'dem': 'dem_path',
        'checkpoint_dir': 'checkpoint_dir',
        'output_dir': 'output_dir',
        'lr': 'lr',
        'weight_decay': 'weight_decay',
        'epochs': 'max_epochs',
        'batch_size': 'batch_size',
        'patience': 'patience',
        'base_filters': 'base_filters',
        'loss': 'loss_type',
        'under_weight': 'under_weight',
        'filter_mode': 'filter_mode',
        'sampler_type': 'sampler_type',
        'max_precip': 'max_precip',
    }

    for yaml_key, cfg_key in field_map.items():
        if yaml_key in base:
            cfg[cfg_key] = base[yaml_key]

    # Direct pass-through keys
    for key in ['fields', 'use_dem', 'use_mask', 'use_temporal_pos',
                'use_feature_masks', 'log_target', 'no_sampler', 'exclude_stations',
                'n_encoder_blocks', 'spatial_head', 'latent_dim', 'z_collapse', 'zarr_cache_mb',
                'skip_ablation', 'seed']:
        if key in base:
            cfg[key] = base[key]
    # 'features' is a YAML alias for 'fields'
    if 'features' in base:
        cfg['fields'] = base['features']

    # Apply experiment-specific overrides
    for yaml_key, cfg_key in field_map.items():
        if yaml_key in experiment:
            cfg[cfg_key] = experiment[yaml_key]

    for key in ['fields', 'use_dem', 'use_mask', 'use_temporal_pos',
                'use_feature_masks', 'log_target', 'no_sampler', 'exclude_stations',
                'n_encoder_blocks', 'spatial_head', 'latent_dim', 'z_collapse', 'zarr_cache_mb',
                'skip_ablation', 'seed']:
        if key in experiment:
            cfg[key] = experiment[key]
    if 'features' in experiment:
        cfg['fields'] = experiment['features']

    # Set defaults for optional boolean fields
    cfg.setdefault('use_dem', True)
    cfg.setdefault('use_mask', True)
    cfg.setdefault('use_temporal_pos', True)
    cfg.setdefault('use_feature_masks', False)
    cfg.setdefault('log_target', True)
    cfg.setdefault('no_sampler', False)
    cfg.setdefault('filter_mode', 'blunt')
    cfg.setdefault('sampler_type', 'moderate')
    cfg.setdefault('exclude_stations', [])

    # Resolve width preset -> base_filters
    width = experiment.get('width', base.get('width'))
    if width:
        if width in WIDTH_PRESETS:
            cfg['base_filters'] = WIDTH_PRESETS[width]
            cfg['width_preset'] = width
        else:
            raise ValueError(f"Unknown width preset '{width}'. Options: {list(WIDTH_PRESETS.keys())}")

    return cfg


def print_experiment_plan(base, experiments):
    """Print a summary of all planned experiments."""
    print(f"\n{'='*70}")
    print("  EXPERIMENT PLAN")
    print(f"{'='*70}")
    print(f"\n  Base config:")
    for k, v in base.items():
        print(f"    {k}: {v}")

    print(f"\n  Experiments ({len(experiments)}):")
    print(f"  {'#':<4} {'Name':<30} {'Model':<7} {'Features':<20} {'Loss':<12} {'Notes':<15}")
    print(f"  {'-'*4} {'-'*30} {'-'*7} {'-'*20} {'-'*12} {'-'*15}")

    for i, exp in enumerate(experiments, 1):
        name = exp.get('name', f'experiment_{i}')
        features = exp.get('features', base.get('features', 'dualpol+vertical'))
        loss = exp.get('loss', base.get('loss', 'mae'))
        model_type = exp.get('model_type', base.get('model_type', 'unet'))

        notes = []
        if exp.get('no_sampler') or base.get('no_sampler'):
            notes.append('no_samp')
        if exp.get('log_target') is False:
            notes.append('raw')
        elif exp.get('log_target') is True or base.get('log_target', True):
            pass
        if exp.get('use_dem') is False:
            notes.append('no_dem')
        notes_str = ', '.join(notes) if notes else 'log'

        # Compute channel count
        cfg = build_cfg(base, exp)
        if model_type in ('stack_3d', 'unet_3d'):
            from models.stack_3d.dataset import resolve_fields as resolve_fields_3d
            from models.stack_3d.dataset import compute_n_input_channels as n_ch_3d
            fields = resolve_fields_3d(cfg.get('fields'))
            n_ch = n_ch_3d(
                fields, cfg.get('use_mask', True),
                cfg.get('use_temporal_pos', True), cfg.get('use_dem', True),
                cfg.get('use_feature_masks', False),
            )
        else:
            fields = resolve_fields(cfg.get('fields'))
            n_ch = compute_n_input_channels(
                fields, cfg.get('use_mask', True),
                cfg.get('use_temporal_pos', True), cfg.get('use_dem', True),
                cfg.get('use_feature_masks', False),
            )

        print(f"  {i:<4} {name:<30} {model_type:<7} {str(features):<20} {loss:<12} {notes_str:<15} ({n_ch}ch)")

    print(f"\n{'='*70}\n")


def _run_unet_experiment(cfg, name, skip_ablation=False):
    """Run a single U-Net experiment (train + eval, optionally + ablation)."""
    from models.unet.train import train
    from models.unet.evaluate import evaluate

    best_ckpt, run_dir = train(cfg, run_name=name)

    metrics, run_dir = evaluate(
        checkpoint_path=best_ckpt,
        checkpoint_dir=cfg['checkpoint_dir'],
        pickle_path=cfg['pickle_path'],
        dem_path=cfg['dem_path'],
        output_dir=cfg.get('output_dir', 'evaluation_figures/unet_dualpol'),
        run_dir=run_dir,
    )

    if skip_ablation:
        print("  ⏭ Skipping ablation (skip_ablation enabled)")
    else:
        from models.unet.ablation import run_ablation
        run_ablation(
            checkpoint_path=best_ckpt,
            checkpoint_dir=cfg['checkpoint_dir'],
            pickle_path=cfg['pickle_path'],
            dem_path=cfg['dem_path'],
            run_dir=run_dir,
        )

    return best_ckpt, run_dir, metrics


def _run_stack_experiment(cfg, name):
    """Run a single Stack CNN experiment (train + eval)."""
    from models.stack.train import train
    from models.stack.evaluate import evaluate

    best_ckpt, run_dir = train(cfg, run_name=name)

    metrics, run_dir = evaluate(
        checkpoint_path=best_ckpt,
        checkpoint_dir=cfg['checkpoint_dir'],
        pickle_path=cfg['pickle_path'],
        dem_path=cfg['dem_path'],
        output_dir=cfg.get('output_dir', 'evaluation_figures/stack_dualpol'),
        run_dir=run_dir,
    )

    return best_ckpt, run_dir, metrics


def _run_stack_3d_experiment(cfg, name):
    """Run a single 3D Stack CNN experiment (train + eval)."""
    from models.stack_3d.train import train
    from models.stack_3d.evaluate import evaluate

    best_ckpt, run_dir = train(cfg, run_name=name)

    metrics, run_dir = evaluate(
        checkpoint_path=best_ckpt,
        checkpoint_dir=cfg['checkpoint_dir'],
        pickle_path=cfg['pickle_path'],
        dem_path=cfg['dem_path'],
        output_dir=cfg.get('output_dir', 'evaluation_figures/stack_3d_dualpol'),
        run_dir=run_dir,
    )

    return best_ckpt, run_dir, metrics


def _run_unet_3d_experiment(cfg, name):
    """Run a single 3D U-Net experiment (train + eval)."""
    from models.unet_3d.train import train
    from models.unet_3d.evaluate import evaluate

    best_ckpt, run_dir = train(cfg, run_name=name)

    metrics, run_dir = evaluate(
        checkpoint_path=best_ckpt,
        checkpoint_dir=cfg['checkpoint_dir'],
        pickle_path=cfg['pickle_path'],
        dem_path=cfg['dem_path'],
        output_dir=cfg.get('output_dir', 'evaluation_figures/unet_3d_dualpol'),
        run_dir=run_dir,
    )

    return best_ckpt, run_dir, metrics


def run_experiments(yaml_path, dry_run=False, only=None, skip_ablation=False):
    """Run all experiments defined in the YAML file."""
    base, experiments = load_experiment_config(yaml_path)
    print_experiment_plan(base, experiments)

    if dry_run:
        print("  [DRY RUN] — no experiments will be executed.")
        return

    # Filter to specific experiments if --only specified
    if only:
        only_set = set(only)
        experiments = [e for e in experiments if e.get('name') in only_set]
        if not experiments:
            print(f"  No matching experiments found for: {only}")
            return
        print(f"  Running {len(experiments)} selected experiment(s): {[e['name'] for e in experiments]}\n")

    results_summary = []
    total_start = time.time()

    for i, experiment in enumerate(experiments, 1):
        name = experiment.get('name', f'experiment_{i}')
        cfg = build_cfg(base, experiment)
        model_type = experiment.get('model_type', base.get('model_type', 'unet'))

        # Set checkpoint dir per model type if not explicitly set
        if 'checkpoint_dir' not in experiment and 'checkpoint_dir' not in base:
            if model_type == 'stack':
                cfg['checkpoint_dir'] = 'models/checkpoints/stack_dualpol/'
            elif model_type == 'stack_3d':
                cfg['checkpoint_dir'] = 'models/checkpoints/stack_3d_dualpol/'
            elif model_type == 'unet_3d':
                cfg['checkpoint_dir'] = 'models/checkpoints/unet_3d_dualpol/'
            else:
                cfg['checkpoint_dir'] = 'models/checkpoints/unet_dualpol/'

        print(f"\n{'#'*70}")
        print(f"  EXPERIMENT {i}/{len(experiments)}: {name} [{model_type}]")
        print(f"{'#'*70}")

        exp_start = time.time()

        try:
            if model_type == 'stack':
                best_ckpt, run_dir, metrics = _run_stack_experiment(cfg, name)
            elif model_type == 'stack_3d':
                best_ckpt, run_dir, metrics = _run_stack_3d_experiment(cfg, name)
            elif model_type == 'unet_3d':
                best_ckpt, run_dir, metrics = _run_unet_3d_experiment(cfg, name)
            else:
                exp_skip_ablation = skip_ablation or cfg.get('skip_ablation', False)
                best_ckpt, run_dir, metrics = _run_unet_experiment(cfg, name, skip_ablation=exp_skip_ablation)

            elapsed = time.time() - exp_start
            results_summary.append({
                'name': name,
                'model_type': model_type,
                'status': 'SUCCESS',
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'run_dir': run_dir,
                'elapsed_min': elapsed / 60,
            })

            print(f"\n  ✓ {name} [{model_type}] complete — R²={metrics['r2']:.4f}, MAE={metrics['mae']:.3f} ({elapsed/60:.1f} min)")

        except Exception as e:
            elapsed = time.time() - exp_start
            results_summary.append({
                'name': name,
                'model_type': model_type,
                'status': f'FAILED: {str(e)}',
                'r2': None,
                'mae': None,
                'rmse': None,
                'run_dir': None,
                'elapsed_min': elapsed / 60,
            })
            print(f"\n  ✗ {name} [{model_type}] FAILED after {elapsed/60:.1f} min: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n\n{'='*70}")
    print("  EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"\n  {'#':<4} {'Name':<30} {'Model':<7} {'Status':<7} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'Time':>8}")
    print(f"  {'-'*4} {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for i, r in enumerate(results_summary, 1):
        r2_str = f"{r['r2']:.4f}" if r['r2'] is not None else 'N/A'
        mae_str = f"{r['mae']:.3f}" if r['mae'] is not None else 'N/A'
        rmse_str = f"{r['rmse']:.3f}" if r['rmse'] is not None else 'N/A'
        status = 'OK' if 'SUCCESS' in r['status'] else 'FAIL'
        mtype = r.get('model_type', 'unet')
        print(f"  {i:<4} {r['name']:<30} {mtype:<7} {status:<7} {r2_str:>8} {mae_str:>8} {rmse_str:>8} {r['elapsed_min']:>6.1f}m")

    print(f"\n{'='*70}\n")

    # Save summary JSON
    summary_path = Path(base.get('checkpoint_dir', 'models/checkpoints')) / 'experiment_summary.json'
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'yaml_file': str(yaml_path),
        'total_elapsed_min': total_elapsed / 60,
        'results': results_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"  ✓ Summary saved to: {summary_path}")

    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run batch experiments from a YAML config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', type=str, help='Path to experiments YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without running')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Only run experiments with these names')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip the per-experiment feature ablation (unet only) for faster sweeps')
    args = parser.parse_args()

    run_experiments(args.config, dry_run=args.dry_run, only=args.only,
                    skip_ablation=args.skip_ablation)
