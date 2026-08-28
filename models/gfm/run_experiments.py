"""
run_experiments.py — Batch experiment runner for the TerraMind/GFM model.

Mirrors models/run_experiments.py (used for unet/stack) but tailored to the GFM,
which trains via PyTorch Lightning + TerraTorch. Each experiment gets its own
timestamped run directory with a config.json, checkpoints, and evaluation figures.

Usage:
    python -m models.gfm.run_experiments models/gfm/experiments.yaml
    python -m models.gfm.run_experiments models/gfm/experiments.yaml --dry-run
    python -m models.gfm.run_experiments models/gfm/experiments.yaml --only loss_huber ctx19

Example YAML:
    base:
      pickle: dataset/outputs/3d/radar_gauge_dataset_subml_daygroup_offsets_9500.pkl
      dem: dem/preserve_dem_10m_utm.tif
      epochs: 100
      batch_size: 8
      patience: 20
      lr: 1.0e-5
      output_size: 9
      log_target: true

    experiments:
      - name: loss_mse
        loss: mse
      - name: loss_huber
        loss: huber
      - name: raw_space
        log_target: false
"""

import argparse
import json
import time
import yaml
from datetime import datetime
from pathlib import Path

from models.gfm.train import DEFAULT_CONFIG
from models.gfm.dataset import DEFAULT_FIELDS, compute_radar_channels
from models.unet.dataset import resolve_fields


# YAML key -> cfg key remapping (keys that differ in name)
FIELD_MAP = {
    'pickle':         'pickle_path',
    'dem':            'dem_path',
    'checkpoint_dir': 'checkpoint_dir',
    'output_dir':     'output_dir',
    'lr':             'lr',
    'radar_lr':       'radar_lr',
    'weight_decay':   'weight_decay',
    'epochs':         'max_epochs',
    'batch_size':     'batch_size',
    'patience':       'patience',
    'loss':           'loss_type',
    'filter_mode':    'filter_mode',
}

# Keys passed straight through with the same name
PASSTHROUGH = [
    'output_size', 'output_bias', 'log_target', 'no_sampler',
    'use_feature_masks', 'seed', 'wandb_key', 'under_weight', 'use_wandb',
    'save_top_k', 'backbone_pretrained', 'dem_init', 'dem_norm',
]


def load_experiment_config(yaml_path):
    """Load and validate experiment configuration from YAML."""
    with open(yaml_path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    base = raw.get('base', {})
    experiments = raw.get('experiments', [])
    if not experiments:
        raise ValueError("No experiments defined in YAML file")
    return base, experiments


def expand_seeds(base, experiments):
    """Expand each experiment into one variant per seed for spread estimation.

    A seed list can be given at base level (``seeds: [42, 43, 44]``) or per
    experiment (overrides base). Each variant gets ``seed`` set and its name
    suffixed ``_s<seed>``. If no seed list is present, experiments pass through
    unchanged (single run using whatever ``seed`` is already set).
    """
    base_seeds = base.get('seeds')
    expanded = []
    for exp in experiments:
        seeds = exp.get('seeds', base_seeds)
        if not seeds:
            expanded.append(exp)
            continue
        for s in seeds:
            variant = {k: v for k, v in exp.items() if k != 'seeds'}
            variant['seed'] = s
            variant['name'] = f"{exp.get('name', 'experiment')}_s{s}"
            expanded.append(variant)
    return expanded


def build_cfg(base, experiment):
    """Merge GFM DEFAULT_CONFIG with base config and experiment-specific overrides."""
    cfg = dict(DEFAULT_CONFIG)

    def apply(src):
        for yaml_key, cfg_key in FIELD_MAP.items():
            if yaml_key in src:
                cfg[cfg_key] = src[yaml_key]
        for key in PASSTHROUGH:
            if key in src:
                cfg[key] = src[key]
        # 'features' is a YAML alias for 'fields'
        if 'features' in src:
            cfg['fields'] = src['features']
        if 'fields' in src:
            cfg['fields'] = src['fields']

    apply(base)
    apply(experiment)
    return cfg


def print_experiment_plan(base, experiments):
    """Print a summary of all planned experiments."""
    print(f"\n{'='*78}")
    print("  GFM EXPERIMENT PLAN")
    print(f"{'='*78}")
    print(f"\n  Base config:")
    for k, v in base.items():
        print(f"    {k}: {v}")

    print(f"\n  Experiments ({len(experiments)}):")
    print(f"  {'#':<4} {'Name':<28} {'Seed':<5} {'Pre':<5} {'DEMinit':<9} {'DEMnorm':<9} "
          f"{'Loss':<7} {'OutSz':<6}")
    print(f"  {'-'*4} {'-'*28} {'-'*5} {'-'*5} {'-'*9} {'-'*9} {'-'*7} {'-'*6}")

    for i, exp in enumerate(experiments, 1):
        name = exp.get('name', f'experiment_{i}')
        cfg  = build_cfg(base, exp)
        seed = cfg.get('seed')
        pre  = 'yes' if cfg.get('backbone_pretrained', True) else 'no'
        dem_init = cfg.get('dem_init', 'pretrained') if cfg.get('backbone_pretrained', True) else 'scratch'
        print(f"  {i:<4} {name:<28} {str(seed):<5} {pre:<5} {dem_init:<9} "
              f"{cfg.get('dem_norm', 'minmax'):<9} {cfg['loss_type']:<7} {cfg['output_size']:<6}")

    print(f"\n{'='*78}\n")


def _run_gfm_experiment(cfg, name):
    """Run a single GFM experiment (train + eval).

    All experiments share one long-lived process, so we explicitly release the
    model/optimizer/dataset and the CUDA caching-allocator pool afterwards.
    Without this, GPU memory fragments and resident RAM (the ~10 GB in-memory
    sample list) lingers across runs, which gradually slows the whole sweep.
    """
    from models.gfm.train import train
    from models.gfm.evaluate import evaluate

    best_ckpt, run_dir = train(cfg, run_name=name)

    metrics, test_summary, run_dir = evaluate(
        checkpoint_path=best_ckpt,
        checkpoint_dir=cfg['checkpoint_dir'],
        pickle_path=cfg['pickle_path'],
        output_dir=cfg.get('output_dir', 'evaluation_figures/terramind_dualpol'),
        output_size=cfg['output_size'],
        log_target=cfg.get('log_target', True),
        fields=cfg.get('fields'),
        use_feature_masks=cfg.get('use_feature_masks', True),
        filter_mode=cfg.get('filter_mode', 'blunt'),
        dem_norm=cfg.get('dem_norm', 'minmax'),
        run_dir=run_dir,
    )
    return best_ckpt, run_dir, metrics, test_summary


def _cleanup_between_experiments():
    """Reclaim CPU/GPU memory so it doesn't accumulate across the sweep."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_experiments(yaml_path, dry_run=False, only=None):
    """Run all GFM experiments defined in the YAML file."""
    base, experiments = load_experiment_config(yaml_path)
    experiments = expand_seeds(base, experiments)
    print_experiment_plan(base, experiments)

    if dry_run:
        print("  [DRY RUN] — no experiments will be executed.")
        return

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

        # Default checkpoint dir if not set anywhere
        if 'checkpoint_dir' not in experiment and 'checkpoint_dir' not in base:
            cfg['checkpoint_dir'] = 'models/checkpoints/terramind_dualpol/'

        print(f"\n{'#'*78}")
        print(f"  GFM EXPERIMENT {i}/{len(experiments)}: {name}")
        print(f"{'#'*78}")

        exp_start = time.time()
        try:
            best_ckpt, run_dir, metrics, test_summary = _run_gfm_experiment(cfg, name)
            elapsed = time.time() - exp_start
            test_summary = test_summary or {}
            results_summary.append({
                'name': name,
                'status': 'SUCCESS',
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'test_r2_naive':  test_summary.get('r2_naive'),
                'test_r2_jensen': test_summary.get('r2_jensen'),
                'test_mae_naive': test_summary.get('mae_naive'),
                'run_dir': run_dir,
                'elapsed_min': elapsed / 60,
            })
            test_str = (f", test_R²={test_summary['r2_jensen']:.4f} (J)"
                        if test_summary.get('r2_jensen') is not None
                        else (f", test_R²={test_summary['r2_naive']:.4f}"
                              if test_summary.get('r2_naive') is not None else ""))
            print(f"\n  ✓ {name} complete — val_R²={metrics['r2']:.4f}{test_str}, "
                  f"MAE={metrics['mae']:.3f} ({elapsed/60:.1f} min)")
        except Exception as e:
            elapsed = time.time() - exp_start
            results_summary.append({
                'name': name,
                'status': f'FAILED: {str(e)}',
                'r2': None, 'mae': None, 'rmse': None,
                'test_r2_naive': None, 'test_r2_jensen': None, 'test_mae_naive': None,
                'run_dir': None,
                'elapsed_min': elapsed / 60,
            })
            print(f"\n  ✗ {name} FAILED after {elapsed/60:.1f} min: {e}")
            import traceback
            traceback.print_exc()
        finally:
            _cleanup_between_experiments()

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n\n{'='*78}")
    print("  GFM EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*78}")
    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"\n  {'#':<4} {'Name':<28} {'Status':<7} {'valR²':>8} {'testR²':>8} {'tR²(J)':>8} {'MAE':>8} {'Time':>8}")
    print(f"  {'-'*4} {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i, r in enumerate(results_summary, 1):
        def fmt(v, p=4):
            return f"{v:.{p}f}" if v is not None else 'N/A'
        status = 'OK' if 'SUCCESS' in r['status'] else 'FAIL'
        print(f"  {i:<4} {r['name']:<28} {status:<7} {fmt(r['r2']):>8} "
              f"{fmt(r.get('test_r2_naive')):>8} {fmt(r.get('test_r2_jensen')):>8} "
              f"{fmt(r['mae'], 3):>8} {r['elapsed_min']:>6.1f}m")
    print(f"\n{'='*78}\n")

    # Timestamped filename so re-running a sweep never clobbers earlier results.
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    summary_dir = Path(base.get('checkpoint_dir', 'models/checkpoints/terramind_dualpol'))
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f'experiment_summary_{stamp}.json'
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'yaml_file': str(yaml_path),
        'total_elapsed_min': total_elapsed / 60,
        'results': results_summary,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"  ✓ Summary saved to: {summary_path}")

    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run batch GFM experiments from a YAML config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', type=str, help='Path to experiments YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without running')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Only run experiments with these names')
    args = parser.parse_args()

    run_experiments(args.config, dry_run=args.dry_run, only=args.only)
