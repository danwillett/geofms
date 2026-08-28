"""
run_unet.py — Orchestration script for the U-Net precipitation model.

Run from the project root:
    python -m models.unet.run_unet --mode all --run-name unet_base
    python -m models.unet.run_unet --mode train --loss huber --features dualpol+vertical
    python -m models.unet.run_unet --mode eval --run-dir models/checkpoints/unet_dualpol/2026-05-28_...
    python -m models.unet.run_unet --mode all --features reflectivity vil echo_top_height --no-dem --no-log
"""


import argparse

from models.unet.dataset import FEATURE_PRESETS, FIELD_NORMS

CONFIG = {
    "pickle_path":     "dataset/outputs/3d/radar_gauge_dataset_vertical_9500.pkl",
    "dem_path":        "dem/preserve_dem_10m_utm.tif",
    "checkpoint_dir":  "models/checkpoints/unet_dualpol/",
    "output_dir":      "evaluation_figures/unet_dualpol",
    "lr":              5e-5,
    "weight_decay":    1e-4,
    "max_epochs":      100,
    "batch_size":      32,
    "patience":        20,
    "base_filters":    64,
    "add_bias":        False,
    "loss_type":       "mae",
    "max_precip":      100.0,
}


def run_train(cfg, run_name=None):
    from models.unet.train import train

    print("\n" + "=" * 60)
    print("  STEP 1 — TRAINING")
    print("=" * 60)

    best_ckpt, run_dir = train(cfg, run_name=run_name)
    return best_ckpt, run_dir


def run_eval(cfg, checkpoint_path=None, run_dir=None):
    from models.unet.evaluate import evaluate

    print("\n" + "=" * 60)
    print("  STEP 2 — EVALUATION")
    print("=" * 60)

    metrics, run_dir = evaluate(
        checkpoint_path=checkpoint_path,
        checkpoint_dir=cfg["checkpoint_dir"],
        pickle_path=cfg["pickle_path"],
        dem_path=cfg["dem_path"],
        output_dir=cfg["output_dir"],
        run_dir=run_dir,
    )
    return metrics, run_dir


def run_ablation_step(cfg, checkpoint_path=None, run_dir=None):
    from models.unet.ablation import run_ablation

    print("\n" + "=" * 60)
    print("  STEP 3 — ABLATION")
    print("=" * 60)

    results = run_ablation(
        checkpoint_path=checkpoint_path,
        checkpoint_dir=cfg["checkpoint_dir"],
        pickle_path=cfg["pickle_path"],
        dem_path=cfg["dem_path"],
        run_dir=run_dir,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run U-Net precipitation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Feature presets:
  {', '.join(f'{k} ({len(v)} fields)' for k, v in FEATURE_PRESETS.items())}

Individual fields:
  {', '.join(FIELD_NORMS.keys())}

Examples:
  python -m models.unet.run_unet --features dualpol+vertical --filter-mode radar
  python -m models.unet.run_unet --features reflectivity vil echo_top_height --no-dem
  python -m models.unet.run_unet --features all --loss huber --no-log
""")
    parser.add_argument("--mode", choices=["train", "eval", "ablation", "all"], default="all")
    parser.add_argument("--run-name", default=None, help="Short description suffix for the run folder")
    parser.add_argument("--run-dir", default=None, help="Existing run directory (for eval/ablation on a previous run)")

    # Data
    parser.add_argument("--pickle", default=None)
    parser.add_argument("--dem", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    # Feature selection
    parser.add_argument("--features", nargs="+", default=None,
                        help="Fields or presets to include (e.g. 'dualpol+vertical', 'reflectivity vil kdp')")
    parser.add_argument("--no-dem", action="store_true", help="Exclude DEM elevation channel")
    parser.add_argument("--no-mask", action="store_true", help="Exclude validity mask channels")
    parser.add_argument("--no-temporal-pos", action="store_true", help="Exclude temporal position channels")
    parser.add_argument("--no-log", action="store_true", help="Use raw mm targets (no log1p transform)")

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--base-filters", type=int, default=None, help="Base filter count for U-Net (default: 64)")
    parser.add_argument("--width", choices=["shallow", "normal", "wide", "extra_wide", "extra_extra_wide"], default=None,
                        help="Width preset: shallow(32), normal(64), wide(96), extra_wide(128), extra_extra_wide(160)")
    parser.add_argument("--n-encoder-blocks", type=int, default=None,
                        help="Number of encoder/decoder blocks (auto-selected from patch size if omitted)")
    parser.add_argument("--loss", choices=["mae", "mse", "huber", "weighted_mae", "weighted_mae_sq"], default=None)
    parser.add_argument("--no-sampler", action="store_true", help="Disable weighted sampler")
    parser.add_argument("--sampler-type", choices=["light", "moderate", "heavy"], default="moderate",
                        help="Sampler intensity preset (default: moderate)")
    parser.add_argument("--exclude-stations", nargs="+", default=[], help="Station names to exclude")
    parser.add_argument("--filter-mode", choices=["blunt", "radar"], default="blunt",
                        help="Filter mode: blunt (station-based caps) or radar (physics-based)")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.pickle:        cfg["pickle_path"]    = args.pickle
    if args.dem:           cfg["dem_path"]        = args.dem
    if args.checkpoint_dir: cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.output_dir:    cfg["output_dir"]      = args.output_dir
    if args.lr:            cfg["lr"]              = args.lr
    if args.epochs:        cfg["max_epochs"]      = args.epochs
    if args.batch_size:    cfg["batch_size"]      = args.batch_size
    if args.patience:      cfg["patience"]        = args.patience
    if args.base_filters:  cfg["base_filters"]    = args.base_filters
    if args.width:
        from models.run_experiments import WIDTH_PRESETS
        cfg["base_filters"] = WIDTH_PRESETS[args.width]
        cfg["width_preset"] = args.width
    if args.n_encoder_blocks: cfg["n_encoder_blocks"] = args.n_encoder_blocks
    if args.loss:          cfg["loss_type"]       = args.loss
    if args.no_sampler:    cfg["no_sampler"]      = True
    cfg["sampler_type"] = args.sampler_type
    if args.exclude_stations: cfg["exclude_stations"] = args.exclude_stations
    cfg["filter_mode"] = args.filter_mode

    # Feature configuration
    if args.features:
        # If single preset name, pass as string; otherwise pass as list
        if len(args.features) == 1 and args.features[0] in FEATURE_PRESETS:
            cfg["fields"] = args.features[0]
        else:
            cfg["fields"] = args.features
    cfg["use_dem"] = not args.no_dem
    cfg["use_mask"] = not args.no_mask
    cfg["use_temporal_pos"] = not args.no_temporal_pos
    cfg["log_target"] = not args.no_log

    print("\n" + "=" * 60)
    print("  U-Net — Precipitation Prediction")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Dataset:        {cfg['pickle_path']}")
    print(f"  Features:       {cfg.get('fields', 'dualpol+vertical (default)')}")
    print(f"  DEM:            {cfg['use_dem']}")
    print(f"  Log target:     {cfg['log_target']}")
    print(f"  Loss:           {cfg.get('loss_type', 'mae')}")
    print(f"  Filter mode:    {cfg['filter_mode']}")
    print(f"  Base filters:   {cfg['base_filters']}")
    print(f"  Checkpoint dir: {cfg['checkpoint_dir']}")
    if args.run_name:
        print(f"  Run name:       {args.run_name}")
    if args.run_dir:
        print(f"  Run dir:        {args.run_dir}")
    print("=" * 60)

    best_ckpt = args.checkpoint
    run_dir = args.run_dir

    if args.mode in ("train", "all"):
        best_ckpt, run_dir = run_train(cfg, run_name=args.run_name)

    if args.mode in ("eval", "all"):
        if best_ckpt:
            print(f"Loading {best_ckpt}")
        _, run_dir = run_eval(cfg, checkpoint_path=best_ckpt, run_dir=run_dir)

    # if args.mode in ("ablation", "all"):
    #     run_ablation_step(cfg, checkpoint_path=best_ckpt, run_dir=run_dir)

    print(f"\n✓ Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
