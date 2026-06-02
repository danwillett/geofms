"""
run_unet.py — Orchestration script for the U-Net precipitation model.

Run from the project root:
    python -m models.unet.run_unet --mode all --run-name unet_base
    python -m models.unet.run_unet --mode train --loss weighted_mae --run-name wmae_unet
    python -m models.unet.run_unet --mode eval --run-dir models/checkpoints/unet_dualpol/2026-05-22_...
"""


import argparse

CONFIG = {
    "pickle_path":     "dataset/outputs/radar_gauge_dataset_with_offsets_9500.pkl",
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
    parser = argparse.ArgumentParser(description="Run U-Net precipitation model")
    parser.add_argument("--mode", choices=["train", "eval", "ablation", "all"], default="all")
    parser.add_argument("--run-name", default=None, help="Short description suffix for the run folder")
    parser.add_argument("--run-dir", default=None, help="Existing run directory (for eval/ablation on a previous run)")
    parser.add_argument("--pickle", default=None)
    parser.add_argument("--dem", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--base-filters", type=int, default=None, help="Base filter count for U-Net (default: 64)")
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
    if args.loss:          cfg["loss_type"]       = args.loss
    if args.no_sampler:    cfg["no_sampler"]      = True
    cfg["sampler_type"] = args.sampler_type
    if args.exclude_stations: cfg["exclude_stations"] = args.exclude_stations
    cfg["filter_mode"] = args.filter_mode

    print("\n" + "=" * 60)
    print("  U-Net — Precipitation Prediction")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Dataset:        {cfg['pickle_path']}")
    print(f"  Base filters:   {cfg['base_filters']}")
    print(f"  Checkpoint dir: {cfg['checkpoint_dir']}")
    print(f"  Output dir:     {cfg['output_dir']}")
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
        print(f"Loading {best_ckpt}")
        _, run_dir = run_eval(cfg, checkpoint_path=best_ckpt, run_dir=run_dir)

    if args.mode in ("ablation", "all"):
        run_ablation_step(cfg, checkpoint_path=best_ckpt, run_dir=run_dir)

    print(f"\n✓ Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
