"""
run_stack_10min.py — Orchestration script for the 10-minute single-scan CNN model.

Run from the project root:
    python -m models.stack_10min.run_stack --mode all --run-name baseline
    python -m models.stack_10min.run_stack --mode train --run-name huber_no_sampler --no-sampler
    python -m models.stack_10min.run_stack --mode eval --run-dir models/checkpoints/stack_10min/2026-05-26_...
"""

import argparse

CONFIG = {
    "pickle_path":     "dataset/outputs/10min/radar_gauge_10min.pkl",
    "dem_path":        "dem/preserve_dem_10m_utm.tif",
    "checkpoint_dir":  "models/checkpoints/stack_10min",
    "output_dir":      "evaluation_figures/stack_10min",
    "lr":              1e-4,
    "weight_decay":    1e-4,
    "max_epochs":      100,
    "batch_size":      64,
    "patience":        15,
    "latent_dim":      256,
    "loss_type":       "huber",
    "max_precip":      50.0,
}


def run_train(cfg, run_name=None):
    from models.stack_10min.train import train

    print("\n" + "=" * 60)
    print("  STEP 1 — TRAINING (10-min)")
    print("=" * 60)

    best_ckpt, run_dir = train(cfg, run_name=run_name)
    return best_ckpt, run_dir


def run_eval(cfg, checkpoint_path=None, run_dir=None):
    from models.stack_10min.evaluate import evaluate

    print("\n" + "=" * 60)
    print("  STEP 2 — EVALUATION (10-min)")
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


def main():
    parser = argparse.ArgumentParser(description="Run 10-min single-scan precipitation model")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--pickle", default=None)
    parser.add_argument("--dem", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--loss", choices=["huber", "mae", "mse", "weighted_mae"], default=None)
    parser.add_argument("--no-sampler", action="store_true")
    parser.add_argument("--sampler-type", choices=["light", "moderate", "heavy"], default="moderate")
    parser.add_argument("--exclude-stations", nargs="+", default=[])
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.pickle:         cfg["pickle_path"]    = args.pickle
    if args.dem:            cfg["dem_path"]        = args.dem
    if args.checkpoint_dir: cfg["checkpoint_dir"]  = args.checkpoint_dir
    if args.output_dir:     cfg["output_dir"]      = args.output_dir
    if args.lr:             cfg["lr"]              = args.lr
    if args.epochs:         cfg["max_epochs"]      = args.epochs
    if args.batch_size:     cfg["batch_size"]      = args.batch_size
    if args.patience:       cfg["patience"]        = args.patience
    if args.loss:           cfg["loss_type"]       = args.loss
    if args.no_sampler:     cfg["no_sampler"]      = True
    cfg["sampler_type"] = args.sampler_type
    if args.exclude_stations: cfg["exclude_stations"] = args.exclude_stations

    print("\n" + "=" * 60)
    print("  10-min Single-Scan CNN — Precipitation Prediction")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Dataset:        {cfg['pickle_path']}")
    print(f"  Checkpoint dir: {cfg['checkpoint_dir']}")
    print(f"  Loss:           {cfg['loss_type']}")
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
        _, run_dir = run_eval(cfg, checkpoint_path=best_ckpt, run_dir=run_dir)

    print(f"\n✓ Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
