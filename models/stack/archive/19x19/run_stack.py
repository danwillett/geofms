"""
run_stack.py — Orchestration script for the Stack CNN precipitation model.

Run from the project root:
    python -m models.stack.run_stack --mode all --run-name mae_3block
    python -m models.stack.run_stack --mode train --run-name mse_wider
    python -m models.stack.run_stack --mode eval --run-dir models/checkpoints/stack_dualpol/2026-05-21_12-04_mae_3block
"""


import argparse

CONFIG = {
    "pickle_path":     "dataset/outputs/radar_gauge_dataset_with_offsets_9500.pkl",
    "dem_path":        "dem/preserve_dem_10m_utm.tif",
    "checkpoint_dir":  "models/checkpoints/stack_dualpol/linear/",
    "output_dir":      "evaluation_figures/stack_dualpol",
    "lr":              5e-5,
    "weight_decay":    1e-4,
    "max_epochs":      100,
    "batch_size":      32,
    "patience":        20,
    "latent_dim":      512,
    "add_bias":        False,
    "loss_type":       "mae",
    "max_precip":      100.0,
}


def run_train(cfg, run_name=None):
    from models.stack.train import train

    print("\n" + "=" * 60)
    print("  STEP 1 — TRAINING")
    print("=" * 60)

    best_ckpt, run_dir = train(cfg, run_name=run_name)
    return best_ckpt, run_dir


def run_eval(cfg, checkpoint_path=None, run_dir=None):
    from models.stack.evaluate import evaluate

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
    from models.stack.ablation import run_ablation

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
    parser = argparse.ArgumentParser(description="Run Stack CNN precipitation model")
    parser.add_argument("--mode", choices=["train", "eval", "ablation", "all"], default="all")
    parser.add_argument("--run-name", default=None, help="Short description suffix for the run folder (e.g. 'mae_3block')")
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
    parser.add_argument("--loss", choices=["mae", "mse", "weighted_mae", "weighted_mae_sq"], default=None)
    parser.add_argument("--add-bias", action="store_true")
    parser.add_argument("--no-sampler", action="store_true", help="Disable weighted sampler")
    parser.add_argument("--exclude-stations", nargs="+", default=[], help="Station names to exclude")
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
    if args.loss:          cfg["loss_type"]       = args.loss
    if args.add_bias:      cfg["add_bias"]        = True
    if args.no_sampler:    cfg["no_sampler"]      = True
    if args.exclude_stations: cfg["exclude_stations"] = args.exclude_stations

    print("\n" + "=" * 60)
    print("  Stack CNN — Precipitation Prediction")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Dataset:        {cfg['pickle_path']}")
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
        _, run_dir = run_eval(cfg, checkpoint_path=best_ckpt, run_dir=run_dir)

    if args.mode in ("ablation", "all"):
        run_ablation_step(cfg, checkpoint_path=best_ckpt, run_dir=run_dir)

    print(f"\n✓ Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
