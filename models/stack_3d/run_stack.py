"""
run_stack.py — Orchestration script for the 3D Stack CNN precipitation model.

Run from the project root:
    python -m models.stack_3d.run_stack --mode all --run-name stack3d_mae
    python -m models.stack_3d.run_stack --mode train --run-name stack3d_mae
"""

import argparse

from models.stack_3d.train import DEFAULT_CONFIG as CONFIG


def run_train(cfg, run_name=None):
    from models.stack_3d.train import train

    print("\n" + "=" * 60)
    print("  STEP 1 — TRAINING")
    print("=" * 60)

    best_ckpt, run_dir = train(cfg, run_name=run_name)
    return best_ckpt, run_dir


def run_eval(cfg, checkpoint_path=None, run_dir=None):
    from models.stack_3d.evaluate import evaluate

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
    print("\n  Ablation not yet implemented for stack_3d — skipping.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Run 3D Stack CNN precipitation model")
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
    parser.add_argument("--sampler-type", choices=["light", "moderate", "heavy"], default="moderate",
                        help="Sampler intensity preset (default: moderate)")
    parser.add_argument("--scalar-output", action="store_true", help="Predict single scalar instead of spatial map")
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
    cfg["sampler_type"] = args.sampler_type
    if args.scalar_output: cfg["scalar_output"]   = True
    if args.exclude_stations: cfg["exclude_stations"] = args.exclude_stations

    print("\n" + "=" * 60)
    print("  Stack 3D CNN — Precipitation Prediction")
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
