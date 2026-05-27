"""
run_gfm.py — Orchestration script for the TerraMind precipitation model.

Run from the project root:

    # Train only
    python -m models.gfm.run_gfm --mode train

    # Evaluate a specific checkpoint
    python -m models.gfm.run_gfm --mode eval --checkpoint checkpoints/terramind_dualpol/best-epoch=32-val_loss=0.0000.ckpt

    # Train then immediately evaluate the best checkpoint
    python -m models.gfm.run_gfm --mode all

All paths are relative to the project root.
"""

import argparse
from pathlib import Path


# ── CONFIG ──────────────────────────────────────────────────────────────────────
# Edit these defaults to match your environment; CLI args override them.
CONFIG = {
    # Dataset pickle produced by deep_learning/prepare_radar_gauge_data.py
    "pickle_path":     "dataset/outputs/radar_gauge_dataset_tr22_24_26_vl_23_25.pkl",

    # Where to save / find checkpoints
    "checkpoint_dir":  "models/checkpoints/terramind_dualpol/tr22_24_26_vl_23_25",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    # Where to save evaluation figures
    "output_dir":      "models/evaluation_figures/terramind_dualpol/tr22_24_26_vl_23_25",

    # Training hyperparameters
    "lr":              1e-5,
    "max_epochs":      100,
    "batch_size":      8,
    "patience":        10,

    # W&B — leave None to use the WANDB_API_KEY environment variable
    "wandb_key":       None,
}
# ────────────────────────────────────────────────────────────────────────────────


def run_train(cfg: dict) -> str | None:
    """Train the model and return the path to the best checkpoint."""
    from models.gfm.train import train

    print("\n" + "=" * 60)
    print("  STEP 1 — TRAINING")
    print("=" * 60)

    best_ckpt = train(
        pickle_path    = cfg["pickle_path"],
        checkpoint_dir = cfg["checkpoint_dir"],
        lr             = cfg["lr"],
        max_epochs     = cfg["max_epochs"],
        batch_size     = cfg["batch_size"],
        patience       = cfg["patience"],
        wandb_key      = cfg["wandb_key"],
    )
    return best_ckpt


def run_eval(cfg: dict, checkpoint_path: str = None) -> dict:
    """Evaluate the model and return metrics dict."""
    from models.gfm.evaluate import evaluate

    print("\n" + "=" * 60)
    print("  STEP 2 — EVALUATION")
    print("=" * 60)

    metrics = evaluate(
        checkpoint_path = checkpoint_path,
        checkpoint_dir  = cfg["checkpoint_dir"],
        pickle_path     = cfg["pickle_path"],
        output_dir      = cfg["output_dir"],
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run TerraMind GFM precipitation model (train / eval / all)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "all"],
        default="all",
        help="Which step(s) to run (default: all)",
    )
    parser.add_argument("--pickle",       default=None, help="Override pickle path")
    parser.add_argument("--checkpoint",   default=None, help="Specific .ckpt to evaluate")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--output-dir",   default=None, help="Override evaluation output directory")
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--patience",     type=int,   default=None)
    parser.add_argument("--wandb-key",    default=None)

    args = parser.parse_args()

    # Merge CLI overrides into config
    cfg = dict(CONFIG)
    if args.pickle:        cfg["pickle_path"]    = args.pickle
    if args.checkpoint_dir: cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.output_dir:    cfg["output_dir"]     = args.output_dir
    if args.lr:            cfg["lr"]             = args.lr
    if args.epochs:        cfg["max_epochs"]     = args.epochs
    if args.batch_size:    cfg["batch_size"]     = args.batch_size
    if args.patience:      cfg["patience"]       = args.patience
    if args.wandb_key:     cfg["wandb_key"]      = args.wandb_key

    print("\n" + "=" * 60)
    print("  TerraMind GFM — Precipitation Prediction")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Dataset:        {cfg['pickle_path']}")
    print(f"  Checkpoint dir: {cfg['checkpoint_dir']}")
    print(f"  Output dir:     {cfg['output_dir']}")
    print("=" * 60)

    best_ckpt = args.checkpoint  # may be None

    if args.mode in ("train", "all"):
        best_ckpt = run_train(cfg)

    if args.mode in ("eval", "all"):
        run_eval(cfg, checkpoint_path=best_ckpt)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()