"""
train.py — Train the TerraMind precipitation model.

Run from the project root:
    python -m models.gfm.train

Or with custom args:
    python -m models.gfm.train --pickle path/to/dataset.pkl --epochs 50 --batch-size 16
"""

import os
import argparse
import torch
import wandb
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)

from models.gfm.model import build_task
from models.gfm.dataset import RadarDEMDataModule, create_heavy_rain_sampler

from dotenv import load_dotenv
load_dotenv()

# ── DEFAULT CONFIG ──────────────────────────────────────────────────────────────
DEFAULT_PICKLE   = "deep_learning/radar_gauge_dataset_9x9.pkl"
DEFAULT_CKPT_DIR = "checkpoints/terramind_dualpol"
DEFAULT_LR       = 1e-5
DEFAULT_EPOCHS   = 100
DEFAULT_BATCH    = 8
DEFAULT_PATIENCE = 20
# ───────────────────────────────────────────────────────────────────────────────


def train(
    pickle_path: str  = DEFAULT_PICKLE,
    checkpoint_dir: str = DEFAULT_CKPT_DIR,
    lr: float           = DEFAULT_LR,
    max_epochs: int     = DEFAULT_EPOCHS,
    batch_size: int     = DEFAULT_BATCH,
    patience: int       = DEFAULT_PATIENCE,
    wandb_key: str      = None,
):
    # ── PATHS ───────────────────────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── PRECISION ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        precision = "16-mixed"
    elif torch.backends.mps.is_available():
        precision = "32"
    else:
        precision = "32"

    print(f"Using precision: {precision}")
    print(f"Checkpoint dir: {ckpt_dir.resolve()}")

    # ── DATA ────────────────────────────────────────────────────────────────────
    datamodule = RadarDEMDataModule(
        pickle_path=pickle_path,
        weight_sampler=create_heavy_rain_sampler,
        batch_size=batch_size,
    )

    # ── MODEL / TASK ────────────────────────────────────────────────────────────
    task = build_task(lr=lr, output_bias=3.0)

    # ── WANDB ────────────────────────────────────────────────────────────────────
    if wandb_key:
        wandb.login(key=wandb_key)
    elif os.environ.get("WANDB_API_KEY"):
        wandb.login()
    else:
        print("⚠ No WANDB_API_KEY set — logging disabled. "
              "Set env var or pass --wandb-key to enable.")

    wandb_logger = WandbLogger(
        project="geofms-precipitation",
        log_model="all",
    ) if (wandb_key or os.environ.get("WANDB_API_KEY")) else None

    # ── CALLBACKS ────────────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor=task.monitor,
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=task.monitor,
        min_delta=0.0,
        patience=patience,
    )

    callbacks = [
        RichProgressBar(),
        checkpoint_callback,
        early_stopping_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── TRAINER ──────────────────────────────────────────────────────────────────
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=max_epochs,
        default_root_dir=str(ckpt_dir),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    # ── FIT ──────────────────────────────────────────────────────────────────────
    trainer.fit(model=task, datamodule=datamodule)

    best = checkpoint_callback.best_model_path
    print(f"\n✓ Training complete.")
    print(f"  Best checkpoint : {best or 'N/A'}")
    print(f"  Last checkpoint : {ckpt_dir / 'last.ckpt'}")
    return best


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TerraMind precipitation model")
    parser.add_argument("--pickle",      default=DEFAULT_PICKLE,   help="Path to dataset pickle")
    parser.add_argument("--ckpt-dir",    default=DEFAULT_CKPT_DIR, help="Checkpoint output directory")
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--wandb-key",   default=None, help="W&B API key (or set WANDB_API_KEY env var)")
    args = parser.parse_args()

    train(
        pickle_path    = args.pickle,
        checkpoint_dir = args.ckpt_dir,
        lr             = args.lr,
        max_epochs     = args.epochs,
        batch_size     = args.batch_size,
        patience       = args.patience,
        wandb_key      = args.wandb_key,
    )