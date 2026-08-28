"""
train.py — Train the TerraMind/GFM precipitation model.

Mirrors the cfg-dict interface of models/stack/train.py and models/unet/train.py
so the GFM can be driven by models/gfm/run_experiments.py from a YAML config.

Run from the project root:
    python -m models.gfm.train

Or with custom args:
    python -m models.gfm.train --pickle path/to/dataset.pkl --epochs 50 --output-size 9
"""

import os
import json
import argparse
import torch
import wandb
from datetime import datetime
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)

from models.gfm.model import build_task
from models.gfm.dataset import (
    RadarDEMDataModule, create_heavy_rain_sampler, DEFAULT_FIELDS,
    compute_radar_channels,
)
from models.unet.dataset import resolve_fields
from models.unet.train import set_seed

from dotenv import load_dotenv
load_dotenv()

# ── DEFAULT CONFIG ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'pickle_path':       'dataset/outputs/3d/radar_gauge_dataset_subml_daygroup_offsets_9500.pkl',
    'dem_path':          'dem/preserve_dem_10m_utm.tif',
    'checkpoint_dir':    'models/checkpoints/terramind_dualpol',
    'lr':                1e-5,
    'radar_lr':          1e-4,
    'weight_decay':      1e-4,
    'max_epochs':        100,
    'batch_size':        8,
    'patience':          20,
    'output_size':       9,        # radar crop size: 5/9/13/19 for a 19×19 pickle
    'output_bias':       3.0,
    'loss_type':         'mse',    # mse | mae | rmse | huber | asym_mae | asym_huber
    'under_weight':      2.0,      # underprediction multiplier for asym_* losses
    'log_target':        True,
    'no_sampler':        False,
    'filter_mode':       'blunt',
    'use_feature_masks': True,
    'fields':            None,     # None -> DEFAULT_FIELDS (17-field subml set)
    'backbone_pretrained': True,   # False -> same tiny arch, random init (scratch control)
    'dem_init':          'pretrained',  # 'pretrained' | 'random' (wipe pretrained DEM embed)
    'dem_norm':          'minmax',      # 'minmax' (local [0,1]) | 'terramind' (standardized m)
    'seed':              None,
    'use_wandb':         True,     # set False to train without W&B logging
    'save_top_k':        3,        # keep the N best checkpoints (history of best epochs)
}
# ───────────────────────────────────────────────────────────────────────────────


class EpochSummary(Callback):
    """Print a concise per-epoch summary line, U-Net/Stack-style.

    Works regardless of W&B: pulls metrics from trainer.callback_metrics after
    each validation epoch so you can watch train/val loss and val R² progress
    in the console even with logging disabled.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics

        def g(*keys):
            for k in keys:
                v = m.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return float('nan')

        lr = (trainer.optimizers[0].param_groups[0]['lr']
              if trainer.optimizers else float('nan'))
        print(
            f"  Epoch {trainer.current_epoch:3d} | "
            f"train_loss={g('train/loss', 'train/loss_epoch'):.4f} | "
            f"val_loss={g('val/loss', 'val/loss_epoch'):.4f} | "
            f"val_R²={g('val/R2_Score'):.3f} | "
            f"val_MAE={g('val/MAE'):.4f} | "
            f"lr={lr:.1e}"
        )


def create_run_dir(base_dir, run_name=None):
    """Create a timestamped run directory for experiment tracking."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    folder_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = Path(base_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir, cfg, n_radar_channels, train_samples, val_samples):
    """Save experiment configuration to config.json in the run directory."""
    fields = cfg.get('fields') or DEFAULT_FIELDS
    if isinstance(fields, str):
        fields = resolve_fields(fields)

    config_data = {
        'timestamp':                  datetime.now().isoformat(),
        'model_type':                 'gfm',
        'backbone':                   'terramind_v1_tiny',
        'backbone_pretrained':        cfg.get('backbone_pretrained', True),
        'dem_init':                   cfg.get('dem_init', 'pretrained'),
        'dem_norm':                   cfg.get('dem_norm', 'minmax'),
        'seed':                       cfg.get('seed'),
        'loss_type':                  cfg.get('loss_type', 'mse'),
        'under_weight':               cfg.get('under_weight', 2.0),
        'lr':                         cfg.get('lr'),
        'radar_lr':                   cfg.get('radar_lr'),
        'weight_decay':               cfg.get('weight_decay'),
        'batch_size':                 cfg.get('batch_size'),
        'patience':                   cfg.get('patience'),
        'max_epochs':                 cfg.get('max_epochs'),
        'output_size':                cfg.get('output_size'),
        'output_bias':                cfg.get('output_bias'),
        'log_target':                 cfg.get('log_target', True),
        'no_sampler':                 cfg.get('no_sampler', False),
        'filter_mode':                cfg.get('filter_mode', 'blunt'),
        'use_feature_masks':          cfg.get('use_feature_masks', True),
        'pickle_path':                cfg.get('pickle_path'),
        'dem_path':                   cfg.get('dem_path'),
        'fields':                     fields,
        'n_radar_channels':           n_radar_channels,
        'train_samples_after_filter': train_samples,
        'val_samples_after_filter':   val_samples,
    }

    config_path = run_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"  ✓ Saved config to: {config_path}")
    return config_data


def train(cfg: dict = None, run_name: str = None):
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}

    seed = cfg.get('seed')
    if seed is not None:
        set_seed(int(seed))

    # ── PATHS ───────────────────────────────────────────────────────────────────
    run_dir = create_run_dir(cfg['checkpoint_dir'], run_name)

    # ── PRECISION ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        precision = "16-mixed"
    else:
        precision = "32"

    print(f"\n{'='*60}")
    print("  TERRAMIND / GFM — Precipitation Prediction")
    print(f"{'='*60}")
    print(f"  Dataset:     {cfg['pickle_path']}")
    print(f"  Epochs:      {cfg['max_epochs']}")
    print(f"  Output size: {cfg['output_size']}")
    print(f"  Loss:        {cfg['loss_type']}")
    print(f"  Log target:  {cfg['log_target']}")
    print(f"  Pretrained:  {cfg.get('backbone_pretrained', True)}")
    print(f"  DEM init:    {cfg.get('dem_init', 'pretrained')}")
    print(f"  DEM norm:    {cfg.get('dem_norm', 'minmax')}")
    print(f"  Seed:        {seed if seed is not None else 'unseeded (random)'}")
    print(f"  Precision:   {precision}")
    print(f"  Run dir:     {run_dir}")
    print(f"{'='*60}\n")

    # ── DATA ────────────────────────────────────────────────────────────────────
    datamodule = RadarDEMDataModule(
        pickle_path=cfg['pickle_path'],
        dem_path=cfg['dem_path'],
        output_size=cfg['output_size'],
        fields=cfg.get('fields'),
        use_feature_masks=cfg.get('use_feature_masks', True),
        log_target=cfg.get('log_target', True),
        dem_norm=cfg.get('dem_norm', 'minmax'),
        weight_sampler=None if cfg.get('no_sampler', False) else create_heavy_rain_sampler,
        batch_size=cfg['batch_size'],
        filter_mode=cfg.get('filter_mode', 'blunt'),
    )
    # Run setup early so we know the channel count before building the model
    datamodule.setup()
    n_train = len(datamodule.train_ds.samples)
    n_val   = len(datamodule.val_ds.samples)

    # ── MODEL / TASK ────────────────────────────────────────────────────────────
    task = build_task(
        lr=cfg['lr'],
        output_bias=cfg['output_bias'],
        n_radar_channels=datamodule.n_radar_channels,
        output_size=cfg['output_size'],
        loss=cfg.get('loss_type', 'mse'),
        weight_decay=cfg.get('weight_decay', 1e-4),
        radar_lr=cfg.get('radar_lr', 1e-4),
        under_weight=cfg.get('under_weight', 2.0),
        backbone_pretrained=cfg.get('backbone_pretrained', True),
        dem_init=cfg.get('dem_init', 'pretrained'),
    )

    # ── CONFIG TRACKING ──────────────────────────────────────────────────────────
    save_config(run_dir, cfg, datamodule.n_radar_channels, n_train, n_val)

    # ── WANDB ────────────────────────────────────────────────────────────────────
    wandb_key = cfg.get('wandb_key')
    use_wandb = cfg.get('use_wandb', True)
    if not use_wandb:
        print("  W&B logging disabled (use_wandb=False).")
        wandb_logger = None
    else:
        if wandb_key:
            wandb.login(key=wandb_key)
        elif os.environ.get("WANDB_API_KEY"):
            wandb.login()
        else:
            print("⚠ No WANDB_API_KEY set — logging disabled. "
                  "Set env var or pass wandb_key to enable.")

        # log_model=False: don't upload checkpoints as artifacts (avoids the
        # multi-hundred-MB disk write + background upload at every epoch).
        wandb_logger = WandbLogger(
            project="geofms-precipitation",
            name=run_name,
            log_model=False,
        ) if (wandb_key or os.environ.get("WANDB_API_KEY")) else None

    # ── CALLBACKS ────────────────────────────────────────────────────────────────
    # Keep the N best checkpoints (a history of the best epochs), each uniquely
    # named by epoch. auto_insert_metric_name=False avoids the broken
    # '{val_loss}' template (the monitored key is 'val/loss', whose slash can't
    # go in a filename) that previously produced 'val_loss=0.0000' names.
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename='best-epoch={epoch:02d}',
        monitor=task.monitor,
        mode='min',
        save_top_k=cfg.get('save_top_k', 3),
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor=task.monitor,
        min_delta=0.0,
        patience=cfg['patience'],
    )
    callbacks = [
        RichProgressBar(),
        EpochSummary(),
        checkpoint_callback,
        early_stopping_callback,
    ]
    # LearningRateMonitor requires a logger; EpochSummary already prints the LR,
    # so only attach it when W&B logging is active.
    if wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # ── TRAINER ──────────────────────────────────────────────────────────────────
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        # logger=False (not None) prevents Lightning from creating a default
        # TensorBoard logger, which would write tfevents to disk and re-enable
        # TerraTorch's broken val-plot path.
        logger=wandb_logger if wandb_logger is not None else False,
        max_epochs=cfg['max_epochs'],
        default_root_dir=str(run_dir),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    # ── FIT ──────────────────────────────────────────────────────────────────────
    trainer.fit(model=task, datamodule=datamodule)

    best = checkpoint_callback.best_model_path

    # ── UPDATE CONFIG WITH FINAL TRAINING INFO ────────────────────────────────────
    config_path = run_dir / 'config.json'
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config_data['best_checkpoint'] = best or None
    config_data['best_val_loss']   = float(checkpoint_callback.best_model_score) \
        if checkpoint_callback.best_model_score is not None else None
    config_data['final_epoch']     = int(trainer.current_epoch)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    if wandb_logger is not None:
        wandb.finish()

    print(f"\n✓ Training complete.")
    print(f"  Run directory:   {run_dir}")
    print(f"  Best checkpoint: {best or 'N/A'}")
    return best, str(run_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TerraMind/GFM precipitation model")
    parser.add_argument("--pickle",      default=DEFAULT_CONFIG['pickle_path'], help="Path to dataset pickle")
    parser.add_argument("--ckpt-dir",    default=DEFAULT_CONFIG['checkpoint_dir'], help="Checkpoint output directory")
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG['max_epochs'])
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG['patience'])
    parser.add_argument("--output-size", type=int,   default=DEFAULT_CONFIG['output_size'],
                        help="Radar crop size fed to ViT (5/9/13/19; default 9)")
    parser.add_argument("--loss",        default=DEFAULT_CONFIG['loss_type'],
                        choices=['mse', 'mae', 'rmse', 'huber', 'asym_mae', 'asym_huber'])
    parser.add_argument("--under-weight", type=float, default=DEFAULT_CONFIG['under_weight'],
                        help="Underprediction multiplier for asym_* losses")
    parser.add_argument("--no-log",      action="store_true", help="Train in raw mm-space (disable log1p target)")
    parser.add_argument("--no-sampler",  action="store_true", help="Disable heavy-rain weighted sampler")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Build tiny arch with random init (scratch control)")
    parser.add_argument("--dem-init",    default=DEFAULT_CONFIG['dem_init'],
                        choices=['pretrained', 'random'],
                        help="Keep pretrained DEM embedding or re-initialize it")
    parser.add_argument("--dem-norm",    default=DEFAULT_CONFIG['dem_norm'],
                        choices=['minmax', 'terramind'],
                        help="DEM normalization: local minmax or TerraMind standardized metres")
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--run-name",    default=None, help="Short description suffix for the run folder")
    parser.add_argument("--wandb-key",   default=None, help="W&B API key")
    parser.add_argument("--no-wandb",    action="store_true", help="Disable W&B logging entirely")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg['pickle_path']    = args.pickle
    cfg['checkpoint_dir'] = args.ckpt_dir
    cfg['lr']             = args.lr
    cfg['max_epochs']     = args.epochs
    cfg['batch_size']     = args.batch_size
    cfg['patience']       = args.patience
    cfg['output_size']    = args.output_size
    cfg['loss_type']      = args.loss
    cfg['under_weight']   = args.under_weight
    cfg['log_target']     = not args.no_log
    cfg['no_sampler']     = args.no_sampler
    cfg['backbone_pretrained'] = not args.no_pretrained
    cfg['dem_init']       = args.dem_init
    cfg['dem_norm']       = args.dem_norm
    cfg['seed']           = args.seed
    cfg['wandb_key']      = args.wandb_key
    cfg['use_wandb']      = not args.no_wandb

    train(cfg, run_name=args.run_name)
