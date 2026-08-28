"""
train.py — Train the from-scratch (untrained) multimodal GFM.

Mirrors the cfg-dict interface of models/gfm/train.py so it can be driven by
models/gfm_untrained/run_experiments.py from a YAML config.

Run from the project root:
    python -m models.gfm_untrained.train
    python -m models.gfm_untrained.train --patch-size 2 --layout grouped --epochs 150
"""

import os
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor,
)

from models.gfm_untrained.model import build_model
from models.gfm_untrained.dataset import (
    RadarDEMDataModule, create_heavy_rain_sampler, DEFAULT_FIELDS,
)
from models.unet.dataset import resolve_fields
from models.unet.train import set_seed

from dotenv import load_dotenv
load_dotenv()

# ── DEFAULT CONFIG ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'pickle_path':        'dataset/outputs/3d/radar_gauge_dataset_subml_daygroup_offsets_9500.pkl',
    'dem_path':           'dem/preserve_dem_10m_utm.tif',
    'checkpoint_dir':     'models/checkpoints/gfm_untrained',
    # optimisation
    'lr':                 3e-4,
    'weight_decay':       0.05,
    'max_epochs':         150,
    'warmup_epochs':      10,
    'batch_size':         32,
    'patience':           25,
    'precision':          'auto',   # auto -> bf16-mixed if supported, else 16-mixed/32
    'gradient_clip_val':  1.0,      # 0 disables; stabilises from-scratch ViT + dropout
    # data / target
    'output_size':        18,      # native tile size (clip of the 19×19 pickle)
    'log_target':         True,
    'no_sampler':         False,
    'filter_mode':        'blunt',
    'use_feature_masks':  True,
    'fields':             None,    # None -> DEFAULT_FIELDS (17-field subml set)
    # architecture
    'patch_size':         1,       # 1 -> per-pixel tokens; 2 -> 2×2 local mixing
    'dim':                128,
    'depth':              4,
    'num_heads':          4,
    'modality_layout':    'grouped',  # 'single' | 'grouped'
    'modality_drop_rate': 0.0,
    'use_dem':            True,        # False -> drop DEM entirely (ablation)
    'output_bias':        2.0,
    # loss
    'loss_type':          'huber',  # mse | mae | rmse | huber | asym_mae | asym_huber
    'under_weight':       2.0,
    # misc
    'seed':               None,
    'use_wandb':          False,
    'save_top_k':         3,
}
# ───────────────────────────────────────────────────────────────────────────────


class EpochSummary(Callback):
    """Print a concise per-epoch summary (train/val loss, val R²/MAE, LR)."""

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
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    folder_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = Path(base_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir, cfg, modality_channels, n_train, n_val):
    fields = cfg.get('fields') or DEFAULT_FIELDS
    if isinstance(fields, str):
        fields = resolve_fields(fields)

    config_data = {
        'timestamp':                  datetime.now().isoformat(),
        'model_type':                 'gfm_untrained',
        'backbone':                   'terramind_vit_scratch',
        'pretrained':                 False,
        'seed':                       cfg.get('seed'),
        # architecture
        'patch_size':                 cfg.get('patch_size'),
        'dim':                        cfg.get('dim'),
        'depth':                      cfg.get('depth'),
        'num_heads':                  cfg.get('num_heads'),
        'modality_layout':            cfg.get('modality_layout'),
        'modality_drop_rate':         cfg.get('modality_drop_rate'),
        'use_dem':                    cfg.get('use_dem', True),
        'modality_channels':          modality_channels,
        'output_bias':                cfg.get('output_bias'),
        # data / target
        'output_size':                cfg.get('output_size'),
        'log_target':                 cfg.get('log_target', True),
        'no_sampler':                 cfg.get('no_sampler', False),
        'filter_mode':                cfg.get('filter_mode', 'blunt'),
        'use_feature_masks':          cfg.get('use_feature_masks', True),
        'fields':                     fields,
        # loss / optim
        'loss_type':                  cfg.get('loss_type'),
        'under_weight':               cfg.get('under_weight'),
        'lr':                         cfg.get('lr'),
        'weight_decay':               cfg.get('weight_decay'),
        'warmup_epochs':              cfg.get('warmup_epochs'),
        'precision':                  cfg.get('precision'),
        'gradient_clip_val':          cfg.get('gradient_clip_val'),
        'batch_size':                 cfg.get('batch_size'),
        'patience':                   cfg.get('patience'),
        'max_epochs':                 cfg.get('max_epochs'),
        # paths
        'pickle_path':                cfg.get('pickle_path'),
        'dem_path':                   cfg.get('dem_path'),
        'train_samples_after_filter': n_train,
        'val_samples_after_filter':   n_val,
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"  ✓ Saved config to: {run_dir / 'config.json'}")
    return config_data


def train(cfg: dict = None, run_name: str = None):
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}

    seed = cfg.get('seed')
    if seed is not None:
        set_seed(int(seed))

    run_dir = create_run_dir(cfg['checkpoint_dir'], run_name)

    # Precision: prefer bf16 on capable GPUs. bf16 has fp32's dynamic range, so
    # (unlike fp16/16-mixed) it does not overflow during from-scratch ViT warmup
    # or aggressive modality dropout — eliminating the nan crashes we hit. Falls
    # back to 16-mixed if the GPU lacks bf16, or 32 on CPU. Override via cfg.
    precision = cfg.get('precision', 'auto')
    if precision == 'auto':
        if torch.cuda.is_available():
            precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        else:
            precision = "32"
    grad_clip = cfg.get('gradient_clip_val', 1.0)

    print(f"\n{'='*60}")
    print("  GFM-UNTRAINED — From-scratch multimodal ViT")
    print(f"{'='*60}")
    print(f"  Dataset:      {cfg['pickle_path']}")
    print(f"  Tile/patch:   {cfg['output_size']} px, patch_size={cfg['patch_size']}")
    print(f"  Layout:       {cfg['modality_layout']}  (mod_drop={cfg['modality_drop_rate']}, use_dem={cfg.get('use_dem', True)})")
    print(f"  ViT:          dim={cfg['dim']}, depth={cfg['depth']}, heads={cfg['num_heads']}")
    print(f"  Loss:         {cfg['loss_type']}  (log_target={cfg['log_target']})")
    print(f"  Seed:         {seed if seed is not None else 'unseeded'}")
    print(f"  Precision:    {precision}  (grad_clip={grad_clip})")
    print(f"  Run dir:      {run_dir}")
    print(f"{'='*60}\n")

    # ── DATA ────────────────────────────────────────────────────────────────────
    datamodule = RadarDEMDataModule(
        pickle_path=cfg['pickle_path'],
        dem_path=cfg['dem_path'],
        output_size=cfg['output_size'],
        fields=cfg.get('fields'),
        use_feature_masks=cfg.get('use_feature_masks', True),
        log_target=cfg.get('log_target', True),
        modality_layout=cfg.get('modality_layout', 'grouped'),
        use_dem=cfg.get('use_dem', True),
        weight_sampler=None if cfg.get('no_sampler', False) else create_heavy_rain_sampler,
        batch_size=cfg['batch_size'],
        filter_mode=cfg.get('filter_mode', 'blunt'),
    )
    datamodule.setup()
    n_train = len(datamodule.train_ds.samples)
    n_val   = len(datamodule.val_ds.samples)

    # ── MODEL ─────────────────────────────────────────────────────────────────--
    model = build_model(datamodule.modality_channels, cfg)

    # ── CONFIG TRACKING ──────────────────────────────────────────────────────────
    save_config(run_dir, cfg, datamodule.modality_channels, n_train, n_val)

    # ── WANDB (optional) ───────────────────────────────────────────────────────--
    wandb_logger = None
    if cfg.get('use_wandb', False):
        import wandb
        from lightning.pytorch.loggers import WandbLogger
        if cfg.get('wandb_key'):
            wandb.login(key=cfg['wandb_key'])
        elif os.environ.get("WANDB_API_KEY"):
            wandb.login()
        if cfg.get('wandb_key') or os.environ.get("WANDB_API_KEY"):
            wandb_logger = WandbLogger(project="geofms-precipitation",
                                       name=run_name, log_model=False)
        else:
            print("⚠ use_wandb=True but no WANDB_API_KEY — continuing without W&B.")

    # ── CALLBACKS ────────────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename='best-epoch={epoch:02d}',
        monitor=model.monitor,
        mode='min',
        save_top_k=cfg.get('save_top_k', 3),
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        RichProgressBar(),
        EpochSummary(),
        checkpoint_callback,
        EarlyStopping(monitor=model.monitor, min_delta=0.0, patience=cfg['patience']),
    ]
    if wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # ── TRAINER ──────────────────────────────────────────────────────────────────
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision=precision,
        gradient_clip_val=grad_clip,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else False,
        max_epochs=cfg['max_epochs'],
        default_root_dir=str(run_dir),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model=model, datamodule=datamodule)

    best = checkpoint_callback.best_model_path

    # ── UPDATE CONFIG WITH FINAL TRAINING INFO ────────────────────────────────────
    config_path = run_dir / 'config.json'
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config_data['best_checkpoint'] = best or None
    config_data['best_val_loss']   = (float(checkpoint_callback.best_model_score)
                                      if checkpoint_callback.best_model_score is not None else None)
    config_data['final_epoch']     = int(trainer.current_epoch)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    if wandb_logger is not None:
        import wandb
        wandb.finish()

    print(f"\n✓ Training complete.")
    print(f"  Run directory:   {run_dir}")
    print(f"  Best checkpoint: {best or 'N/A'}")
    return best, str(run_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train from-scratch multimodal GFM")
    p.add_argument("--pickle",      default=DEFAULT_CONFIG['pickle_path'])
    p.add_argument("--ckpt-dir",    default=DEFAULT_CONFIG['checkpoint_dir'])
    p.add_argument("--lr",          type=float, default=DEFAULT_CONFIG['lr'])
    p.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG['max_epochs'])
    p.add_argument("--warmup",      type=int,   default=DEFAULT_CONFIG['warmup_epochs'])
    p.add_argument("--batch-size",  type=int,   default=DEFAULT_CONFIG['batch_size'])
    p.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG['patience'])
    p.add_argument("--output-size", type=int,   default=DEFAULT_CONFIG['output_size'],
                   help="Native tile size (clip of the pickle patch; default 18)")
    p.add_argument("--patch-size",  type=int,   default=DEFAULT_CONFIG['patch_size'], choices=[1, 2, 3, 6, 9])
    p.add_argument("--dim",         type=int,   default=DEFAULT_CONFIG['dim'])
    p.add_argument("--depth",       type=int,   default=DEFAULT_CONFIG['depth'])
    p.add_argument("--heads",       type=int,   default=DEFAULT_CONFIG['num_heads'])
    p.add_argument("--layout",      default=DEFAULT_CONFIG['modality_layout'], choices=['single', 'grouped'])
    p.add_argument("--mod-drop",    type=float, default=DEFAULT_CONFIG['modality_drop_rate'])
    p.add_argument("--no-dem",      action="store_true", help="Drop the DEM modality entirely (ablation)")
    p.add_argument("--loss",        default=DEFAULT_CONFIG['loss_type'],
                   choices=['mse', 'mae', 'rmse', 'huber', 'asym_mae', 'asym_huber'])
    p.add_argument("--under-weight", type=float, default=DEFAULT_CONFIG['under_weight'])
    p.add_argument("--precision",   default=DEFAULT_CONFIG['precision'],
                   help="auto | bf16-mixed | 16-mixed | 32")
    p.add_argument("--grad-clip",   type=float, default=DEFAULT_CONFIG['gradient_clip_val'],
                   help="Gradient clip value (0 disables)")
    p.add_argument("--no-log",      action="store_true", help="Train in raw mm-space")
    p.add_argument("--no-sampler",  action="store_true", help="Disable heavy-rain sampler")
    p.add_argument("--seed",        type=int, default=None)
    p.add_argument("--run-name",    default=None)
    p.add_argument("--wandb-key",   default=None)
    p.add_argument("--use-wandb",   action="store_true")
    args = p.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        'pickle_path':        args.pickle,
        'checkpoint_dir':     args.ckpt_dir,
        'lr':                 args.lr,
        'max_epochs':         args.epochs,
        'warmup_epochs':      args.warmup,
        'batch_size':         args.batch_size,
        'patience':           args.patience,
        'output_size':        args.output_size,
        'patch_size':         args.patch_size,
        'dim':                args.dim,
        'depth':              args.depth,
        'num_heads':          args.heads,
        'modality_layout':    args.layout,
        'modality_drop_rate': args.mod_drop,
        'use_dem':            not args.no_dem,
        'loss_type':          args.loss,
        'under_weight':       args.under_weight,
        'precision':          args.precision,
        'gradient_clip_val':  args.grad_clip,
        'log_target':         not args.no_log,
        'no_sampler':         args.no_sampler,
        'seed':               args.seed,
        'wandb_key':          args.wandb_key,
        'use_wandb':          args.use_wandb,
    })
    train(cfg, run_name=args.run_name)
