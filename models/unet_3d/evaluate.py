"""
evaluate.py — Evaluate a trained 3D U-Net precipitation model.

Reuses the plotting/inference/test helpers from models.stack_3d.evaluate; only the
model construction differs (PrecipUNet3D instead of the stack encoder/decoder).

Run from the project root:
    python -m models.unet_3d.evaluate --run-dir models/checkpoints/unet_3d_dualpol/<run>
    python -m models.unet_3d.evaluate --checkpoint .../best-epoch=XX-val_loss=X.XXXX.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader

from models.unet_3d.model import PrecipUNet3D
from models.stack_3d.dataset import RadarGauge3DDataset, resolve_fields, compute_n_input_channels
from models.stack_3d.train import (
    filter_bad_samples, filter_biased_extremes, filter_nan_radar,
    filter_suspect_station_days, filter_stations,
)
from models.stack_3d.evaluate import (
    find_checkpoint, run_inference, compute_metrics, print_report,
    write_eval_results, plot_evaluation, plot_station_bias, evaluate_test,
)

DEFAULT_DEM      = 'dem/preserve_dem_10m_utm.tif'
DEFAULT_CKPT_DIR = 'models/checkpoints/unet_3d_dualpol'
DEFAULT_OUTPUT   = 'evaluation_figures/unet_3d_dualpol'


def load_model(checkpoint_path, device):
    print(f"Loading model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})

    fields = resolve_fields(cfg.get('fields'))
    in_channels = compute_n_input_channels(
        fields, cfg.get('use_mask', True), cfg.get('use_temporal_pos', True),
        cfg.get('use_dem', True), cfg.get('use_feature_masks', False))

    model = PrecipUNet3D(
        in_channels=in_channels,
        base_filters=cfg.get('base_filters', 32),
        dropout_rate=cfg.get('dropout_rate', 0.15),
        n_encoder_blocks=cfg.get('n_encoder_blocks', 3),
        z_collapse=cfg.get('z_collapse', 'mean'),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    return model, cfg


def evaluate(checkpoint_path=None, checkpoint_dir=None, pickle_path=None,
             dem_path=None, output_dir=None, run_dir=None, exclude_stations=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = output_dir or DEFAULT_OUTPUT

    ckpt = find_checkpoint(checkpoint_path, checkpoint_dir or DEFAULT_CKPT_DIR, run_dir)

    if not run_dir:
        ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
        run_dir = ckpt_data.get('run_dir')

    model, cfg = load_model(ckpt, device)

    pickle_path = cfg.get('pickle_path') or pickle_path
    dem_path = cfg.get('dem_path') or dem_path
    exclude = exclude_stations or cfg.get('exclude_stations', [])

    ds_kwargs = dict(
        dem_path=dem_path,
        fields=cfg.get('fields'),
        use_dem=cfg.get('use_dem', True),
        use_mask=cfg.get('use_mask', True),
        use_temporal_pos=cfg.get('use_temporal_pos', True),
        log_target=cfg.get('log_target', True),
        use_feature_masks=cfg.get('use_feature_masks', False),
    )
    val_ds = RadarGauge3DDataset(pickle_path, split='val', augment=False, **ds_kwargs)
    val_ds.samples = filter_stations(val_ds.samples, exclude)
    val_ds.samples = filter_nan_radar(val_ds.samples)
    filter_mode = cfg.get('filter_mode', 'blunt')
    if filter_mode != 'radar':
        val_ds.samples = filter_biased_extremes(val_ds.samples)
        val_ds.samples = filter_bad_samples(val_ds.samples)
    val_ds.samples = filter_suspect_station_days(val_ds.samples)

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    log_target = cfg.get('log_target', True)
    preds_mm, targets_mm, station_names = run_inference(model, val_loader, device, log_target=log_target)
    metrics = compute_metrics(preds_mm, targets_mm)
    print_report(preds_mm, targets_mm, metrics)
    plot_evaluation(preds_mm, targets_mm, metrics, output_dir, run_dir=run_dir)
    plot_station_bias(preds_mm, targets_mm, station_names, output_dir, run_dir=run_dir)
    write_eval_results(run_dir, preds_mm, targets_mm, metrics)

    evaluate_test(model, cfg, pickle_path, dem_path, device, output_dir, run_dir)

    return metrics, run_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net precipitation model")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CKPT_DIR)
    parser.add_argument('--run-dir', default=None, help='Run directory (auto-finds checkpoint and saves results there)')
    parser.add_argument('--pickle', default=None)
    parser.add_argument('--dem', default=DEFAULT_DEM)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        pickle_path=args.pickle,
        dem_path=args.dem,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
    )
