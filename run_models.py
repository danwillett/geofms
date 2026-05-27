"""
run_models.py — Batch runner for model experiments.

Runs all defined scenarios sequentially, continuing to the next if one fails.

Usage:
    python run_models.py
    python run_models.py --model stack
    python run_models.py --model unet
"""

import subprocess
import sys
import time
from datetime import datetime

STACK_RUNS = [
    {"loss": "weighted_mae", "sampler": "light",    "name": "wmae_light"},
    {"loss": "weighted_mae", "sampler": "moderate", "name": "wmae_moderate"},
    {"loss": "weighted_mae", "sampler": "heavy",    "name": "wmae_heavy"},
    {"loss": "mae",          "sampler": "light",    "name": "mae_light"},
    {"loss": "mae",          "sampler": "moderate", "name": "mae_moderate"},
    {"loss": "mae",          "sampler": "heavy",    "name": "mae_heavy"},
]

UNET_RUNS = [
    {"loss": "weighted_mae", "sampler": "light",    "name": "wmae_light"},
    {"loss": "weighted_mae", "sampler": "moderate", "name": "wmae_moderate"},
    {"loss": "weighted_mae", "sampler": "heavy",    "name": "wmae_heavy"},
    {"loss": "mae",          "sampler": "light",    "name": "mae_light"},
    {"loss": "mae",          "sampler": "moderate", "name": "mae_moderate"},
    {"loss": "mae",          "sampler": "heavy",    "name": "mae_heavy"},
    {"loss": "mse",          "sampler": "light",    "name": "mse_light"},
    {"loss": "mse",          "sampler": "moderate", "name": "mse_moderate"},
    {"loss": "mse",          "sampler": "heavy",    "name": "mse_heavy"},
]


def build_command(model_type, run_cfg):
    if model_type == "stack":
        module = "models.stack.run_stack"
    else:
        module = "models.unet.run_unet"

    cmd = [
        sys.executable, "-m", module,
        "--mode", "all",
        "--loss", run_cfg["loss"],
        "--sampler-type", run_cfg["sampler"],
        "--run-name", run_cfg["name"],
    ]
    return cmd


def run_experiment(model_type, run_cfg, idx, total):
    cmd = build_command(model_type, run_cfg)
    header = f"[{idx}/{total}] {model_type.upper()} — {run_cfg['name']} (loss={run_cfg['loss']}, sampler={run_cfg['sampler']})"

    print("\n" + "=" * 70)
    print(f"  {header}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"\n  ✓ PASSED — {header} ({elapsed:.0f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n  ✗ FAILED — {header} ({elapsed:.0f}s)")
        print(f"    Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n  ⚠ INTERRUPTED — {header}")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch run model experiments")
    parser.add_argument("--model", choices=["stack", "unet", "all"], default="all",
                        help="Which model(s) to run (default: all)")
    args = parser.parse_args()

    runs = []
    if args.model in ("unet", "all"):
        runs += [("unet", r) for r in UNET_RUNS]
    
    if args.model in ("stack", "all"):
        runs += [("stack", r) for r in STACK_RUNS]
    

    total = len(runs)
    results = []

    print(f"\n{'='*70}")
    print(f"  BATCH RUN — {total} experiments queued")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    for i, (model_type, run_cfg) in enumerate(runs, 1):
        try:
            success = run_experiment(model_type, run_cfg, i, total)
            results.append((model_type, run_cfg["name"], success))
        except KeyboardInterrupt:
            print("\n\nBatch interrupted by user.")
            break

    # Summary
    print(f"\n\n{'='*70}")
    print("  BATCH SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for _, _, s in results if s)
    failed = sum(1 for _, _, s in results if not s)
    print(f"  Completed: {len(results)}/{total}")
    print(f"  Passed: {passed}  |  Failed: {failed}")
    print()
    for model_type, name, success in results:
        status = "✓" if success else "✗"
        print(f"    {status} {model_type}/{name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
