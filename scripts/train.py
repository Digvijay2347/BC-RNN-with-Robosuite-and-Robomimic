"""
CPU-Optimized Training Pipeline for Contact-Rich BC & BC-RNN Benchmarking.
Trains 3 experimental conditions:
  1. Model A: bc_no_ft       (Baseline: Standard BC-MLP, position-only)
  2. Model B: bc_with_ft     (Ablation 1: BC-MLP with Instantaneous Force/Torque)
  3. Model C: bc_rnn_with_ft (Ablation 2: BC-RNN with Force/Torque History, H=10)

Extracts metrics, saves standard checkpoints (model_best.pth),
logs telemetry to results/training_history.json, and plots loss curves.
"""

import os
import sys
import time
import json
import glob
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import robomimic.config as Config
import robomimic.utils.torch_utils as TorchUtils
import robomimic.scripts.train as TrainRunner


MODELS_CONFIG = {
    "bc_no_ft": {
        "name": "Model A: BC (No F/T)",
        "config_file": "configs/config_bc_no_ft.json",
        "color": "#1f77b4",  # Blue
    },
    "bc_with_ft": {
        "name": "Model B: BC (With F/T)",
        "config_file": "configs/config_bc_with_ft.json",
        "color": "#ff7f0e",  # Orange
    },
    "bc_rnn_with_ft": {
        "name": "Model C: BC-RNN (F/T History)",
        "config_file": "configs/config_bc_rnn_with_ft.json",
        "color": "#2ca02c",  # Green
    },
}


def train_single_model(model_key, epochs=50, dataset_path="datasets/square/ph/low_dim_with_ft.hdf5"):
    info = MODELS_CONFIG[model_key]
    cfg_path = info["config_file"]
    
    print("\n" + "=" * 72)
    print(f"  STARTING TRAINING: {info['name']}")
    print(f"  Configuration : {cfg_path}")
    print(f"  Epochs        : {epochs}")
    print(f"  Device        : CPU (Torch CPU Backend)")
    print("=" * 72)

    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)

    config = Config.config_factory(cfg_dict["algo_name"], dic=cfg_dict)
    
    # Configure absolute paths and parameters
    output_dir = os.path.abspath(os.path.join("models", model_key))
    os.makedirs(output_dir, exist_ok=True)

    config.unlock()
    config.train.data = os.path.abspath(dataset_path)
    config.train.output_dir = output_dir
    config.train.cuda = False
    config.train.num_epochs = epochs
    config.experiment.logging.terminal_output_to_txt = False
    config.lock()

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    start_time = time.time()
    
    # Run training
    TrainRunner.train(config, device=device)
    elapsed = time.time() - start_time
    print(f"\n[DONE] Training finished for {info['name']} in {elapsed:.1f}s ({elapsed/60:.2f} min).")

    # Locate generated log file and model checkpoints
    run_dirs = glob.glob(os.path.join(output_dir, config.experiment.name, "*"))
    if not run_dirs:
        # Fallback check
        run_dirs = glob.glob(os.path.join(output_dir, "*"))

    latest_run = max(run_dirs, key=os.path.getmtime) if run_dirs else output_dir
    checkpoints_dir = os.path.join(latest_run, "models")
    
    # Find best checkpoint
    best_ckpt = None
    all_ckpts = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    for ckpt in all_ckpts:
        if "best_validation" in os.path.basename(ckpt):
            best_ckpt = ckpt
            break
    if not best_ckpt and all_ckpts:
        best_ckpt = all_ckpts[-1]

    # Copy to a standardized location: models/{model_key}/model_best.pth
    std_ckpt_path = os.path.join("models", f"{model_key}_best.pth")
    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copyfile(best_ckpt, std_ckpt_path)
        print(f"  Standardized Checkpoint Saved -> {std_ckpt_path}")

    # Parse training log for curves
    log_file = os.path.join(latest_run, "logs", "log.txt")
    train_losses, val_losses, epochs_logged = [], [], []

    # If Tensorboard logs exist, read from tb or robomimic's logged json
    # Or fallback to reading validation checkpoints / stdout
    for ep in range(1, epochs + 1):
        epochs_logged.append(ep)

    return {
        "model_key": model_key,
        "name": info["name"],
        "checkpoint": std_ckpt_path,
        "elapsed_sec": elapsed,
        "latest_run": latest_run,
    }


def parse_and_plot_curves(run_results, save_path="results/training_curves.png"):
    print("\n" + "=" * 72)
    print("  PARSING TENSORBOARD LOGS & GENERATING LOSS CURVES")
    print("=" * 72)

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    history_summary = {}

    for res in run_results:
        m_key = res["model_key"]
        info = MODELS_CONFIG[m_key]
        color = info["color"]
        run_dir = res["latest_run"]

        tb_files = glob.glob(os.path.join(run_dir, "logs", "tb", "events.out.tfevents*"))
        if not tb_files:
            continue

        ea = EventAccumulator(tb_files[0])
        ea.Reload()

        train_loss = [s.value for s in ea.Scalars("Train/Loss")]
        val_loss = [s.value for s in ea.Scalars("Valid/Loss")]
        epochs = list(range(1, len(train_loss) + 1))

        history_summary[m_key] = {
            "name": info["name"],
            "initial_train_loss": float(train_loss[0]),
            "final_train_loss": float(train_loss[-1]),
            "best_val_loss": float(min(val_loss)),
            "best_val_epoch": int(val_loss.index(min(val_loss)) + 1),
            "train_loss_history": [float(x) for x in train_loss],
            "val_loss_history": [float(x) for x in val_loss],
        }

        # Plot curves
        axes[0].plot(epochs, train_loss, label=info["name"], color=color, lw=2.2)
        axes[1].plot(epochs, val_loss, label=info["name"], color=color, lw=2.2)

    axes[0].set_title("Training Loss (L2 Action Regression)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", framealpha=0.9)

    axes[1].set_title("Validation Loss (Held-Out Human Demonstrations)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [SAVED] Publication-Quality Loss Curves -> {save_path}")

    # Save detailed numeric telemetry
    json_path = os.path.join("results", "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history_summary, f, indent=4)
    print(f"  [SAVED] Training History Telemetry -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CPU-optimized training for BC & BC-RNN.")
    parser.add_argument("--model", type=str, default="all", choices=["all", "bc_no_ft", "bc_with_ft", "bc_rnn_with_ft"],
                        help="Model to train (or 'all' for all 3)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--dataset", type=str, default="datasets/square/ph/low_dim_with_ft.hdf5", help="Path to HDF5 dataset")
    args = parser.parse_args()

    targets = list(MODELS_CONFIG.keys()) if args.model == "all" else [args.model]
    results = []

    total_start = time.time()
    for m in targets:
        res = train_single_model(m, epochs=args.epochs, dataset_path=args.dataset)
        results.append(res)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 72)
    print("  ALL REQUESTED MODELS TRAINED SUCCESSFULLY!")
    print(f"  Total Pipeline Time: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    print("=" * 72)

    # Save summary manifest
    summary_path = os.path.join("results", "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Saved Training Manifest -> {summary_path}")

    # Plot loss curves
    parse_and_plot_curves(results)


if __name__ == "__main__":
    main()
