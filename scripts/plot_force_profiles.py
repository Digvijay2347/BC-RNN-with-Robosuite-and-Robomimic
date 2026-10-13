"""
scripts/plot_force_profiles.py
==============================
Generates comparative force profile curves over time for Phase 6, comparing:
- Model A (Standard BC Baseline)
- Model B (BC with Instant F/T)
- Model C (BC-RNN with F/T History)

Outputs:
    results/force_profile_comparison.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    json_path = "results/evaluation_results.json"
    save_path = "results/force_profile_comparison.png"

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Contact Force Profile Comparison: BC vs. F/T-Aware Policies", fontsize=16, fontweight="bold", y=0.98)

    models_info = [
        ("bc_no_ft", "Model A: Standard BC (No F/T)", "#1f77b4", "-"),
        ("bc_with_ft", "Model B: BC with Instant F/T", "#ff7f0e", "-"),
        ("bc_rnn_with_ft", "Model C: BC-RNN with F/T History", "#2ca02c", "-"),
    ]

    # Panel 1: Single representative episode trajectory (Episode 0)
    ax1 = axes[0]
    for key, label, color, style in models_info:
        if key in data and data[key]["episodes"]:
            ep0_forces = data[key]["episodes"][0]["forces"]
            steps = np.arange(len(ep0_forces))
            ax1.plot(steps, ep0_forces, label=label, color=color, linestyle=style, linewidth=2.0, alpha=0.9)

    ax1.axhline(25.0, color="#c0392b", linestyle="--", linewidth=1.2, label="Jamming Force Threshold (25 N)")
    ax1.axhline(5.0, color="#7f8c8d", linestyle=":", linewidth=1.0, label="Contact Detection Threshold (5 N)")
    ax1.set_title("Single Episode Trajectory (Episode 0: Seed 42)", fontweight="bold", pad=10)
    ax1.set_ylabel(r"Contact Force $\|F\|_2$ (N)", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(-2, 110)

    # Panel 2: Mean contact force across all 50 evaluation rollouts with +/- 1 std error band
    ax2 = axes[1]
    for key, label, color, style in models_info:
        if key in data and data[key]["episodes"]:
            # Matrix of shape (50, 400)
            all_forces = [ep["forces"] for ep in data[key]["episodes"]]
            force_mat = np.array(all_forces)
            mean_f = np.mean(force_mat, axis=0)
            std_f = np.std(force_mat, axis=0)
            steps = np.arange(force_mat.shape[1])

            ax2.plot(steps, mean_f, label=f"{label} (Mean)", color=color, linewidth=2.2)
            ax2.fill_between(steps, np.maximum(0, mean_f - 0.5 * std_f), mean_f + 0.5 * std_f, color=color, alpha=0.18)

    ax2.axhline(25.0, color="#c0392b", linestyle="--", linewidth=1.2, label="Jamming Force Threshold (25 N)")
    ax2.set_title("Population Mean & Variance Across 50 Evaluation Seeds", fontweight="bold", pad=10)
    ax2.set_xlabel("Environment Steps (Control Frequency: 20 Hz)", fontweight="bold")
    ax2.set_ylabel(r"Mean Contact Force $\|F\|_2$ (N)", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(-2, 130)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] Force profile plot saved to: {save_path}")


if __name__ == "__main__":
    main()
