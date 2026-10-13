"""
scripts/evaluate.py
===================
Phase 5: Closed-Loop Simulation Evaluation Pipeline for Contact-Rich Manipulation.

Evaluates trained imitation learning checkpoints (Model A: BC No-F/T, Model B: BC With-F/T,
Model C: BC-RNN With-F/T) inside the robosuite `NutAssemblySquare` simulation.

Features:
- Deterministic paired testing: Evaluates all models across identical random seeds/initial states.
- Timestep-level telemetry logging:
  * Insertion Success (binary)
  * Completion Horizon (steps)
  * 3D Contact Force & Torque profiles (||F||_2, ||T||_2)
  * Jamming / Wedging detection (sustained high force with near-zero motion)
- Optional offscreen MP4 video recording of evaluation rollouts.
- Generates 4-panel comparative publication-quality visualizations and JSON reports.

Usage:
    python scripts/evaluate.py --n_episodes 50
    python scripts/evaluate.py --models bc_rnn_with_ft --n_episodes 10
    python scripts/evaluate.py --n_episodes 20 --video_episodes 1
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

# Define evaluation models
MODELS = {
    "bc_no_ft": {
        "name": "Model A (BC Baseline)",
        "arch": "Standard BC (MLP, No F/T)",
        "checkpoint": "models/bc_no_ft_best.pth",
        "color": "#1f77b4",  # Blue
    },
    "bc_with_ft": {
        "name": "Model B (BC Instant F/T)",
        "arch": "BC (MLP, Instant F/T)",
        "checkpoint": "models/bc_with_ft_best.pth",
        "color": "#ff7f0e",  # Orange
    },
    "bc_rnn_with_ft": {
        "name": "Model C (BC-RNN F/T History)",
        "arch": "BC-RNN (LSTM, F/T History H=10)",
        "checkpoint": "models/bc_rnn_with_ft_best.pth",
        "color": "#2ca02c",  # Green
    },
}

# Physical thresholds for contact analysis
FORCE_CONTACT_THRESH = 5.0     # N: Threshold above which contact is considered active
FORCE_JAMMING_THRESH = 25.0    # N: Threshold considered high sustained contact
MOTION_STALL_THRESH = 0.0015   # meters (1.5mm): Displacement considered stationary per step
JAMMING_MIN_STEPS = 15         # consecutive steps of stall under high force to flag a wedge


def set_seed(seed):
    """Set random seed across Python, NumPy, and PyTorch for bit-exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
    policy,
    env,
    seed,
    horizon=400,
    record_video=False,
    camera_name="agentview",
    video_dims=(384, 384),
):
    """
    Executes a single closed-loop rollout episode.
    """
    set_seed(seed)
    obs = env.reset()
    policy.start_episode()

    frames = []
    if record_video:
        frame = env.render(mode="rgb_array", height=video_dims[0], width=video_dims[1], camera_name=camera_name)
        frames.append(frame)

    prev_pos = obs.get("robot0_eef_pos", np.zeros(3)).copy()

    step_forces = []
    step_torques = []
    stall_counter = 0
    max_consecutive_stall = 0
    success = False
    total_reward = 0.0

    for step in range(horizon):
        # Query policy action
        act = policy(obs)

        # Step environment
        next_obs, reward, done, info = env.step(act)
        total_reward += float(reward)

        # Contact forces & torques
        force = env.env.robots[0].ee_force
        torque = env.env.robots[0].ee_torque
        f_mag = float(np.linalg.norm(force))
        t_mag = float(np.linalg.norm(torque))
        step_forces.append(f_mag)
        step_torques.append(t_mag)

        # Check motion displacement
        curr_pos = next_obs.get("robot0_eef_pos", np.zeros(3))
        disp = float(np.linalg.norm(curr_pos - prev_pos))
        prev_pos = curr_pos.copy()

        # Jamming detection (high force with negligible displacement)
        if f_mag >= FORCE_JAMMING_THRESH and disp < MOTION_STALL_THRESH:
            stall_counter += 1
            if stall_counter > max_consecutive_stall:
                max_consecutive_stall = stall_counter
        else:
            stall_counter = 0

        # Video recording
        if record_video:
            frame = env.render(mode="rgb_array", height=video_dims[0], width=video_dims[1], camera_name=camera_name)
            frames.append(frame)

        # Success check
        is_succ = env.is_success()["task"]
        if is_succ:
            success = True
            obs = next_obs
            break

        obs = next_obs

    steps_taken = step + 1
    forces_arr = np.array(step_forces) if step_forces else np.array([0.0])
    torques_arr = np.array(step_torques) if step_torques else np.array([0.0])

    peak_force = float(np.max(forces_arr))
    mean_force = float(np.mean(forces_arr))
    rms_force = float(np.sqrt(np.mean(forces_arr ** 2)))
    peak_torque = float(np.max(torques_arr))

    contact_steps = int(np.sum(forces_arr > FORCE_CONTACT_THRESH))
    jammed = bool(max_consecutive_stall >= JAMMING_MIN_STEPS and not success)

    episode_data = {
        "seed": seed,
        "success": bool(success),
        "steps": int(steps_taken),
        "total_reward": float(total_reward),
        "peak_force": peak_force,
        "mean_force": mean_force,
        "rms_force": rms_force,
        "peak_torque": peak_torque,
        "contact_ratio": float(contact_steps / steps_taken),
        "max_consecutive_stall": int(max_consecutive_stall),
        "jammed": jammed,
        "forces": [round(float(f), 2) for f in forces_arr],
    }

    return episode_data, frames


def evaluate_model(
    model_key,
    n_episodes=50,
    horizon=400,
    seed_start=42,
    video_episodes=0,
    video_dir="results",
):
    """
    Evaluates a specific trained model across n_episodes.
    """
    m_info = MODELS[model_key]
    ckpt_path = m_info["checkpoint"]

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    print("\n" + "=" * 76)
    print(f"  EVALUATING: {m_info['name']}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Rollouts: {n_episodes} episodes | Max Horizon: {horizon} steps")
    print("=" * 76)

    # Load policy and create simulator environment
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device="cpu", verbose=False)
    has_video = video_episodes > 0

    env = EnvUtils.create_env_from_metadata(
        env_meta=ckpt_dict["env_metadata"],
        render=False,
        render_offscreen=has_video,
    )

    results = []
    t0 = time.time()

    pbar = tqdm(range(n_episodes), desc=f"Evaluating {model_key}", unit="ep")
    for ep_idx in pbar:
        seed = seed_start + ep_idx
        record_this_ep = (ep_idx < video_episodes)

        ep_data, frames = run_episode(
            policy=policy,
            env=env,
            seed=seed,
            horizon=horizon,
            record_video=record_this_ep,
        )
        ep_data["episode_idx"] = ep_idx
        results.append(ep_data)

        # Save video if recorded
        if record_this_ep and frames:
            import imageio
            os.makedirs(video_dir, exist_ok=True)
            v_name = f"eval_{model_key}_ep{ep_idx}.mp4"
            v_path = os.path.join(video_dir, v_name)
            imageio.mimsave(v_path, frames, fps=20)
            print(f"\n  [Video Saved] {v_path}")

        # Update tqdm live progress
        successes = sum(1 for r in results if r["success"])
        curr_succ_rate = (successes / len(results)) * 100.0
        pbar.set_postfix({
            "Success": f"{successes}/{len(results)} ({curr_succ_rate:.1f}%)",
            "Avg F": f"{np.mean([r['peak_force'] for r in results]):.1f}N",
        })

    elapsed = time.time() - t0

    # Aggregate metrics
    succ_list = [r["success"] for r in results]
    success_rate = (sum(succ_list) / len(results)) * 100.0
    all_steps = [r["steps"] for r in results]
    succ_steps = [r["steps"] for r in results if r["success"]]
    peak_forces = [r["peak_force"] for r in results]
    rms_forces = [r["rms_force"] for r in results]
    jammed_list = [r["jammed"] for r in results]
    jamming_rate = (sum(jammed_list) / len(results)) * 100.0

    summary = {
        "model_key": model_key,
        "name": m_info["name"],
        "arch": m_info["arch"],
        "n_episodes": n_episodes,
        "success_rate": round(success_rate, 2),
        "success_count": sum(succ_list),
        "mean_steps_all": round(float(np.mean(all_steps)), 2),
        "median_steps_all": round(float(np.median(all_steps)), 2),
        "mean_steps_on_success": round(float(np.mean(succ_steps)), 2) if succ_steps else None,
        "mean_peak_force": round(float(np.mean(peak_forces)), 2),
        "max_peak_force": round(float(np.max(peak_forces)), 2),
        "mean_rms_force": round(float(np.mean(rms_forces)), 2),
        "jamming_rate": round(jamming_rate, 2),
        "total_time_sec": round(elapsed, 2),
        "fps": round((sum(all_steps) / elapsed), 2),
    }

    print("\n--- Summary for " + m_info["name"] + " ---")
    print(f"  Success Rate:          {summary['success_rate']}% ({summary['success_count']}/{n_episodes})")
    print(f"  Mean Steps (All):      {summary['mean_steps_all']} steps")
    if summary['mean_steps_on_success'] is not None:
        print(f"  Mean Steps (Success):  {summary['mean_steps_on_success']} steps")
    print(f"  Mean Peak Force:       {summary['mean_peak_force']} N (Max: {summary['max_peak_force']} N)")
    print(f"  Mean RMS Force:        {summary['mean_rms_force']} N")
    print(f"  Jamming / Wedging:     {summary['jamming_rate']}%")
    print(f"  Evaluation Time:       {summary['total_time_sec']}s ({summary['fps']} steps/s)")

    return {
        "summary": summary,
        "episodes": results,
    }


def generate_evaluation_plots(eval_data, save_path="results/evaluation_metrics.png"):
    """
    Generates a 4-panel publication-ready comparison plot:
    1. Success Rate (%)
    2. Peak & RMS Contact Force (N)
    3. Horizon / Steps to Completion
    4. Jamming & Wedging Frequency (%)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Contact-Rich Nut Assembly: Policy Evaluation Comparison", fontsize=16, fontweight="bold", y=0.98)

    model_names = []
    success_rates = []
    mean_peaks = []
    mean_rms = []
    jam_rates = []
    colors = []
    all_step_data = []

    for m_key in ["bc_no_ft", "bc_with_ft", "bc_rnn_with_ft"]:
        if m_key not in eval_data:
            continue
        data = eval_data[m_key]
        s = data["summary"]
        model_names.append(s["name"].replace("Model ", "M"))
        success_rates.append(s["success_rate"])
        mean_peaks.append(s["mean_peak_force"])
        mean_rms.append(s["mean_rms_force"])
        jam_rates.append(s["jamming_rate"])
        colors.append(MODELS[m_key]["color"])

        # Steps data
        for ep in data["episodes"]:
            all_step_data.append({
                "Model": s["name"].replace("Model ", "M"),
                "Steps": ep["steps"],
                "Success": "Success" if ep["success"] else "Failure",
            })

    # Panel 1: Success Rate (%)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_names, success_rates, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_title("Task Success Rate (%) [Higher is Better]", fontweight="bold", pad=12)
    ax1.set_ylabel("Success Rate (%)", fontweight="bold")
    ax1.set_ylim(0, 105)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    # Panel 2: Peak & RMS Contact Force (N)
    ax2 = axes[0, 1]
    x_pos = np.arange(len(model_names))
    width = 0.35
    b_peak = ax2.bar(x_pos - width / 2, mean_peaks, width, label="Mean Peak Force (N)", color="#e74c3c", edgecolor="black", linewidth=1.1)
    b_rms = ax2.bar(x_pos + width / 2, mean_rms, width, label="Mean RMS Force (N)", color="#3498db", edgecolor="black", linewidth=1.1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names)
    ax2.set_title("Contact Force Intensity (N) [Lower is Safer]", fontweight="bold", pad=12)
    ax2.set_ylabel("Force Magnitude (N)", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    for bar in b_peak:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}N", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in b_rms:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}N", ha="center", va="bottom", fontsize=10)

    # Panel 3: Steps to Completion (Box Plot / Stripplot)
    ax3 = axes[1, 0]
    import pandas as pd
    if all_step_data:
        df_steps = pd.DataFrame(all_step_data)
        sns.boxplot(data=df_steps, x="Model", y="Steps", hue="Model", ax=ax3, palette=colors, width=0.45, legend=False, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df_steps, x="Model", y="Steps", ax=ax3, color="black", alpha=0.4, size=5, jitter=0.2)
    ax3.set_title("Episode Horizon Distribution (Steps)", fontweight="bold", pad=12)
    ax3.set_ylabel("Total Steps (Max 400)", fontweight="bold")
    ax3.set_ylim(0, 420)

    # Panel 4: Jamming / Wedging Rate (%)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(model_names, jam_rates, color=["#e67e22", "#d35400", "#c0392b"], width=0.55, edgecolor="black", linewidth=1.2)
    ax4.set_title("Jamming / Wedging Rate (%) [Lower is Better]", fontweight="bold", pad=12)
    ax4.set_ylabel("Jamming Rate (%)", fontweight="bold")
    ax4.set_ylim(0, max(100, max(jam_rates) + 15) if jam_rates else 100)
    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n[Figure Saved] Evaluation metrics plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Closed-Loop Policy Evaluation for Contact-Rich BC")
    parser.add_argument("--models", type=str, default="all", choices=["all", "bc_no_ft", "bc_with_ft", "bc_rnn_with_ft"],
                        help="Which model(s) to evaluate (default: all)")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of rollouts per model (default: 50)")
    parser.add_argument("--horizon", type=int, default=400, help="Maximum steps per episode (default: 400)")
    parser.add_argument("--seed_start", type=int, default=42, help="Base seed for reproducibility (default: 42)")
    parser.add_argument("--video_episodes", type=int, default=1, help="Number of rollouts to record as MP4 videos (default: 1)")
    parser.add_argument("--save_results", type=str, default="results/evaluation_results.json",
                        help="Path to save evaluation JSON results")
    parser.add_argument("--plot", type=str, default="results/evaluation_metrics.png",
                        help="Path to save comparison plots")
    args = parser.parse_args()

    models_to_eval = list(MODELS.keys()) if args.models == "all" else [args.models]

    all_eval_data = {}

    # If results file already exists, load existing data so partial runs can be merged
    if os.path.isfile(args.save_results):
        try:
            with open(args.save_results, "r") as f:
                all_eval_data = json.load(f)
        except Exception:
            all_eval_data = {}

    start_all = time.time()

    for m_key in models_to_eval:
        res = evaluate_model(
            model_key=m_key,
            n_episodes=args.n_episodes,
            horizon=args.horizon,
            seed_start=args.seed_start,
            video_episodes=args.video_episodes,
            video_dir="results",
        )
        all_eval_data[m_key] = res

        # Save incremental JSON after each model
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(all_eval_data, f, indent=4)
        print(f"  [Saved Incremental Results] {args.save_results}")

    # Generate comparison plots
    generate_evaluation_plots(all_eval_data, save_path=args.plot)

    total_eval_time = time.time() - start_all

    # Print final comparative summary table
    print("\n" + "=" * 80)
    print("                    FINAL COMPARATIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<28} | {'Success':<10} | {'Mean Steps':<11} | {'Mean Peak F':<12} | {'Jam Rate':<9}")
    print("-" * 80)
    for m_key in ["bc_no_ft", "bc_with_ft", "bc_rnn_with_ft"]:
        if m_key in all_eval_data:
            s = all_eval_data[m_key]["summary"]
            print(f"{s['name']:<28} | {s['success_rate']:>5.1f}%     | {s['mean_steps_all']:>7.1f}   | {s['mean_peak_force']:>7.1f} N   | {s['jamming_rate']:>5.1f}%")
    print("=" * 80)
    print(f"Total Evaluation Pipeline Runtime: {total_eval_time / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
