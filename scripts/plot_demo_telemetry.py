"""
Telemetry Plotter for Human Demonstration Trajectories.
Visualizes:
1. End-Effector 3D trajectory (X, Y, Z position).
2. 3-Axis Contact Forces (Fx, Fy, Fz) and Net Force Magnitude ||F||.
3. 3-Axis Contact Torques (Tx, Ty, Tz).
4. Gripper finger displacement & action commands.
Saves a publication-quality plot into results/ for visual inspection.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def plot_demonstration_telemetry(hdf5_path, demo_id="demo_0", save_path="results/demo_0_telemetry.png"):
    print("=" * 70)
    print(f"  PLOTTING TELEMETRY FOR {demo_id} FROM {hdf5_path}")
    print("=" * 70)

    if not os.path.exists(hdf5_path):
        print(f"ERROR: Dataset not found at: {hdf5_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        if f"data/{demo_id}" not in f:
            print(f"ERROR: {demo_id} not found in dataset!")
            return

        demo = f[f"data/{demo_id}"]
        obs = demo["obs"]
        actions = np.array(demo["actions"])
        
        # Extract signals
        eef_pos = np.array(obs["robot0_eef_pos"])      # (T, 3)
        forces = np.array(obs["robot0_eef_force"])      # (T, 3)
        torques = np.array(obs["robot0_eef_torque"])    # (T, 3)
        gripper = np.array(obs["robot0_gripper_qpos"])  # (T, 2)
        T = eef_pos.shape[0]
        timesteps = np.arange(T)

        force_norm = np.linalg.norm(forces, axis=-1)
        torque_norm = np.linalg.norm(torques, axis=-1)

    # 1. Print structured phase breakdown to console
    print(f"\nDemonstration Summary for {demo_id}:")
    print(f"  - Total Timesteps: {T} steps (~{T/20:.2f} seconds at 20 Hz)")
    print(f"  - Peak Force ||F||: {np.max(force_norm):.2f} N")
    print(f"  - Mean Force ||F||: {np.mean(force_norm):.2f} N")
    print(f"  - Peak Torque ||T||: {np.max(torque_norm):.2f} Nm")
    print(f"  - Initial EEF Z-height: {eef_pos[0, 2]:.3f} m")
    print(f"  - Final Insertion Z-height: {eef_pos[-1, 2]:.3f} m")

    # Detect contact phase (when force exceeds 10 N threshold)
    contact_steps = np.where(force_norm > 10.0)[0]
    if len(contact_steps) > 0:
        first_contact = contact_steps[0]
        print(f"  - Initial Peg Contact detected at step: {first_contact} (t = {first_contact/20:.2f}s)")
    else:
        first_contact = T // 2

    # 2. Generate 4-panel matplotlib visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Demonstration Telemetry Breakdown: {demo_id} (Nut Assembly Square)", fontsize=14, fontweight="bold")

    # Subplot 1: End-Effector 3D Position
    ax1 = axes[0]
    ax1.plot(timesteps, eef_pos[:, 0], label="EEF X (m)", color="#1f77b4", lw=2)
    ax1.plot(timesteps, eef_pos[:, 1], label="EEF Y (m)", color="#2ca02c", lw=2)
    ax1.plot(timesteps, eef_pos[:, 2], label="EEF Z (height, m)", color="#d62728", lw=2.5)
    ax1.axvline(x=first_contact, color="gray", linestyle="--", alpha=0.7, label="Peg Contact Phase")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("1. End-Effector Cartesian Trajectory (Approach -> Grasp -> Transport -> Insertion)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Subplot 2: Contact Forces
    ax2 = axes[1]
    ax2.plot(timesteps, forces[:, 0], label="Force Fx (N)", color="#17becf", alpha=0.7)
    ax2.plot(timesteps, forces[:, 1], label="Force Fy (N)", color="#bcbd22", alpha=0.7)
    ax2.plot(timesteps, forces[:, 2], label="Force Fz (Vertical, N)", color="#ff7f0e", lw=1.8)
    ax2.plot(timesteps, force_norm, label="Net Force ||F|| (N)", color="#d62728", lw=2)
    ax2.axvline(x=first_contact, color="gray", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Force (N)")
    ax2.set_title("2. Wrist Force Sensing (Spikes denote peg contact & alignment resistance)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9)

    # Subplot 3: Contact Torques
    ax3 = axes[2]
    ax3.plot(timesteps, torques[:, 0], label="Torque Tx (Nm)", color="#9467bd", alpha=0.7)
    ax3.plot(timesteps, torques[:, 1], label="Torque Ty (Nm)", color="#8c564b", alpha=0.7)
    ax3.plot(timesteps, torques[:, 2], label="Torque Tz (Nm)", color="#e377c2", alpha=0.7)
    ax3.plot(timesteps, torque_norm, label="Net Torque ||T|| (Nm)", color="#1f77b4", lw=2)
    ax3.axvline(x=first_contact, color="gray", linestyle="--", alpha=0.7)
    ax3.set_ylabel("Torque (Nm)")
    ax3.set_title("3. Wrist Torque Sensing (Rotational moments during hole alignment)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.9)

    # Subplot 4: Gripper State & Action Command
    ax4 = axes[3]
    ax4.plot(timesteps, gripper[:, 0], label="Finger 1 Width (m)", color="#2ca02c", lw=2)
    ax4.plot(timesteps, gripper[:, 1], label="Finger 2 Width (m)", color="#98df8a", lw=2)
    ax4.plot(timesteps, actions[:, -1], label="Gripper Action (+1 Close / -1 Open)", color="#7f7f7f", linestyle=":", lw=2)
    ax4.axvline(x=first_contact, color="gray", linestyle="--", alpha=0.7)
    ax4.set_xlabel("Simulation Timestep (20 Hz, 50 ms/step)")
    ax4.set_ylabel("Gripper Width / Action")
    ax4.set_title("4. Gripper State (Nut grasp confirmation)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n[SAVED] Visual Telemetry Plot: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    demo_to_plot = sys.argv[1] if len(sys.argv) > 1 else "demo_0"
    dataset_path = "datasets/square/ph/low_dim_with_ft.hdf5"
    plot_demonstration_telemetry(dataset_path, demo_id=demo_to_plot)
