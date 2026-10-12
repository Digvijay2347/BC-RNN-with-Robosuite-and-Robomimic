"""
Human Demonstration 3D Playback Script.
Replays a recorded human demonstration trajectory from the HDF5 dataset
inside the interactive MuJoCo 3D simulator window with real-time force display.

Usage:
  # Play demo_0 in interactive 3D window:
  python scripts/playback_demo.py --demo demo_0

  # Play another demo (e.g. demo_5):
  python scripts/playback_demo.py --demo demo_5

  # Export demonstration to MP4 video:
  python scripts/playback_demo.py --demo demo_0 --video results/demo_0_playback.mp4
"""

import os
import sys
import time
import argparse
import h5py
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import robosuite as suite
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


def playback_demonstration(hdf5_path, demo_name="demo_0", video_path=None):
    print("=" * 72)
    print(f"  REPLAYING HUMAN DEMONSTRATION: {demo_name}")
    print(f"  Dataset: {hdf5_path}")
    print("=" * 72)

    if not os.path.exists(hdf5_path):
        print(f"ERROR: Dataset not found: {hdf5_path}")
        return

    # Extract trajectory actions and states
    with h5py.File(hdf5_path, "r") as f:
        if f"data/{demo_name}" not in f:
            print(f"ERROR: Trajectory {demo_name} not found in dataset!")
            return

        demo_grp = f[f"data/{demo_name}"]
        actions = np.array(demo_grp["actions"])
        states = np.array(demo_grp["states"])
        T = actions.shape[0]

        env_args = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)

    has_video = video_path is not None
    if has_video:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        import imageio
        video_writer = imageio.get_writer(video_path, fps=20)
        print(f"Recording video to: {video_path}")

    # Create environment with matching OSC_POSE controller
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=(not has_video),
        has_offscreen_renderer=has_video,
        use_camera_obs=has_video,
        camera_names=["frontview"] if has_video else None,
        camera_heights=512 if has_video else None,
        camera_widths=512 if has_video else None,
        control_freq=20,
    )

    # Initialize environment to the demonstration's initial simulator state
    env.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    if not has_video and hasattr(env, "viewer") and env.viewer is not None:
        env.viewer.set_camera(-1)  # Free camera
        env.viewer.update()
        viewer = env.viewer.viewer
        if hasattr(viewer, "cam"):
            viewer.cam.type = 0
            viewer.cam.lookat = [0.0, 0.0, 0.90]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -22.0

    print(f"\nReplaying {T} timesteps (~{T/20:.1f}s)...")
    print(f"Watch the Franka Panda grasp the square nut and insert it onto the peg!\n")

    try:
        for t in range(T):
            obs, reward, done, info = env.step(actions[t])
            
            force = env.robots[0].ee_force
            f_arr = force["right"] if isinstance(force, dict) else force
            f_norm = np.linalg.norm(f_arr)
            z_pos = obs["robot0_eef_pos"][2]

            # Status log every 10 steps
            if (t + 1) % 15 == 0 or t == T - 1:
                print(f"  Step {t+1:03d}/{T} | Height: {z_pos:.4f} m | Contact Force: {f_norm:6.2f} N")

            if has_video:
                img = obs["frontview_image"]
                video_writer.append_data(np.flipud(img))
            else:
                env.viewer.update()
                time.sleep(0.04)  # ~25 FPS replay speed

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        if has_video:
            video_writer.close()
            print(f"\n[SAVED] Video exported successfully to: {video_path}")
        env.close()
        print("\nReplay finished cleanly!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay human demonstration in 3D.")
    parser.add_argument("--demo", type=str, default="demo_0", help="Demo ID (e.g. demo_0, demo_10)")
    parser.add_argument("--dataset", type=str, default="datasets/square/ph/low_dim_with_ft.hdf5", help="Path to HDF5")
    parser.add_argument("--video", type=str, default=None, help="Optional output MP4 video path")
    args = parser.parse_args()

    playback_demonstration(hdf5_path=args.dataset, demo_name=args.demo, video_path=args.video)
