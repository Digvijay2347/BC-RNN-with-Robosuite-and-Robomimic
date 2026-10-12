"""
Interactive 3D Visualizer for Franka Emika Panda in Robosuite.
Launches an interactive GUI window on your desktop where you can watch
the robot in real-time, rotate the camera, pan, and zoom.

Controls:
- Left-click + drag: Rotate camera
- Right-click + drag: Pan camera
- Scroll wheel: Zoom in / out
- Press ESC or close the window to exit
"""

import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import robosuite as suite


def run_interactive_viewer():
    print("=" * 70)
    print("  LAUNCHING INTERACTIVE FRANKA PANDA 3D VIEWER")
    print("=" * 70)
    print("Controls in the 3D window:")
    print("  - Left Click + Drag  : Rotate view")
    print("  - Right Click + Drag : Pan view")
    print("  - Scroll Wheel       : Zoom in / out")
    print("  - Close window / Ctrl+C to stop.")
    print("=" * 70)

    # Initialize environment with active on-screen renderer
    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="frontview",
        control_freq=20,
    )

    obs = env.reset()
    env.render()
    print("Viewer opened! Stepping robot arm in real time...")

    try:
        step = 0
        while True:
            # Generate a smooth oscillatory motion so you can see the arm articulating
            action = np.zeros(env.action_dim)
            t = step * 0.05
            action[0] = 0.15 * np.sin(t)        # dx
            action[1] = 0.15 * np.cos(t)        # dy
            action[2] = -0.05 * np.sin(t * 0.5) # dz
            
            # Periodically toggle gripper
            action[6] = 1.0 if (step // 60) % 2 == 0 else -1.0

            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.03)  # ~30 FPS rate
            step += 1

    except KeyboardInterrupt:
        print("\nViewer closed by user.")
    finally:
        env.close()
        print("Environment cleaned up successfully.")


if __name__ == "__main__":
    run_interactive_viewer()
