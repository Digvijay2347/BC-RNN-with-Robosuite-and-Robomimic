"""
Interactive 3D Visualizer for Franka Emika Panda in Robosuite.
Launches an interactive MuJoCo GUI window with unlocked Free Camera controls.

Mouse & Keyboard Controls:
- Left-Click + Drag  : Orbit / Rotate camera around the workcell
- Right-Click + Drag : Pan camera vertically & horizontally
- Shift + Right-Click: Pan camera laterally
- Scroll Wheel       : Zoom in / Zoom out
- Spacebar           : Pause / Resume simulation
- Tab                : Toggle MuJoCo settings sidebar
- F1                 : Display built-in MuJoCo help HUD
- [ / ]              : Cycle between free camera and fixed camera angles
"""

import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import robosuite as suite


def run_interactive_viewer():
    print("=" * 72)
    print("  LAUNCHING FULLY INTERACTIVE FRANKA PANDA 3D VIEWER")
    print("=" * 72)
    print("Camera Navigation (Unlocked Free-Cam):")
    print("  - Left-Click + Drag        : Orbit / Rotate camera around workcell")
    print("  - Right-Click + Drag       : Pan camera (vertical/horizontal)")
    print("  - Shift + Right-Click Drag : Pan camera horizontally")
    print("  - Scroll Wheel             : Zoom in / Zoom out")
    print("  - Spacebar                 : Pause / Resume robot motion")
    print("  - Tab                      : Toggle MuJoCo sidebar UI")
    print("  - F1                       : Open built-in Help Overlay")
    print("=" * 72)

    # 1. Initialize environment with on-screen renderer and free camera
    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,  # Use Free Camera mode
        control_freq=20,
    )

    obs = env.reset()

    # 2. Unlock free camera (camera_id = -1 -> cam.type = 0 Free Camera)
    if hasattr(env, "viewer") and env.viewer is not None:
        env.viewer.set_camera(-1)
        env.viewer.update()

        # Set optimal initial viewing distance and angle looking at the peg and nut
        viewer = env.viewer.viewer
        if hasattr(viewer, "cam"):
            viewer.cam.type = 0  # mjCAMERA_FREE
            viewer.cam.lookat = [0.0, 0.0, 0.90]  # center on the table peg & nut
            viewer.cam.distance = 1.65             # ideal zoom distance
            viewer.cam.azimuth = 145.0             # 3/4 perspective angle
            viewer.cam.elevation = -22.0           # slight top-down angle

    print("\nViewer window is open! You can now freely rotate, pan, and zoom.")
    print("Close the 3D window or press Ctrl+C in this terminal to exit.\n")

    step = 0
    try:
        # Loop while the interactive window is open
        while True:
            if hasattr(env, "viewer") and env.viewer is not None and env.viewer.viewer is not None:
                if not env.viewer.viewer.is_running():
                    break

            # Smooth oscillatory demo motion
            action = np.zeros(env.action_dim)
            t = step * 0.04
            action[0] = 0.12 * np.sin(t)         # dx
            action[1] = 0.12 * np.cos(t)         # dy
            action[2] = -0.04 * np.sin(t * 0.5)  # dz
            
            # Periodically open and close gripper
            action[6] = 1.0 if (step // 50) % 2 == 0 else -1.0

            obs, reward, done, info = env.step(action)
            env.viewer.update()
            time.sleep(0.02)  # ~50 FPS
            step += 1

    except KeyboardInterrupt:
        print("\nViewer stopped by user.")
    finally:
        env.close()
        print("Viewer closed cleanly.")


if __name__ == "__main__":
    run_interactive_viewer()
