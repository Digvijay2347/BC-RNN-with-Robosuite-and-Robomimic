"""
Render high-resolution snapshots of the Franka Emika Panda arm in the
NutAssemblySquare environment. Saves images to results/ for viewing.
"""

import os
import sys
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import robosuite as suite


def capture_snapshots():
    os.makedirs("results", exist_ok=True)
    print("Initializing environment to capture high-res Franka Panda views...")

    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "agentview", "sideview"],
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
    )

    obs = env.reset()

    # Robosuite camera images are returned upside-down from OpenGL convention, so flip vertically
    cameras = ["frontview", "agentview", "sideview"]
    for cam in cameras:
        img_key = f"{cam}_image"
        if img_key in obs:
            raw_img = obs[img_key]
            # Flip upside down to correct orientation
            img_arr = np.flipud(raw_img)
            out_path = os.path.join("results", f"panda_{cam}.png")
            Image.fromarray(img_arr).save(out_path)
            print(f"  [SAVED] {cam:<12} -> {out_path} (512x512)")

    env.close()
    print("\nSnapshots captured successfully in the results/ folder!")


if __name__ == "__main__":
    capture_snapshots()
