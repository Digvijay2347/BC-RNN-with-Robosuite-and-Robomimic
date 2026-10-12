"""
Sanity Check Script for Robosuite & Robomimic Setup.
Verifies that:
1. Robosuite NutAssemblySquare environment instantiates with Panda robot.
2. Robomimic EnvRobosuite wrapper correctly formats low-dimensional observations.
3. Proprioception, object, and Force/Torque sensor keys (robot0_eef_force, robot0_eef_torque) are active.
4. Stepping the environment produces valid physics updates without error.
"""

import sys
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import robosuite as suite
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_robosuite import EnvRobosuite


def run_sanity_check():
    print("=" * 70)
    print("  RUNNING ROBOSUITE & ROBOMIMIC FORCE/TORQUE SENSOR SANITY CHECK")
    print("=" * 70)

    # 1. Initialize Robomimic Observation Specifications
    print("\n[1/4] Configuring Robomimic observation modalities...")
    required_low_dim_keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_eef_force",
        "robot0_eef_torque",
        "object",
    ]
    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "low_dim": required_low_dim_keys
        }
    })

    # 2. Instantiate Robomimic EnvRobosuite wrapper for NutAssemblySquare
    print("\n[2/4] Instantiating EnvRobosuite('NutAssemblySquare', robots='Panda')...")
    env = EnvRobosuite(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # 3. Reset and Validate Observation Keys
    print("\n[3/4] Resetting environment and validating observation keys...")
    obs = env.reset()

    all_keys_valid = True
    for key in required_low_dim_keys:
        if key in obs and obs[key] is not None:
            val = np.asarray(obs[key])
            sample_str = np.array2string(np.round(val, 3), precision=3, suppress_small=True)
            print(f"  [FOUND] {key:<22} shape: {val.shape}, sample: {sample_str}")
        else:
            print(f"  [MISSING] {key}")
            all_keys_valid = False

    if not all_keys_valid:
        print("\nERROR: One or more required observation keys were not found.")
        sys.exit(1)

    # 4. Simulation Rollout Test
    print("\n[4/4] Executing 10 simulation control steps...")
    for step in range(1, 11):
        # Action: [dx, dy, dz, droll, dpitch, dyaw, grasp]
        action = np.zeros(env.action_dimension)
        action[2] = -0.05  # slight downward velocity towards table
        
        obs, reward, done, info = env.step(action)
        force = obs["robot0_eef_force"]
        torque = obs["robot0_eef_torque"]
        force_norm = float(np.linalg.norm(force))
        torque_norm = float(np.linalg.norm(torque))
        z_pos = float(obs["robot0_eef_pos"][2])

        print(f"  Step {step:02d} | EEF z-pos: {z_pos:.4f} m | ||F||: {force_norm:.3f} N | ||Torque||: {torque_norm:.3f} Nm")

    print("\n" + "=" * 70)
    print("  ALL SANITY CHECKS PASSED: Robosuite + Robomimic F/T Pipeline Ready!")
    print("=" * 70)


if __name__ == "__main__":
    run_sanity_check()
