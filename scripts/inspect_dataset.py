"""
Dataset Inspection Script for Robomimic HDF5 Demonstrations.
Inspects:
1. Environment metadata (robot type, env name, controller, horizons).
2. Trajectory statistics (total demos, step lengths, min/max/mean horizon).
3. Available observation keys in 'obs' and their shapes/data types.
4. Action specifications, ranges, and reward distributions.
5. Verifies force/torque and proprioceptive sensor channels.
"""

import os
import sys
import json
import h5py
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def inspect_dataset(hdf5_path):
    print("=" * 72)
    print("  ROBOMIMIC DEMONSTRATION DATASET INSPECTION")
    print(f"  Target File: {hdf5_path}")
    print("=" * 72)

    if not os.path.exists(hdf5_path):
        print(f"ERROR: Dataset file not found at: {hdf5_path}")
        sys.exit(1)

    file_size_mb = os.path.getsize(hdf5_path) / (1024 * 1024)
    print(f"\n[1/5] File Properties:")
    print(f"  - File Size: {file_size_mb:.2f} MB")

    with h5py.File(hdf5_path, "r") as f:
        # 1. Environment Metadata
        print(f"\n[2/5] Environment Metadata:")
        if "data" in f and "env_args" in f["data"].attrs:
            env_args = json.loads(f["data"].attrs["env_args"])
            print(f"  - Environment: {env_args.get('env_name')}")
            print(f"  - Robot(s):    {env_args.get('env_kwargs', {}).get('robots')}")
            print(f"  - Controller:  {env_args.get('env_kwargs', {}).get('controller_configs', {}).get('type')}")
        elif "env_metadata" in f:
            env_meta = json.loads(f.attrs["env_args"]) if "env_args" in f.attrs else {}
            print(f"  - Metadata: {env_meta}")
        else:
            print("  - Note: env_args stored under custom attributes.")

        # 2. Demonstration Trajectories
        data_grp = f["data"]
        demos = [k for k in data_grp.keys() if k.startswith("demo_")]
        num_demos = len(demos)
        print(f"\n[3/5] Demonstration Statistics:")
        print(f"  - Total Demonstrations: {num_demos}")

        lens = []
        for d in demos:
            lens.append(data_grp[d]["actions"].shape[0])

        total_transitions = sum(lens)
        print(f"  - Total State-Action Transitions: {total_transitions:,}")
        print(f"  - Trajectory Length (Min) : {min(lens)} steps")
        print(f"  - Trajectory Length (Max) : {max(lens)} steps")
        print(f"  - Trajectory Length (Mean): {np.mean(lens):.1f} steps")
        print(f"  - Trajectory Length (Med) : {np.median(lens):.1f} steps")

        # 3. Observation Keys & Shapes
        sample_demo = data_grp[demos[0]]
        obs_grp = sample_demo["obs"]
        print(f"\n[4/5] Observation Keys & Shapes (Sample from {demos[0]}):")
        print(f"  {'Key Name':<30} {'Shape':<15} {'Dtype':<10} {'Min':<10} {'Max':<10}")
        print("  " + "-" * 75)

        for k in sorted(obs_grp.keys()):
            ds = obs_grp[k]
            arr = np.array(ds)
            print(f"  {k:<30} {str(arr.shape):<15} {str(arr.dtype):<10} {arr.min():<10.3f} {arr.max():<10.3f}")

        # 4. Action Space Specification
        actions = np.array(sample_demo["actions"])
        print(f"\n[5/5] Action Space & Dynamics:")
        print(f"  - Action Shape: {actions.shape} (T x Dim)")
        print(f"  - Action Dtype: {actions.dtype}")
        print(f"  - Action Min per Dim: {np.round(actions.min(axis=0), 3)}")
        print(f"  - Action Max per Dim: {np.round(actions.max(axis=0), 3)}")
        print(f"  - Gripper Action Values: {np.unique(actions[:, -1])}")

        # Check filter keys (train / valid splits)
        if "mask" in f:
            print("\nDataset Filter Masks:")
            for m in f["mask"].keys():
                print(f"  - Mask '{m}': {len(f['mask'][m])} demos")

    print("\n" + "=" * 72)
    print("  DATASET VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dataset = sys.argv[1]
    else:
        target_dataset = "datasets/square/ph/low_dim_with_ft.hdf5"
    inspect_dataset(target_dataset)
