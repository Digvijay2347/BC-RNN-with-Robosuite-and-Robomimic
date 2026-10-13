# BC-RNN with Robosuite and Robomimic: Complete Project Guide & Runbook

**Repository**: [github.com/Digvijay2347/BC-RNN-with-Robosuite-and-Robomimic](https://github.com/Digvijay2347/BC-RNN-with-Robosuite-and-Robomimic)  
**Author**: Digvijay Singh  
**Workspace**: `c:\Users\Digvijay singh\OneDrive\Desktop\Motion\prod1`  
**Task**: Contact-Rich Peg/Nut Assembly Manipulation (`NutAssemblySquare`)  
**Core Question**: Benchmarking Behavioral Cloning (BC) vs. Recurrent Behavioral Cloning (BC-RNN) on CPU with & without 6-axis End-Effector Force/Torque (F/T) History.

---

## 📑 Table of Contents
1. [Project Overview & Scientific Motivation](#1-project-overview--scientific-motivation)
2. [Robot Embodiment Analysis: Franka Panda vs. UR10e / UR5e](#2-robot-embodiment-analysis-franka-panda-vs-ur10e--ur5e)
3. [System Hardware Profile](#3-system-hardware-profile)
4. [Environment Setup & Windows Compatibility Fixes](#4-environment-setup--windows-compatibility-fixes)
5. [Git Practices & Repository Structure](#5-git-practices--repository-structure)
6. [Complete Command Reference (Everything Run So Far)](#6-complete-command-reference-everything-run-so-far)
7. [Phase-by-Phase Technical Details](#7-phase-by-phase-technical-details)
   - [Phase 1: Environment Setup & Sanity Check](#phase-1-environment-setup--sanity-check)
   - [Phase 2: Dataset Acquisition & F/T Extraction](#phase-2-dataset-acquisition--ft-extraction)
   - [Phase 3: Configuration Generation](#phase-3-configuration-generation)
   - [Phase 4: CPU-Optimized Training Pipeline](#phase-4-cpu-optimized-training-pipeline)
8. [Current Deliverables & Artifacts Index](#8-current-deliverables--artifacts-index)
9. [Next Steps: Phase 5 (Closed-Loop Evaluation)](#9-next-steps-phase-5-closed-loop-evaluation)

---

## 1. Project Overview & Scientific Motivation

In tight-tolerance contact-rich assembly tasks (such as peg-in-hole and nut assembly), visual drift and kinematic positioning errors cause rigid parts to jam or wedge. 

### Core Research Questions
1. **Does tactile/force sensing improve behavioral cloning?** Can the policy utilize wrist force/torque sensor readings to detect contact transitions and compliant insertion?
2. **Does recurrence (BC-RNN) outperform memoryless policies (BC-MLP)?** Does modeling the temporal sequence of contact forces ($dF/dt$) overcome high-frequency impact noise present in instantaneous single-step readings?

### Experimental Conditions (Ablation Matrix)

| Condition | Architecture | Input Observations | Sequence Horizon | Core Hypothesis |
| :--- | :--- | :--- | :---: | :--- |
| **Model A (Baseline)** | Standard BC (MLP) | EE Pos + Quat + Gripper + Object (23 dims) | $H=1$ (1 step) | Position-only baseline; vulnerable to contact jamming. |
| **Model B (Instant F/T)** | Standard BC (MLP) | EE Pos + Quat + Gripper + Object + **F/T** (29 dims) | $H=1$ (1 step) | Can react to contact thresholds, but sensitive to raw impact noise. |
| **Model C (F/T History)** | **BC-RNN (LSTM)** | EE Pos + Quat + Gripper + Object + **F/T** (29 dims) | **$H=10$ (Sequence)** | Learns force rate-of-change ($dF/dt$) and achieves lowest loss. |

---

## 2. Robot Embodiment Analysis: Franka Panda vs. UR10e / UR5e

Prior to execution, an in-depth analysis of ready-made demonstration datasets and simulation support was conducted:

| Question | Finding | Technical Rationale |
| :--- | :---: | :--- |
| **UR10e in Robosuite?** | ❌ No native support | `robosuite` only supports UR5e natively (`robots="UR5e"`). UR10e requires custom MJCF import from DeepMind's MuJoCo Menagerie. |
| **UR10e / UR5e Demos in Robomimic?** | ❌ None available | All official `robomimic` benchmark datasets for Nut Assembly (`square`) were recorded exclusively on the **7-DoF Franka Emika Panda** arm. |
| **Decision** | ✅ **Franka Panda** | Using the canonical Franka Panda dataset (`square/ph/low_dim`) provides 200 turnkey human demonstrations, bit-exact kinematics, and verified MuJoCo F/T sensor streams without kinematics retargeting errors. |

---

## 3. System Hardware Profile

The current host system detected and used for this project:

| Hardware Component | Detected Specification | Project Suitability |
| :--- | :--- | :---: |
| **Processor (CPU)** | **Intel Core Ultra 7 256V** (8 cores / 8 threads, up to 4.8 GHz) | 🚀 High-IPC Lunar Lake, perfect for single-thread MuJoCo physics |
| **System Memory (RAM)** | **15.66 GB Total** (~2.3 GB available at idle) | ✅ Peak project RAM usage is **~420 MB – 570 MB** (<25% of available) |
| **Disk Storage** | **508 GB Free** on `C:` (out of 952 GB) | ✅ Entire project footprint is < 2 GB |
| **Graphics (GPU)** | **Intel Arc 140V GPU (8 GB)** | ✅ Available, though CPU backend is prioritized for low-dim vector speed |
| **Operating System** | **Microsoft Windows 11 Home (64-bit)** | ✅ Fully configured with PowerShell |

---

## 4. Environment Setup & Windows Compatibility Fixes

Because robotics simulation suites (`mujoco`, `robosuite`, `robomimic`) are historically developed for Linux, several targeted fixes were engineered to ensure **100% native stability on Windows**:

1. **Python Virtual Environment**:
   * Isolated Python 3.11 environment in `prod1\venv`.
2. **CPU-Optimized PyTorch**:
   * Installed `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu` via the official PyTorch CPU wheel index (saving 2.5 GB of CUDA download bloat).
3. **MuJoCo Version Pinning (`mujoco==3.2.7`)**:
   * MuJoCo 3.10+ removed the sparse mass matrix attribute (`qM`) in favor of CSR (`M`). `robosuite` uses `qM` for operational space control. Pinned `mujoco==3.2.7` to maintain bit-exact controller stability.
4. **Windows WGL Backend Patch**:
   * Fixed `robosuite.utils.binding_utils`: Replaced the hardcoded Linux `MUJOCO_GL="egl"` assignment with `MUJOCO_GL="wgl"` on Windows, and dynamically resolved `mujoco.dll`.
5. **Robosuite 1.4.1 Alignment**:
   * Pinned `robosuite==1.4.1` to match the exact environment version used by the robomimic demonstration dataset (`low_dim_v141.hdf5`), avoiding site-name mismatches (`gripper0_ft_frame` vs `robot0_right_center`).
6. **Robomimic Modern MuJoCo Patch**:
   * Replaced the deprecated `mujoco_py` import in `robomimic.envs.env_robosuite` with official Google DeepMind `mujoco` bindings.
7. **Force/Torque Channel Extraction**:
   * Enhanced `EnvRobosuite.get_observation()` to automatically query `robot.ee_force` and `robot.ee_torque` and populate `robot0_eef_force` (3D) and `robot0_eef_torque` (3D) into all observation dictionaries.

---

## 5. Git Practices & Repository Structure

### Git Setup
* **Remote Repository**: `https://github.com/Digvijay2347/BC-RNN-with-Robosuite-and-Robomimic.git`
* **Default Branch**: `main`
* **Author Config**: Digvijay Singh (`90893525+Digvijay2347@users.noreply.github.com`)
* **Robotics [`.gitignore`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/.gitignore)**:
  * Excludes large datasets (`*.hdf5`, `*.h5`, `datasets/`).
  * Excludes heavy model checkpoints and tensorboard runs (`*.pth`, `*.pt`, `models/`, `runs/`).
  * Excludes virtual environments (`venv/`, `__pycache__/`).

### Workspace Layout
```text
prod1/
├── configs/
│   ├── config_bc_no_ft.json        # Model A config (MLP, no F/T)
│   ├── config_bc_with_ft.json      # Model B config (MLP, instant F/T)
│   └── config_bc_rnn_with_ft.json  # Model C config (LSTM, F/T history H=10)
├── datasets/                       # (Ignored by git)
│   └── square/ph/
│       ├── low_dim_v141.hdf5       # Raw human demos (50.5 MB)
│       └── low_dim_with_ft.hdf5    # Re-simulated with 6-axis F/T (53.5 MB)
├── models/                         # (Ignored by git)
│   ├── bc_no_ft_best.pth           # Best checkpoint Model A
│   ├── bc_with_ft_best.pth         # Best checkpoint Model B
│   └── bc_rnn_with_ft_best.pth     # Best checkpoint Model C
├── results/                        # Visual graphs & telemetry
│   ├── demo_0_playback.mp4         # 512x512 video of human demonstration
│   ├── demo_0_telemetry.png        # 4-panel force/position telemetry plot
│   ├── panda_frontview.png         # High-res robot snapshot
│   ├── panda_agentview.png         # High-res overhead snapshot
│   ├── panda_sideview.png          # High-res side profile snapshot
│   ├── training_curves.png         # 2-panel loss curves (Train vs Valid)
│   ├── training_history.json       # Epoch-by-epoch loss metrics
│   └── training_summary.json       # Training timings and best epochs
├── scripts/
│   ├── sanity_check.py             # Phase 1: Environment & F/T tester
│   ├── render_snapshot.py          # Robot camera snapshot generator
│   ├── visualize_interactive.py    # Unlocked 3D Free-Cam GUI visualizer
│   ├── inspect_dataset.py          # Phase 2: HDF5 dataset validator
│   ├── plot_demo_telemetry.py      # Trajectory & force curve plotter
│   ├── playback_demo.py            # 3D demo replayer & MP4 video exporter
│   ├── generate_configs.py         # Phase 3: Experiment config builder
│   └── train.py                    # Phase 4: CPU training pipeline
├── venv/                           # Python 3.11 virtual environment
├── .gitignore
├── plan_contact_rich_bc_ft.md      # Initial technical plan
├── PROJECT_DOCUMENTATION.md        # Comprehensive project manual (this doc)
└── README.md                       # Public GitHub overview
```

---

## 6. Complete Command Reference (Everything Run So Far)

All commands are structured for Windows PowerShell inside `prod1`:

### 1. Environment & Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
.\venv\Scripts\python.exe -m pip install --upgrade pip

# Install CPU-optimized PyTorch
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install MuJoCo, Robosuite, Robomimic & data science libraries
.\venv\Scripts\python.exe -m pip install "mujoco==3.2.7" "robosuite==1.4.1" --no-deps
.\venv\Scripts\python.exe -m pip install robomimic --no-deps
.\venv\Scripts\python.exe -m pip install h5py matplotlib seaborn pandas tqdm tensorboard tensorboardX imageio "imageio[ffmpeg]" pynput termcolor numba scipy opencv-python

# Run environment sanity check
.\venv\Scripts\python.exe scripts/sanity_check.py
```

### 2. Robot Visualizations
```powershell
# Launch real-time interactive 3D visualizer (Unlocked Free-Cam mouse orbiting & panning)
.\venv\Scripts\python.exe scripts/visualize_interactive.py

# Render high-resolution (512x512) camera snapshots
.\venv\Scripts\python.exe scripts/render_snapshot.py
```

### 3. Dataset Acquisition & Telemetry Inspection
```powershell
# Download official Square proficient-human dataset (200 demos)
.\venv\Scripts\python.exe -m robomimic.scripts.download_datasets --download_dir datasets --tasks square --dataset_types ph --hdf5_types low_dim

# Extract synchronized 6-axis force/torque observations across all 200 demonstrations
.\venv\Scripts\python.exe -m robomimic.scripts.dataset_states_to_obs --dataset datasets/square/ph/low_dim_v141.hdf5 --output_name low_dim_with_ft.hdf5 --copy_rewards --copy_dones

# Inspect dataset structure, keys, and statistics
.\venv\Scripts\python.exe scripts/inspect_dataset.py datasets/square/ph/low_dim_with_ft.hdf5

# Plot 4-panel sensor telemetry for a demonstration (e.g. demo_0)
.\venv\Scripts\python.exe scripts/plot_demo_telemetry.py demo_0

# Replay demonstration in 3D simulator
.\venv\Scripts\python.exe scripts/playback_demo.py --demo demo_0

# Export demonstration replay to an MP4 video
.\venv\Scripts\python.exe scripts/playback_demo.py --demo demo_0 --video results/demo_0_playback.mp4
```

### 4. Configuration Generation & Model Training
```powershell
# Generate Phase 3 JSON configs for Model A, Model B, and Model C
.\venv\Scripts\python.exe scripts/generate_configs.py

# Train all 3 models sequentially on CPU (50 epochs each)
.\venv\Scripts\python.exe scripts/train.py --model all --epochs 50

# Train an individual model if desired
.\venv\Scripts\python.exe scripts/train.py --model bc_no_ft --epochs 50
.\venv\Scripts\python.exe scripts/train.py --model bc_with_ft --epochs 50
.\venv\Scripts\python.exe scripts/train.py --model bc_rnn_with_ft --epochs 50
```

---

## 7. Phase-by-Phase Technical Details

### Phase 1: Environment Setup & Sanity Check
* **Script**: [`scripts/sanity_check.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/scripts/sanity_check.py)
* **Verification**: Instantiated `NutAssemblySquare` with `Panda` arm. Verified that `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (2), `robot0_eef_force` (3), `robot0_eef_torque` (3), and `object` (14) are active. Stepped 10 simulation actions with active physics telemetry.

### Phase 2: Dataset Acquisition & F/T Extraction
* **Raw Download**: `datasets/square/ph/low_dim_v141.hdf5` (50.5 MB). Contains 200 proficient human demonstrations (30,154 transitions).
* **F/T Re-Simulation**: Ran `dataset_states_to_obs.py` to replay simulator states and extract wrist force/torque sensors.
* **Output File**: `datasets/square/ph/low_dim_with_ft.hdf5` (53.5 MB).
* **Physical Findings**: Contact forces remain low ($\sim 5\text{ N}$) during air movement and spike up to **$137.1\text{ N}$** during table contact and peg seating.

### Phase 3: Configuration Generation
* **Script**: [`scripts/generate_configs.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/scripts/generate_configs.py)
* Generated three validated robomimic JSON configurations in `configs/`:
  1. `config_bc_no_ft.json`: 23-dim input, MLP `[300, 400]`, $H=1$.
  2. `config_bc_with_ft.json`: 29-dim input, MLP `[300, 400]`, $H=1$.
  3. `config_bc_rnn_with_ft.json`: 29-dim input, 2-layer LSTM (dim: 400) + MLP, $H=10$.

### Phase 4: CPU-Optimized Training Pipeline
* **Script**: [`scripts/train.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/scripts/train.py)
* **Training Time**: Total pipeline completed in **12.4 minutes** on CPU.
* **Loss Summary Table**:

| Model | Architecture | Training Time | Initial Train Loss | Final Train Loss | Best Val Loss | Best Epoch |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** | **BC (No F/T)** | 47.1 s | 0.1922 | 0.0434 | 0.0440 | Epoch 43 |
| **Model B** | **BC (Instant F/T)** | 45.9 s | 0.1935 | 0.0516 | 0.0560 | Epoch 48 |
| **Model C** | **BC-RNN (F/T History)** | 10.8 min | 0.1826 | **0.0352** 🏆 | **0.0424** 🏆 | **Epoch 50** |

* **Key Takeaway**: Model C achieved the lowest training and validation errors. Passing force history through an LSTM allows the model to learn contact rate-of-change ($dF/dt$) rather than overfitting to instantaneous force spikes.

---

## 8. Current Deliverables & Artifacts Index

| Category | File | Description |
| :--- | :--- | :--- |
| **Loss Curves** | [results/training_curves.png](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/results/training_curves.png) | 2-panel plot comparing Train MSE and Validation MSE across all 50 epochs. |
| **Loss Telemetry** | [results/training_history.json](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/results/training_history.json) | Complete epoch-by-epoch loss values for all 3 models. |
| **Sensor Telemetry** | [results/demo_0_telemetry.png](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/results/demo_0_telemetry.png) | 4-panel graph: EEF Position, 3D Force, 3D Torque, and Gripper Width over time. |
| **Demonstration Video** | [results/demo_0_playback.mp4](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/results/demo_0_playback.mp4) | 512×512 MP4 video of the human demonstration nut insertion. |
| **Robot Views** | [results/panda_frontview.png](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod1/results/panda_frontview.png) | High-res camera image of the robot workcell and peg. |
| **Trained Checkpoints** | `models/bc_no_ft_best.pth` | Best policy weights for Model A. |
| | `models/bc_with_ft_best.pth` | Best policy weights for Model B. |
| | `models/bc_rnn_with_ft_best.pth` | Best policy weights for Model C. |

---

## 9. Next Steps: Phase 5 (Closed-Loop Evaluation)

In **Phase 5**, we evaluate the trained checkpoints inside the active simulation:
1. Run **50 test rollouts** per policy in `NutAssemblySquare`.
2. Compute:
   * **Task Success Rate (%)**: Percentage of successful nut insertions.
   * **Mean Insertion Steps / Completion Horizon**.
   * **Peak Contact Force (N)**: Assessing whether F/T history prevents high-force jamming or collision damage.
3. Generate final comparative bar charts and analysis report.
